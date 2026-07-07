import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from functools import partial

import einops
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import einsum, nn
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from policy_models.projector import PerceiverResampler

from policy_models.edm_diffusion.score_wrappers import GCDenoiser
from policy_models.module.clip_lang_encoder import LangClip
from policy_models.edm_diffusion.gc_sampling import *
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from policy_models.module.Video_Former import Video_Former_2D,Video_Former_3D
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from policy_models.module.diffusion_extract import Diffusion_feature_extractor
from transformers import AutoTokenizer, CLIPTextModelWithProjection

_PROJECT_ROOT = Path(__file__).absolute().parents[1]
if _PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT.as_posix())

# Roots of the vendored hidden2dino / hidden2dpa decoupler packages.
# These directories also hold the pretrained run subfolders (runs_*/run_*) used at eval time.
HIDDEN2DINO_ROOT = _PROJECT_ROOT / "hidden2dino"
HIDDEN2DPA_ROOT = _PROJECT_ROOT / "hidden2dpa"

from hidden2dino.model_spail_tem_attention import HiddenToDinoModel, HiddenToDinoModelWithRef
from hidden2dpa.model_spail_tem_attention import HiddenToDA3Model, HiddenToDA3ModelWithRef


logger = logging.getLogger(__name__)


def _sanitize_st_decoupler_kwargs(model_kwargs: Dict, label: str) -> Dict:
    """Filter run metadata and adapt config keys to the ST decoupler constructors."""
    allowed = {"C_in", "C_out", "T", "H", "W", "hidden_dim", "num_layers", "num_heads", "dropout"}
    cleaned = {k: v for k, v in model_kwargs.items() if k in allowed}
    if "C_out" not in cleaned and "token_dim" in model_kwargs:
        cleaned["C_out"] = model_kwargs["token_dim"]
    if "C_out" not in cleaned:
        raise KeyError(f"{label} config is missing required output-dim key ('C_out' or 'token_dim').")
    return cleaned

def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    # Disable FlashAttention in all diffusers attention blocks by reverting to the
    # vanilla processor, which uses standard matmul kernels and is stable on H100.
    pipeline.unet.set_attn_processor(AttnProcessor())
    return pipeline, None, pipeline.feature_extractor, pipeline.scheduler, pipeline.video_processor, \
        pipeline.image_encoder, pipeline.vae, pipeline.unet


class VPP_Policy(pl.LightningModule):
    """
    The lightning module used for training.
    """

    def __init__(
            self,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            latent_dim: int = 512,
            multistep: int = 10,
            sampler_type: str = 'ddim',
            num_sampling_steps: int = 10,
            sigma_data: float = 0.5,
            sigma_min: float = 0.001,
            sigma_max: float = 80,
            noise_scheduler: str = 'exponential',
            sigma_sample_density_type: str = 'loglogistic',
            use_lr_scheduler: bool = True,
            act_window_size: int = 10,
            use_text_not_embedding: bool = False,
            seed: int = 42,
            pretrained_model_path: str = '',
            text_encoder_path: str = '',
            use_position_encoding: bool = True,
            use_gripper_features: bool = False,
            Former_depth: int = 3,
            Former_heads: int = 8,
            Former_dim_head: int = 64,
            Former_num_time_embeds: int = 1,
            num_latents: int = 3,
            use_Former: str = '3d',
            timestep: int = 20,
            max_length: int = 20,
            extract_layer_idx: int = 1,
            use_all_layer: bool = False,
            obs_seq_len: int = 1,
            action_dim: int = 7,
            action_seq_len: int = 10,
            use_pipeline_cpu_offload: bool = False,
            debug_hidden2dino: bool = False,
            debug_hidden2dino_dir: Optional[str] = None,
            hidden2dino_use_ref_override: Optional[bool] = True,
            hidden2dpa_use_ref_override: Optional[bool] = True,
            use_gt_dino_condition: bool = False,
            gt_dino_chunk: int = 32,
            bypass_video_former: bool = False,
            without_svd: bool = False,
            use_hidden_dino_concat: bool = True,
            use_hidden_dino_dpa_concat: bool = True,
            use_hidden_dpa_concat: bool = False,
            hidden2dino_ckpt: str = '',
            hidden2dpa_ckpt: str = '',
            dinov2_path: str = '',
            da3_path: str = '',
    ):
        super(VPP_Policy, self).__init__()
        self.dinov2_path = dinov2_path
        self.da3_path = da3_path
        self.latent_dim = latent_dim
        self.use_all_layer = use_all_layer
        self.use_position_encoding = use_position_encoding
        self.use_gripper_features = False

        self.act_window_size = act_window_size
        self.action_dim = action_dim

        self.timestep = timestep
        self.extract_layer_idx = extract_layer_idx
        self.use_Former = use_Former
        self.Former_num_time_embeds = Former_num_time_embeds
        self.max_length = max_length
        self.use_gt_dino_condition = use_gt_dino_condition
        self.gt_dino_chunk = gt_dino_chunk
        self.bypass_video_former = bypass_video_former
        # Keep eval-time constructor compatible with the wo_hidden2dino policy.
        self.without_svd = False
        self.use_hidden_dpa_concat = False
        self.use_hidden_dino_concat = True
        self.use_hidden_dino_dpa_concat = True
        self.num_latents = num_latents
        self.bypass_proj: Optional[nn.Module] = None

        # Default dims; will be updated after reading configs
        self.hidden2dino_out_dim = 768
        self.hidden2dpa_out_dim = 768
        self.hidden2dpa_expected_T = self.Former_num_time_embeds
        self.hidden2dpa_spatial_hw = (16, 16)
        self.hidden2dpa_use_ref = False
        self.hidden2dpa_model_kwargs: Dict = {}
        self.hidden2dpa_run_dir = None
        self.hidden2dpa_config_path = None
        self.hidden2dpa_ckpt_path = None
        self.da3_model_dir = None
        self.da3_backbone_ckpt = None

        # Preload Hidden2DPA metadata for condition_dim computation
        if self.use_hidden_dino_dpa_concat:
            if not hidden2dpa_ckpt:
                raise ValueError("--hidden2dpa_ckpt is required.")
            self.hidden2dpa_ckpt_path = Path(hidden2dpa_ckpt).expanduser().resolve()
            self.hidden2dpa_run_dir = self.hidden2dpa_ckpt_path.parent
            self.hidden2dpa_config_path = self.hidden2dpa_run_dir / "model_config_resolved.yaml"
            if not self.hidden2dpa_config_path.exists() or not self.hidden2dpa_ckpt_path.exists():
                raise FileNotFoundError(
                    f"Hidden2DPA resources missing at {hidden2dpa_run_dir}. "
                    f"Expected config={self.hidden2dpa_config_path} and ckpt={self.hidden2dpa_ckpt_path}."
                )
            with self.hidden2dpa_config_path.open("r", encoding="utf-8") as cfg_file:
                hidden2dpa_cfg = yaml.safe_load(cfg_file) or {}
            self.hidden2dpa_model_kwargs = _sanitize_st_decoupler_kwargs(
                hidden2dpa_cfg.get("model", {}),
                label="Hidden2DPA",
            )
            # Keep DA3 backbone source aligned with the hidden2dpa training run.
            self.da3_model_dir = hidden2dpa_cfg.get("model_dir", None)
            self.da3_backbone_ckpt = hidden2dpa_cfg.get("backbone_ckpt", None)
            self.hidden2dpa_expected_T = self.hidden2dpa_model_kwargs.get("T", self.hidden2dpa_expected_T)
            self.hidden2dpa_spatial_hw = (
                self.hidden2dpa_model_kwargs.get("H", self.hidden2dpa_spatial_hw[0]),
                self.hidden2dpa_model_kwargs.get("W", self.hidden2dpa_spatial_hw[1]),
            )
            self.hidden2dpa_out_dim = self.hidden2dpa_model_kwargs.get("C_out", self.hidden2dpa_out_dim)
            self.hidden2dpa_use_ref = True

        if self.use_gt_dino_condition:
            self.Former_num_time_embeds = self.act_window_size

        condition_dim_list = [1280,1280,1280,640]
        sum_dim = 0
        for i in range(extract_layer_idx+1):
            sum_dim = sum_dim + condition_dim_list[i+1]
        condition_dim = condition_dim_list[extract_layer_idx+1] if not self.use_all_layer else sum_dim
        if self.use_hidden_dino_concat or self.use_hidden_dino_dpa_concat:
            condition_dim = condition_dim + self.hidden2dino_out_dim
        if self.use_hidden_dino_dpa_concat:
            condition_dim = condition_dim + self.hidden2dpa_out_dim
        self.condition_dim = condition_dim
        if self.bypass_video_former or not (self.use_hidden_dino_concat or self.use_hidden_dino_dpa_concat):
            self.perceiver_resampler = PerceiverResampler(input_dim=768, output_dim=384, num_queries=224)

        if self.bypass_video_former:
            self.Video_Former = None
            self.bypass_proj = nn.Linear(condition_dim, latent_dim)
        elif use_Former=='3d':
            self.Video_Former = Video_Former_3D(
                dim=latent_dim,
                depth=Former_depth,
                dim_head=Former_dim_head,
                heads=Former_heads,
                num_frame=self.Former_num_time_embeds,
                num_time_embeds=self.Former_num_time_embeds,
                num_latents=num_latents,
                condition_dim=condition_dim,
                use_temporal=True,
             )
        elif use_Former == '2d':
            self.Video_Former = Video_Former_2D(
                    dim=latent_dim,
                    depth=Former_depth,
                    dim_head=Former_dim_head,
                    heads=Former_heads,
                    num_frame=self.Former_num_time_embeds,
                    num_time_embeds=self.Former_num_time_embeds,
                    num_latents=num_latents,
                    condition_dim=condition_dim,
                 )
        else:
            self.Video_Former = nn.Linear(condition_dim,latent_dim)

        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler
        # goal encoders
        self.language_goal = LangClip(
            model_name='ViT-B/32',
            pretrained_path=text_encoder_path or None,
        ).to(self.device)

        pipeline, tokenizer, feature_extractor, train_scheduler, vae_processor, text_encoder, vae, unet = load_primary_models(
            pretrained_model_path , eval = True)

        text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_path)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=False)

        text_encoder = text_encoder.to(self.device).eval()

        for param in pipeline.image_encoder.parameters():
            param.requires_grad = False
        for param in text_encoder.parameters():
            param.requires_grad = False

        for param in pipeline.vae.parameters():
            param.requires_grad = False
        for param in pipeline.unet.parameters():
            param.requires_grad = False

        self.use_pipeline_cpu_offload = use_pipeline_cpu_offload
        self.debug_hidden2dino = debug_hidden2dino
        self.debug_hidden2dino_dir = (
            Path(debug_hidden2dino_dir).expanduser().resolve()
            if debug_hidden2dino_dir is not None else (Path.cwd() / "debug_hidden2dino")
        )
        self._hidden2dino_debug_saved = 0
        self._hidden2dino_debug_limit = 10
        self._dino_dpa_debug_saved = 0
        self._dino_dpa_debug_limit = 20
        self._has_trainer_warning_logged = False

        # Initialize DINOv2 encoder for extracting real DINO features
        self._dino_encoder = None
        self._dino_encoder_initialized = False

        # Initialize DPA (DA3) encoder for extracting real DPA features
        self._dpa_encoder = None
        self._dpa_encoder_initialized = False

        if not self.use_pipeline_cpu_offload:
            pipeline = pipeline.to(self.device)

        pipeline.unet.eval()

        self.TVP_encoder = Diffusion_feature_extractor(pipeline=pipeline,
                                                        tokenizer=tokenizer,
                                                        text_encoder=text_encoder,
                                                        position_encoding = self.use_position_encoding)

        if not self.use_pipeline_cpu_offload:
            self.TVP_encoder = self.TVP_encoder.to(self.device)
        else:
            # Keep only necessary components on the target device.
            self.TVP_encoder.text_encoder = self.TVP_encoder.text_encoder.to(self.device)
        if not hidden2dino_ckpt:
            raise ValueError("--hidden2dino_ckpt is required.")
        hidden2dino_ckpt_path = Path(hidden2dino_ckpt).expanduser().resolve()
        hidden2dino_run_dir = hidden2dino_ckpt_path.parent
        hidden2dino_config_path = hidden2dino_run_dir / "model_config_resolved.yaml"
        if not hidden2dino_config_path.exists() or not hidden2dino_ckpt_path.exists():
            raise FileNotFoundError(
                f"Hidden2DINO resources missing at {hidden2dino_run_dir}. "
                f"Expected config={hidden2dino_config_path} and ckpt={hidden2dino_ckpt_path}."
            )
        with hidden2dino_config_path.open("r", encoding="utf-8") as cfg_file:
            hidden2dino_cfg = yaml.safe_load(cfg_file) or {}
        hidden_model_kwargs = _sanitize_st_decoupler_kwargs(
            hidden2dino_cfg.get("model", {}),
            label="Hidden2DINO",
        )
        self.hidden2dino_expected_T = hidden_model_kwargs.get("T", self.Former_num_time_embeds)
        self.hidden2dino_spatial_hw = (
            hidden_model_kwargs.get("H", 16),
            hidden_model_kwargs.get("W", 16),
        )
        self.hidden2dino_out_dim = hidden_model_kwargs.get("C_out", self.hidden2dino_out_dim)
        self.hidden2dino_use_ref = True

        
        self.gt_dino_encoder = None
        self.gt_dino_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        self.gt_dino_std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)

        if self.use_gt_dino_condition:
            self.hidden2dino_use_ref = False
            self.hidden2dino = None
            self.hidden2dino_proj = nn.Linear(self.hidden2dino_out_dim, self.condition_dim).to(self.device)
            for param in self.hidden2dino_proj.parameters():
                param.requires_grad = True
            self.gt_dino_encoder = self._init_gt_dino_encoder()
            #self.gt_dino_cat_proj = nn.Linear(self.condition_dim * 2, self.condition_dim).to(self.device)
            self._gt_debug_counter = 0
        else:
            base_hidden2dino = HiddenToDinoModel(**hidden_model_kwargs)
            if self.hidden2dino_use_ref:
                ref_dim = hidden_model_kwargs.get("C_out", 768)
                self.hidden2dino = HiddenToDinoModelWithRef(base_hidden2dino, ref_dim=ref_dim)
            else:
                self.hidden2dino = base_hidden2dino

            checkpoint = torch.load(hidden2dino_ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)

            def strip_prefix(sd: dict, prefix: str) -> dict:
                return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

            if not self.hidden2dino_use_ref:
                has_base_prefix = any(k.startswith("base_model.") for k in state_dict.keys())
                if has_base_prefix:
                    state_dict = strip_prefix(state_dict, "base_model.")

            self.hidden2dino.load_state_dict(state_dict, strict=True)
            self.hidden2dino.eval()
            for param in self.hidden2dino.parameters():
                param.requires_grad = False
            self.hidden2dino = self.hidden2dino.to(self.device)
            #self.hidden2dino_proj = nn.Linear(self.hidden2dino_out_dim, self.condition_dim).to(self.device)
            #for param in self.hidden2dino_proj.parameters():
            #    param.requires_grad = True

        # Initialize Hidden2DPA model (Depth Anything tokens) when requested
        self.hidden2dpa = None
        if self.use_hidden_dino_dpa_concat:
            base_hidden2dpa = HiddenToDA3Model(**self.hidden2dpa_model_kwargs)
            if self.hidden2dpa_use_ref:
                ref_dim = self.hidden2dpa_model_kwargs.get("C_out", self.hidden2dpa_out_dim)
                self.hidden2dpa = HiddenToDA3ModelWithRef(base_hidden2dpa, ref_dim=ref_dim)
            else:
                self.hidden2dpa = base_hidden2dpa

            checkpoint_dpa = torch.load(self.hidden2dpa_ckpt_path, map_location="cpu")
            state_dict_dpa = checkpoint_dpa.get("model", checkpoint_dpa)
            if not self.hidden2dpa_use_ref:
                has_base_prefix_dpa = any(k.startswith("base_model.") for k in state_dict_dpa.keys())
                if has_base_prefix_dpa:
                    state_dict_dpa = {k[len("base_model.") :]: v for k, v in state_dict_dpa.items() if k.startswith("base_model.")}
            self.hidden2dpa.load_state_dict(state_dict_dpa, strict=True)
            self.hidden2dpa.eval()
            for param in self.hidden2dpa.parameters():
                param.requires_grad = False
            self.hidden2dpa = self.hidden2dpa.to(self.device)

        # policy network
        self.model = GCDenoiser(action_dim = action_dim,
                                obs_dim=latent_dim,
                                goal_dim=512,
                                num_tokens=num_latents,
                                goal_window_size = 1,
                                obs_seq_len = obs_seq_len,
                                act_seq_len = action_seq_len,
                                device=self.device,
                                sigma_data=0.5).to(self.device)

        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters()
        # diffusion stuff
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep
        self.latent_goal = None
        self.plan = None
        self.last_predictive_feature = None
        self.last_hidden2dino_tokens = None
        self.last_hidden2dpa_tokens = None
        self.use_text_not_embedding = use_text_not_embedding
        # print_model_parameters(self.perceptual_encoder.perceiver_resampler)
        # for clip loss ground truth plot
        self.ema_callback_idx = None

        for param in self.model.inner_model.proprio_emb.parameters():
            param.requires_grad = False
        for param in self.model.inner_model.goal_emb.parameters():
            param.requires_grad = False
        self.model.inner_model.pos_emb.requires_grad = False

    def log(self, name, value, *args, **kwargs):  # type: ignore[override]
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            self._has_trainer_warning_logged = True
            if torch.is_tensor(value):
                if value.numel() == 1:
                    return value.detach().item()
                return value.detach().cpu()
            return value
        return super().log(name, value, *args, **kwargs)

    def process_device(self):
        if self.use_pipeline_cpu_offload and hasattr(self.TVP_encoder.pipeline, "enable_model_cpu_offload"):
            self.TVP_encoder.pipeline.enable_model_cpu_offload(device=self.device)
        else:
            self.TVP_encoder.pipeline = self.TVP_encoder.pipeline.to(self.device)
            self.TVP_encoder.text_encoder = self.TVP_encoder.text_encoder.to(self.device)
        if getattr(self, "hidden2dino", None) is not None:
            self.hidden2dino = self.hidden2dino.to(self.device)
        if getattr(self, "hidden2dpa", None) is not None:
            self.hidden2dpa = self.hidden2dpa.to(self.device)
        if hasattr(self, "hidden2dino_proj"):
            self.hidden2dino_proj = self.hidden2dino_proj.to(self.device)
        if getattr(self, "gt_dino_cat_proj", None) is not None:
            self.gt_dino_cat_proj = self.gt_dino_cat_proj.to(self.device)
        if getattr(self, "bypass_proj", None) is not None:
            self.bypass_proj = self.bypass_proj.to(self.device)
        if getattr(self, "perceiver_resampler", None) is not None:
            self.perceiver_resampler = self.perceiver_resampler.to(self.device)

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        '''optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
            weight_decay=self.optimizer_config.transformer_weight_decay
        )'''
        optim_groups = [
            {
                "params": self.model.inner_model.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay,
            }
        ]
        #if getattr(self, "hidden2dino", None)
        #if self.Video_Former is not None:
        if getattr(self, "Video_Former", None) is not None:
            optim_groups.append(
                {
                    "params": self.Video_Former.parameters(),
                    "weight_decay": self.optimizer_config.transformer_weight_decay,
                }
            )
        if getattr(self, "bypass_proj", None) is not None:
            optim_groups.append(
                {
                    "params": self.bypass_proj.parameters(),
                    "weight_decay": self.optimizer_config.transformer_weight_decay,
                }
            )
        if getattr(self, "perceiver_resampler", None) is not None:
            optim_groups.append(
                {
                    "params": self.perceiver_resampler.parameters(),
                    "weight_decay": self.optimizer_config.transformer_weight_decay,
                }
            )
        if getattr(self, "hidden2dino_proj", None) is not None:
            optim_groups.append(
                {
                    "params": self.hidden2dino_proj.parameters(),
                    "weight_decay": self.optimizer_config.transformer_weight_decay,
                }
            )
       


        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate,
                                      betas=self.optimizer_config.betas)

        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def on_before_zero_grad(self, optimizer=None):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
            total_param_norm += p.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        total_param_norm = total_param_norm ** 0.5

        self.log("train/grad_norm", total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/param_norm", total_param_norm, on_step=True, on_epoch=False, sync_dist=True)


    def training_step(self, dataset_batch: Dict[str, Dict],) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss for the MDT Agent.
        The training loss consists of the score matching loss of the diffusion model
        and the contrastive loss of the CLIP model for the multimodal encoder.

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            loss tensor
        """
        total_loss, action_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )
        predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)

        act_loss, sigmas, noise = self.diffusion_loss(
            predictive_feature,
            latent_goal,
            dataset_batch["actions"],
        )

        action_loss += act_loss
        total_loss += act_loss

        total_bs = dataset_batch["actions"].shape[0]

        self._log_training_metrics(action_loss, total_loss, total_bs)
        return total_loss

    @torch.no_grad()
    def validation_step(self, dataset_batch: Dict[str, Dict]) -> Dict[
        str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.
        During the validation step, the diffusion model predicts the next action sequence given the current state

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal module, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
            # Compute the required embeddings
        predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)

        # predict the next action sequence
        action_pred = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            predictive_feature,
            latent_goal,
            inference=True,
        )
        dataset_batch["actions"] = dataset_batch["actions"].to(action_pred.device)
        # compute the mse action loss
        pred_loss = torch.nn.functional.mse_loss(action_pred, dataset_batch["actions"])
        val_total_act_loss_pp += pred_loss

        output[f"idx:"] = dataset_batch["idx"]
        output["validation_loss"] = val_total_act_loss_pp
        return output

    def _run_hidden2dino(self, features: torch.Tensor, ref_dino: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert diffusion features to DINO-style tokens using the frozen hidden2dino model."""

        if features.ndim != 5:
            raise ValueError(f"Expected features with 5 dims (B, T, C, H, W), got {features.shape}")

        B, T, C, H, W = features.shape
        expected_h, expected_w = self.hidden2dino_spatial_hw
        if H != expected_h or W != expected_w:
            features = features.view(B * T, C, H, W)
            features = F.interpolate(
                features,
                size=(expected_h, expected_w),
                mode="bilinear",
                align_corners=False,
            )
            features = features.view(B, T, C, expected_h, expected_w)
            H, W = expected_h, expected_w
            
            if ref_dino is not None:
                 # ref_dino: (B, C, 1, H_ref, W_ref)
                 # Interpolate ref_dino if needed
                 if ref_dino.shape[-2:] != (expected_h, expected_w):
                     ref_B, ref_C, ref_T, ref_H, ref_W = ref_dino.shape
                     ref_dino = ref_dino.view(ref_B * ref_T, ref_C, ref_H, ref_W)
                     ref_dino = F.interpolate(
                         ref_dino,
                         size=(expected_h, expected_w),
                         mode="bilinear",
                         align_corners=False,
                     )
                     ref_dino = ref_dino.view(ref_B, ref_C, ref_T, expected_h, expected_w)

        if T < self.hidden2dino_expected_T:
            raise ValueError(
                f"Hidden2DINO expects at least {self.hidden2dino_expected_T} frames, got {T}"
            )
        if T != self.hidden2dino_expected_T:
            features = features[:, :self.hidden2dino_expected_T]
            T = features.shape[1]
        features = features.permute(0, 2, 1, 3, 4).contiguous().to(torch.float32)
        
        with torch.no_grad():
            if self.hidden2dino_use_ref:
                if ref_dino is None:
                    raise ValueError("Hidden2DINO model expects ref_dino (use_ref_frame=True), but none provided.")
                # Ensure ref_dino is on correct device and dtype
                ref_dino = ref_dino.to(device=features.device, dtype=features.dtype)
                dino_feats = self.hidden2dino(features, ref_dino)
            else:
                dino_feats = self.hidden2dino(features)
                
        dino_feats = dino_feats.permute(0, 2, 3, 4, 1).contiguous()
        tokens = dino_feats.view(B, T, -1, dino_feats.shape[-1])
        #tokens = self.hidden2dino_proj(tokens)

        if self.debug_hidden2dino and self._hidden2dino_debug_saved < self._hidden2dino_debug_limit:
            # Add source type (static or gripper) to help with debugging
            source_type = getattr(self, '_current_source_type', 'unknown')
            self._save_hidden2dino_debug(features, dino_feats, tokens, source_type)
            self._hidden2dino_debug_saved += 1
        return tokens

    def _run_hidden2dpa(self, features: torch.Tensor, ref_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert diffusion features to DPA (Depth Anything) tokens using the frozen hidden2dpa model."""
        if self.hidden2dpa is None:
            raise RuntimeError("Hidden2DPA model is not initialized. Set use_hidden_dino_dpa_concat=True to enable it.")
        if features.ndim != 5:
            raise ValueError(f"Expected features with 5 dims (B, T, C, H, W), got {features.shape}")

        B, T, C, H, W = features.shape
        expected_h, expected_w = self.hidden2dpa_spatial_hw
        expected_c = self.hidden2dpa_model_kwargs.get("C_in", C)

        if H != expected_h or W != expected_w:
            features = features.view(B * T, C, H, W)
            features = F.interpolate(
                features,
                size=(expected_h, expected_w),
                mode="bilinear",
                align_corners=False,
            )
            features = features.view(B, T, C, expected_h, expected_w)
            H, W = expected_h, expected_w

        if C != expected_c:
            raise ValueError(f"Hidden2DPA expects channel dimension {expected_c}, got {C}")

        if T < self.hidden2dpa_expected_T:
            raise ValueError(
                f"Hidden2DPA expects at least {self.hidden2dpa_expected_T} frames, got {T}"
            )
        if T != self.hidden2dpa_expected_T:
            features = features[:, : self.hidden2dpa_expected_T]
            T = features.shape[1]

        features = features.permute(0, 2, 1, 3, 4).contiguous().to(torch.float32)

        with torch.no_grad():
            if self.hidden2dpa_use_ref:
                dpa_outputs = self.hidden2dpa(features, ref_tokens)
            else:
                dpa_outputs = self.hidden2dpa(features)

        # HiddenToDA3Model returns a list of (tokens, cam_tokens) tuples
        if isinstance(dpa_outputs, list):
            tokens = dpa_outputs[0][0]
        else:
            tokens = dpa_outputs
        return tokens
    


    def _bypass_video_former(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.bypass_proj is None:
            raise ValueError("Bypass projection is not initialized.")
        if tokens.ndim != 4:
            raise ValueError(f"Bypass expects tokens with shape (B, T, L, C), got {tokens.shape}")
        B, T, L, C = tokens.shape
        tokens = tokens.view(B, T * L, C)
        if tokens.shape[1] != self.num_latents:
            tokens = tokens.permute(0, 2, 1)
            tokens = F.interpolate(tokens, size=self.num_latents, mode="linear", align_corners=False)
            tokens = tokens.permute(0, 2, 1)
        tokens = self.bypass_proj(tokens)
        return tokens

    def _init_gt_dino_encoder(self):
        import torch.hub
        dinov2_dir = self.dinov2_path or os.environ.get(
            "S_VAM_TORCH_HUB_DIR",
            str(_PROJECT_ROOT / "checkpoints" / "torch_hub"),
        )
        torch.hub.set_dir(dinov2_dir)
        encoder = torch.hub.load(
            os.path.join(dinov2_dir, "facebookresearch_dinov2_main"),
            "dinov2_vitb14_reg",
            verbose=True,
            source="local",
        )
        encoder.head = torch.nn.Identity()
        encoder = encoder.to(self.device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder

    def _prepare_tokens_from_dino(self, dino_feats: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = dino_feats.shape
        tokens = (
            dino_feats.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(B, T, -1, C)
        )
        tokens = self.hidden2dino_proj(tokens)
        return tokens

    def _extract_gt_dino_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        if self.gt_dino_encoder is None:
            raise ValueError("GT DINO encoder not initialized.")
        B, T, C, H, W = frames.shape
        frames = frames.to(self.device).float()
        if frames.max() > 1.5:
            frames = frames / 255.0
        if frames.min() < -0.1:
            frames = (frames + 1.0) / 2.0
        frames = frames.clamp(0.0, 1.0)
        frames = frames.view(B * T, C, H, W)
        frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
        mean = self.gt_dino_mean.to(frames.device)
        std = self.gt_dino_std.to(frames.device)
        frames = (frames - mean) / std

        outputs = []
        for start in range(0, frames.shape[0], self.gt_dino_chunk):
            chunk = frames[start : start + self.gt_dino_chunk]
            feats = self.gt_dino_encoder.forward_features(chunk)
            tokens = feats["x_norm_patchtokens"]
            outputs.append(tokens)
        patch_tokens = torch.cat(outputs, dim=0)
        num_tokens = patch_tokens.shape[1]
        patch_size = int(num_tokens ** 0.5)
        patch_tokens = patch_tokens.view(B, T, patch_size, patch_size, -1)
        patch_tokens = patch_tokens.permute(0, 4, 1, 2, 3).contiguous()
        return self._prepare_tokens_from_dino(patch_tokens)

    def _save_hidden2dino_debug(
        self,
        hidden_inputs: torch.Tensor,
        dino_raw: torch.Tensor,
        dino_tokens: torch.Tensor,
        source_type: str = 'unknown',
    ) -> None:
        try:
            self.debug_hidden2dino_dir.mkdir(parents=True, exist_ok=True)
            vis_dir = self.debug_hidden2dino_dir / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to create hidden2dino debug dir %s: %s", self.debug_hidden2dino_dir, exc)
            return

        idx = self._hidden2dino_debug_saved
        path = self.debug_hidden2dino_dir / f"hidden2dino_debug_{idx:04d}_{source_type}.pt"
        payload = {
            "inputs_BCTHW": hidden_inputs.detach().cpu(),
            "dino_raw_BTHWC": dino_raw.detach().cpu(),
            "dino_tokens_BTLC": dino_tokens.detach().cpu(),
            "source_type": source_type,
        }
        try:
            torch.save(payload, path)
            # Generate visualizations for the first sample in batch
            self._visualize_hidden2dino_features(
                dino_raw[0].detach().cpu(),  # Take first sample
                hidden_inputs[0].detach().cpu(),
                vis_dir / f"sample_{idx:04d}_{source_type}",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to save/visualize hidden2dino debug to %s: %s", path, exc)

    @staticmethod
    def _token_vector_to_square_map(token_norm: np.ndarray) -> np.ndarray:
        token_norm = np.asarray(token_norm, dtype=np.float32).reshape(-1)
        if token_norm.size == 0:
            return np.zeros((1, 1), dtype=np.float32)
        side = int(np.sqrt(token_norm.size))
        if side * side == token_norm.size:
            return token_norm.reshape(side, side)
        return token_norm.reshape(1, -1)

    @staticmethod
    def _norm_map_from_ref_dino(ref_dino: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if ref_dino is None:
            return None
        t = ref_dino.detach().to(torch.float32).cpu()
        # expected (B, C, T, H, W)
        if t.ndim == 5:
            m = torch.norm(t[0, :, 0], dim=0).numpy()
            return m
        return None

    @staticmethod
    def _norm_map_from_tokens(tokens: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if tokens is None:
            return None
        t = tokens.detach().to(torch.float32).cpu()
        # expected (B, T, L, C) or (B, L, C)
        if t.ndim == 4:
            vec = torch.norm(t[0, 0], dim=-1).numpy()
            return VPP_Policy._token_vector_to_square_map(vec)
        if t.ndim == 3:
            vec = torch.norm(t[0], dim=-1).numpy()
            return VPP_Policy._token_vector_to_square_map(vec)
        return None

    @staticmethod
    def _norm_maps_from_sequence_tokens(tokens: Optional[torch.Tensor]) -> list:
        if tokens is None:
            return []
        t = tokens.detach().to(torch.float32).cpu()
        maps = []
        if t.ndim == 4:  # (B, T, L, C)
            for i in range(t.shape[1]):
                vec = torch.norm(t[0, i], dim=-1).numpy()
                maps.append(VPP_Policy._token_vector_to_square_map(vec))
        elif t.ndim == 3:  # (B, L, C)
            vec = torch.norm(t[0], dim=-1).numpy()
            maps.append(VPP_Policy._token_vector_to_square_map(vec))
        return maps

    @staticmethod
    def _pca_rgb_from_tokens_2d(token_feats: torch.Tensor) -> np.ndarray:
        # token_feats: (L, C)
        x = token_feats.to(torch.float32)
        if x.ndim != 2:
            return np.zeros((1, 1, 3), dtype=np.float32)
        l, c = x.shape
        if l <= 0 or c <= 0:
            return np.zeros((1, 1, 3), dtype=np.float32)
        q = min(3, c, max(1, l - 1))
        x = x - x.mean(dim=0, keepdim=True)
        if q >= 1:
            try:
                _, _, v = torch.pca_lowrank(x, q=q)
                y = x @ v[:, :q]
            except Exception:
                y = x[:, :q]
        else:
            y = x[:, :1]
        if y.shape[1] < 3:
            y = torch.cat([y, torch.zeros((l, 3 - y.shape[1]), dtype=y.dtype)], dim=1)
        y = y[:, :3].cpu().numpy()
        side = int(np.sqrt(l))
        if side * side == l:
            rgb = y.reshape(side, side, 3)
        else:
            rgb = y.reshape(1, l, 3)
        # Per-channel normalize to [0,1]
        for i in range(3):
            ch = rgb[..., i]
            rgb[..., i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
        return rgb.astype(np.float32)

    @staticmethod
    def _pca_rgb_from_ref_dino(ref_dino: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if ref_dino is None:
            return None
        t = ref_dino.detach().to(torch.float32).cpu()
        # expected (B, C, T, H, W)
        if t.ndim != 5:
            return None
        c, h, w = t.shape[1], t.shape[3], t.shape[4]
        feat = t[0, :, 0].permute(1, 2, 0).reshape(h * w, c)  # (H*W, C)
        rgb = VPP_Policy._pca_rgb_from_tokens_2d(feat)
        return rgb.reshape(h, w, 3)

    @staticmethod
    def _pca_rgbs_from_sequence_tokens(tokens: Optional[torch.Tensor]) -> list:
        if tokens is None:
            return []
        t = tokens.detach().to(torch.float32).cpu()
        rgbs = []
        if t.ndim == 4:  # (B, T, L, C)
            for i in range(t.shape[1]):
                rgbs.append(VPP_Policy._pca_rgb_from_tokens_2d(t[0, i]))
        elif t.ndim == 3:  # (B, L, C)
            rgbs.append(VPP_Policy._pca_rgb_from_tokens_2d(t[0]))
        return rgbs

    @staticmethod
    def _tokens_2d_from_ref_dino(ref_dino: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if ref_dino is None:
            return None
        t = ref_dino.detach().to(torch.float32).cpu()
        if t.ndim != 5:
            return None
        c, h, w = t.shape[1], t.shape[3], t.shape[4]
        return t[0, :, 0].permute(1, 2, 0).reshape(h * w, c)

    @staticmethod
    def _tokens_2d_list_from_sequence(tokens: Optional[torch.Tensor]) -> list:
        if tokens is None:
            return []
        t = tokens.detach().to(torch.float32).cpu()
        outs = []
        if t.ndim == 4:  # (B, T, L, C)
            for i in range(t.shape[1]):
                outs.append(t[0, i])
        elif t.ndim == 3:  # (B, L, C)
            outs.append(t[0])
        return outs

    @staticmethod
    def _global_pca_rgb_from_token_list(token_list: list) -> list:
        valid = [x for x in token_list if isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[0] > 0 and x.shape[1] > 0]
        if not valid:
            return []
        c_min = min(int(x.shape[1]) for x in valid)
        xs = [x[:, :c_min].to(torch.float32) for x in valid]
        all_x = torch.cat(xs, dim=0)
        all_x = all_x - all_x.mean(dim=0, keepdim=True)
        q = min(3, c_min, max(1, all_x.shape[0] - 1))
        try:
            _, _, v = torch.pca_lowrank(all_x, q=q)
            basis = v[:, :q]
        except Exception:
            basis = torch.eye(c_min, dtype=all_x.dtype)[:, :q]

        rgbs = []
        global_stack = []
        for x in xs:
            y = (x - all_x.mean(dim=0, keepdim=True)) @ basis
            if y.shape[1] < 3:
                y = torch.cat([y, torch.zeros((y.shape[0], 3 - y.shape[1]), dtype=y.dtype)], dim=1)
            y = y[:, :3].cpu().numpy()
            l = y.shape[0]
            side = int(np.sqrt(l))
            if side * side == l:
                rgb = y.reshape(side, side, 3)
            else:
                rgb = y.reshape(1, l, 3)
            rgbs.append(rgb.astype(np.float32))
            global_stack.append(rgb.reshape(-1, 3))

        merged = np.concatenate(global_stack, axis=0)
        ch_min = merged.min(axis=0, keepdims=True)
        ch_max = merged.max(axis=0, keepdims=True)
        denom = ch_max - ch_min + 1e-6
        normed = []
        for rgb in rgbs:
            out = (rgb - ch_min) / denom
            normed.append(np.clip(out, 0.0, 1.0))
        return normed

    @staticmethod
    def _save_ref_pred_strip_png(
        out_path,
        title: str,
        ref_map: Optional[np.ndarray],
        pred_maps: list,
    ) -> None:
        import matplotlib.pyplot as plt

        max_pred = min(len(pred_maps), 16)
        total = 1 + max_pred
        cols = min(6, total)
        rows = int(np.ceil(total / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.array(axes).reshape(-1)

        def _draw(ax, arr, name):
            if arr is None:
                ax.text(0.5, 0.5, "None", ha="center", va="center")
                ax.set_title(name)
                ax.axis("off")
                return
            m = np.asarray(arr, dtype=np.float32)
            if m.ndim == 3 and m.shape[-1] == 3:
                m = np.clip(m, 0.0, 1.0)
                ax.imshow(m, aspect="auto")
                im = None
            else:
                m = (m - m.min()) / (m.max() - m.min() + 1e-6)
                im = ax.imshow(m, cmap="viridis", aspect="auto")
            ax.set_title(name)
            ax.axis("off")
            if im is not None:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        _draw(axes[0], ref_map, "ref")
        for i in range(max_pred):
            _draw(axes[i + 1], pred_maps[i], f"pred_t{i:02d}")
        for j in range(total, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def _save_dino_dpa_debug(
        self,
        *,
        ref_dino: Optional[torch.Tensor],
        dino_tokens: Optional[torch.Tensor],
        ref_dpa: Optional[torch.Tensor],
        dpa_tokens: Optional[torch.Tensor],
        source: str,
    ) -> None:
        if not self.debug_hidden2dino:
            return
        if self._dino_dpa_debug_saved >= self._dino_dpa_debug_limit:
            return
        out_dir = self.debug_hidden2dino_dir / "dino_dpa_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = self._dino_dpa_debug_saved

        payload = {
            "ref_dino": ref_dino.detach().cpu() if ref_dino is not None else None,
            "dino_tokens": dino_tokens.detach().cpu() if dino_tokens is not None else None,
            "ref_dpa": ref_dpa.detach().cpu() if ref_dpa is not None else None,
            "dpa_tokens": dpa_tokens.detach().cpu() if dpa_tokens is not None else None,
            "source": source,
        }
        torch.save(payload, out_dir / f"dino_dpa_debug_{idx:04d}_{source}.pt")

        try:
            import matplotlib.pyplot as plt

            maps = [
                ("ref_dino", self._norm_map_from_ref_dino(ref_dino)),
                ("hidden2dino_tokens", self._norm_map_from_tokens(dino_tokens)),
                ("ref_dpa", self._norm_map_from_tokens(ref_dpa)),
                ("hidden2dpa_tokens", self._norm_map_from_tokens(dpa_tokens)),
            ]
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            for ax, (name, arr) in zip(axes.flat, maps):
                if arr is None:
                    ax.text(0.5, 0.5, "None", ha="center", va="center")
                    ax.set_title(name)
                    ax.axis("off")
                    continue
                arr = np.asarray(arr, dtype=np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                im = ax.imshow(arr, cmap="viridis", aspect="auto")
                ax.set_title(f"{name} norm")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"DINO/DPA debug [{source}]")
            fig.tight_layout()
            fig.savefig(out_dir / f"dino_dpa_debug_{idx:04d}_{source}.png", dpi=150)
            plt.close(fig)

            # Extra: ref + first 16 predicted frames with a shared PCA basis (global PCA).
            dpa_ref_tok = self._tokens_2d_list_from_sequence(ref_dpa)
            dpa_pred_tok = self._tokens_2d_list_from_sequence(dpa_tokens)
            dpa_tok_all = dpa_ref_tok[:1] + dpa_pred_tok[:16]
            dpa_rgb_all = self._global_pca_rgb_from_token_list(dpa_tok_all)
            dpa_ref = dpa_rgb_all[0] if len(dpa_rgb_all) > 0 else None
            dpa_pred_seq = dpa_rgb_all[1:]
            self._save_ref_pred_strip_png(
                out_dir / f"dino_dpa_ref_pred16_{idx:04d}_{source}.png",
                f"DPA ref vs pred(16) Global-PCA-RGB [{source}]",
                dpa_ref,
                dpa_pred_seq,
            )

            dino_ref_tok = self._tokens_2d_from_ref_dino(ref_dino)
            dino_pred_tok = self._tokens_2d_list_from_sequence(dino_tokens)
            dino_tok_all = ([dino_ref_tok] if dino_ref_tok is not None else []) + dino_pred_tok[:16]
            dino_rgb_all = self._global_pca_rgb_from_token_list(dino_tok_all)
            dino_ref = dino_rgb_all[0] if (dino_ref_tok is not None and len(dino_rgb_all) > 0) else None
            dino_pred_seq = dino_rgb_all[1:] if dino_ref is not None else dino_rgb_all
            self._save_ref_pred_strip_png(
                out_dir / f"dino_ref_pred16_{idx:04d}_{source}.png",
                f"DINO ref vs pred(16) Global-PCA-RGB [{source}]",
                dino_ref,
                dino_pred_seq,
            )
        except Exception as exc:
            logger.warning("Failed to save dino/dpa visualization: %s", exc)

        self._dino_dpa_debug_saved += 1

    def _visualize_hidden2dino_features(
        self,
        dino_features: torch.Tensor,  # (T, H, W, C)
        hidden_features: torch.Tensor,  # (C, T, H, W)
        output_dir: Path,
    ) -> None:
        """Generate comprehensive visualizations of input images, DINO features, and predictions"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            output_dir.mkdir(parents=True, exist_ok=True)
            
            T, H, W, C = dino_features.shape
            
            # Generate comprehensive visualization with input images
            self._generate_comprehensive_visualization(dino_features, hidden_features, output_dir)
            
            # Generate PCA visualization across all frames
            self._generate_pca_visualization(dino_features, hidden_features, output_dir)
            
        except Exception as exc:
            logger.warning("Failed to generate visualizations: %s", exc)

    def _generate_comprehensive_visualization(
        self,
        dino_features: torch.Tensor,  # (T, H, W, C)
        hidden_features: torch.Tensor,  # (C, T, H, W)
        output_dir: Path,
    ) -> None:
        """Generate comprehensive visualizations with input images and all features"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.decomposition import PCA
            
            T, H, W, C_dino = dino_features.shape
            
            # Get input images if available
            has_input_images = hasattr(self, '_current_batch_images')
            
            # Create a comprehensive visualization for each frame
            for t in range(T):
                # Determine the number of rows based on available data
                n_rows = 2  # minimum: hidden states + predicted dino
                if has_input_images and t == 0:
                    n_rows = 4  # add input images and input dino for first frame
                
                fig = plt.figure(figsize=(20, 5 * n_rows))
                
                current_row = 1
                
                # Row 1: Input images (only for first frame)
                if has_input_images and t == 0:
                    # Show input RGB static image
                    ax1 = plt.subplot(n_rows, 4, 1)
                    rgb_static = self._current_batch_images['rgb_static']
                    # Handle different shape formats
                    if rgb_static.ndim == 4 and rgb_static.shape[0] == 1:  # (1, C, H, W)
                        rgb_static = rgb_static.squeeze(0)  # Remove batch dimension
                    if rgb_static.shape[0] == 3:  # CHW format
                        rgb_static = rgb_static.permute(1, 2, 0)  # Convert to HWC
                    elif rgb_static.shape[-1] != 3:  # Not in HWC format
                        logger.warning(f"Unexpected RGB static shape: {rgb_static.shape}")
                    rgb_static_np = rgb_static.numpy()
                    # Normalize to [0, 1]
                    rgb_static_np = (rgb_static_np - rgb_static_np.min()) / (rgb_static_np.max() - rgb_static_np.min() + 1e-6)
                    ax1.imshow(rgb_static_np)
                    ax1.set_title("Input RGB Static")
                    ax1.axis("off")
                    
                    # Show input RGB gripper image
                    ax2 = plt.subplot(n_rows, 4, 2)
                    rgb_gripper = self._current_batch_images['rgb_gripper']
                    # Handle different shape formats
                    if rgb_gripper.ndim == 4 and rgb_gripper.shape[0] == 1:  # (1, C, H, W)
                        rgb_gripper = rgb_gripper.squeeze(0)  # Remove batch dimension
                    if rgb_gripper.shape[0] == 3:  # CHW format
                        rgb_gripper = rgb_gripper.permute(1, 2, 0)  # Convert to HWC
                    elif rgb_gripper.shape[-1] != 3:  # Not in HWC format
                        logger.warning(f"Unexpected RGB gripper shape: {rgb_gripper.shape}")
                    rgb_gripper_np = rgb_gripper.numpy()
                    # Normalize to [0, 1]
                    rgb_gripper_np = (rgb_gripper_np - rgb_gripper_np.min()) / (rgb_gripper_np.max() - rgb_gripper_np.min() + 1e-6)
                    ax2.imshow(rgb_gripper_np)
                    ax2.set_title("Input RGB Gripper")
                    ax2.axis("off")
                    
                    # Compute and show DINO features for input image
                    ax3 = plt.subplot(n_rows, 4, 3)
                    # Get first frame of perceptual features before hidden2dino
                    input_features = self._current_batch_images['perceptual_features_before_h2d']
                    if input_features.ndim == 4:  # (T, C, H, W)
                        input_feat = input_features[0]  # First frame
                    else:
                        input_feat = input_features
                    
                    # Compute feature norm
                    input_feat_norm = torch.norm(input_feat, dim=0).numpy()
                    input_feat_norm = (input_feat_norm - input_feat_norm.min()) / (input_feat_norm.max() - input_feat_norm.min() + 1e-6)
                    im3 = ax3.imshow(input_feat_norm, cmap="viridis")
                    ax3.set_title("Input Image Diffusion Features")
                    ax3.axis("off")
                    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
                    
                    # PCA visualization of input features
                    ax4 = plt.subplot(n_rows, 4, 4)
                    # Compute PCA for input features
                    C_in, H_in, W_in = input_feat.shape
                    input_flat = input_feat.permute(1, 2, 0).reshape(-1, C_in).numpy()
                    if input_flat.shape[0] > 3 and C_in > 3:
                        pca = PCA(n_components=3)
                        input_pca = pca.fit_transform(input_flat).reshape(H_in, W_in, 3)
                        # Normalize each channel to [0, 1]
                        for i in range(3):
                            channel = input_pca[..., i]
                            input_pca[..., i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
                    else:
                        input_pca = np.zeros((H_in, W_in, 3))
                    ax4.imshow(input_pca)
                    ax4.set_title("Input Features PCA RGB")
                    ax4.axis("off")
                    
                    current_row = 5
                
                # Row 2: Hidden states for current frame
                ax5 = plt.subplot(n_rows, 4, current_row)
                hidden_norm = torch.norm(hidden_features[:, t], dim=0).numpy()
                hidden_norm = (hidden_norm - hidden_norm.min()) / (hidden_norm.max() - hidden_norm.min() + 1e-6)
                im5 = ax5.imshow(hidden_norm, cmap="viridis")
                ax5.set_title(f"Hidden States Norm - Frame {t}")
                ax5.axis("off")
                plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
                
                # Hidden states PCA
                ax6 = plt.subplot(n_rows, 4, current_row + 1)
                C_hid, H_hid, W_hid = hidden_features.shape[0], hidden_features.shape[2], hidden_features.shape[3]
                hidden_flat = hidden_features[:, t].permute(1, 2, 0).reshape(-1, C_hid).numpy()
                if hidden_flat.shape[0] > 3 and C_hid > 3:
                    pca = PCA(n_components=3)
                    hidden_pca = pca.fit_transform(hidden_flat).reshape(H_hid, W_hid, 3)
                    for i in range(3):
                        channel = hidden_pca[..., i]
                        hidden_pca[..., i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
                else:
                    hidden_pca = np.zeros((H_hid, W_hid, 3))
                ax6.imshow(hidden_pca)
                ax6.set_title(f"Hidden States PCA RGB - Frame {t}")
                ax6.axis("off")
                
                # Row 3: Predicted DINO features
                ax7 = plt.subplot(n_rows, 4, current_row + 2)
                dino_norm = torch.norm(dino_features[t], dim=-1).numpy()
                dino_norm = (dino_norm - dino_norm.min()) / (dino_norm.max() - dino_norm.min() + 1e-6)
                im7 = ax7.imshow(dino_norm, cmap="viridis")
                ax7.set_title(f"Predicted DINO Norm - Frame {t}")
                ax7.axis("off")
                plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
                
                # Predicted DINO PCA
                ax8 = plt.subplot(n_rows, 4, current_row + 3)
                dino_flat = dino_features[t].reshape(-1, C_dino).numpy()
                if dino_flat.shape[0] > 3 and C_dino > 3:
                    pca = PCA(n_components=3)
                    dino_pca = pca.fit_transform(dino_flat).reshape(H, W, 3)
                    for i in range(3):
                        channel = dino_pca[..., i]
                        dino_pca[..., i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
                else:
                    dino_pca = np.zeros((H, W, 3))
                ax8.imshow(dino_pca)
                ax8.set_title(f"Predicted DINO PCA RGB - Frame {t}")
                ax8.axis("off")
                
                # Add main title
                if t == 0:
                    fig.suptitle("Input Images → Diffusion Features → Hidden States → Predicted DINO", fontsize=16)
                else:
                    fig.suptitle(f"Hidden States → Predicted DINO - Frame {t}", fontsize=16)
                
                plt.tight_layout()
                fig.savefig(output_dir / f"comprehensive_frame_{t:02d}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            # Generate global PCA visualization combining input and predicted DINO
            self._generate_global_dino_pca_visualization(dino_features, output_dir)
                
        except Exception as exc:
            logger.warning("Failed to generate comprehensive visualization: %s", exc)

    def _generate_pca_visualization(
        self,
        dino_features: torch.Tensor,  # (T, H, W, C)
        hidden_features: torch.Tensor,  # (C, T, H, W) 
        output_dir: Path,
    ) -> None:
        """Generate PCA RGB visualization of features"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            T, H, W, C_dino = dino_features.shape
            C_hidden = hidden_features.shape[0]
            
            # Reshape for PCA
            dino_flat = dino_features.reshape(-1, C_dino).float()
            hidden_flat = hidden_features.permute(1, 2, 3, 0).reshape(-1, C_hidden).float()
            
            # Compute PCA for DINO features
            mean_dino = dino_flat.mean(dim=0, keepdim=True)
            centered_dino = dino_flat - mean_dino
            _, _, V_dino = torch.pca_lowrank(centered_dino, q=3)
            dino_pca = (centered_dino @ V_dino[:, :3]).reshape(T, H, W, 3)
            
            # Normalize to [0, 1]
            for i in range(3):
                channel = dino_pca[..., i]
                dino_pca[..., i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
            
            # Compute PCA for hidden features
            mean_hidden = hidden_flat.mean(dim=0, keepdim=True)
            centered_hidden = hidden_flat - mean_hidden
            _, _, V_hidden = torch.pca_lowrank(centered_hidden, q=3)
            hidden_pca = (centered_hidden @ V_hidden[:, :3]).reshape(T, H, W, 3)
            
            # Normalize to [0, 1]
            for i in range(3):
                channel = hidden_pca[..., i]
                hidden_pca[..., i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
            
            # Save a few PCA frames
            frames_to_save = min(4, T)
            frame_indices = np.linspace(0, T-1, frames_to_save, dtype=int)
            
            for i, t in enumerate(frame_indices):
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                axes[0].imshow(hidden_pca[t].numpy())
                axes[0].set_title(f"Hidden PCA RGB - Frame {t}")
                axes[0].axis("off")
                
                axes[1].imshow(dino_pca[t].numpy())
                axes[1].set_title(f"DINO PCA RGB - Frame {t}")
                axes[1].axis("off")
                
                fig.suptitle(f"PCA Visualization - Frame {t}")
                fig.tight_layout()
                fig.savefig(output_dir / f"frame_{t:02d}_pca.png", dpi=150)
                plt.close(fig)
                
        except Exception as exc:
            logger.warning("Failed to generate PCA visualization: %s", exc)
            
    def _generate_global_dino_pca_visualization(
        self,
        predicted_dino_features: torch.Tensor,  # (T, H, W, C)
        output_dir: Path,
    ) -> None:
        """Generate global PCA visualization combining input DINO and predicted DINO features"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.decomposition import PCA
            
            T, H, W, C = predicted_dino_features.shape
            
            # Check if we have input features and images available
            if not hasattr(self, '_current_batch_images'):
                return
            
            # Check if we have stored the actual input DINO features
            if not hasattr(self, '_input_dino_features'):
                return
            
            # Get the stored input DINO features
            input_dino_features = self._input_dino_features  # Should be (T_input, H, W, C)
            
            # Move to CPU for visualization
            input_dino_features = input_dino_features.cpu()
            T_input, H_input, W_input, C_input = input_dino_features.shape
            
            # Resize input DINO features to match predicted size if needed
            if H_input != H or W_input != W:
                # Reshape to batch format for interpolation
                input_dino_resized = input_dino_features.permute(0, 3, 1, 2)  # (T, C, H, W)
                input_dino_resized = torch.nn.functional.interpolate(
                    input_dino_resized,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
                input_dino_resized = input_dino_resized.permute(0, 2, 3, 1)  # (T, H, W, C)
            else:
                input_dino_resized = input_dino_features
            
            # Concatenate all features for global PCA
            # Input frames: use all available input frames
            input_flat = input_dino_resized.reshape(-1, C_input).numpy()
            
            # Predicted frames: all frames
            predicted_flat = predicted_dino_features.reshape(-1, C).numpy()
            
            # Combine for global PCA (ensure same feature dimension)
            if C_input == C:
                all_features = np.concatenate([input_flat, predicted_flat], axis=0)
            else:
                logger.warning(f"Feature dimension mismatch: input {C_input} vs predicted {C}. Using predicted dimension.")
                # Project input features to match predicted dimension
                if C_input > C:
                    # Simple linear projection
                    input_proj = input_flat @ np.random.randn(C_input, C) / np.sqrt(C_input)
                else:
                    # Pad with zeros
                    input_proj = np.pad(input_flat, ((0, 0), (0, C - C_input)), mode='constant')
                all_features = np.concatenate([input_proj, predicted_flat], axis=0)
                
            # Compute global PCA
            pca = PCA(n_components=3)
            pca.fit(all_features)
            
            # Transform input and predicted features separately
            if C_input == C:
                input_pca = pca.transform(input_flat).reshape(T_input, H, W, 3)
            else:
                # Transform with projection
                input_pca = pca.transform(input_proj).reshape(T_input, H, W, 3)
                
            predicted_pca = pca.transform(predicted_flat).reshape(T, H, W, 3)
            
            # Normalize to [0, 1] for visualization
            def normalize_pca(pca_features):
                for i in range(3):
                    channel = pca_features[..., i]
                    min_val, max_val = channel.min(), channel.max()
                    if max_val > min_val:
                        pca_features[..., i] = (channel - min_val) / (max_val - min_val)
                return pca_features
            
            input_pca = normalize_pca(input_pca)
            predicted_pca = normalize_pca(predicted_pca)
            
            # Create visualization
            total_frames = T_input + T
            cols = min(5, total_frames)
            rows = (total_frames + cols - 1) // cols
            
            fig = plt.figure(figsize=(4 * cols, 4 * rows))
            
            # Plot input frames
            for i in range(T_input):
                ax = plt.subplot(rows, cols, i + 1)
                ax.imshow(input_pca[i])
                ax.set_title(f"Input DINO Frame {i}\n(Global PCA)", fontsize=10)
                ax.axis("off")
            
            # Plot predicted frames
            for t in range(T):
                ax = plt.subplot(rows, cols, T_input + t + 1)
                ax.imshow(predicted_pca[t])
                ax.set_title(f"Predicted DINO Frame {t}\n(Global PCA)", fontsize=10)
                ax.axis("off")
                
            plt.suptitle("Global PCA: Input DINO + Predicted DINO Features", fontsize=14)
            plt.tight_layout()
            fig.savefig(output_dir / "global_dino_pca_visualization.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Create comprehensive comparison visualization
            # Show: Input RGB | Input DINO | Hidden States | Predicted DINO
            num_frames_to_show = min(4, T_input, T)
            fig, axes = plt.subplots(num_frames_to_show, 4, figsize=(20, 5 * num_frames_to_show))
            
            if num_frames_to_show == 1:
                axes = axes.reshape(1, -1)
            
            for t in range(num_frames_to_show):
                # Column 1: Input RGB (if available)
                if hasattr(self, '_current_batch_images') and 'rgb_static' in self._current_batch_images:
                    rgb_img = self._current_batch_images['rgb_static']
                    if rgb_img.ndim == 4 and rgb_img.shape[0] == 1:
                        rgb_img = rgb_img.squeeze(0)
                    if rgb_img.shape[0] == 3:  # CHW to HWC
                        rgb_img = rgb_img.permute(1, 2, 0)
                    rgb_np = rgb_img.numpy()
                    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-6)
                    axes[t, 0].imshow(rgb_np)
                    axes[t, 0].set_title(f"Input RGB - Frame {t}")
                else:
                    axes[t, 0].text(0.5, 0.5, "No RGB Image", ha='center', va='center', transform=axes[t, 0].transAxes)
                    axes[t, 0].set_title(f"Input RGB - Frame {t}")
                axes[t, 0].axis("off")
                
                # Column 2: Input DINO (Global PCA)
                if t < T_input:
                    axes[t, 1].imshow(input_pca[t])
                    axes[t, 1].set_title(f"Input DINO - Frame {t}\n(Global PCA)")
                else:
                    axes[t, 1].text(0.5, 0.5, "No Input DINO", ha='center', va='center', transform=axes[t, 1].transAxes)
                    axes[t, 1].set_title(f"Input DINO - Frame {t}")
                axes[t, 1].axis("off")
                
                # Column 3: SVD Hidden States (if available)
                if hasattr(self, '_current_batch_images') and 'perceptual_features_before_h2d' in self._current_batch_images:
                    hidden_features = self._current_batch_images['perceptual_features_before_h2d']
                    if hidden_features.ndim == 4 and t < hidden_features.shape[0]:  # (T, C, H, W)
                        hidden_norm = torch.norm(hidden_features[t], dim=0).numpy()
                        hidden_norm = (hidden_norm - hidden_norm.min()) / (hidden_norm.max() - hidden_norm.min() + 1e-6)
                        im = axes[t, 2].imshow(hidden_norm, cmap="viridis")
                        plt.colorbar(im, ax=axes[t, 2], fraction=0.046, pad=0.04)
                        axes[t, 2].set_title(f"SVD Hidden States - Frame {t}")
                    else:
                        axes[t, 2].text(0.5, 0.5, "No Hidden States", ha='center', va='center', transform=axes[t, 2].transAxes)
                        axes[t, 2].set_title(f"SVD Hidden States - Frame {t}")
                else:
                    axes[t, 2].text(0.5, 0.5, "No Hidden States", ha='center', va='center', transform=axes[t, 2].transAxes)
                    axes[t, 2].set_title(f"SVD Hidden States - Frame {t}")
                axes[t, 2].axis("off")
                
                # Column 4: Predicted DINO (Global PCA)
                if t < T:
                    axes[t, 3].imshow(predicted_pca[t])
                    axes[t, 3].set_title(f"Predicted DINO - Frame {t}\n(Global PCA)")
                else:
                    axes[t, 3].text(0.5, 0.5, "No Prediction", ha='center', va='center', transform=axes[t, 3].transAxes)
                    axes[t, 3].set_title(f"Predicted DINO - Frame {t}")
                axes[t, 3].axis("off")
            
            plt.suptitle("Comprehensive Comparison: Input RGB → Input DINO → SVD Hidden States → Predicted DINO", fontsize=14)
            plt.tight_layout()
            fig.savefig(output_dir / "global_comprehensive_comparison.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as exc:
            logger.warning("Failed to generate global DINO PCA visualization: %s", exc)
            
    def _init_dino_encoder(self):
        """Initialize DINOv2 encoder for extracting real DINO features"""
        if not self._dino_encoder_initialized:
            try:
                logger.info("Loading DINOv2 ViT-B/14 model for feature extraction...")
                import torch.hub
                dinov2_dir = self.dinov2_path or os.environ.get(
                    "S_VAM_TORCH_HUB_DIR",
                    str(_PROJECT_ROOT / "checkpoints" / "torch_hub"),
                )
                torch.hub.set_dir(dinov2_dir)
                self._dino_encoder = torch.hub.load(
                    os.path.join(dinov2_dir, "facebookresearch_dinov2_main"),
                    "dinov2_vitb14_reg",
                    verbose=True,
                    source="local",
                )
                self._dino_encoder.head = torch.nn.Identity()
                self._dino_encoder = self._dino_encoder.to(self.device)
                self._dino_encoder.eval()
                for p in self._dino_encoder.parameters():
                    p.requires_grad = False
                self._dino_encoder_initialized = True
                logger.info("DINOv2 model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load DINOv2 encoder: {e}")
                self._dino_encoder = None
                
    def _init_dpa_encoder(self):
        """Initialize DA3 DinoV2 backbone for extracting real DPA features.

        Reuses load_da3_backbone from hidden2dpa/train_new.py.
        """
        if not self._dpa_encoder_initialized:
            try:
                import importlib
                import sys
                HIDDEN2DPA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hidden2dpa')
                DA3_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Depth-Anything-3', 'src')
                for p in [DA3_SRC, HIDDEN2DPA_DIR]:
                    if p in sys.path:
                        sys.path.remove(p)
                    sys.path.insert(0, p)
                # Force `model` to resolve to hidden2dpa.model. train_new uses `from model import ...`.
                sys.modules.pop("hidden2dpa.train_new", None)
                sys.modules.pop("model", None)
                sys.modules["model"] = importlib.import_module("hidden2dpa.model")
                from hidden2dpa.train_new import load_da3_backbone

                logger.info("Loading DA3 backbone for DPA feature extraction...")
                da3_model_dir = self.da3_path or os.environ.get(
                    "S_VAM_DA3_MODEL_DIR",
                    str(_PROJECT_ROOT / "checkpoints" / "da3-large"),
                )
                if not Path(da3_model_dir).exists():
                    da3_model_dir = getattr(self, 'da3_model_dir', 'depth-anything/DA3-LARGE')
                da3_ckpt = getattr(self, 'da3_backbone_ckpt', None)
                self._dpa_encoder = load_da3_backbone(
                    device=self.device,
                    ckpt_path=da3_ckpt,
                    model_dir=da3_model_dir,
                )
                self._dpa_encoder_initialized = True
                logger.info("DA3 DPA backbone loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load DPA encoder: {e}")
                self._dpa_encoder = None

    @torch.no_grad()
    def _extract_real_dino_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract real DINO features from input images using DINOv2"""
        if self._dino_encoder is None:
            self._init_dino_encoder()
            
        if self._dino_encoder is None:
            logger.warning("DINOv2 encoder not available")
            return None
            
        # images shape: (B, T, C, H, W) or (B, C, H, W)
        if images.ndim == 5:
            B, T, C, H, W = images.shape
            images_flat = images.view(B * T, C, H, W)
        else:
            B, C, H, W = images.shape
            T = 1
            images_flat = images
            
        # DINOv2 expects 224x224 images, resize if needed
        if H != 224 or W != 224:
            images_resized = F.interpolate(
                images_flat,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
        else:
            images_resized = images_flat
            
        # Extract features
        features_dict = self._dino_encoder.forward_features(images_resized)
        # Get patch tokens (excluding CLS token)
        patch_features = features_dict['x_norm_patchtokens']  # (B*T, num_patches, dim)
        
        # Reshape to spatial format
        # For ViT-B/14, we have 16x16 patches for 224x224 images
        num_patches = patch_features.shape[1]
        patch_h = patch_w = int(np.sqrt(num_patches))
        dim = patch_features.shape[2]
        
        # Reshape to (B*T, H, W, C)
        dino_features = patch_features.view(B * T, patch_h, patch_w, dim)
        
        if images.ndim == 5:
            # Reshape back to (B, T, H, W, C)
            dino_features = dino_features.view(B, T, patch_h, patch_w, dim)
            
        return dino_features

    @torch.no_grad()
    def _extract_real_dpa_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract real DPA features from first frame using DA3 backbone.

        Reuses extract_da3_targets from hidden2dpa/train_new.py.

        Args:
            images: (B, T, C, H, W) or (B, C, H, W) — only the first frame is used.

        Returns:
            ref_tokens: (B, 1, num_patches, C_token) suitable for HiddenToDA3ModelWithRef.
        """
        if self._dpa_encoder is None:
            self._init_dpa_encoder()

        if self._dpa_encoder is None:
            logger.warning("DPA encoder not available")
            return None

        # Force `model_spail_tem_attention` and `model` (used by hidden2dpa.train_new_st via
        # implicit-relative imports) to resolve to the hidden2dpa versions, otherwise Python
        # may pick up `hidden2dino/model_spail_tem_attention.py` from sys.path.
        import importlib as _importlib
        sys.modules.pop("model_spail_tem_attention", None)
        sys.modules["model_spail_tem_attention"] = _importlib.import_module(
            "hidden2dpa.model_spail_tem_attention"
        )
        sys.modules.pop("model", None)
        sys.modules["model"] = _importlib.import_module("hidden2dpa.model")
        from hidden2dpa.train_new_st import extract_da3_targets

        # images shape: (B, T, C, H, W) or (B, C, H, W)
        if images.ndim == 5:
            B, T, C, H, W = images.shape
            first_frame = images[:, 0:1]  # (B, 1, C, H, W)
        else:
            B, C, H, W = images.shape
            first_frame = images.unsqueeze(1)  # (B, 1, C, H, W)

        # Resize to 224x224 if needed (DA3 backbone expects 14-patch grid)
        if H != 224 or W != 224:
            first_frame = F.interpolate(
                first_frame.view(B, C, H, W),
                size=(224, 224),
                mode='bilinear',
                align_corners=False,
            ).unsqueeze(1)  # (B, 1, C, 224, 224)

        # extract_da3_targets expects (B, T, 3, H, W), returns List[(B, T, num_patches, C_token)]
        tokens_list = extract_da3_targets(first_frame, self._dpa_encoder)
        # Use stage 0 first-frame tokens as reference: (B, 1, num_patches, C_token)
        ref_tokens = tokens_list[0][:, :1]

        return ref_tokens


    def _prepare_lang_goal(self, lang_value, target_dtype: torch.dtype) -> torch.Tensor:
        """
        Ensure language embeddings are torch tensors on the correct device with shape (B, 1, C).
        """
        if lang_value is None:
            raise ValueError("Language embeddings are required but were not provided.")
        if not torch.is_tensor(lang_value):
            lang_value = torch.as_tensor(lang_value)
        lang_value = lang_value.to(self.device, dtype=target_dtype)
        if lang_value.ndim == 1:
            lang_value = lang_value.unsqueeze(0).unsqueeze(0)
        elif lang_value.ndim == 2:
            lang_value = lang_value.unsqueeze(1)
        return lang_value
            
    def extract_predictive_feature(self, dataset_batch):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        # 1. extract the revelant visual observations
        rgb_obs_dict = dataset_batch["rgb_obs"]
        rgb_static = rgb_obs_dict['rgb_static'].to(self.device)
        rgb_gripper = rgb_obs_dict.get('rgb_gripper', None)
        use_gripper = self.use_gripper_features and (rgb_gripper is not None)
        if use_gripper:
            rgb_gripper = rgb_gripper.to(self.device)
        rgb_static_future = rgb_obs_dict.get('rgb_static_future')
        if rgb_static_future is not None:
            rgb_static_future = rgb_static_future.to(self.device)
        rgb_gripper_future = rgb_obs_dict.get('rgb_gripper_future')
        if use_gripper and rgb_gripper_future is not None:
            rgb_gripper_future = rgb_gripper_future.to(self.device)
        # 3. we compute the language goal if the language modality is in the scope
        modality = "lang"
        if self.use_text_not_embedding:
            latent_goal = self.language_goal(dataset_batch["lang_text"]).to(rgb_static.dtype)
        else:
            lang_embeddings = dataset_batch.get("lang", None)
            latent_goal = self._prepare_lang_goal(lang_embeddings, rgb_static.dtype)

        language = dataset_batch["lang_text"]

        num_frames = self.Former_num_time_embeds
        rgb_static = rgb_static.to(self.device)
        batch = rgb_static.shape[0]
        use_concat_branch = False

        use_gt = self.use_gt_dino_condition

        with torch.no_grad():
            input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0) if use_gripper else rgb_static
            language_for_encoder = language + language if use_gripper else language
            perceptual_features = self.TVP_encoder(
                input_rgb,
                language_for_encoder,
                self.timestep,
                self.extract_layer_idx,
                all_layer=self.use_all_layer,
                step_time=1,
                max_length=self.max_length,
            )
        #perceptual_features = perceptual_features[:, :num_frames]
        #perceptual_features = einops.rearrange(perceptual_features, 'b f c h w-> b f c (h w)')
        #perceptual_features = einops.rearrange(perceptual_features, 'b f c l-> b f l c')
        perceptual_features = perceptual_features[:, :num_frames]
        if use_gripper:
            perceptual_features_static, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
        else:
            perceptual_features_static, gripper_feature = perceptual_features, None
        perceptual_features_static_token = einops.rearrange(perceptual_features_static, 'b f c h w-> b f c (h w)')
        perceptual_features_static_token = einops.rearrange(perceptual_features_static_token, 'b f c l-> b f l c')
        if use_gripper:
            gripper_feature_token = einops.rearrange(gripper_feature, 'b f c h w-> b f c (h w)')
            gripper_feature_token = einops.rearrange(gripper_feature_token, 'b f c l-> b f l c')
            perceptual_features = torch.cat([perceptual_features_static_token, gripper_feature_token], dim=2)
        else:
            perceptual_features = perceptual_features_static_token
        
        
        

        ref_dino = None
        if self.hidden2dino_use_ref:
            if rgb_static.ndim == 5:
                ref_images = rgb_static[:, 0:1]
            else:
                ref_images = rgb_static.unsqueeze(1)
            ref_dino = self._extract_real_dino_features(ref_images)
            if ref_dino is not None:
                ref_dino = ref_dino.permute(0, 4, 1, 2, 3)

        dpa_tokens = None
        ref_dpa = None
        if self.use_hidden_dino_dpa_concat:
            if rgb_static.ndim == 5:
                ref_images = rgb_static[:, 0:1]
            else:
                ref_images = rgb_static.unsqueeze(1)
            ref_dpa = self._extract_real_dpa_features(ref_images)
            if ref_dpa is None:
                logger.warning("ref_dpa is None in extract_predictive_feature.")
            dpa_source = perceptual_features_static
            dpa_tokens = self._run_hidden2dpa(dpa_source,ref_dpa)

        dino_tokens = self._run_hidden2dino(perceptual_features_static, ref_dino=ref_dino)
        self._save_dino_dpa_debug(
            ref_dino=ref_dino,
            dino_tokens=dino_tokens,
            ref_dpa=ref_dpa,
            dpa_tokens=dpa_tokens,
            source="train",
        )
        use_concat_branch = (self.use_hidden_dino_concat or self.use_hidden_dino_dpa_concat) and (not self.bypass_video_former)
        if use_concat_branch:
            if self.use_hidden_dino_dpa_concat:
                concat_source = perceptual_features_static_token
            else:
                concat_source = torch.cat([perceptual_features_static_token, gripper_feature_token], dim=3) if use_gripper else perceptual_features_static_token
            concat_list = [concat_source, dino_tokens]
            if self.use_hidden_dino_dpa_concat and dpa_tokens is not None:
                concat_list.append(dpa_tokens)
            perceptual_features = torch.cat(concat_list, dim=3)
            perceptual_features = self.Video_Former(perceptual_features)
        else:
            perceptual_features = dino_tokens.to(torch.float32)
            if use_gripper:
                gripper_feature = gripper_feature_token.to(torch.float32)
            else:
                gripper_feature = None
            if self.bypass_video_former:
                perceptual_features = self._bypass_video_former(perceptual_features)
            else:
                if use_gripper:
                    gripper_feature = self.Video_Former(gripper_feature)

            if (not use_concat_branch) or self.bypass_video_former:
                if self.use_Former=='linear':
                    perceptual_features = rearrange(perceptual_features, 'b T q d -> b (T q) d')
                perceptual_features = self.perceiver_resampler(perceptual_features)
            if use_gripper:
                perceptual_features = torch.cat([perceptual_features, gripper_feature], dim=1)

        predictive_feature = {'state_images': perceptual_features}
        predictive_feature['modality'] = modality
        if 'state_obs' in dataset_batch.keys():
            predictive_feature['state_obs'] = dataset_batch['state_obs'].to(self.device)
        
        return predictive_feature, latent_goal


    def _log_training_metrics(self, action_loss, total_loss, total_bs):
        """
        Log the training metrics.
        """
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

    def _log_validation_metrics(self, pred_loss, img_gen_loss, val_total_act_loss_pp):
        """
        Log the validation metrics.
        """
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / len(self.trainer.datamodule.modalities),  # type:ignore
            sync_dist=True,
        )
        self.log(f"val_act/img_gen_loss_pp", img_gen_loss, sync_dist=True)

    def diffusion_loss(
            self,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        self.model.train()
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        loss, _ = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas)
        return loss, sigmas, noise

    def denoise_actions(  # type: ignore
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            inference: Optional[bool] = False,
            extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10
        self.model.eval()
        if len(latent_goal.shape) < len(
                perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
            latent_goal = latent_goal.unsqueeze(1)  # .expand(-1, seq_len, -1)
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)

        x = torch.randn((len(latent_goal), self.act_window_size, self.action_dim), device=self.device) * self.sigma_max

        actions = self.sample_loop(sigmas, x, input_state, latent_goal, latent_plan, self.sampler_type, extra_args)

        return actions

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps * 1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def sample_loop(
            self,
            sigmas,
            x_t: torch.Tensor,
            state: torch.Tensor,
            goal: torch.Tensor,
            latent_plan: torch.Tensor,
            sampler_type: str,
            extra_args={},
    ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x: extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min,
                              disable=True)
        # ODE deterministic
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas),
                                  disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7,
                                     self.device)  # rho=7 is the default from EDM karras
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0
        self.last_predictive_feature = None

    def forward(self,batch):
        return self.training_step(batch)
        #def training_step(self, batch: Dict[str, Dict], batch_idx: int,
        #                  dataloader_idx: int = 0) -> torch.Tensor

    def eval_forward(self, obs, goal):
        """
        Method for doing inference with the model.
        """
        if self.use_text_not_embedding:
            if 'lang_text' not in goal:
                raise KeyError("Expected 'lang_text' in goal when use_text_not_embedding is True.")
            latent_goal = self.language_goal(goal["lang_text"]).to(torch.float32).to(self.device)
#            latent_goal = F.normalize(latent_goal, p=2, dim=-1, eps=1e-6)
        else:
            lang_embeddings = goal.get("lang", None)
            latent_goal = self._prepare_lang_goal(lang_embeddings, torch.float32)

        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"].get('rgb_gripper', None)
        use_gripper = self.use_gripper_features and (rgb_gripper is not None)
        language = goal["lang_text"]
       

        num_frames = self.Former_num_time_embeds
        rgb_static = rgb_static.to(self.device)
        if use_gripper:
            rgb_gripper = rgb_gripper.to(self.device)
        batch = rgb_static.shape[0]
        use_concat_branch = False

        with torch.no_grad():
            input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0) if use_gripper else rgb_static
            language_for_encoder = [language] + [language] if use_gripper else [language]
            perceptual_features = self.TVP_encoder(
                input_rgb,
                language_for_encoder,
                self.timestep,
                self.extract_layer_idx,
                all_layer=self.use_all_layer,
                step_time=1,
                max_length=self.max_length,
            )
        
        perceptual_features = perceptual_features[:, :num_frames]
        perceptual_features_for_record = perceptual_features.clone().detach().cpu()
        if use_gripper:
            perceptual_features_static, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
        else:
            perceptual_features_static, gripper_feature = perceptual_features, None
        perceptual_features_static_token = einops.rearrange(perceptual_features_static, 'b f c h w-> b f c (h w)')
        perceptual_features_static_token = einops.rearrange(perceptual_features_static_token, 'b f c l-> b f l c')
        if use_gripper:
            gripper_feature_token = einops.rearrange(gripper_feature, 'b f c h w-> b f c (h w)')
            gripper_feature_token = einops.rearrange(gripper_feature_token, 'b f c l-> b f l c')
            perceptual_features = torch.cat([perceptual_features_static_token, gripper_feature_token], dim=2)
        else:
            perceptual_features = perceptual_features_static_token
        
        
        

        ref_dino = None
        if self.hidden2dino_use_ref:
            if rgb_static.ndim == 5:
                ref_images = rgb_static[:, 0:1]
            else:
                ref_images = rgb_static.unsqueeze(1)
            ref_dino = self._extract_real_dino_features(ref_images)
            if ref_dino is not None:
                ref_dino = ref_dino.permute(0, 4, 1, 2, 3)

        dpa_tokens = None
        ref_dpa = None
        if self.use_hidden_dino_dpa_concat:
            if rgb_static.ndim == 5:
                ref_images = rgb_static[:, 0:1]
            else:
                ref_images = rgb_static.unsqueeze(1)
            ref_dpa = self._extract_real_dpa_features(ref_images)
            dpa_source = perceptual_features_static
            dpa_tokens = self._run_hidden2dpa(dpa_source,ref_dpa)

        dino_tokens = self._run_hidden2dino(perceptual_features_static, ref_dino=ref_dino)
        self._save_dino_dpa_debug(
            ref_dino=ref_dino,
            dino_tokens=dino_tokens,
            ref_dpa=ref_dpa,
            dpa_tokens=dpa_tokens,
            source="eval",
        )
        self.last_hidden2dino_tokens = dino_tokens.detach().cpu()
        self.last_hidden2dpa_tokens = dpa_tokens.detach().cpu() if dpa_tokens is not None else None
        use_concat_branch = (self.use_hidden_dino_concat or self.use_hidden_dino_dpa_concat) and (not self.bypass_video_former)
        if use_concat_branch:
            if self.use_hidden_dino_dpa_concat:
                concat_source = perceptual_features_static_token
            else:
                concat_source = torch.cat([perceptual_features_static_token, gripper_feature_token], dim=3) if use_gripper else perceptual_features_static_token
            concat_list = [concat_source, dino_tokens]
            if self.use_hidden_dino_dpa_concat and dpa_tokens is not None:
                concat_list.append(dpa_tokens)
            perceptual_features = torch.cat(concat_list, dim=3)
            perceptual_features = self.Video_Former(perceptual_features)
        else:
            perceptual_features = dino_tokens.to(torch.float32)
            gripper_feature = gripper_feature_token.to(torch.float32) if use_gripper else None
            if self.bypass_video_former:
                perceptual_features = self._bypass_video_former(perceptual_features)
            else:
                if use_gripper:
                    gripper_feature = self.Video_Former(gripper_feature)

            if (not use_concat_branch) or self.bypass_video_former:
                if self.use_Former=='linear':
                    perceptual_features = rearrange(perceptual_features, 'b T q d -> b (T q) d')
                perceptual_features = self.perceiver_resampler(perceptual_features)
            if use_gripper:
                perceptual_features = torch.cat([perceptual_features, gripper_feature], dim=1)

        predictive_feature = {'state_images': perceptual_features}
        predictive_feature['modality'] = "lang"
        self.last_predictive_feature = {
            "state_images": perceptual_features_for_record,
            "modality": "lang",
        }
        if 'state_obs' in obs.keys():
            predictive_feature['state_obs'] = obs['state_obs'].to(self.device)
            self.last_predictive_feature["state_obs"] = obs['state_obs'].detach().cpu()
        act_seq = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            predictive_feature,
            latent_goal,
            inference=True,
        )
        return act_seq

    def step(self, obs, goal):
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions.
        We only compute the sequence once every self.multistep steps.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        if self.rollout_step_counter % self.multistep == 0:
            pred_action_seq = self.eval_forward(obs, goal)

            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return current_action

    def on_train_start(self) -> None:

        self.model.to(dtype=self.dtype)

        if self.Video_Former is not None:
            self.Video_Former.to(dtype=self.dtype)
        if self.bypass_proj is not None:
            self.bypass_proj.to(dtype=self.dtype)
        self.language_goal.to(dtype=self.dtype)
        #self.vae.to(dtype=self.dtype)
        self.TVP_encoder.to(dtype=self.dtype)

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")


    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)