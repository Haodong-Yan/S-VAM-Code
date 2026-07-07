import torch.nn as nn
import torch.nn.functional as F
# import cv2
from torchvision.utils import save_image

import logging
from pathlib import Path
import sys
from typing import List, Union
import os
import wandb
from time import time
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import torch
import torch.multiprocessing as mp
from glob import glob
from copy import deepcopy
from collections import OrderedDict

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer

from policy_models.utils.utils import (
    get_git_commit_hash,
    get_last_checkpoint,
    initialize_pretrained_weights,
    print_system_env_info,
)
try:
    import swanlab
except ImportError:
    swanlab = None
from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
#from torch.nn.parallel import DistributedDataParallel as DDP

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # Skip DINOv2 encoder parameters as they are frozen and not part of EMA
        if '_dino_encoder' in name:
            continue
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        else:
            # Skip parameters that don't exist in EMA (e.g., dynamically added components)
            pass


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def normalize_clip_language(batch):
    """
    L2-normalize CLIP language embeddings to stabilize downstream layers.
    """
    if not isinstance(batch, dict):
        return
    lang = batch.get("lang", None)
    if isinstance(lang, torch.Tensor):
        batch["lang"] = F.normalize(lang, p=2, dim=-1, eps=1e-6)


def configure_future_rgb_inputs(cfg: DictConfig) -> None:
    """
    When GT-DINO conditioning is enabled, extend the datamodule configuration so it
    also streams future RGB frames aligned with each predicted action.
    """
    datamodule_cfg = getattr(cfg, "datamodule", None)
    model_cfg = getattr(cfg, "model", None)
    if datamodule_cfg is None or model_cfg is None:
        return
    use_gt = bool(getattr(model_cfg, "use_gt_dino_condition", False))
    datamodule_cfg.enable_future_rgb = use_gt
    if not use_gt:
        return

    lang_dataset_cfg = getattr(getattr(datamodule_cfg, "datasets", None), "lang_dataset", None)
    if lang_dataset_cfg is not None:
        worker_cap = int(os.environ.get("GT_DINO_MAX_WORKERS", 4))
        current_workers = getattr(lang_dataset_cfg, "num_workers", None)
        if isinstance(current_workers, int) and current_workers > worker_cap:
            print(
                f"[configure_future_rgb_inputs] Reducing lang_dataset.num_workers from "
                f"{current_workers} to {worker_cap} for GT-DINO mode to limit shared memory usage."
            )
            lang_dataset_cfg.num_workers = worker_cap

    future_pairs = [
        ("rgb_static", "rgb_static_future"),
        ("rgb_gripper", "rgb_gripper_future"),
    ]

    rgb_obs_list = list(datamodule_cfg.observation_space.rgb_obs)
    for _, future in future_pairs:
        if future not in rgb_obs_list:
            rgb_obs_list.append(future)
    datamodule_cfg.observation_space.rgb_obs = rgb_obs_list

    # align VideoFormer temporal slots with action horizon
    act_window = getattr(model_cfg, "act_window_size", None)
    if act_window is not None:
        model_cfg.Former_num_time_embeds = act_window

    transforms_cfg = datamodule_cfg.get("transforms", None)
    if transforms_cfg is None:
        return
    for split_key in ("train", "val"):
        if split_key not in transforms_cfg:
            continue
        split_cfg = transforms_cfg[split_key]
        for base, future in future_pairs:
            if base in split_cfg and future not in split_cfg:
                split_cfg[future] = deepcopy(split_cfg[base])


def create_logger(logging_dir, log_filename="log.txt"):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/{log_filename}")]
    )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################


#@hydra.main(config_path="./policy_conf", config_name="VPP_Calvinabc_train")
def train(cfg: DictConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    logger = logging.getLogger(__name__)
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = accelerator.device
    # new added
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    current_datetime = datetime.now()
    current_date_tag = current_datetime.strftime("%Y%m%d")
    run_tag = current_datetime.strftime("%Y%m%d_%H%M%S")
    checkpoint_interval = cfg.get("checkpoint_interval", 20000) if hasattr(cfg, "get") else 20000
    log_interval_steps = cfg.get("log_every", 50) if hasattr(cfg, "get") else 50
    default_loss_dir = Path(cfg.log_dir) / "loss"
    cfg_loss_dir = cfg.get("loss_dir", None) if hasattr(cfg, "get") else None
    loss_root_dir = Path(cfg_loss_dir) if cfg_loss_dir else default_loss_dir
    loss_dir = loss_root_dir / run_tag
    loss_history = []
    experiment_dir = f"{cfg.log_dir}/{run_tag}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    eval_dir = f"{experiment_dir}/eval"
    swanlab_run = None
    swanlab_project = getattr(cfg, "swanlab_project", getattr(cfg, "benchmark_name", "default"))
    swanlab_config = {
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "log_interval": log_interval_steps,
        "checkpoint_interval": checkpoint_interval,
    }
    try:
        swanlab_config["learning_rate"] = cfg.model.optimizer.learning_rate
    except Exception:
        pass


    if accelerator.is_main_process:
        os.makedirs(cfg.log_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        loss_dir.mkdir(parents=True, exist_ok=True)
        logger = create_logger(experiment_dir, f"log_{run_tag}.txt")
        logger.info(f"Experiment directory created at {experiment_dir}")
        config_dump_path = Path(experiment_dir) / "training_config.yaml"
        try:
            with config_dump_path.open("w", encoding="utf-8") as cfg_file:
                cfg_file.write(OmegaConf.to_yaml(cfg))
            logger.info(f"Training config saved to {config_dump_path}")
        except Exception as err:
            logger.error(f"Failed to write training config: {err}")
        #logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
        logger.info(f"Checkpoint & log interval set to every {checkpoint_interval:,} steps")
        if swanlab is not None:
            try:
                swanlab_run = swanlab.init(
                    project=swanlab_project,
                    name=run_tag,
                    config=swanlab_config,
                )
                logger.info(f"swanlab experiment initialized under project {swanlab_project}")
            except Exception as err:
                logger.error(f"Failed to initialize swanlab experiment: {err}")
    accelerator.wait_for_everyone()

    configure_future_rgb_inputs(cfg)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    if accelerator.is_main_process:
        logger.info(f"Global batch size {cfg.batch_size:,} num_processes ({accelerator.num_processes})")
    chk = get_last_checkpoint(Path.cwd())
    train_loader = datamodule.train_dataloader()["lang"]
    val_loader = datamodule.val_dataloader()["lang"]
    # Load Model
    model = hydra.utils.instantiate(cfg.model)
    if "pretrain_chk" in cfg:
        initialize_pretrained_weights(model, cfg)

    if cfg.use_ckpt_path:
        state_dict = torch.load(cfg.ckpt_path, map_location='cpu')
        # print('state_dict_key:', state_dict['model'].keys())
        print('load_from_ckpt:',cfg.ckpt_path)
        # c = []
        # hydra.initialize(config_path="../../conf")
        # hydra.main(config_name="config_abc.yaml")(lambda x: c.append(x))()
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict['model'])

    model = model.to(device)
    model.process_device()


    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = model.configure_optimizers()["optimizer"]
    Ir_scheduler = model.configure_optimizers()["lr_scheduler"]["scheduler"]

    model.on_train_start()
    if accelerator.is_main_process:
        logger.info(f"model parameter init")
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()
    model.train()
    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)
   # model = DDP(model, find_unused_parameters=True)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    eval_batch = None
    best_eval_loss = 1e8
    current_epoch = 0

    def log_and_save_interval(reason="interval", save_checkpoint=True):
        nonlocal running_loss, log_steps, start_time, current_epoch
        if log_steps == 0:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time()
        duration = max(end_time - start_time, 1e-6)
        steps_per_sec = log_steps / duration
        if torch.is_tensor(running_loss):
            avg_loss_value = (running_loss / log_steps).detach().item()
        else:
            avg_loss_value = running_loss / log_steps

        if accelerator.is_main_process:
            logger.info(
                f"(run={run_tag}) (step={train_steps:07d}) [{reason}] Train total Loss : {avg_loss_value:.6f}, "
                f"Train Steps/Sec: {steps_per_sec:.2f}"
            )
            if swanlab_run is not None:
                swanlab.log(
                    {
                        "train_loss": avg_loss_value,
                        "train_steps_per_sec": steps_per_sec,
                        "global_step": train_steps,
                    }
                )
            loss_history.append((train_steps, avg_loss_value))
            if save_checkpoint:
                scheduler_state = None
                if Ir_scheduler is not None and hasattr(Ir_scheduler, "state_dict"):
                    try:
                        scheduler_state = Ir_scheduler.state_dict()
                    except Exception as err:
                        logger.error(f"Failed to serialize lr scheduler state: {err}")
                checkpoint = {
                    "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                    "args": cfg,
                    "optimizer": opt.state_dict(),
                    "train_steps": train_steps,
                    "epoch": current_epoch,
                }
                if scheduler_state is not None:
                    checkpoint["lr_scheduler"] = scheduler_state
                checkpoint_path = f"{checkpoint_dir}/{run_tag}_step{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved step checkpoint to {checkpoint_path}")
                
                # Generate and save loss curve every checkpoint interval
                if loss_history:
                    steps, losses = zip(*loss_history)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(steps, losses, marker="o", linewidth=2, markersize=4)
                    ax.set_xlabel("Train Steps")
                    ax.set_ylabel("Avg Loss")
                    ax.set_title(f"Training Loss Curve ({run_tag}) - Step {train_steps}")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    loss_curve_path = loss_dir / f"loss_curve_step{train_steps:07d}.png"
                    fig.tight_layout()
                    fig.savefig(loss_curve_path, dpi=200)
                    plt.close(fig)
                    logger.info(f"Loss curve saved to {loss_curve_path}")

        running_loss = 0
        log_steps = 0
        start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {cfg.max_epochs} epochs...")

    nan_detected = False
    for epoch in range(cfg.max_epochs):
        current_epoch = epoch
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        running_loss = 0

        for idx,data_batch in enumerate(train_loader):
            # normalize_clip_language(data_batch)
            with accelerator.autocast():
                loss = model(data_batch)

            loss_is_nan=torch.tensor(0.0,device=accelerator.device)
            if not torch.isfinite(loss):
                loss_is_nan=torch.tensor(1.0,device=accelerator.device)
            global_is_nan=accelerator.reduce(loss_is_nan,reduction="sum")

            if global_is_nan>0:
                loss_value=loss.detach().item()
   
                if accelerator.is_main_process:
                    accelerator.print(
                        f"[NaN/Inf Loss] epoch={epoch}, step={train_steps}, batch_idx={idx}, loss={loss_value}"
                    )
                    accelerator.print(f"[NaN/Inf Loss] Batch snapshot: {data_batch}")
                if accelerator.is_main_process:
                    nan_batch_path = Path(experiment_dir) / f"nan_batch_step{train_steps:07d}_idx{idx:05d}.pt"
                    try:
                        cpu_batch = {}
                        for key, value in data_batch.items():
                            if torch.is_tensor(value):
                                cpu_batch[key] = value.detach().cpu()
                            elif isinstance(value, (list, tuple)):
                                cpu_batch[key] = [
                                    item.detach().cpu() if torch.is_tensor(item) else item for item in value
                                ]
                            else:
                                cpu_batch[key] = value
                        torch.save(cpu_batch, nan_batch_path)
                        logger.info(f"Saved NaN batch snapshot to {nan_batch_path}")
                    except Exception as err:
                        logger.error(f"Failed to save NaN batch snapshot: {err}")
                log_and_save_interval(reason="nan_detected", save_checkpoint=True)
                if accelerator.is_main_process:
                    logger.info("NaN detected. Latest metrics and checkpoint saved before exiting.")
                nan_detected = True
                break
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            Ir_scheduler.step()
            update_ema(ema, model)
            running_loss += loss
            log_steps += 1
            train_steps += 1
            if train_steps > 0 and train_steps % checkpoint_interval == 0:
                log_and_save_interval(reason="checkpoint_interval", save_checkpoint=True)
            elif train_steps > 0 and train_steps % log_interval_steps == 0:
                log_and_save_interval(reason="log_interval", save_checkpoint=False)
            
        if nan_detected:
            break

        model.eval()

        total_val_loss = 0.0
        val_steps=0

        if accelerator.is_main_process:
            logger.info(f"Finished training epoch {epoch}")
            logger.info(f"started validation epoch {epoch}")

        for val_batch in val_loader:
            #normalize_clip_language(val_batch)
            with torch.no_grad():
                if hasattr(model, 'module'):
                    val_dict=model.module.validation_step(val_batch)
                else:
                    val_dict=model.validation_step(val_batch)

                loss_tensor=val_dict["validation_loss"]
                gathered_loss=accelerator.gather(loss_tensor)
                total_val_loss+=gathered_loss.mean().item()
                val_steps+=1
        
        model.train()
        if accelerator.is_main_process:
            avg_val_loss = total_val_loss/max(val_steps,1)

            logger.info(f"Validation Loss: {avg_val_loss:.6f}")
            if swanlab_run is not None:
                swanlab.log(
                    {
                        "val_loss": avg_val_loss,
                        "epoch": epoch,
                        "global_step": train_steps,
                    }
                )

            scheduler_state = None
            if Ir_scheduler is not None and hasattr(Ir_scheduler, "state_dict"):
                try:
                    scheduler_state = Ir_scheduler.state_dict()
                except Exception as err:
                    logger.error(f"Failed to serialize lr scheduler state: {err}")
            checkpoint = {
                "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                "args": cfg,
                "optimizer": opt.state_dict(),
                "train_steps": train_steps,
                "epoch": current_epoch,
            }
            if scheduler_state is not None:
                checkpoint["lr_scheduler"] = scheduler_state
            
            if avg_val_loss < best_eval_loss:
                checkpoint_path = f"{checkpoint_dir}/{run_tag}_{train_steps:07d}_{avg_val_loss:.3f}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                best_eval_loss = avg_val_loss
            last_path = f"{checkpoint_dir}/{run_tag}_last.pt"
            torch.save(checkpoint, last_path)
           

    log_and_save_interval(reason="final_flush", save_checkpoint=True)

    if accelerator.is_main_process and loss_history:
        steps, losses = zip(*loss_history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, losses, marker="o", linewidth=2, markersize=4)
        ax.set_xlabel("Train Steps")
        ax.set_ylabel("Avg Loss")
        ax.set_title(f"Training Loss Curve ({run_tag})")
        ax.grid(True, linestyle="--", alpha=0.5)
        loss_curve_path = loss_dir / "loss_curve.png"
        fig.tight_layout()
        fig.savefig(loss_curve_path, dpi=200)
        plt.close(fig)
        csv_path = loss_dir / "loss_history.csv"
        with open(csv_path, "w") as f:
            f.write("step,loss\n")
            for step, loss_value in loss_history:
                f.write(f"{step},{loss_value}\n")
        logger.info(f"Loss curve saved to {loss_curve_path}")
        logger.info(f"Loss history csv saved to {csv_path}")

    if accelerator.is_main_process and swanlab_run is not None:
        finish_fn = getattr(swanlab, "finish", None)
        if callable(finish_fn):
            finish_fn()

    # Setup accelerator:

def setup_logger(cfg: DictConfig, model: LightningModule):
    """
    Set up the logger (tensorboard or wandb) from hydra config.

    Args:
        cfg: Hydra config
        model: LightningModule

    Returns:
        logger
    """
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = pathlib_cwd.parent.name + "/" + pathlib_cwd.name
        cfg.logger.id = cfg.logger.name.replace("/", "_")
        train_logger = hydra.utils.instantiate(cfg.logger)
        # train_logger.watch(model)
    else:
        train_logger = hydra.utils.instantiate(cfg.logger)
    return train_logger

if __name__ == "__main__":
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--text_encoder_path", type=str, default="")
    parser.add_argument("--hidden2dino_ckpt", type=str, required=True, help="Path to hidden2dino checkpoint.")
    parser.add_argument("--hidden2dpa_ckpt", type=str, required=True, help="Path to hidden2dpa checkpoint.")
    parser.add_argument("--dinov2_path", type=str, required=True, help="Path to DINOv2 torch hub directory.")
    parser.add_argument("--da3_path", type=str, required=True, help="Path to Depth Anything 3 model directory.")
    parser.add_argument("--root_data_dir", type=str, default="")
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default=None,
        help="Optional comma separated list of CUDA device indices to expose (e.g. '2' or '2,3').",
    )
    parser.add_argument(
        "--pipeline_cpu_offload",
        action="store_true",
        help="Enable CPU offloading for the diffusion pipeline to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--debug_hidden2dino",
        action="store_true",
        help="Dump intermediate Hidden2DINO tensors for debugging.",
    )
    parser.add_argument(
        "--debug_hidden2dino_dir",
        type=str,
        default=None,
        help="Directory to store Hidden2DINO debug tensors (defaults to ./debug_hidden2dino).",
    )
    parser.add_argument(
        "--use_gt_dino_condition",
        action="store_true",
        help="Use ground-truth DINO features from RGB frames as VideoFormer condition (upper bound).",
    )
    parser.add_argument(
        "--gt_dino_chunk",
        type=int,
        default=32,
        help="Chunk size when extracting ground-truth DINO features.",
    )
    parser.add_argument(
        "--bypass_video_former",
        action="store_true",
        help="Skip VideoFormer and feed (downsampled) DINO tokens directly into the diffusion policy.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Override log directory (experiment folders will be created inside).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override global batch size.",
    )
    args = parser.parse_args()
    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_devices)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    from hydra import compose, initialize

    with initialize(config_path="./policy_conf", job_name="VPP_Calvinabc_train"):
        cfg = compose(config_name="VPP_Calvinabc_train")
    
    # Allow adding new keys to config
    OmegaConf.set_struct(cfg, False)

    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.text_encoder_path
    cfg.root_data_dir = args.root_data_dir
    cfg.datamodule.root_data_dir = args.root_data_dir
    if args.pipeline_cpu_offload:
        cfg.model.use_pipeline_cpu_offload = True
    if args.debug_hidden2dino:
        cfg.model.debug_hidden2dino = True
    if args.debug_hidden2dino_dir is not None:
        cfg.model.debug_hidden2dino_dir = args.debug_hidden2dino_dir
    
    if args.use_gt_dino_condition:
        cfg.model.use_gt_dino_condition = True
    if args.gt_dino_chunk is not None:
        cfg.model.gt_dino_chunk = args.gt_dino_chunk
    if args.bypass_video_former:
        cfg.model.bypass_video_former = True
    if args.log_dir:
        cfg.log_dir = args.log_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.model.hidden2dino_ckpt = args.hidden2dino_ckpt
    cfg.model.hidden2dpa_ckpt = args.hidden2dpa_ckpt
    cfg.model.dinov2_path = args.dinov2_path
    cfg.model.da3_path = args.da3_path

    train(cfg)