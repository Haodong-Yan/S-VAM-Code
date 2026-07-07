"""Training script: hidden states -> Depth Anything 3 (DualDPT) patch tokens."""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

# Reuse dataset from hidden2dino
HIDDEN2DINO_DIR = os.path.join(PARENT_DIR, "hidden2dino")
if HIDDEN2DINO_DIR not in sys.path:
    sys.path.append(HIDDEN2DINO_DIR)

# Depth-Anything-3 src path
DA3_SRC = os.path.join(PARENT_DIR, "Depth-Anything-3", "src")
if DA3_SRC not in sys.path:
    sys.path.append(DA3_SRC)

from hidden_feature_dataset import HiddenFeatureDataset  # type: ignore
from model_spail_tem_attention import HiddenToDA3Model, HiddenToDA3ModelWithRef
from depth_anything_3.api import DepthAnything3  # type: ignore
from depth_anything_3.model.dinov2.dinov2 import DinoV2  # type: ignore
from depth_anything_3.utils.logger import logger as da3_logger  # type: ignore
import logging

# Silence DA3 backbone info logs (e.g., reference view selection)
if hasattr(da3_logger, "setLevel"):
    da3_logger.setLevel(logging.WARNING)
elif hasattr(da3_logger, "info"):
    da3_logger.info = lambda *args, **kwargs: None


DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "configs", "model_default.yaml")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_hf_cache_snapshot_dir(model_dir: str) -> str:
    """Resolve HuggingFace cache root to an actual snapshot directory.

    People sometimes pass a HF cache folder like:
      .../hub/models--org--repo
    but `from_pretrained` expects either a repo id or a *model folder* that
    directly contains `config.json` and `model.safetensors`.
    """
    p = Path(model_dir).expanduser()
    if not p.is_dir():
        return model_dir

    # Already a valid local model dir.
    if (p / "config.json").is_file():
        return str(p)

    snapshots = p / "snapshots"
    if snapshots.is_dir():
        # Prefer the commit referenced by refs/main if present.
        ref_main = p / "refs" / "main"
        if ref_main.is_file():
            try:
                commit = ref_main.read_text(encoding="utf-8").strip()
                cand = snapshots / commit
                if (cand / "config.json").is_file():
                    return str(cand)
            except Exception:
                pass

        # Fallback: pick the most recently modified snapshot containing config.json.
        try:
            candidates = [d for d in snapshots.iterdir() if d.is_dir() and (d / "config.json").is_file()]
            if candidates:
                candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                return str(candidates[0])
        except Exception:
            pass

    # Leave unchanged (will error later with a clearer message).
    return str(p)


@torch.no_grad()
def load_da3_backbone(
    device: torch.device,
    ckpt_path: str | None = None,
    model_dir: str = "depth-anything/DA3NESTED-GIANT-LARGE",
) -> nn.Module:
    """Load DA3 DinoV2 backbone (vitb, cat_token=True) used by DualDPT."""
    if ckpt_path:
        print("[Init] Loading Depth Anything 3 DinoV2 backbone (vitb/14) from checkpoint", flush=True)
        backbone = DinoV2(
            name="vitb",
            out_layers=[11, 15, 19, 23],
            alt_start=4,
            qknorm_start=4,
            rope_start=4,
            cat_token=True,
        )
        print(f"[Init] Loading backbone checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            for key in ["model", "state_dict", "weights"]:
                if key in state:
                    state = state[key]
                    break
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"[Init] Backbone loaded with missing={len(missing)}, unexpected={len(unexpected)}")
        backbone = backbone.to(device)
    else:
        resolved_model_dir = _resolve_hf_cache_snapshot_dir(model_dir)
        if resolved_model_dir != model_dir:
            print(f"[Init] Resolved model_dir '{model_dir}' -> '{resolved_model_dir}'", flush=True)
        print(
            f"[Init] Loading DA3 backbone via DepthAnything3.from_pretrained('{resolved_model_dir}')",
            flush=True,
        )
        try:
            if Path(resolved_model_dir).is_dir():
                try:
                    da3 = DepthAnything3.from_pretrained(resolved_model_dir, local_files_only=True)
                except TypeError:
                    da3 = DepthAnything3.from_pretrained(resolved_model_dir)
            else:
                da3 = DepthAnything3.from_pretrained(resolved_model_dir)
        except Exception as e:
            raise RuntimeError(
                "Failed to load DepthAnything3 weights. If you passed a HuggingFace cache root folder "
                "(e.g., .../hub/models--depth-anything--DA3NESTED-GIANT-LARGE), please point to a snapshot "
                "subfolder that contains config.json/model.safetensors, e.g. "
                f"'{model_dir}/snapshots/<commit_hash>'."
            ) from e
        da3 = da3.to(device=device)
        net = da3.model
        if hasattr(net, "backbone"):
            backbone = net.backbone.to(device)
        elif hasattr(net, "da3") and hasattr(net.da3, "backbone"):
            backbone = net.da3.backbone.to(device)
        else:
            raise AttributeError("Loaded DA3 model has no accessible backbone attribute.")

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print("[Init] DA3 backbone ready.")
    return backbone


@torch.no_grad()
def extract_da3_targets(
    frames: torch.Tensor,
    backbone: nn.Module,
) -> List[torch.Tensor]:
    """Extract DA3 patch tokens from frames using DinoV2 backbone.

    Args:
        frames: (B, T, 3, H, W) normalized to ImageNet mean/std.
    Returns:
        List of tokens per stage; each: (B, T, num_patches, C_token)
    """
    outputs, _ = backbone(frames)
    tokens_list: List[torch.Tensor] = []
    for tokens, _cam_token in outputs:
        # tokens shape: (B, T, num_patches, C_token) after get_intermediate_layers
        tokens_list.append(tokens)
    return tokens_list


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    split_file: str | None,
    train_ratio: float,
    use_dummy: bool,
    use_gripper: bool,
    use_base_camera: bool,
) -> Tuple[HiddenFeatureDataset, HiddenFeatureDataset | None, DataLoader, DataLoader | None]:
    train_dataset = HiddenFeatureDataset(
        data_dir=data_dir,
        split="train",
        dummy=use_dummy,
        split_file=split_file,
        train_ratio=train_ratio,
        use_gripper=use_gripper,
        use_base_camera=use_base_camera,
    )

    if use_dummy:
        val_dataset = None
    else:
        val_dataset = HiddenFeatureDataset(
            data_dir=data_dir,
            split="val",
            dummy=False,
            split_file=split_file,
            train_ratio=train_ratio,
            use_gripper=use_gripper,
            use_base_camera=use_base_camera,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not use_dummy,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if val_dataset is None or len(val_dataset) == 0:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=True,
        )

    return train_dataset, val_dataset, train_loader, val_loader


def _compute_stage_loss(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
) -> torch.Tensor:
    if len(preds) != len(targets):
        raise ValueError(f"Stage count mismatch: pred={len(preds)}, tgt={len(targets)}")
    losses = []
    for p, t in zip(preds, targets):
        if p.shape[-1] != t.shape[-1]:
            raise ValueError(
                f"Channel mismatch pred={p.shape[-1]} tgt={t.shape[-1]} — please set model token_dim to {t.shape[-1]} in config/args."
            )
        losses.append(F.mse_loss(p, t.to(p.dtype)))
    return sum(losses) / len(losses)


def _get_num_stages(model: nn.Module) -> int:
    if hasattr(model, "num_stages"):
        return int(getattr(model, "num_stages"))
    if hasattr(model, "base_model") and hasattr(model.base_model, "num_stages"):
        return int(getattr(model.base_model, "num_stages"))
    raise AttributeError("Cannot infer num_stages from model.")


def _resolve_stage_idx(requested_idx: int, num_stages: int) -> int:
    if num_stages <= 0:
        raise ValueError(f"num_stages should be > 0, got {num_stages}")
    if requested_idx < 0:
        requested_idx += num_stages
    if requested_idx < 0 or requested_idx >= num_stages:
        raise ValueError(
            f"target_stage_idx={requested_idx} out of range, expected [-{num_stages}, {num_stages - 1}]"
        )
    return requested_idx


def train_one_epoch(
    model: nn.Module,
    backbone: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    use_ref_frame: bool,
    target_stage_idx: int,
) -> float:
    model.train()
    total_loss = 0.0
    num_samples = 0
    progress = tqdm(loader, desc="Train", leave=False)

    for frames, hidden_states in progress:
        frames = frames.to(device)
        hidden_states = hidden_states.to(device)
        first_frame = frames[:, 0:1, :, :, :]
        

        with torch.no_grad():
            target_stages = extract_da3_targets(frames, backbone)
            stage_idx = _resolve_stage_idx(target_stage_idx, len(target_stages))
            targets = target_stages[stage_idx]
            ref_tokens_from_first_frame = extract_da3_targets(first_frame, backbone)
            ref_tokens_from_first_frame = ref_tokens_from_first_frame[stage_idx][:, :1]
        optimizer.zero_grad(set_to_none=True)

        ref_tokens: torch.Tensor | None = None
        if use_ref_frame:
            # use stage 0 first-frame tokens as reference: (B, 1, H*W, token_dim)
            ref_tokens = ref_tokens_from_first_frame 

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            if use_ref_frame:
                

                preds_full = model(hidden_states, ref_tokens)  # list of (tokens, cam_token)
            else:
                preds_full = model(hidden_states)  # list of (tokens, cam_token)
            
            print("debug",targets.mean(), targets.std())
            print("debug",preds_full.mean(), preds_full.std())
            #preds_tokens = [p[0] for p in preds_full]
            loss = F.mse_loss(preds_full, targets)
            print("debug",loss.item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = frames.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, num_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    backbone: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    use_amp: bool,
    use_ref_frame: bool,
    target_stage_idx: int,
) -> float | None:
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    num_samples = 0
    progress = tqdm(loader, desc="Val", leave=False)

    for frames, hidden_states in progress:
        frames = frames.to(device)
        hidden_states = hidden_states.to(device)
        first_frame = frames[:, 0:1, :, :, :]

        target_stages = extract_da3_targets(frames, backbone)
        stage_idx = _resolve_stage_idx(target_stage_idx, len(target_stages))
        targets = target_stages[stage_idx]
        ref_tokens_from_first_frame = extract_da3_targets(first_frame, backbone)
        ref_tokens_from_first_frame = ref_tokens_from_first_frame[stage_idx][:, :1]
        ref_tokens: torch.Tensor | None = ref_tokens_from_first_frame if use_ref_frame else None

        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            if use_ref_frame:

                preds_full = model(hidden_states, ref_tokens)
            else:
                preds_full = model(hidden_states)
            
            loss = F.mse_loss(preds_full, targets)

        batch_size = frames.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, num_samples)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    output_dir: str,
    tag: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save(
        {
            "epoch": epoch,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_path,
    )
    print(f"[Checkpoint] Saved to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hidden -> DepthAnything3 token training")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root (sample_xxx)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--use_dummy", action="store_true", help="Use dummy data for quick debug")
    parser.add_argument("--config", type=str, default=None, help="Model hyperparam YAML")
    parser.add_argument("--split_file", type=str, default=None, help="Split file, default data_dir/split_indices.json")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio (0-1)")
    parser.add_argument("--use_gripper", action="store_true", help="Use gripper view samples")
    parser.add_argument("--no_base_camera", action="store_true", help="Disable base camera samples")
    parser.add_argument("--backbone_ckpt", type=str, default=None, help="Optional DA3 backbone checkpoint")
    parser.add_argument("--model_dir", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE", help="DA3 pretrained model dir or HF repo id")
    parser.add_argument("--use_ref_frame", action="store_true", help="Use first-frame DA3 tokens as condition")
    parser.add_argument(
        "--target_stage_idx",
        type=int,
        default=-1,
        help="DA3 stage to supervise. 0=shallowest, -1=deepest(last).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Device: {device}")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_file=args.split_file,
        train_ratio=args.train_ratio,
        use_dummy=args.use_dummy,
        use_gripper=args.use_gripper,
        use_base_camera=not args.no_base_camera,
    )

    if not args.use_dummy:
        split_manifest = getattr(train_dataset, "split_file", None)
        print(
            f"[Data] split file: {split_manifest or os.path.join(args.data_dir, 'split_indices.json')}"
        )
        print(f"[Data] train samples: {len(train_dataset)}, val samples: {len(val_dataset) if val_dataset else 0}")

    # Infer shapes from first sample
    example_hidden = train_dataset[0][1]
    C_in, T, H, W = example_hidden.shape
    print("DEBUG", C_in, T, H, W)
    token_dim = 2048  # DinoV2 vitb with cat_token=True

    config_path = args.config or DEFAULT_CONFIG_PATH
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}
    model_cfg = config_data.get("model", {})

    base_model = HiddenToDA3Model(
        C_in=C_in,
        C_out=model_cfg.get("token_dim", token_dim),
        T=T,
        H=H,
        W=W,
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        num_layers=model_cfg.get("num_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    if args.use_ref_frame:
        model = HiddenToDA3ModelWithRef(base_model, ref_dim=model_cfg.get("token_dim", token_dim))
    else:
        model = base_model

    # base_model 已经在 device 上，但 use_ref_frame 时新建的 adapter 默认仍在 CPU。
    # 按 hidden2dino/train.py 的写法，确保 wrap 后的整体模型都搬到 device，避免 error_213855。
    model = model.to(device)

    print(f"[Init] Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"[Init] Requested target_stage_idx={args.target_stage_idx}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    backbone = load_da3_backbone(device, ckpt_path=None, model_dir=args.model_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    resolved_config = {
        "model": {
            "hidden_dim": model_cfg.get("hidden_dim", 1024),
            "num_layers": model_cfg.get("num_layers", 4),
            "num_heads": model_cfg.get("num_heads", 8),
            "dropout": model_cfg.get("dropout", 0.1),
            "C_in": C_in,
            "token_dim": model_cfg.get("token_dim", token_dim),
            "T": T,
            "H": H,
            "W": W,
            "num_stages": model_cfg.get("num_stages", 4),
            "camera_pool": model_cfg.get("camera_pool", "mean"),
            "use_stage_embed": bool(model_cfg.get("use_stage_embed", True)),
        },
        "source_config": os.path.abspath(config_path),
        "split_file": getattr(train_dataset, "split_file", None),
        "train_ratio": args.train_ratio,
        "use_gripper": args.use_gripper,
        "use_base_camera": not args.no_base_camera,
        "backbone_ckpt": args.backbone_ckpt,
        "model_dir": args.model_dir,
        "use_ref_frame": args.use_ref_frame,
        "target_stage_idx": args.target_stage_idx,
    }
    with open(os.path.join(run_dir, "model_config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, allow_unicode=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss = train_one_epoch(
            model,
            backbone,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp=args.use_amp,
            use_ref_frame=args.use_ref_frame,
            target_stage_idx=args.target_stage_idx,
        )
        print(f"[Train] loss={train_loss:.6f}")
        writer.add_scalar("train/loss", train_loss, epoch)

        val_loss = evaluate(
            model,
            backbone,
            val_loader,
            device,
            use_amp=args.use_amp,
            use_ref_frame=args.use_ref_frame,
            target_stage_idx=args.target_stage_idx,
        )
        if val_loss is not None:
            print(f"[Val] loss={val_loss:.6f}")
            writer.add_scalar("val/loss", val_loss, epoch)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, epoch, best_val, run_dir, "best")

        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, best_val, run_dir, f"epoch{epoch}")

    save_checkpoint(model, optimizer, args.epochs, best_val, run_dir, "last")
    writer.close()
    print(f"Training finished, logs at {run_dir}")


if __name__ == "__main__":
    main()


