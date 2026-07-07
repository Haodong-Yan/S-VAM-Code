import torch
from torch.utils.data import Dataset

import random
import numpy as np
from einops import rearrange
from glob import glob
import os
import json
import numpy as np
# from numba import njit, prange
import copy
from sklearn.decomposition import PCA
from torchvision import transforms as TF
import torch.nn.functional as F
import os
from sklearn.decomposition import PCA
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def preprocess_frame(frame_tensor):
    """
    Defines the standard preprocessing pipeline for DINOv2 for video frames.
    """
    target_height = 14*16
    target_width = 14*16
    transform = TF.Compose([
        TF.Resize(size=(target_height, target_width), interpolation=TF.InterpolationMode.BICUBIC, antialias=True),
        TF.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    return transform(frame_tensor)


def _sample_name_to_id(sample_name: str) -> int | None:
    base_name = sample_name.strip()
    if base_name.endswith("_gripper"):
        base_name = base_name[:-8]
    if base_name.startswith("sample_"):
        base_name = base_name[len("sample_"):]
    if base_name.isdigit():
        return int(base_name)
    return None


class HiddenFeatureDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split: str = "train",
        dummy: bool = True,
        split_file: str | None = None,
        train_ratio: float = 0.9,
        use_gripper: bool = False,
        use_base_camera: bool = True,
        return_frame_paths: bool = False,
        sample_ids: str | None = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.dummy = dummy
        self.train_ratio = max(0.0, min(train_ratio, 1.0))
        self.split_file = split_file or os.path.join(data_dir, "split_indices.json")
        self.split_created = False
        self.use_gripper = use_gripper
        self.use_base_camera = use_base_camera
        self.return_frame_paths = return_frame_paths
        self.sample_ids = sample_ids or ""

        # 兼容两种数据布局：
        # (A) 旧：data_dir/sample_xxx/...
        # (B) 新：data_dir/train/sample_xxx/... 以及 data_dir/val/sample_xxx/...
        split_subdir = os.path.join(data_dir, split)
        self._use_split_subdir = os.path.isdir(split_subdir)
        self._scan_root = split_subdir if self._use_split_subdir else data_dir

        if self.dummy:
            self.samples = [(None, None)]
            self.sample_names = ["dummy"]
            self.selected_names = self.sample_names
        else:
            sample_dirs = sorted(glob(os.path.join(self._scan_root, "sample_*")))
            all_entries = []
            for sample_dir in sample_dirs:
                # 1. 原始样本
                if self.use_base_camera:
                    frames_path = os.path.join(sample_dir, "full_video_frames.npy")
                    hidden_path = os.path.join(sample_dir, "one_step_features.npy")
                    if os.path.isfile(frames_path) and os.path.isfile(hidden_path):
                        sample_name = os.path.basename(sample_dir)
                        all_entries.append((frames_path, hidden_path, sample_name))
                
                # 2. Gripper 样本
                if self.use_gripper:
                    frames_path_gripper = os.path.join(sample_dir, "full_video_frames_gripper.npy")
                    hidden_path_gripper = os.path.join(sample_dir, "one_step_features_gripper.npy")
                    if os.path.isfile(frames_path_gripper) and os.path.isfile(hidden_path_gripper):
                        sample_name_gripper = os.path.basename(sample_dir) + "_gripper"
                        all_entries.append((frames_path_gripper, hidden_path_gripper, sample_name_gripper))

            if not all_entries:
                raise FileNotFoundError(
                    f"在路径 {self._scan_root} 下未找到有效的样本 "
                    f"(full_video_frames.npy + one_step_features.npy 或 gripper 版本)"
                )

            self.sample_names = [name for _, _, name in all_entries]
            # 如果 data_dir 有显式 split 子目录，则认为 train/val 已经在目录层面划分好了，
            # 不再依赖 split_indices.json 过滤。
            if self._use_split_subdir:
                self.samples = [(frames_path, hidden_path) for frames_path, hidden_path, _ in all_entries]
                self.selected_names = [name for _, _, name in all_entries]
            else:
                self._ensure_split_file()
                with open(self.split_file, "r", encoding="utf-8") as f:
                    split_config = json.load(f)

                if self.split not in split_config:
                    raise ValueError(f"split_indices 中不存在 split='{self.split}'")

                allowed = set(split_config[self.split])
                self.samples = []
                self.selected_names = []
                for frames_path, hidden_path, name in all_entries:
                    is_allowed = name in allowed
                    # 兼容：如果 split 只有 base 样本名，但现在加载了 gripper 样本，则检查去掉后缀后是否在 allowed 中
                    if not is_allowed and name.endswith("_gripper"):
                        base_name = name[:-8]  # remove "_gripper"
                        if base_name in allowed:
                            is_allowed = True
                    if is_allowed:
                        self.samples.append((frames_path, hidden_path))
                        self.selected_names.append(name)

                if not self.samples:
                    raise ValueError(f"split='{self.split}' 未匹配到任何样本，请检查 {self.split_file}")

            self._filter_by_sample_ids()

        # 为兼容旧代码保留属性
        self.generated_video_files = [frames for frames, _ in self.samples]
        self.noisy_latents_files = [hidden for _, hidden in self.samples]

    def _filter_by_sample_ids(self) -> None:
        requested_tokens = [token.strip() for token in self.sample_ids.split(",") if token.strip()]
        if not requested_tokens:
            return

        requested_ids = {sample_id for token in requested_tokens if (sample_id := _sample_name_to_id(token)) is not None}
        filtered_samples = []
        filtered_names = []
        for (frames_path, hidden_path), sample_name in zip(self.samples, self.selected_names):
            sample_id = _sample_name_to_id(sample_name)
            if sample_name in requested_tokens or (sample_id is not None and sample_id in requested_ids):
                filtered_samples.append((frames_path, hidden_path))
                filtered_names.append(sample_name)

        missing_tokens = []
        for token in requested_tokens:
            token_id = _sample_name_to_id(token)
            matched = token in filtered_names
            if not matched and token_id is not None:
                matched = any(_sample_name_to_id(name) == token_id for name in filtered_names)
            if not matched:
                missing_tokens.append(token)

        if missing_tokens:
            missing_str = ", ".join(missing_tokens)
            raise ValueError(f"split='{self.split}' 未匹配到指定样本: {missing_str}")

        self.samples = filtered_samples
        self.selected_names = filtered_names
        self.sample_names = list(filtered_names)

    def _ensure_split_file(self) -> None:
        if os.path.isfile(self.split_file):
            return

        os.makedirs(os.path.dirname(self.split_file), exist_ok=True)
        total = len(self.sample_names)
        if total == 0:
            raise ValueError("没有可供划分的样本")

        train_count = max(1, int(round(total * self.train_ratio)))
        if train_count >= total and total > 1:
            train_count = total - 1

        train_names = self.sample_names[:train_count]
        val_names = self.sample_names[train_count:]

        if not val_names and total > 1:
            val_names = [train_names.pop()]

        split_config = {
            "train": train_names,
            "val": val_names,
        }

        with open(self.split_file, "w", encoding="utf-8") as f:
            json.dump(split_config, f, ensure_ascii=False, indent=2)

        self.split_created = True
        print(
            f"[HiddenFeatureDataset] 已创建划分文件: {self.split_file} | train={len(train_names)}, val={len(val_names)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.dummy:
            frames_dummy = torch.randn(16, 3, 224, 224)
            hidden_dummy = torch.randn(1280, 16, 16, 16)
            if self.return_frame_paths:
                return frames_dummy.to(torch.float32), hidden_dummy.to(torch.float32), ""
            return frames_dummy.to(torch.float32), hidden_dummy.to(torch.float32)

        frames_path, hidden_path = self.samples[index]

        frames_np = np.load(frames_path)
        frames_np = frames_np

        # (1, T, H, W, 3) -> (T, 3, H, W)
        #frames_np = np.squeeze(frames_np, axis=0)
        frames_pt = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
        # 兼容 uint8 保存（0-255）
        if frames_pt.numel() > 0 and frames_pt.max() > 1.5:
            frames_pt = frames_pt / 255.0

        processed_frames = []
        for frame in frames_pt:
            processed_frames.append(preprocess_frame(frame))
        frames = torch.stack(processed_frames, dim=0)
        #frames = frames[:8]

        hidden_np = np.load(hidden_path)
        # (1, T, S, C) -> (T, H, W, C)
        #hidden_np = np.squeeze(hidden_np, axis=0)
        T, tokens, C = hidden_np.shape
        H = W = int(tokens ** 0.5)
        if H * W != tokens:
            raise ValueError(
                f"无法将 tokens={tokens} 重塑为方形网格，样本路径: {hidden_path}"
            )
        hidden_states_pt = (
            torch.from_numpy(hidden_np)
            .view(T, H, W, C)
            .permute(3, 0, 1, 2)
            .float()
        )

        if self.return_frame_paths:
            return frames.to(torch.float32), hidden_states_pt.to(torch.float32), frames_path
        return frames.to(torch.float32), hidden_states_pt.to(torch.float32)

if __name__ == "__main__":
    dataset = HiddenFeatureDataset(
        data_dir="./vis_dataset_features_real",
        dummy=False,
        split="train",
        use_gripper=True,
        use_base_camera=False,
    )
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)