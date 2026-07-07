from collections import Counter, defaultdict
from fnmatch import fnmatch
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from copy import deepcopy

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist

from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition, join_vis_lang
from policy_models.utils.utils import get_last_checkpoint
from policy_models.rollout.rollout_video import RolloutVideo

logger = logging.getLogger(__name__)


def _parse_sequence_ids(spec):
    """
    Parse sequence id spec string, e.g.:
    - "10,678,999"
    - "0-5,10,20-22"
    Returns a de-duplicated list preserving input order.
    """
    if spec is None:
        return None
    spec = str(spec).strip()
    if spec == "":
        return None

    ids = []
    seen = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            a = int(a_str.strip())
            b = int(b_str.strip())
            if b < a:
                raise ValueError(f"Invalid range '{token}': end < start.")
            for x in range(a, b + 1):
                if x not in seen:
                    ids.append(x)
                    seen.add(x)
        else:
            x = int(token)
            if x not in seen:
                ids.append(x)
                seen.add(x)
    return ids


def _select_sequence_items(cfg):
    all_sequences = get_sequences(cfg.num_sequences)
    all_items = list(enumerate(all_sequences))
    selected_ids = getattr(cfg, "sequence_ids", None)
    if not selected_ids:
        return all_items

    max_idx = len(all_sequences) - 1
    invalid = [i for i in selected_ids if i < 0 or i > max_idx]
    if invalid:
        raise ValueError(f"sequence_ids out of range [0, {max_idx}]: {invalid[:10]}")

    selected_set = set(selected_ids)
    selected_items = [(idx, seq) for idx, seq in all_items if idx in selected_set]
    # Keep the user-provided order.
    order = {sid: pos for pos, sid in enumerate(selected_ids)}
    selected_items.sort(key=lambda x: order[x[0]])
    return selected_items


def _maybe_save_predictive_feature(model, cfg, feature_root, sequence_idx, subtask_idx, step_idx, subtask_name):
    save_predictive_features = bool(getattr(cfg, "save_predictive_features", False))
    save_video_former_attention = bool(getattr(cfg, "save_video_former_attention", False))
    if not (save_predictive_features or save_video_former_attention):
        return
    if feature_root is None:
        return
    predictive_feature = getattr(model, "last_predictive_feature", None)
    attention_map = None
    video_former = getattr(model, "Video_Former", None)
    if video_former is not None:
        attention_map = getattr(video_former, "last_attention_map", None)
    if predictive_feature is None and attention_map is None:
        return
    payload = dict(predictive_feature) if predictive_feature is not None else {}
    hidden2dino_tokens = getattr(model, "last_hidden2dino_tokens", None)
    hidden2dpa_tokens = getattr(model, "last_hidden2dpa_tokens", None)
    if hidden2dino_tokens is not None:
        payload["hidden2dino_tokens"] = hidden2dino_tokens
    if hidden2dpa_tokens is not None:
        payload["hidden2dpa_tokens"] = hidden2dpa_tokens
    if attention_map is not None:
        payload["video_former_attention_map"] = attention_map.detach().cpu()

    seq_dir = Path(feature_root) / f"sequence_{sequence_idx:04d}" / f"subtask_{subtask_idx:02d}_{subtask_name}"
    os.makedirs(seq_dir, exist_ok=True)
    save_path = seq_dir / f"step_{step_idx:04d}.pt"
    torch.save(payload, save_path)

def _collect_eval_results_paths(train_folder: Path) -> list[Path]:
    """
    Collect evaluation result files for a given action-model folder.

    NOTE: Per user request we do NOT rely on any marker files; skip/resume is
    decided solely by globally scanning existing `results.json` files.
    """
    base = train_folder
    if base.is_file():
        base = base.parent

    roots = {base}
    # If user passes run_folder, also consider run_folder/checkpoints.
    if (base / "checkpoints").is_dir():
        roots.add(base / "checkpoints")
    # If user passes checkpoints folder, also consider the run folder.
    if base.name == "checkpoints":
        roots.add(base.parent)

    results_paths: list[Path] = []
    for root in sorted(roots):
        logs_root = root / "logs"
        if logs_root.is_dir():
            results_paths.extend(sorted(logs_root.glob("*/results.json")))

    # De-duplicate while preserving order
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for p in results_paths:
        if p in seen:
            continue
        seen.add(p)
        unique_paths.append(p)
    return unique_paths


def _load_done_from_logs(train_folder: Path) -> dict[str, str]:
    """
    Scan all discovered `logs/*/results.json` and return {ckpt_name: results_json_path}.
    """
    done: dict[str, str] = {}
    for results_path in _collect_eval_results_paths(train_folder):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            for ckpt_name in data.keys():
                # keep first occurrence (oldest) to avoid churn
                done.setdefault(str(ckpt_name), str(results_path))
        except Exception as e:
            logger.warning("Failed reading results file %s: %s", results_path, e)
    return done


def configure_future_rgb_inputs(cfg):
    datamodule_cfg = getattr(cfg, "datamodule", None)
    model_cfg = getattr(cfg, "model", None)
    if datamodule_cfg is None or model_cfg is None:
        return
    use_gt = bool(getattr(model_cfg, "use_gt_dino_condition", False))
    datamodule_cfg.enable_future_rgb = use_gt
    if not use_gt:
        return

    future_pairs = [
        ("rgb_static", "rgb_static_future"),
        ("rgb_gripper", "rgb_gripper_future"),
    ]

    rgb_obs_list = list(datamodule_cfg.observation_space.rgb_obs)
    for _, future in future_pairs:
        if future not in rgb_obs_list:
            rgb_obs_list.append(future)
    datamodule_cfg.observation_space.rgb_obs = rgb_obs_list

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


def _resolve_wandb_cfg(cfg):
    if not hasattr(cfg, "wandb") or cfg.wandb is None:
        return {}
    wandb_cfg = cfg.wandb
    if isinstance(wandb_cfg, dict):
        return wandb_cfg
    return OmegaConf.to_container(wandb_cfg, resolve=True)


def init_wandb_run(cfg):
    wandb_cfg = _resolve_wandb_cfg(cfg)
    run_name = wandb_cfg.get("name") or getattr(cfg, "model_name", None)
    if not run_name:
        train_folder = getattr(cfg, "train_folder", None)
        run_name = Path(train_folder).name if train_folder else "calvin_eval"
    init_kwargs = {
        "project": wandb_cfg.get("project") or "policy_evaluation",
        "name": run_name,
    }
    for key in ("entity", "group", "mode", "notes", "tags", "job_type", "dir"):
        value = wandb_cfg.get(key)
        if value is not None:
            init_kwargs[key] = value
    cfg_snapshot = {}
    for field in ("train_folder", "num_sequences", "num_videos", "device"):
        if hasattr(cfg, field):
            cfg_snapshot[field] = getattr(cfg, field)
    if cfg_snapshot:
        init_kwargs["config"] = cfg_snapshot
    return wandb.init(**init_kwargs)


def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


def _sanitize_filename_fragment(text: str, max_len: int = 28) -> str:
    fragment = str(text).strip().lower()
    fragment = fragment.replace(" ", "_")
    fragment = re.sub(r"[^a-z0-9_]+", "-", fragment)
    fragment = re.sub(r"-{2,}", "-", fragment).strip("-_")
    if not fragment:
        fragment = "na"
    return fragment[:max_len]


def _build_video_tag_with_prompts_and_status(sequence_idx: int, prompts: list[str], subtask_success: list[bool]) -> str:
    prompt_part = "__".join(_sanitize_filename_fragment(p) for p in prompts) if prompts else "no_prompt"
    status_part = "-".join("s" if ok else "f" for ok in subtask_success) if subtask_success else "no_status"
    return f"_long_horizon/sequence_{sequence_idx}__text_prompt_{prompt_part}__subtask_{status_part}"


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        # If a single checkpoint file is provided, place logs under its parent dir.
        if log_dir.is_file() or log_dir.suffix == ".pt":
            log_dir = log_dir.parent
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=False)
    print(f"logging to {log_dir}")
    return log_dir


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, plan_dicts, cfg, log_dir=None):
    if log_dir is None:
        log_dir = get_log_dir(cfg.train_folder)

    sequence_items = _select_sequence_items(cfg)
    sequences = [seq for _, seq in sequence_items]

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        if isinstance(checkpoint, Path):
            ckpt_name = checkpoint.name
        else:
            ckpt_name = str(checkpoint)
        print(f"Results for Checkpoint {ckpt_name}:")
        avg_seq_len = np.mean(results)
        ranking[ckpt_name] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
        if wandb.run is not None:
            wandb.log(
                {
                    "avrg_performance/avg_seq_len": avg_seq_len,
                    "avrg_performance/chain_sr": chain_sr,
                    "detailed_metrics/task_info": task_info,
                }
            )
        elif cfg.log_wandb:
            logger.warning("W&B run not initialized; skipping metric logging.")
        current_data[ckpt_name] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file, indent=2)
    ranking_all = {}
    for name, data in json_data.items():
        if isinstance(data, dict) and "avg_seq_len" in data:
            ranking_all[name] = data["avg_seq_len"]
    if ranking_all:
        best_name = max(ranking_all, key=ranking_all.get)
        print(
            f"Best model: checkpoint {best_name} with average sequences length of {ranking_all[best_name]}"
        )


def evaluate_policy(model, env, lang_embeddings, cfg, num_videos=0, save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations

    eval_sequence_items = _select_sequence_items(cfg)
    eval_sequences = [seq for _, seq in eval_sequence_items]
    if getattr(cfg, "sequence_ids", None):
        print(
            f"Evaluating selected sequences: {len(eval_sequence_items)} / {cfg.num_sequences} "
            f"(absolute ids, first few: {cfg.sequence_ids[:10]})"
        )
    # Use num_videos < 0 as "record all evaluated sequences".
    if num_videos is not None and num_videos < 0:
        num_videos = len(eval_sequences)

    # video stuff
    if num_videos > 0:
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=save_dir,
            resolution_scale=1,
        )
    else:
        rollout_video = None

    feature_root = None
    if bool(getattr(cfg, "save_predictive_features", False)) and save_dir is not None:
        feature_root = Path(save_dir) / "predictive_features"
        os.makedirs(feature_root, exist_ok=True)

    results = []
    plans = defaultdict(list)

    if not cfg.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for eval_pos, (seq_idx, (initial_state, eval_sequence)) in enumerate(eval_sequence_items):
        record = eval_pos < num_videos
        result, subtask_success, prompt_texts = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            lang_embeddings,
            val_annotations,
            cfg,
            record,
            rollout_video,
            seq_idx,
            feature_root,
        )
        results.append(result)
        should_save_video = True
        if bool(getattr(cfg, "skip_video_on_full_success", False)) and result >= len(eval_sequence):
            should_save_video = False
        if not cfg.debug:
            success_rates = count_success(results)
            average_rate = sum(success_rates) / len(success_rates) * 5
            description = " ".join([f"{j + 1}/5 : {v * 100:.1f}% |" for j, v in enumerate(success_rates)])
            description += f" Average: {average_rate:.1f} |"
            eval_sequences.set_description(description)
        if record:
            if should_save_video:
                if rollout_video.tags:
                    rollout_video.tags[-1] = _build_video_tag_with_prompts_and_status(
                        sequence_idx=seq_idx,
                        prompts=prompt_texts,
                        subtask_success=subtask_success,
                    )
                rollout_video.write_to_tmp()
                rollout_video._log_currentvideos_to_file(seq_idx, save_as_video=True)
            else:
                print(f"Skip saving video for sequence {seq_idx}: full success ({result}/{len(eval_sequence)}).")
            # Keep memory bounded when recording many sequences.
            if rollout_video.videos:
                rollout_video.videos.pop()
            if rollout_video.tags:
                rollout_video.tags.pop()
            if rollout_video.captions:
                rollout_video.captions.pop()
        #break
    #if num_videos > 0:
    #    print('save_video_2:',rollout_video.save_dir)
    #    # log rollout videos
    #    rollout_video._log_videos_to_file(0, save_as_video=True)
    return results, plans


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i, feature_root=None
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
    success_counter = 0
    subtask_success = []
    prompt_texts = []
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_idx, subtask in enumerate(eval_sequence):
        #breakpoint()
        if record:
            rollout_video.new_subtask()
        prompt_texts.append(val_annotations[subtask][0])
        success = rollout(
            env,
            model,
            task_checker,
            cfg,
            subtask,
            lang_embeddings,
            val_annotations,
            record,
            rollout_video,
            feature_root=feature_root,
            sequence_idx=i,
            subtask_idx=subtask_idx,
        )
        if record:
            status = "s" if success else "f"
            prompt_fragment = _sanitize_filename_fragment(val_annotations[subtask][0])
            rollout_video.save_last_subtask_video_to_file(
                tag_suffix=f"subtask_{subtask_idx:02d}__text_prompt_{prompt_fragment}__{status}",
                current_step=i,
                save_as_video=True,
                subfolder=f"subtasks_sequence_{i:04d}",
            )
        subtask_success.append(bool(success))
        if record and bool(getattr(cfg, "video_draw_outcome", False)):
            rollout_video.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter, subtask_success, prompt_texts
    return success_counter, subtask_success, prompt_texts


def rollout(
    env,
    model,
    task_oracle,
    cfg,
    subtask,
    lang_embeddings,
    val_annotations,
    record=False,
    rollout_video=None,
    feature_root=None,
    sequence_idx=0,
    subtask_idx=0,
):
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal['lang_text'] = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for step in range(cfg.ep_len):
        #if step % 10 == 0:
        action = model.step(obs, goal)
        _maybe_save_predictive_feature(
            model=model,
            cfg=cfg,
            feature_root=feature_root,
            sequence_idx=sequence_idx,
            subtask_idx=subtask_idx,
            step_idx=step,
            subtask_name=subtask,
        )
        step_result = env.step(action)
        try:
            step_tuple = tuple(step_result)
        except TypeError:
            raise TypeError(
                f"env.step(action) returned {type(step_result)}; expected an iterable of length 4 or 5."
            ) from None

        result_len = len(step_tuple)
        if result_len == 4:
            obs, _, _, current_info = step_tuple
        elif result_len == 5:
            obs, _, _, _, current_info = step_tuple
        else:
            raise ValueError(
                f"env.step(action) returned {result_len} values, expected 4 or 5."
            )
        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if record:
            # update video
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record and bool(getattr(cfg, "video_render_text", False)):
                rollout_video.add_language_instruction(lang_annotation)
            return True
    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record and bool(getattr(cfg, "video_render_text", False)):
        rollout_video.add_language_instruction(lang_annotation)
    return False


#@hydra.main(config_path="../policy_conf", config_name="calvin_evaluate_all")
def main(cfg):
    configure_future_rgb_inputs(cfg)
    log_wandb = cfg.log_wandb
    wandb_run = None
    if log_wandb and wandb.run is None:
        wandb_run = init_wandb_run(cfg)
    log_dir = get_log_dir(cfg.train_folder)
    if log_wandb:
        os.makedirs(log_dir / "wandb", exist_ok=False)
    print('cfg.device',cfg.device)
    torch.cuda.set_device(cfg.device)
    print(f"DEBUG: torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    print(f"DEBUG: torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        print(f"DEBUG: torch.cuda.current_device()={current}", flush=True)
        print(f"DEBUG: torch.cuda.get_device_name({current})={torch.cuda.get_device_name(current)}", flush=True)
    seed_everything(0, workers=True)  # type:ignore
    # evaluate all checkpoints under ckpt_path (cfg.train_folder)
    ckpt_root = Path(cfg.train_folder)
    ckpt_files = []
    if ckpt_root.is_file():
        ckpt_files = [ckpt_root]
    elif ckpt_root.is_dir():
        candidate_dirs = []
        # if user passes the run folder instead of the checkpoints folder, also support run_folder/checkpoints
        if (ckpt_root / "checkpoints").is_dir():
            candidate_dirs.append(ckpt_root / "checkpoints")
        candidate_dirs.append(ckpt_root)

        for d in candidate_dirs:
            if d.is_dir():
                ckpt_files.extend(sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime))
        # de-duplicate while preserving order
        seen = set()
        ckpt_files = [p for p in ckpt_files if not (p in seen or seen.add(p))]

        if not ckpt_files:
            # fallback to recursive search (useful if checkpoints are nested)
            ckpt_files = sorted(ckpt_root.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
    else:
        raise FileNotFoundError(f"ckpt_path not found: {cfg.train_folder}")

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint *.pt files found under: {cfg.train_folder}")

    ckpt_pattern = str(getattr(cfg, "checkpoint_name_pattern", "*.pt"))
    if ckpt_pattern and ckpt_pattern != "*.pt":
        before = len(ckpt_files)
        ckpt_files = [p for p in ckpt_files if fnmatch(p.name, ckpt_pattern)]
        print(f"Applied checkpoint_name_pattern='{ckpt_pattern}': {len(ckpt_files)}/{before} matched.")

    if not ckpt_files:
        raise FileNotFoundError(
            f"No checkpoint files match pattern '{ckpt_pattern}' under: {cfg.train_folder}"
        )

    print('train_folder', cfg.train_folder)
    print(f'Found {len(ckpt_files)} checkpoints to evaluate.')
    for p in ckpt_files:
        print(f' - {p.name}')

    ckpt_files_to_eval: list[Path] = []
    if bool(getattr(cfg, "force_eval", False)):
        ckpt_files_to_eval = ckpt_files
        print("force_eval=True; evaluating all checkpoints (no skip).")
    else:
        # Skip checkpoints that have already been evaluated (resume-friendly).
        # IMPORTANT: Do not rely on any marker files; only scan existing results.json globally.
        done_from_logs = _load_done_from_logs(ckpt_root)
        done_set = set(done_from_logs.keys())
        skipped_ckpt_files: list[Path] = []
        for p in ckpt_files:
            if p.name in done_set:
                skipped_ckpt_files.append(p)
            else:
                ckpt_files_to_eval.append(p)

        if skipped_ckpt_files:
            print(f"Skipping {len(skipped_ckpt_files)} already-evaluated checkpoints.")
            preview = 20
            for p in skipped_ckpt_files[:preview]:
                print(f" - {p.name}")
            if len(skipped_ckpt_files) > preview:
                print(f" - ... ({len(skipped_ckpt_files) - preview} more)")

        if not ckpt_files_to_eval:
            print("All checkpoints already evaluated; exiting.")
            return

    lang_embeddings = None
    env = None
    data_module = None
    results = {}
    plans = {}
    try:
        # build env/lang embeddings once (shared across checkpoints)
        env, data_module, lang_embeddings = get_default_beso_and_env(
            cfg.train_folder,
            cfg.root_data_dir,
            ckpt_files_to_eval[0],
            env=env,
            lang_embeddings=lang_embeddings,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=cfg.device,
            cfg=cfg,
        )

        for ckpt_file in ckpt_files_to_eval:
            ckpt_name = ckpt_file.name
            print(f"\n==============================")
            print(f"Evaluating checkpoint: {ckpt_name}")
            print(f"Loading model from {ckpt_file}")

            state_dict = torch.load(ckpt_file, map_location='cpu')
            if isinstance(state_dict, dict) and "model" in state_dict:
                model_weights = state_dict["model"]
            elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                model_weights = state_dict["state_dict"]
            else:
                raise KeyError(
                    f"Unsupported checkpoint format for {ckpt_file}. "
                    f"Expected dict with key 'model' (or 'state_dict')."
                )

            device = torch.device(f"cuda:{cfg.device}")
            model = hydra.utils.instantiate(cfg.model)
            model.load_state_dict(model_weights, strict=False)
            model.freeze()
            model = model.cuda(device)
            print(f"DEBUG: Model primary parameter device: {next(model.parameters()).device}", flush=True)

            print(
                cfg.num_sampling_steps,
                cfg.sampler_type,
                cfg.multistep,
                cfg.sigma_min,
                cfg.sigma_max,
                cfg.noise_scheduler,
            )
            model.num_sampling_steps = cfg.num_sampling_steps
            model.sampler_type = cfg.sampler_type
            model.multistep = cfg.multistep
            if cfg.sigma_min is not None:
                model.sigma_min = cfg.sigma_min
            if cfg.sigma_max is not None:
                model.sigma_max = cfg.sigma_max
            if cfg.noise_scheduler is not None:
                model.noise_scheduler = cfg.noise_scheduler

            if cfg.cfg_value != 1:
                raise NotImplementedError("cfg_value != 1 not implemented yet")
            model.process_device()
            model.eval()
            if getattr(model, "Video_Former", None) is not None and hasattr(model.Video_Former, "track_attention_default"):
                model.Video_Former.track_attention_default = bool(getattr(cfg, "save_video_former_attention", False))

            # avoid video overwrite across checkpoints
            ckpt_save_dir = Path(log_dir) / ckpt_name
            results[ckpt_name], plans[ckpt_name] = evaluate_policy(
                model,
                env,
                lang_embeddings,
                cfg,
                num_videos=cfg.num_videos,
                save_dir=ckpt_save_dir,
            )
            # incremental save (so partial results are preserved if the job preempts)
            print_and_save({ckpt_name: results[ckpt_name]}, {ckpt_name: plans[ckpt_name]}, cfg, log_dir=log_dir)

            # free GPU memory between checkpoints
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Respect externally-set CUDA_VISIBLE_DEVICES (e.g. from SLURM); only fall back to GPU 0.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Force PyBullet to use the EGL platform for headless rendering.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from hydra import compose, initialize
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--action_model_folder", type=str, default="")
    parser.add_argument("--clip_model_path", type=str, default="")
    parser.add_argument("--calvin_abc_dir", type=str, default="")
    parser.add_argument("--use_ref_frame", action="store_true", help="Force enable use_ref_frame for hidden2dino")
    parser.add_argument("--no_use_ref_frame", action="store_true", help="Force disable use_ref_frame for hidden2dino")
    parser.add_argument(
        "--use_gt_dino_condition",
        action="store_true",
        help="Enable GT-DINO conditioning when instantiating the policy.",
    )
    parser.add_argument(
        "--gt_dino_chunk",
        type=int,
        default=None,
        help="Override chunk size used while extracting GT-DINO features during evaluation.",
    )
    parser.add_argument(
        "--bypass_video_former",
        action="store_true",
        help="Skip VideoFormer at evaluation time and feed GT-DINO tokens directly to the diffusion policy.",
    )
    parser.add_argument(
        "--use_hidden_dino_concat",
        action="store_true",
        help="Concatenate SVD hidden states with predicted DINO tokens before VideoFormer.",
    )
    parser.add_argument(
        "--use_hidden_dpa_concat",
        action="store_true",
        help="Use hidden DPA tokens (Depth Anything) for concatenation.",
    )

    parser.add_argument(
        "--without_svd",
        action="store_true",
        help="Skip SVD compression for the video features.",
    )
    parser.add_argument(
        "--no_use_dpa_ref_frame",
        action="store_true",
        help="Force disable use_ref_frame for hidden2dpa.",
    )
    parser.add_argument(
        "--use_hidden_dino_dpa_concat",
        action="store_true",
        help="Concatenate SVD hidden states with predicted DINO tokens before VideoFormer.",
    )
    parser.add_argument(
        "--use_dpa_ref_frame",
        action="store_true",
        help="Use DPA reference frame for hidden2dino.",
    )
    parser.add_argument(
        "--disable_gripper_features",
        action="store_true",
        help="Use only static-camera features (disable gripper features) in the policy.",
    )
    parser.add_argument(
        "--force_eval",
        action="store_true",
        help="Re-evaluate checkpoints even if existing evaluation markers/results are found.",
    )
    parser.add_argument(
        "--checkpoint_name_pattern",
        type=str,
        default="*.pt",
        help="Glob pattern used to filter checkpoint filenames, e.g. '*000.pt'.",
    )
    parser.add_argument(
        "--skip_video_on_full_success",
        action="store_true",
        help="Do not save rollout videos for sequences that complete all subtasks successfully.",
    )
    parser.add_argument(
        "--sequence_ids",
        type=str,
        default="",
        help="Evaluate only specific absolute sequence indices. Supports comma/range, e.g. '10,678,999' or '0-5,42'.",
    )
    parser.add_argument(
        "--save_predictive_features",
        action="store_true",
        help="Save model.last_predictive_feature at every rollout step.",
    )
    parser.add_argument(
        "--save_video_former_attention",
        action="store_true",
        help="Save Video Former latent-query attention map at every rollout step.",
    )
    parser.add_argument(
        "--video_render_text",
        action="store_true",
        help="Render language text onto rollout videos.",
    )
    parser.add_argument(
        "--video_draw_outcome",
        action="store_true",
        help="Draw red/green outcome borders in rollout videos.",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=None,
        help="Number of evaluated sequences whose rollout videos to save. "
             "<0 = save all (yaml default), 0 = disable video recording entirely (much faster).",
    )

    args = parser.parse_args()

    with initialize(config_path="../policy_conf", job_name="calvin_evaluate_all_our.yaml"):
        cfg = compose(config_name="calvin_evaluate_all_our.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.train_folder = args.action_model_folder
    cfg.model.text_encoder_path = args.clip_model_path
    cfg.root_data_dir = args.calvin_abc_dir
    cfg.force_eval = args.force_eval
    cfg.checkpoint_name_pattern = args.checkpoint_name_pattern
    cfg.skip_video_on_full_success = args.skip_video_on_full_success
    cfg.sequence_ids = _parse_sequence_ids(args.sequence_ids)
    cfg.save_predictive_features = args.save_predictive_features
    cfg.save_video_former_attention = args.save_video_former_attention
    cfg.video_render_text = args.video_render_text
    cfg.video_draw_outcome = args.video_draw_outcome
    if args.num_videos is not None:
        cfg.num_videos = args.num_videos

    if args.use_ref_frame:
        cfg.model.hidden2dino_use_ref_override = True
    elif args.no_use_ref_frame:
        cfg.model.hidden2dino_use_ref_override = False
    if args.use_gt_dino_condition:
        cfg.model.use_gt_dino_condition = True
    if args.gt_dino_chunk is not None:
        cfg.model.gt_dino_chunk = args.gt_dino_chunk
    if args.bypass_video_former:
        cfg.model.bypass_video_former = True
    if args.use_hidden_dino_concat:
        cfg.model.use_hidden_dino_concat = True
    if args.use_hidden_dpa_concat:
        cfg.model.use_hidden_dpa_concat = True
    if args.disable_gripper_features:
        cfg.model.use_gripper_features = False
    if args.use_hidden_dino_dpa_concat:
        cfg.model.use_hidden_dino_dpa_concat = True
    if args.use_dpa_ref_frame:
        cfg.model.hidden2dpa_use_ref_override = True
    if args.without_svd:
        cfg.model.without_svd = True
    main(cfg)

