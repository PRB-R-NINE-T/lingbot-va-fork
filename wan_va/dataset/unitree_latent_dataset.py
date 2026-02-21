# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Unitree G1 dataset adapter for joint-space actions stored as separate columns."""
import os
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import packaging.version
import torch
from einops import rearrange
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.constants import HF_LEROBOT_HOME

from .lerobot_latent_dataset import recursive_find_file


def construct_unitree(repo_id, config):
    return UnitreeLatentLeRobotDataset(repo_id=repo_id, config=config)


def construct_unitree_multi_processor(config, num_init_worker=4):
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    # Avoid multiprocessing Pool in distributed contexts (CUDA + fork = deadlock)
    if len(repo_list) <= 1 or num_init_worker <= 1:
        datasets_out_lst = [construct_unitree(r, config) for r in repo_list]
    else:
        construct_func = partial(construct_unitree, config=config)
        with Pool(num_init_worker) as pool:
            datasets_out_lst = pool.map(construct_func, repo_list)
    return datasets_out_lst


class MultiUnitreeLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, config, num_init_worker=4):
        self._datasets = construct_unitree_multi_processor(config, num_init_worker)
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(self):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx):
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]


class UnitreeLatentLeRobotDataset(LeRobotDataset):
    """LeRobot dataset for Unitree G1 with joint-space actions in separate columns."""

    def __init__(self, repo_id, config=None):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]

        # Load action columns into a combined tensor view
        self.action_columns = getattr(config, 'action_columns', [
            'action.left_arm', 'action.left_gripper',
            'action.right_arm', 'action.right_gripper'
        ])
        self._hf_torch_view = self.hf_dataset.with_format(
            type='torch',
            columns=self.action_columns,
            output_all_columns=False
        )
        self.parse_meta()

    def parse_meta(self):
        max_frames = getattr(self.config, 'max_episode_frames', None)
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value.get("action_config", [
                {"start_frame": 0, "end_frame": value["length"],
                 "action_text": tasks[0] if tasks else ""}
            ])
            for acfg in action_config:
                orig_end_frame = acfg["end_frame"]
                start_frame = acfg["start_frame"]
                # Cap episode length to fit in GPU memory
                if max_frames and (orig_end_frame - start_frame) > max_frames:
                    end_frame = start_frame + max_frames
                else:
                    end_frame = orig_end_frame
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "orig_end_frame": orig_end_frame,  # Keep original for latent file lookup
                    "action_text": acfg.get("action_text", ""),
                }
                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["orig_end_frame"],
                    cur_meta["episode_index"],
                )
                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index, local_index):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        """Load action data from separate columns and concatenate into flat tensor."""
        batch = self._hf_torch_view[start_frame:end_frame]
        # Concatenate separate action columns into a single (N, action_channels) tensor
        parts = []
        for col in self.action_columns:
            # Column name in the batch is the part after the last dot for nested,
            # or the full name. HF datasets stores them under the full dotted name.
            val = batch[col]
            if val.dim() == 1:
                val = val.unsqueeze(-1)
            parts.append(val)
        action = torch.cat(parts, dim=-1)
        return {'action': action}

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        return self._flatten_latent_dict(out)

    def _cat_video_latents(self, data_dict):
        latent_lst = []
        for key in self.used_video_keys:
            latent = data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent,
                               '(f h w) c -> f h w c',
                               f=latent_num_frames,
                               h=latent_height,
                               w=latent_width)
            latent_lst.append(latent)
        wrist_latent = torch.cat(latent_lst[1:], dim=2)
        cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents=cat_latent,
            text_emb=text_emb,
        )
        return out_dict

    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        """Process joint-space actions: compute relative (delta from first frame) and normalize."""
        if isinstance(action, torch.Tensor):
            action = action.numpy()

        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]

        # Relative joint positions: delta from first frame
        action = action - action[0:1]

        # Pad start with zeros (frame_stride * 4 frames)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)),
                        mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num

        # Pad action dim by 1 (for inverse channel mapping)
        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        # Map to full action_dim space via inverse channel ids
        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]

        # Normalize
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.

        # Reshape to (C, F, N, 1)
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned

        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def _truncate_latent_data(self, data_dict, max_end_frame, frame_stride=4):
        """Truncate latent data to fit max_end_frame if needed."""
        for key in self.used_video_keys:
            frame_ids = data_dict[f"{key}.frame_ids"]
            # Filter frame_ids within the cap
            valid_ids = [fid for fid in frame_ids if fid < max_end_frame]
            # Ensure count is 4k+1
            target_count = ((len(valid_ids) - 1) // 4) * 4 + 1
            if target_count < 1:
                target_count = 1
            valid_ids = valid_ids[:target_count]
            if len(valid_ids) == len(frame_ids):
                continue  # No truncation needed

            # Recompute latent dimensions
            latent_num_frames = (len(valid_ids) - 1) // 4 + 1
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]

            # Truncate the latent tensor
            full_latent = data_dict[f"{key}.latent"]
            # full_latent is (F*H*W, C) flattened
            orig_num_frames = data_dict[f"{key}.latent_num_frames"]
            per_frame = latent_height * latent_width
            truncated = full_latent[:latent_num_frames * per_frame]

            data_dict[f"{key}.latent"] = truncated
            data_dict[f"{key}.latent_num_frames"] = latent_num_frames
            data_dict[f"{key}.video_num_frames"] = len(valid_ids)
            data_dict[f"{key}.frame_ids"] = valid_ids
        return data_dict

    def __getitem__(self, idx):
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        orig_end_frame = cur_meta.get("orig_end_frame", end_frame)
        local_start_frame = start_frame
        local_end_frame = end_frame

        # Load latents using original filename
        ori_data_dict = self._get_range_latent_data(start_frame, orig_end_frame, episode_index)

        # Truncate latents if end_frame was capped
        if end_frame < orig_end_frame:
            ori_data_dict = self._truncate_latent_data(ori_data_dict, end_frame)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        global_start = self._get_global_idx(episode_index, start_frame)
        global_end = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(global_start, global_end)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(
            local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)
