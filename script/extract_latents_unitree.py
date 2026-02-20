#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Fast pipelined latent extraction for Unitree G1 datasets
"""
Extract VAE latents and text embeddings from a LeRobot Unitree G1 dataset.
Uses pipelined CPU video decoding + GPU encoding for maximum throughput.

Usage:
    python script/extract_latents_unitree.py \
        --dataset-path /root/data/UnitreeBagClose \
        --model-path /root/models/lingbot-va-base
"""
import argparse
import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from tqdm import tqdm
from transformers import T5TokenizerFast, UMT5EncoderModel


# --------------- Video decoding (CPU) ---------------

def load_video_frames(video_path, frame_stride=4, target_h=None, target_w=None):
    """Load, subsample, and resize video frames. Runs on CPU."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total_frames = stream.frames

    if total_frames == 0:
        frames_all = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        total_frames = len(frames_all)
    else:
        container.close()
        frames_all = None

    # Compute frame indices to sample
    all_frame_ids = list(range(0, total_frames, frame_stride))
    target_count = ((len(all_frame_ids) - 1) // 4) * 4 + 1
    if target_count < 1:
        target_count = 1
    frame_ids = all_frame_ids[:target_count]

    if frames_all is not None:
        frames = np.stack([frames_all[fid] for fid in frame_ids], axis=0)
    else:
        container = av.open(str(video_path))
        frame_set = set(frame_ids)
        sampled = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in frame_set:
                sampled[i] = frame.to_ndarray(format="rgb24")
            if i > frame_ids[-1]:
                break
        container.close()
        frames = np.stack([sampled[fid] for fid in frame_ids if fid in sampled], axis=0)
        frame_ids = [fid for fid in frame_ids if fid in sampled]

    # Resize
    if target_h is not None and target_w is not None:
        frames_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
        frames_t = F.interpolate(frames_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        frames = frames_t.permute(0, 2, 3, 1).numpy()

    return frames, frame_ids


def decode_worker(task_queue, result_queue, frame_stride):
    """Worker thread that decodes videos from task_queue into result_queue."""
    while True:
        item = task_queue.get()
        if item is None:  # Poison pill
            task_queue.task_done()
            break
        job_id, video_path, target_h, target_w, start_frame, end_frame = item
        try:
            frames, frame_ids = load_video_frames(
                video_path, frame_stride=frame_stride,
                target_h=target_h, target_w=target_w
            )
            # Filter to action segment
            segment_ids = [fid for fid in frame_ids if start_frame <= fid < end_frame]
            if len(segment_ids) < 1:
                result_queue.put((job_id, None, None))
            else:
                target_count = ((len(segment_ids) - 1) // 4) * 4 + 1
                if target_count < 1:
                    target_count = 1
                segment_ids = segment_ids[:target_count]
                segment_frames = np.stack([
                    frames[frame_ids.index(fid)] for fid in segment_ids
                ], axis=0)
                result_queue.put((job_id, segment_frames, segment_ids))
        except Exception as e:
            print(f"Warning: decode failed for {video_path}: {e}")
            result_queue.put((job_id, None, None))
        task_queue.task_done()


# --------------- VAE encoding (GPU) ---------------

def encode_video(vae, frames_np, device, dtype):
    """Encode video frames through WAN VAE. Frames must be 4k+1 count."""
    video = torch.from_numpy(frames_np).float().permute(3, 0, 1, 2).unsqueeze(0)
    video = video / 255.0 * 2.0 - 1.0
    video = video.to(device=device, dtype=dtype)
    with torch.no_grad():
        enc = vae._encode(video)
    mu, _ = torch.chunk(enc, 2, dim=1)
    latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
    return (mu - latents_mean) * (1.0 / latents_std)


def encode_text(tokenizer, text_encoder, text, device, dtype, max_seq_len=512):
    """Encode text prompt using T5 encoder."""
    text = prompt_clean(text)
    inputs = tokenizer(
        [text], padding="max_length", max_length=max_seq_len,
        truncation=True, add_special_tokens=True,
        return_attention_mask=True, return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)
    seq_len = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        embeds = text_encoder(input_ids, mask).last_hidden_state
    embeds = embeds.to(dtype=dtype)
    embeds = embeds[0, :seq_len[0]]
    padded = torch.zeros(max_seq_len, embeds.shape[-1], dtype=dtype, device=device)
    padded[:embeds.shape[0]] = embeds
    return padded


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--num-decode-workers", type=int, default=6)
    parser.add_argument("--cam-keys", nargs="+",
                        default=["observation.images.cam_left_high",
                                 "observation.images.cam_left_wrist",
                                 "observation.images.cam_right_wrist"])
    parser.add_argument("--high-cam", type=str, default="observation.images.cam_left_high")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    dataset_path = Path(args.dataset_path)
    model_path = Path(args.model_path)

    # Load episodes
    episodes = []
    with open(dataset_path / "meta" / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Found {len(episodes)} episodes")

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(str(model_path / "vae"), torch_dtype=dtype)
    vae = vae.to(device).eval()

    # Load text encoder, encode prompts, then free it
    print("Loading text encoder...")
    tokenizer = T5TokenizerFast.from_pretrained(str(model_path / "tokenizer"))
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(model_path / "text_encoder"), torch_dtype=dtype
    ).to(device).eval()

    empty_emb = encode_text(tokenizer, text_encoder, "", device, dtype)
    torch.save(empty_emb.cpu(), dataset_path / "empty_emb.pt")
    print("Saved empty_emb.pt")

    task_texts = set()
    for ep in episodes:
        for t in ep.get("tasks", []):
            task_texts.add(t)
        for ac in ep.get("action_config", []):
            task_texts.add(ac.get("action_text", ""))
    text_emb_cache = {}
    for t in task_texts:
        if t:
            text_emb_cache[t] = encode_text(tokenizer, text_encoder, t, device, dtype).cpu()
    print(f"Encoded {len(text_emb_cache)} text prompts")
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    # Build job list: (job_id, video_path, target_h, target_w, start, end)
    jobs = []  # list of dicts with all metadata per encode job
    for ep in episodes:
        ep_idx = ep["episode_index"]
        ep_chunk = ep_idx // 1000
        action_configs = ep.get("action_config", [
            {"start_frame": 0, "end_frame": ep["length"],
             "action_text": ep["tasks"][0] if ep.get("tasks") else ""}
        ])
        for ac in action_configs:
            sf, ef = ac["start_frame"], ac["end_frame"]
            action_text = ac.get("action_text", "")
            text_emb = text_emb_cache.get(action_text, empty_emb.cpu())
            for cam_key in args.cam_keys:
                if cam_key == args.high_cam:
                    th, tw = args.height, args.width
                else:
                    th, tw = args.height // 2, args.width // 2
                video_path = (dataset_path / "videos" / f"chunk-{ep_chunk:03d}" /
                              cam_key / f"episode_{ep_idx:06d}.mp4")
                if not video_path.exists():
                    continue
                latent_dir = dataset_path / "latents" / f"chunk-{ep_chunk:03d}" / cam_key
                latent_file = latent_dir / f"episode_{ep_idx:06d}_{sf}_{ef}.pth"
                # Skip if already extracted
                if latent_file.exists():
                    continue
                jobs.append({
                    "job_id": len(jobs),
                    "video_path": video_path,
                    "target_h": th, "target_w": tw,
                    "start_frame": sf, "end_frame": ef,
                    "text_emb": text_emb, "action_text": action_text,
                    "latent_file": latent_file,
                    "ep_idx": ep_idx,
                })

    print(f"{len(jobs)} encode jobs ({len(jobs) - len(jobs)} already done, skipped)")
    if not jobs:
        print("Nothing to do!")
        return

    # --- Pipelined processing ---
    # Decode threads fill result_queue; main thread consumes for GPU encode.
    task_queue = queue.Queue(maxsize=args.num_decode_workers * 2)
    result_queue = queue.Queue(maxsize=args.num_decode_workers * 2)

    # Start decode workers
    workers = []
    for _ in range(args.num_decode_workers):
        w = threading.Thread(target=decode_worker,
                             args=(task_queue, result_queue, args.frame_stride),
                             daemon=True)
        w.start()
        workers.append(w)

    # Submit all jobs to decode queue (in a separate thread to avoid blocking)
    def submit_jobs():
        for job in jobs:
            task_queue.put((
                job["job_id"], job["video_path"],
                job["target_h"], job["target_w"],
                job["start_frame"], job["end_frame"],
            ))
        # Poison pills
        for _ in range(args.num_decode_workers):
            task_queue.put(None)

    submit_thread = threading.Thread(target=submit_jobs, daemon=True)
    submit_thread.start()

    # Save executor
    save_executor = ThreadPoolExecutor(max_workers=8)
    save_futures = []

    # Main loop: consume decoded frames, encode on GPU, save
    pbar = tqdm(total=len(jobs), desc="Encoding")
    results_received = 0
    while results_received < len(jobs):
        job_id, segment_frames, segment_ids = result_queue.get()
        results_received += 1
        job = jobs[job_id]

        if segment_frames is None:
            pbar.update(1)
            continue

        # GPU encode
        mu_norm = encode_video(vae, segment_frames, device, dtype)
        latent = mu_norm[0]  # [48, F_lat, H_lat, W_lat]

        latent_data = {
            "latent": latent.permute(1, 2, 3, 0).reshape(-1, 48).to(torch.bfloat16).cpu(),
            "latent_num_frames": latent.shape[1],
            "latent_height": latent.shape[2],
            "latent_width": latent.shape[3],
            "video_num_frames": len(segment_ids),
            "video_height": job["target_h"],
            "video_width": job["target_w"],
            "text_emb": job["text_emb"].to(torch.bfloat16),
            "text": job["action_text"],
            "frame_ids": segment_ids,
            "start_frame": job["start_frame"],
            "end_frame": job["end_frame"],
            "fps": 30 // args.frame_stride,
            "ori_fps": 30,
        }

        os.makedirs(os.path.dirname(job["latent_file"]), exist_ok=True)
        save_futures.append(
            save_executor.submit(torch.save, latent_data, str(job["latent_file"]))
        )
        pbar.update(1)

    pbar.close()
    submit_thread.join()

    # Wait for saves
    print("Flushing saves...")
    for f in save_futures:
        f.result()

    print("Done! Latent extraction complete.")
    print(f"  Latents: {dataset_path / 'latents'}")
    print(f"  Empty emb: {dataset_path / 'empty_emb.pt'}")


if __name__ == "__main__":
    main()
