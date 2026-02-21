# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_unitree_cfg import va_unitree_cfg
import os

va_unitree_train_cfg = EasyDict(__name__='Config: VA unitree train')
va_unitree_train_cfg.update(va_unitree_cfg)

va_unitree_train_cfg.dataset_path = '/root/data/UnitreeBagClose'
va_unitree_train_cfg.empty_emb_path = os.path.join(va_unitree_train_cfg.dataset_path, 'empty_emb.pt')
va_unitree_train_cfg.enable_wandb = True
va_unitree_train_cfg.load_worker = 2
va_unitree_train_cfg.save_interval = 1000
va_unitree_train_cfg.gc_interval = 1
va_unitree_train_cfg.max_episode_frames = 1100  # Cap episode length to fit in GPU memory
va_unitree_train_cfg.cfg_prob = 0.1

# Training parameters
va_unitree_train_cfg.learning_rate = 1e-5
va_unitree_train_cfg.beta1 = 0.9
va_unitree_train_cfg.beta2 = 0.95
va_unitree_train_cfg.weight_decay = 0.1
va_unitree_train_cfg.warmup_steps = 10
va_unitree_train_cfg.batch_size = 1
va_unitree_train_cfg.gradient_accumulation_steps = 1
va_unitree_train_cfg.num_steps = 5000
