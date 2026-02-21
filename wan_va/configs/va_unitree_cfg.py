# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_unitree_cfg = EasyDict(__name__='Config: VA unitree')
va_unitree_cfg.update(va_shared_cfg)

va_unitree_cfg.wan22_pretrained_model_name_or_path = "/root/models/lingbot-va-base"

va_unitree_cfg.attn_window = 72
va_unitree_cfg.frame_chunk_size = 2
va_unitree_cfg.env_type = 'unitree'

va_unitree_cfg.height = 256
va_unitree_cfg.width = 320
va_unitree_cfg.action_dim = 30
va_unitree_cfg.action_per_frame = 16
va_unitree_cfg.obs_cam_keys = [
    'observation.images.cam_left_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_unitree_cfg.guidance_scale = 5
va_unitree_cfg.action_guidance_scale = 1

va_unitree_cfg.num_inference_steps = 25
va_unitree_cfg.video_exec_step = -1
va_unitree_cfg.action_num_inference_steps = 50

va_unitree_cfg.snr_shift = 5.0
va_unitree_cfg.action_snr_shift = 1.0

# Action channel mapping: 16 used channels -> 30-dim action space
# left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
# mapped to positions [0:7, 28, 7:14, 29] in the 30-dim space
va_unitree_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_unitree_cfg.used_action_channel_ids)
] * va_unitree_cfg.action_dim
for i, j in enumerate(va_unitree_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_unitree_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# Action columns to concatenate from LeRobot parquet (in order)
va_unitree_cfg.action_columns = [
    'action.left_arm', 'action.left_gripper',
    'action.right_arm', 'action.right_gripper'
]

va_unitree_cfg.action_norm_method = 'quantiles'
va_unitree_cfg.norm_stat = {
    "q01": [
        -0.9609625227749348, -0.2716635763645172, -0.8289744555950165,
        -0.7256097197532654, -0.36176037788391113, -0.38379054283723235,
        -1.272720992565155,
        -1.3200475871562958, -0.7807004451751709, -0.16485309600830078,
        -1.1904349774122238, -1.0097885727882385, -0.7942505083046854,
        -0.45627284049987793,
    ] + [0.] * 14 + [
        -4.488851430825889,
        -4.478157257661223,
    ],
    "q99": [
        0.32066911086440086, 0.6562500596046448, 0.32020625472068787,
        1.0085519552230835, 0.6170351803302765, 1.1271608285605907,
        0.3247203677892685,
        0.40561947226524353, 0.5394414141774178, 1.2794688642024994,
        1.1274054944515228, 0.3943028002977371, 1.4054273217916489,
        1.3940322399139404,
    ] + [0.] * 14 + [
        0.04668235778808594,
        0.07489824295043945,
    ],
}
