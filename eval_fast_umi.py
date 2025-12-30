# =========================
# Minimal UMI inference demo (single-frame)
# USB RGB + MujocoAR iPhone pose -> obs -> policy -> print action
# =========================

import time
import cv2
import torch
import hydra
import dill
import numpy as np
from scipy.spatial.transform import Rotation as R
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from mujoco_ar import MujocoARConnector
from diffusion_policy.common.pytorch_util import dict_apply
from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action

# -------------------------
# 1. Load policy checkpoint
# -------------------------
ckpt_path = "data/outputs/2025.12.28/16.48.18_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt"
payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
cfg = payload["cfg"]

print("size: : " + cfg.task.obs_horizon)

cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload)

policy = workspace.model
if cfg.training.use_ema:
    policy = workspace.ema_model
policy.eval().to("cuda")

# -------------------------
# 2. Mujoco AR connector (iPhone pose)
# -------------------------
connector = MujocoARConnector()
connector.start()

# -------------------------
# 3. USB camera
# -------------------------
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
assert cap.isOpened(), "USB camera open failed"

ret, frame = cap.read()
assert ret
H, W, _ = frame.shape

# -------------------------
# 4. Fixed gripper width
# -------------------------
FIXED_GRIPPER_WIDTH = np.array([0.04], dtype=np.float32)  # 4cm

# -------------------------
# 5. Episode start pose
# -------------------------
data = connector.get_latest_data()
while data["position"] is None:
    data = connector.get_latest_data()

init_pos = np.array(data["position"], dtype=np.float32)
init_rotvec = R.from_matrix(np.array(data["rotation"], dtype=np.float32)).as_rotvec()
episode_start_pose = [np.concatenate([init_pos, init_rotvec], axis=0)]

# -------------------------
# 6. Inference loop
# -------------------------
print("Start inference loop (Ctrl+C exit)")

while True:
    timestamp = np.array([time.time()], dtype=np.float64)  # T=1

    # --- camera frame ---
    ret, frame = cap.read()
    if not ret:
        continue
    rgb = frame.astype(np.uint8)
    rgb = rgb[None]  # [T=1, H, W, 3]

    # --- iPhone pose ---
    data = connector.get_latest_data()
    if data["position"] is None:
        continue
    pos = np.array(data["position"], dtype=np.float32)[None]  # [T=1, 3]
    rot_axis_angle = R.from_matrix(np.array(data["rotation"], dtype=np.float32)).as_rotvec()[None]  # [T=1, 3]

    # --- construct env_obs (single-frame sim) ---
    env_obs = {
        "camera0_rgb": rgb,
        "robot0_eef_pos": pos,
        "robot0_eef_rot_axis_angle": rot_axis_angle,
        "robot0_gripper_width": FIXED_GRIPPER_WIDTH[None],  # [T=1, 1]
        "timestamp": timestamp  # [T=1]
    }

    # --- convert to policy obs ---
    obs_dict_np = get_real_umi_obs_dict(
        env_obs=env_obs,
        shape_meta=cfg.task.shape_meta,
        obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
        tx_robot1_robot0=None,
        episode_start_pose=episode_start_pose
    )
    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to("cuda"))

    # --- policy inference ---
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
        raw_action = result["action_pred"][0].cpu().numpy()

    # --- decode action ---
    action = get_real_umi_action(
        raw_action=raw_action,
        env_obs=env_obs,
        action_pose_repr=cfg.task.pose_repr.action_pose_repr
    )

    # --- print ---
    print("========== POLICY ACTION ==========")
    print("EEF pos:", action[:3])
    print("EEF rot (axis-angle):", action[3:6])
    print("Gripper width:", action[6])
    print("===================================")

    time.sleep(1 / 30)  # 30Hz