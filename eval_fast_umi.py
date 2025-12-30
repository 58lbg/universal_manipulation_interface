"""
Minimal UMI inference demo
USB RGB + MujocoAR iPhone pose -> obs -> policy -> print action
"""

# =========================
# 基础库
# =========================
import time
import cv2
import torch
import hydra
import dill
import numpy as np
from scipy.spatial.transform import Rotation as R

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# =========================
# Mujoco AR（与你采集代码一致）
# =========================
from mujoco_ar import MujocoARConnector

# =========================
# diffusion / umi
# =========================
from diffusion_policy.common.pytorch_util import dict_apply
from umi.real_world.real_inference_util import (
    get_real_umi_obs_dict,
    get_real_umi_action
)

# =========================
# 1. 加载 policy checkpoint
# =========================
ckpt_path = "data/outputs/2025.12.28/16.48.18_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt"  # ← 修改成你的 ckpt

# 用 dill 加载（与你 eval_real_umi.py 一致）
payload = torch.load(
    open(ckpt_path, "rb"),
    map_location="cpu",
    pickle_module=dill
)

cfg = payload["cfg"]

# 构造 workspace
# cls = __import__(
#     cfg._target_,
#     fromlist=['']
# ).__dict__[cfg._target_.split('.')[-1]]
#
# workspace = cls(cfg)
# workspace.load_payload(payload)

cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# 选择 EMA / 原模型
policy = workspace.model
if cfg.training.use_ema:
    policy = workspace.ema_model

policy.eval()
policy.to("cuda")

# =========================
# 2. 初始化 MujocoARConnector（iPhone 位姿）
# =========================
connector = MujocoARConnector()
connector.start()

# =========================
# 3. 初始化 USB 摄像头
# =========================
cap = cv2.VideoCapture("/dev/video0")  # 与采集代码一致
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
assert cap.isOpened(), "USB camera open failed"

# 读取一帧确认尺寸
ret, frame = cap.read()
assert ret
H, W, _ = frame.shape

# =========================
# 4. 固定夹爪宽度（policy 需要）
# =========================
FIXED_GRIPPER_WIDTH = np.array([0.04], dtype=np.float32)  # 4cm

# =========================
# 5. episode_start_pose（必须）
# =========================
# policy 的相对位姿编码依赖这个
data = connector.get_latest_data()
while data["position"] is None:
    data = connector.get_latest_data()

init_pos = np.array(data["position"], dtype=np.float32)
init_rot_mat = np.array(data["rotation"], dtype=np.float32)
init_rotvec = R.from_matrix(init_rot_mat).as_rotvec()

episode_start_pose = [
    np.concatenate([init_pos, init_rotvec], axis=0)
]

# =========================
# 6. 推理主循环
# =========================
print("Start inference loop (Ctrl+C exit)")

while True:
    # ---------- 时间戳 ----------
    timestamp = np.array([time.time()], dtype=np.float64)

    # ---------- USB 摄像头 ----------
    ret, frame = cap.read()
    if not ret:
        continue

    # BGR → RGB（与采集保持一致）
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    rgb = frame.astype(np.uint8)  # 这里可以改名字，比如 bgr = frame

    # ---------- iPhone 位姿 ----------
    data = connector.get_latest_data()
    if data["position"] is None:
        continue

    pos = np.array(data["position"], dtype=np.float32)
    rot_mat = np.array(data["rotation"], dtype=np.float32)

    # rotation matrix → axis-angle（关键一步）
    rot_axis_angle = R.from_matrix(rot_mat).as_rotvec().astype(np.float32)

    # =========================
    # 构造 env_obs（模拟 env.get_obs()）
    # =========================
    env_obs = {
        # 图像 obs，形状 [T, H, W, 3]
        "camera0_rgb": rgb[None],

        # 末端位姿 obs
        "robot0_eef_pos": pos[None],
        "robot0_eef_rot_axis_angle": rot_axis_angle[None],

        # 夹爪宽度
        "robot0_gripper_width": FIXED_GRIPPER_WIDTH[None],

        # 时间戳
        "timestamp": timestamp
    }

    # =========================
    # 构造 policy obs
    # =========================
    obs_dict_np = get_real_umi_obs_dict(
        env_obs=env_obs,
        shape_meta=cfg.task.shape_meta,
        obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
        tx_robot1_robot0=None,
        episode_start_pose=episode_start_pose
    )

    # numpy → torch → batch
    obs_dict = dict_apply(
        obs_dict_np,
        lambda x: torch.from_numpy(x).unsqueeze(0).to("cuda")
    )

    print(obs_dict.keys())
    # =========================
    # policy 推理
    # =========================
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
        raw_action = result["action_pred"][0].cpu().numpy()

    # =========================
    # 解码 action（与真实执行前一致）
    # =========================
    action = get_real_umi_action(
        raw_action=raw_action,
        env_obs=env_obs,
        action_pose_repr=cfg.task.pose_repr.action_pose_repr
    )

    # =========================
    # 打印结果
    # =========================
    print("========== POLICY ACTION ==========")
    print("EEF pos:", action[:3])
    print("EEF rot (axis-angle):", action[3:6])
    print("Gripper width:", action[6])
    print("===================================")

    time.sleep(1 / 30)  # 30Hz