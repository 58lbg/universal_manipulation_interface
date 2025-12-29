import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

root = zarr.open('./dataset/dp_train_data.zarr.zip')
print(root.tree())

root = zarr.open('./example_demo_session/dataset.zarr.zip')
print(root.tree())


import numpy as np
import cv2
import matplotlib.pyplot as plt

# ===============================
# 1. 打开 zarr 数据
# ===============================
zarr_path = './dataset/dp_train_data.zarr.zip'   # ← 改成你的路径
root = zarr.open(zarr_path, mode="r")

# ===============================
# 2. 找到一帧图像
# ===============================
# ⚠️ 根据你的实际结构改这行
img = root["data"]["camera0_rgb"][0]
# img.shape == (H, W, 3), dtype=uint8

print("image shape:", img.shape, img.dtype)

# ===============================
# 3. 两种方式显示
# ===============================
plt.figure(figsize=(10, 4))

# --- 3.1 直接当 RGB 显示 ---
plt.subplot(1, 2, 1)
plt.title("Assume RGB (no conversion)")
plt.imshow(img)
plt.axis("off")

# --- 3.2 当作 BGR，转成 RGB 再显示 ---
img_bgr_to_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 2)
plt.title("Assume BGR → RGB")
plt.imshow(img_bgr_to_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()