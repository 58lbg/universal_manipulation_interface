import cv2
import yaml
import numpy as np
import os

def load_camera_params(yaml_path):
    """
    从 YAML 读取相机内参和畸变参数
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    fx = cfg['Camera1.fx']
    fy = cfg['Camera1.fy']
    cx = cfg['Camera1.cx']
    cy = cfg['Camera1.cy']

    D = np.array([
        cfg['Camera1.k1'],
        cfg['Camera1.k2'],
        cfg['Camera1.k3'],
        cfg['Camera1.k4']
    ], dtype=np.float64)

    orig_w = cfg['Camera.width']
    orig_h = cfg['Camera.height']

    return fx, fy, cx, cy, D, orig_w, orig_h


def undistort_fisheye(img, fx, fy, cx, cy, D, orig_size, scale=1.0):
    """
    完全矫正鱼眼图片
    img       : 输入图片
    fx,fy,cx,cy: 原始内参
    D         : 畸变参数
    orig_size : 原始标定尺寸
    scale     : 输出图片缩放系数（1.0保持原比例）
    """
    h, w = img.shape[:2]
    orig_w, orig_h = orig_size

    # === 缩放内参到输入图片尺寸 ===
    scale_x = w / orig_w
    scale_y = h / orig_h
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    K = np.array([
        [fx_new, 0, cx_new],
        [0, fy_new, cy_new],
        [0, 0, 1]
    ], dtype=np.float64)

    # === 输出图片尺寸（可以放大或保持原比例） ===
    out_w = int(w * scale)
    out_h = int(h * scale)
    out_size = (out_w, out_h)

    # === 生成新的摄像机矩阵（保留全视野） ===
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=1.0, new_size=out_size
    )

    # === 生成去畸变映射 ===
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, out_size, cv2.CV_16SC2
    )

    # === remap 得到矫正图片 ===
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(base_dir, "setting2.yaml")
    img_path = os.path.join(base_dir, "input2.png")

    # 读取图片
    img = cv2.imread(img_path)
    assert img is not None, "图片读取失败"

    # 读取相机参数
    fx, fy, cx, cy, D, orig_w, orig_h = load_camera_params(yaml_path)

    # 去畸变
    undistorted = undistort_fisheye(img, fx, fy, cx, cy, D, (orig_w, orig_h), scale=1.0)

    # 保存结果
    out_path = os.path.join(base_dir, "undistorted.png")
    cv2.imwrite(out_path, undistorted)

