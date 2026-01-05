import cv2
import yaml
import numpy as np
import os

def load_camera_params(yaml_path):
    """
    读取 YAML 中的相机参数
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

    orig_width = cfg['Camera.width']
    orig_height = cfg['Camera.height']

    return fx, fy, cx, cy, D, orig_width, orig_height


def undistort_fisheye_highres(img, fx, fy, cx, cy, D, orig_size, balance=0.0):
    """
    高分辨率图片去畸变（比例一致）
    img        : 高分辨率输入图片
    fx,fy,cx,cy: 原始内参
    D          : 畸变参数
    orig_size  : 原始标定尺寸 (width, height)
    balance    : 0.0裁剪多，1.0视野大
    """
    h, w = img.shape[:2]
    orig_w, orig_h = orig_size

    # === 缩放内参到高分辨率 ===
    scale_x = w / orig_w
    scale_y = h / orig_h
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    K_new = np.array([
        [fx_new, 0, cx_new],
        [0, fy_new, cy_new],
        [0, 0, 1]
    ], dtype=np.float64)

    img_size = (w, h)

    # 生成去畸变映射
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_new, D, img_size, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_new, D, np.eye(3), new_K, img_size, cv2.CV_16SC2
    )

    undistorted = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return undistorted


if __name__ == "__main__":
    # 当前目录
    base_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(base_dir, "setting2.yaml")
    img_path = os.path.join(base_dir, "input2.png")

    # 读取图片
    img = cv2.imread(img_path)
    assert img is not None, "图片读取失败"

    # 读取原始相机参数
    fx, fy, cx, cy, D, orig_w, orig_h = load_camera_params(yaml_path)

    # 去畸变
    undistorted = undistort_fisheye_highres(
        img, fx, fy, cx, cy, D, (orig_w, orig_h), balance=0.0
    )

    # 保存结果
    out_path = os.path.join(base_dir, "undistorted.png")
    cv2.imwrite(out_path, undistorted)

    # 显示结果
    cv2.imshow("raw", img)
    cv2.imshow("undistorted", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()