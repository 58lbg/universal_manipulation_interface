import cv2
import yaml
import numpy as np
import os

def load_camera_params(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    fx = cfg['Camera1.fx']
    fy = cfg['Camera1.fy']
    cx = cfg['Camera1.cx']
    cy = cfg['Camera1.cy']
    D = np.array([cfg['Camera1.k1'], cfg['Camera1.k2'], cfg['Camera1.k3'], cfg['Camera1.k4']], dtype=np.float64)
    orig_w = cfg['Camera.width']
    orig_h = cfg['Camera.height']
    return fx, fy, cx, cy, D, orig_w, orig_h

def undistort_fisheye_highres(img, fx, fy, cx, cy, D, orig_size, balance=1.0):
    h, w = img.shape[:2]
    orig_w, orig_h = orig_size

    # 缩放内参
    scale_x = w / orig_w
    scale_y = h / orig_h
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    K_new = np.array([[fx_new, 0, cx_new],
                      [0, fy_new, cy_new],
                      [0, 0, 1]], dtype=np.float64)

    # 使用全视野去畸变
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_new, D, (w,h), np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_new, D, np.eye(3), new_K, (w,h), cv2.CV_16SC2)
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 标记中心
    cv2.drawMarker(undistorted, (int(cx_new), int(cy_new)), (0,0,255), cv2.MARKER_CROSS, 50, 2)
    return undistorted

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(base_dir, "setting2.yaml")
    img_path = os.path.join(base_dir, "input2.png")

    img = cv2.imread(img_path)
    assert img is not None, "图片读取失败"

    fx, fy, cx, cy, D, orig_w, orig_h = load_camera_params(yaml_path)
    undistorted = undistort_fisheye_highres(img, fx, fy, cx, cy, D, (orig_w, orig_h), balance=1.0)

    out_path = os.path.join(base_dir, "undistorted.png")
    cv2.imwrite(out_path, undistorted)
