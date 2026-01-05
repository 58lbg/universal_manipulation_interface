from pathlib import Path

import cv2
import yaml
import numpy as np

def load_camera_params(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    fx = cfg['Camera1.fx']
    fy = cfg['Camera1.fy']
    cx = cfg['Camera1.cx']
    cy = cfg['Camera1.cy']

    k1 = cfg['Camera1.k1']
    k2 = cfg['Camera1.k2']
    k3 = cfg['Camera1.k3']
    k4 = cfg['Camera1.k4']

    width = cfg['Camera.width']
    height = cfg['Camera.height']

    # 内参矩阵
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    # Kannala-Brandt (fisheye) 畸变参数
    D = np.array([k1, k2, k3, k4], dtype=np.float64)

    return K, D, (width, height)


def undistort_fisheye(img, K, D, img_size, balance=0.0):
    """
    balance:
        0   → 裁剪多一些，视野小但直
        1   → 保留更多视野，边缘拉伸
    """
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_size, np.eye(3), balance=balance
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, img_size, cv2.CV_16SC2
    )

    undistorted = cv2.remap(
        img, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return undistorted


if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent
    setting_file = file_path.joinpath('setting.yaml')
    yaml_path = str(setting_file.absolute())
    image_path = str(file_path.joinpath("input.png").absolute())

    K, D, img_size = load_camera_params(yaml_path)

    img = cv2.imread(image_path)
    assert img is not None, "图片读取失败"

    undistorted = undistort_fisheye(
        img, K, D, img_size, balance=0.0
    )

    cv2.imwrite("undistorted.jpg", undistorted)

    cv2.imshow("raw", img)
    cv2.imshow("undistorted", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()