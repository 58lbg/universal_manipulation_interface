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

    fx = float(cfg['Camera1.fx'])
    fy = float(cfg['Camera1.fy'])
    cx = float(cfg['Camera1.cx'])
    cy = float(cfg['Camera1.cy'])

    # Kannala-Brandt8 模型通常有8个参数，但这里只有4个
    # 创建完整的8参数数组（后4个为0）
    D = np.zeros(8, dtype=np.float64)
    D[0] = float(cfg['Camera1.k1'])
    D[1] = float(cfg['Camera1.k2'])
    D[2] = float(cfg['Camera1.k3'])
    D[3] = float(cfg['Camera1.k4'])
    # D[4], D[5], D[6], D[7] 保持为0

    orig_w = int(cfg['Camera.width'])
    orig_h = int(cfg['Camera.height'])

    return fx, fy, cx, cy, D, orig_w, orig_h


def undistort_fisheye(img, fx, fy, cx, cy, D, orig_size, scale=1.0):
    """
    矫正鱼眼图片 - 兼容Kannala-Brandt模型
    """
    h, w = img.shape[:2]
    orig_w, orig_h = orig_size

    # 缩放内参到输入图片尺寸
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

    # 输出图片尺寸
    out_w = int(w * scale)
    out_h = int(h * scale)

    # 对于Kannala-Brandt模型，需要使用不同的方法
    try:
        # 方法1：尝试使用OpenCV的omnidir模块（如果可用）
        import cv2.omnidir as od
        undistorted = od.undistortImage(
            img, K, D, cv2.RECTIFY_PERSPECTIVE,
            Knew=K,  # 使用相同的内参
            size=(out_w, out_h)
        )
        return undistorted
    except ImportError:
        # 方法2：使用fisheye模块近似处理（可能不够精确）
        print("警告：未找到omnidir模块，使用fisheye近似处理")

        # 仅使用前4个参数作为fisheye模型的近似
        D_fisheye = D[:4].copy()

        # 估计新的相机矩阵
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D_fisheye, (w, h), np.eye(3),
            balance=1.0,  # 可以调整这个值来控制视野保留程度
            new_size=(out_w, out_h)
        )

        # 生成映射
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D_fisheye, np.eye(3), new_K,
            (out_w, out_h), cv2.CV_16SC2
        )

        # 重映射
        undistorted = cv2.remap(
            img, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        return undistorted


def undistort_standard(img, fx, fy, cx, cy, D, orig_size, scale=1.0):
    """
    另一种方法：使用标准去畸变（假设畸变较小）
    """
    h, w = img.shape[:2]
    orig_w, orig_h = orig_size

    # 缩放内参
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

    # 使用标准去畸变（前4个参数）
    D_std = D[:4]

    out_w = int(w * scale)
    out_h = int(h * scale)

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D_std, (w, h), alpha=0,
        newImgSize=(out_w, out_h)
    )

    undistorted = cv2.undistort(
        img, K, D_std, None, new_K
    )

    return undistorted


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(base_dir, "setting2.yaml")
    img_path = os.path.join(base_dir, "input2.png")

    # 读取图片
    img = cv2.imread(img_path)
    assert img is not None, f"图片读取失败: {img_path}"

    # 读取相机参数
    fx, fy, cx, cy, D, orig_w, orig_h = load_camera_params(yaml_path)

    print("相机参数:")
    print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"畸变参数: {D}")
    print(f"原始尺寸: {orig_w}x{orig_h}")
    print(f"输入图片尺寸: {img.shape[1]}x{img.shape[0]}")

    # 尝试不同的去畸变方法
    try:
        # 方法1：使用fisheye近似
        undistorted = undistort_fisheye(img, fx, fy, cx, cy, D, (orig_w, orig_h), scale=1.0)
        out_path1 = os.path.join(base_dir, "undistorted_fisheye.png")
        cv2.imwrite(out_path1, undistorted)
        print(f"已保存fisheye近似结果到: {out_path1}")
    except Exception as e:
        print(f"fisheye方法失败: {e}")

    # 方法2：使用标准去畸变
    undistorted2 = undistort_standard(img, fx, fy, cx, cy, D, (orig_w, orig_h), scale=1.0)
    out_path2 = os.path.join(base_dir, "undistorted_standard.png")
    cv2.imwrite(out_path2, undistorted2)
    print(f"已保存标准去畸变结果到: {out_path2}")

    # 可视化比较
    combined = np.hstack([img, undistorted2])
    cv2.imshow("原始 vs 去畸变", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()