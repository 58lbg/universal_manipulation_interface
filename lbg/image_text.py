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

    # Kannala-Brandt模型，但只提供了4个参数
    D = np.array([
        float(cfg['Camera1.k1']),
        float(cfg['Camera1.k2']),
        float(cfg['Camera1.k3']),
        float(cfg['Camera1.k4'])
    ], dtype=np.float64)

    orig_w = int(cfg['Camera.width'])
    orig_h = int(cfg['Camera.height'])

    return fx, fy, cx, cy, D, orig_w, orig_h


def undistort_fisheye_simple(img, fx, fy, cx, cy, D, orig_size, scale=1.0, balance=0.5):
    """
    使用OpenCV fisheye模块进行去畸变
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
    out_size = (out_w, out_h)

    # 估计新的相机矩阵
    # balance参数控制视野保留程度：0=完全裁剪，1=保留所有
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3),
        balance=balance,  # 调整这个值
        new_size=out_size
    )

    # 生成映射
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, out_size, cv2.CV_16SC2
    )

    # 重映射
    undistorted = cv2.remap(
        img, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return undistorted, new_K


def undistort_standard(img, fx, fy, cx, cy, D, orig_size, scale=1.0):
    """
    使用标准去畸变方法
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

    out_w = int(w * scale)
    out_h = int(h * scale)

    # 对于畸变参数，使用前4个（标准的径向畸变）
    # alpha参数控制视野保留：0=完全裁剪，1=保留所有
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (w, h), alpha=0.5,  # 调整alpha值
        newImgSize=(out_w, out_h)
    )

    undistorted = cv2.undistort(img, K, D, None, new_K)

    return undistorted, new_K


def main():
    base_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(base_dir, "setting2.yaml")
    img_path = os.path.join(base_dir, "input2.png")

    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图片 {img_path}")
        return

    # 读取相机参数
    fx, fy, cx, cy, D, orig_w, orig_h = load_camera_params(yaml_path)

    print("=" * 50)
    print("相机参数:")
    print(f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"畸变参数: k1={D[0]:.6f}, k2={D[1]:.6f}, k3={D[2]:.6f}, k4={D[3]:.6f}")
    print(f"原始标定尺寸: {orig_w}x{orig_h}")
    print(f"输入图片尺寸: {img.shape[1]}x{img.shape[0]}")
    print("=" * 50)

    # 方法1：使用fisheye方法（适用于鱼眼畸变）
    print("\n方法1：使用fisheye去畸变")
    for balance in [0.0, 0.3, 0.6, 1.0]:
        undistorted1, K1 = undistort_fisheye_simple(
            img, fx, fy, cx, cy, D, (orig_w, orig_h),
            scale=1.0, balance=balance
        )
        out_path1 = os.path.join(base_dir, f"undistorted_fisheye_balance{balance:.1f}.png")
        cv2.imwrite(out_path1, undistorted1)
        print(f"  balance={balance:.1f}: 已保存到 {out_path1}")
        print(f"    新内参矩阵:")
        print(f"    [{K1[0, 0]:.2f}, 0, {K1[0, 2]:.2f}]")
        print(f"    [0, {K1[1, 1]:.2f}, {K1[1, 2]:.2f}]")
        print(f"    [0, 0, 1]")

    # 方法2：使用标准方法（适用于轻微畸变）
    print("\n方法2：使用标准去畸变")
    for alpha in [0.0, 0.3, 0.6, 1.0]:
        undistorted2, K2 = undistort_standard(
            img, fx, fy, cx, cy, D, (orig_w, orig_h),
            scale=1.0
        )
        out_path2 = os.path.join(base_dir, f"undistorted_standard_alpha{alpha:.1f}.png")
        cv2.imwrite(out_path2, undistorted2)
        print(f"  alpha={alpha:.1f}: 已保存到 {out_path2}")

    # 创建对比图
    print("\n创建对比图...")
    h, w = img.shape[:2]

    # 选择效果最好的一个进行对比
    best_fisheye, _ = undistort_fisheye_simple(img, fx, fy, cx, cy, D, (orig_w, orig_h), balance=0.6)
    best_standard, _ = undistort_standard(img, fx, fy, cx, cy, D, (orig_w, orig_h))

    # 调整大小以便显示
    if w > 1200:
        scale_factor = 1200 / w
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img_resized = cv2.resize(img, (new_w, new_h))
        best_fisheye_resized = cv2.resize(best_fisheye, (new_w, new_h))
        best_standard_resized = cv2.resize(best_standard, (new_w, new_h))
    else:
        img_resized = img
        best_fisheye_resized = best_fisheye
        best_standard_resized = best_standard

    # 创建对比图像
    top_row = np.hstack([img_resized, best_fisheye_resized])
    bottom_row = np.hstack([best_standard_resized, np.zeros_like(best_standard_resized)])
    comparison = np.vstack([top_row, bottom_row])

    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "原始图像", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "fisheye矫正 (balance=0.6)", (w // 2 + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "标准矫正", (10, h // 2 + 30), font, 1, (255, 255, 255), 2)

    out_comparison = os.path.join(base_dir, "comparison.png")
    cv2.imwrite(out_comparison, comparison)

    # # 显示结果
    # cv2.imshow("对比: 原始 | fisheye矫正 | 标准矫正", comparison)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # print(f"\n对比图已保存到: {out_comparison}")
    # print("\n建议:")
    # print("1. 查看生成的图片，选择效果最好的方法")
    # print("2. 观察直线是否变直来判断矫正效果")
    # print("3. 调整balance/alpha参数以获得最佳视野")


if __name__ == "__main__":
    main()