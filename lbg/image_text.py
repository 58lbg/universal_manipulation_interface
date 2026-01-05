import cv2
import numpy as np
import yaml
import os


def detect_and_evaluate_straightness(img):
    """
    检测图像中的直线，评估直线度
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    if lines is None:
        return 0, img.copy()

    # 计算直线的平均角度方差（衡量直线度的指标）
    angles = []
    line_img = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 计算角度方差（越小越直）
    if len(angles) > 1:
        angle_variance = np.var(angles)
    else:
        angle_variance = 1000  # 大值表示不好

    return angle_variance, line_img


def optimize_distortion_params(img, initial_D, fx, fy, cx, cy, orig_size):
    """
    自动优化畸变参数
    """
    h, w = img.shape[:2]
    orig_w, orig_h = orig_size

    best_D = initial_D.copy()
    best_score = float('inf')
    best_img = None

    # 尝试在初始参数周围搜索
    search_range = 0.05  # 搜索范围
    steps = 5  # 搜索步数

    print("开始自动优化畸变参数...")

    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                for l in range(steps):
                    # 生成新的畸变参数
                    test_D = initial_D.copy()
                    test_D[0] = initial_D[0] + (i - steps // 2) * search_range / steps
                    test_D[1] = initial_D[1] + (j - steps // 2) * search_range / steps
                    test_D[2] = initial_D[2] + (k - steps // 2) * search_range / steps
                    test_D[3] = initial_D[3] + (l - steps // 2) * search_range / steps

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

                    # 去畸变
                    undistorted = cv2.undistort(img, K, test_D[:4], None)

                    # 评估直线度
                    score, _ = detect_and_evaluate_straightness(undistorted)

                    if score < best_score:
                        best_score = score
                        best_D = test_D.copy()
                        best_img = undistorted.copy()

                        print(f"找到更好的参数: k1={test_D[0]:.6f}, k2={test_D[1]:.6f}, "
                              f"k3={test_D[2]:.6f}, k4={test_D[3]:.6f}, 评分={score:.4f}")

    print(f"\n最佳参数:")
    print(f"k1={best_D[0]:.10f}")
    print(f"k2={best_D[1]:.10f}")
    print(f"k3={best_D[2]:.10f}")
    print(f"k4={best_D[3]:.10f}")
    print(f"直线度评分: {best_score:.4f}")

    return best_D, best_img, best_score


def main_optimization():
    base_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(base_dir, "setting2.yaml")
    img_path = os.path.join(base_dir, "input2.png")

    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return

    # 读取相机参数
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    fx = float(cfg['Camera1.fx'])
    fy = float(cfg['Camera1.fy'])
    cx = float(cfg['Camera1.cx'])
    cy = float(cfg['Camera1.cy'])

    initial_D = np.array([
        float(cfg['Camera1.k1']),
        float(cfg['Camera1.k2']),
        float(cfg['Camera1.k3']),
        float(cfg['Camera1.k4'])
    ], dtype=np.float64)

    orig_w = int(cfg['Camera.width'])
    orig_h = int(cfg['Camera.height'])

    print("原始参数:")
    print(f"k1={initial_D[0]:.6f}, k2={initial_D[1]:.6f}, "
          f"k3={initial_D[2]:.6f}, k4={initial_D[3]:.6f}")

    # 评估原始图像的直线度
    original_score, original_lines = detect_and_evaluate_straightness(img)
    print(f"原始图像直线度评分: {original_score:.4f}")

    # 使用原始参数去畸变
    scale_x = img.shape[1] / orig_w
    scale_y = img.shape[0] / orig_h
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    K = np.array([
        [fx_new, 0, cx_new],
        [0, fy_new, cy_new],
        [0, 0, 1]
    ], dtype=np.float64)

    undistorted_original = cv2.undistort(img, K, initial_D[:4], None)
    original_undistorted_score, original_undistorted_lines = detect_and_evaluate_straightness(undistorted_original)
    print(f"使用原始参数去畸变后的直线度评分: {original_undistorted_score:.4f}")

    # 自动优化参数
    best_D, best_img, best_score = optimize_distortion_params(
        img, initial_D, fx, fy, cx, cy, (orig_w, orig_h)
    )

    # 显示结果对比
    comparison = np.hstack([
        cv2.resize(img, (400, 300)),
        cv2.resize(undistorted_original, (400, 300)),
        cv2.resize(best_img, (400, 300))
    ])

    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "原始", (50, 50), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, f"原始参数(评分:{original_undistorted_score:.2f})",
                (450, 50), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, f"优化参数(评分:{best_score:.2f})",
                (850, 50), font, 1, (0, 255, 0), 2)

    # 保存结果
    out_path = os.path.join(base_dir, "optimized_correction.png")
    cv2.imwrite(out_path, best_img)

    comparison_path = os.path.join(base_dir, "comparison_optimized.png")
    cv2.imwrite(comparison_path, comparison)

    # 保存优化后的参数
    param_path = os.path.join(base_dir, "optimized_params.yaml")
    with open(param_path, 'w') as f:
        f.write(f"# 优化后的畸变参数\n")
        f.write(f"Camera1.k1: {best_D[0]:.10f}\n")
        f.write(f"Camera1.k2: {best_D[1]:.10f}\n")
        f.write(f"Camera1.k3: {best_D[2]:.10f}\n")
        f.write(f"Camera1.k4: {best_D[3]:.10f}\n")

    print(f"\n优化后的图像已保存到: {out_path}")
    print(f"对比图已保存到: {comparison_path}")
    print(f"优化后的参数已保存到: {param_path}")

    # 显示结果
    # cv2.imshow("对比: 原始 | 原始参数去畸变 | 优化参数去畸变", comparison)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main_optimization()