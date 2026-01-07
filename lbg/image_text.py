import cv2
import yaml
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading


class DistortionCorrector:
    def __init__(self, img_path, yaml_path):
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        self.fx, self.fy, self.cx, self.cy, self.D, self.orig_w, self.orig_h = self.load_camera_params(yaml_path)
        self.current_D = self.D.copy()

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("畸变参数调整工具")

        # 创建画布显示图片
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # 畸变参数滑块
        self.sliders = []
        param_names = ['k1', 'k2', 'k3', 'k4']
        for i, name in enumerate(param_names):
            tk.Label(control_frame, text=name).grid(row=i, column=0, sticky='w')

            # 创建滑块，范围比原始值大一些
            initial_value = float(self.D[i]) if i < len(self.D) else 0.0
            slider = tk.Scale(
                control_frame,
                from_=initial_value - 0.1,
                to=initial_value + 0.1,
                resolution=0.0001,
                orient=tk.HORIZONTAL,
                length=200,
                command=lambda val, idx=i: self.update_param(idx, float(val))
            )
            slider.set(initial_value)
            slider.grid(row=i, column=1, pady=5)
            self.sliders.append(slider)

        # Alpha参数滑块
        tk.Label(control_frame, text="Alpha").grid(row=4, column=0, sticky='w')
        self.alpha_slider = tk.Scale(
            control_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_alpha
        )
        self.alpha_slider.set(0.0)
        self.alpha_slider.grid(row=4, column=1, pady=5)

        # 缩放因子滑块
        tk.Label(control_frame, text="Scale").grid(row=5, column=0, sticky='w')
        self.scale_slider = tk.Scale(
            control_frame,
            from_=0.5,
            to=2.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_scale
        )
        self.scale_slider.set(1.0)
        self.scale_slider.grid(row=5, column=1, pady=5)

        # 重置按钮
        reset_btn = tk.Button(control_frame, text="重置参数", command=self.reset_params)
        reset_btn.grid(row=6, column=0, columnspan=2, pady=10)

        # 保存按钮
        save_btn = tk.Button(control_frame, text="保存结果", command=self.save_result)
        save_btn.grid(row=7, column=0, columnspan=2, pady=10)

        # 当前参数显示
        self.param_label = tk.Label(control_frame, text="", font=("Courier", 10))
        self.param_label.grid(row=8, column=0, columnspan=2, pady=10)

        self.current_alpha = 0.0
        self.current_scale = 1.0

        # 初始显示
        self.update_display()

    def load_camera_params(self, yaml_path):
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        fx = float(cfg['Camera1.fx'])
        fy = float(cfg['Camera1.fy'])
        cx = float(cfg['Camera1.cx'])
        cy = float(cfg['Camera1.cy'])

        D = np.array([
            float(cfg['Camera1.k1']),
            float(cfg['Camera1.k2']),
            float(cfg['Camera1.k3']),
            float(cfg['Camera1.k4'])
        ], dtype=np.float64)

        orig_w = int(cfg['Camera.width'])
        orig_h = int(cfg['Camera.height'])

        return fx, fy, cx, cy, D, orig_w, orig_h

    def undistort_image(self):
        h, w = self.img.shape[:2]

        # 缩放内参
        scale_x = w / self.orig_w
        scale_y = h / self.orig_h
        fx_new = self.fx * scale_x
        fy_new = self.fy * scale_y
        cx_new = self.cx * scale_x
        cy_new = self.cy * scale_y

        K = np.array([
            [fx_new, 0, cx_new],
            [0, fy_new, cy_new],
            [0, 0, 1]
        ], dtype=np.float64)

        out_w = int(w * self.current_scale)
        out_h = int(h * self.current_scale)

        # 获取新的相机矩阵
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, self.current_D[:4], (w, h),
            alpha=self.current_alpha,
            newImgSize=(out_w, out_h)
        )

        # 去畸变
        undistorted = cv2.undistort(
            self.img, K, self.current_D[:4], None, new_K
        )

        return undistorted

    def update_param(self, idx, value):
        self.current_D[idx] = value
        self.update_display()

    def update_alpha(self, value):
        self.current_alpha = float(value)
        self.update_display()

    def update_scale(self, value):
        self.current_scale = float(value)
        self.update_display()

    def reset_params(self):
        self.current_D = self.D.copy()
        for i, slider in enumerate(self.sliders):
            if i < len(self.current_D):
                slider.set(self.current_D[i])
        self.alpha_slider.set(0.0)
        self.scale_slider.set(1.0)
        self.update_display()

    def save_result(self):
        undistorted = self.undistort_image()
        base_dir = os.path.dirname(__file__)
        out_path = os.path.join(base_dir, "manual_corrected1.png")
        cv2.imwrite(out_path, undistorted)

        # 保存参数
        param_path = os.path.join(base_dir, "corrected_params1.yaml")
        with open(param_path, 'w') as f:
            f.write(f"# 手动调整后的畸变参数\n")
            f.write(f"k1: {self.current_D[0]:.10f}\n")
            f.write(f"k2: {self.current_D[1]:.10f}\n")
            f.write(f"k3: {self.current_D[2]:.10f}\n")
            f.write(f"k4: {self.current_D[3]:.10f}\n")
            f.write(f"alpha: {self.current_alpha:.3f}\n")
            f.write(f"scale: {self.current_scale:.3f}\n")

        print(f"结果已保存到: {out_path}")
        print(f"参数已保存到: {param_path}")

    def update_display(self):
        # 在后台线程中处理图像，避免界面卡顿
        def process():
            undistorted = self.undistort_image()

            # 转换为RGB并调整大小以适应画布
            undistorted_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(undistorted_rgb)

            # 保持宽高比缩放
            canvas_width = self.canvas.winfo_width() or 800
            canvas_height = self.canvas.winfo_height() or 600

            img_width, img_height = img_pil.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            new_size = (int(img_width * scale), int(img_height * scale))
            img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img_pil)

            # 在主线程中更新显示
            self.root.after(0, lambda: self._update_canvas(img_tk))

        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()

    def _update_canvas(self, img_tk):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # 保持引用

        # 更新参数显示
        param_text = f"k1: {self.current_D[0]:.6f}\n"
        param_text += f"k2: {self.current_D[1]:.6f}\n"
        param_text += f"k3: {self.current_D[2]:.6f}\n"
        param_text += f"k4: {self.current_D[3]:.6f}\n"
        param_text += f"alpha: {self.current_alpha:.3f}\n"
        param_text += f"scale: {self.current_scale:.3f}"
        self.param_label.config(text=param_text)

    def run(self):
        self.root.mainloop()


def main():
    base_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(base_dir, "setting1.yaml")
    img_path = os.path.join(base_dir, "input.png")

    try:
        corrector = DistortionCorrector(img_path, yaml_path)
        corrector.run()
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()