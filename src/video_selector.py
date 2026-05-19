"""
============================================================
视频选择模块 —— GUI 界面选择待处理的视频文件
============================================================

提供基于 tkinter 的图形界面，扫描 data/ 目录下所有 .mp4
文件，让用户通过列表选择要处理的视频，并可设置卷积核大小。

依赖：
  - tkinter（Python 内置）
  - pathlib
============================================================
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path


def scan_video_files(data_dir="data") -> list:
    """扫描 data 目录下所有 .mp4 文件，返回 (文件名, 完整路径) 列表"""
    video_dir = Path(data_dir)
    if not video_dir.exists():
        return []

    videos = []
    for f in sorted(video_dir.iterdir()):
        if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
            videos.append((f.stem, str(f.resolve())))
    return videos


def select_video_gui(title="选择待处理的视频", default_conv_size=3, default_scale_factor=2) -> tuple:
    """
    弹出 GUI 窗口让用户选择视频文件和卷积核大小区间。

    参数:
        title: 窗口标题
        default_conv_size: 默认卷积核大小（作为区间默认值）
        default_scale_factor: 默认放大倍数

    返回:
        (selected_name, selected_path, conv_size_min, conv_size_max, enable_transposed,
         enable_transposed_x, enable_scale, scale_factor, enable_diff)
         或 (None, None, None, None, None, None, None, None, None) 如果用户取消
    """
    videos = scan_video_files()
    if not videos:
        messagebox.showerror("错误", "data/ 目录下没有找到任何视频文件！")
        return None, None, None, None, None, None, None

    selected = [None, None]  # [name, path]

    root = tk.Tk()
    conv_size_min = tk.IntVar(value=max(1, default_conv_size - 1))
    conv_size_max = tk.IntVar(value=default_conv_size)
    enable_diff = tk.BooleanVar(value=True)
    enable_scale = tk.BooleanVar(value=True)
    scale_factor = tk.IntVar(value=default_scale_factor)
    root.title(title)
    root.geometry("600x520")
    root.minsize(400, 400)

    # 设置窗口居中
    root.update_idletasks()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - 600) // 2
    y = (screen_h - 520) // 2
    root.geometry(f"+{x}+{y}")

    # 主框架
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 标题标签
    label = ttk.Label(main_frame, text="请选择要处理的视频文件：",
                      font=("微软雅黑", 12, "bold"))
    label.pack(pady=(0, 10))

    # 列表框架
    list_frame = ttk.Frame(main_frame)
    list_frame.pack(fill=tk.BOTH, expand=True)

    # 滚动条
    scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)

    # 视频列表
    listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                         font=("Consolas", 11),
                         selectmode=tk.SINGLE,
                         activestyle='none',
                         borderwidth=1,
                         relief=tk.SOLID)
    scrollbar.config(command=listbox.yview)

    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 填充列表
    for i, (name, path) in enumerate(videos):
        display_text = f"  {i+1:2d}. {name}"
        listbox.insert(tk.END, display_text)

    # 信息标签（显示选中视频的详细信息）
    info_var = tk.StringVar()
    info_var.set(f"共找到 {len(videos)} 个视频文件")
    info_label = ttk.Label(main_frame, textvariable=info_var,
                           font=("微软雅黑", 9), foreground="gray")
    info_label.pack(pady=(5, 0))

    # ========== 卷积核参数设置区域 ==========
    conv_frame = ttk.LabelFrame(main_frame, text="卷积核参数设置", padding="10")
    conv_frame.pack(fill=tk.X, pady=(10, 0))

    # 卷积核大小区间选择（最小值 ~ 最大值）
    conv_size_frame = ttk.Frame(conv_frame)
    conv_size_frame.pack(fill=tk.X, pady=(5, 0))

    ttk.Label(conv_size_frame, text="卷积核大小区间：",
              font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=(0, 10))

    # 最小值 Spinbox
    ttk.Label(conv_size_frame, text="最小值",
              font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 3))
    conv_min_spinbox = ttk.Spinbox(conv_size_frame, from_=1, to=20,
                                   textvariable=conv_size_min, width=4,
                                   font=("微软雅黑", 10))
    conv_min_spinbox.pack(side=tk.LEFT, padx=(0, 10))

    # 最大值 Spinbox
    ttk.Label(conv_size_frame, text="最大值",
              font=("微软雅黑", 9)).pack(side=tk.LEFT, padx=(0, 3))
    conv_max_spinbox = ttk.Spinbox(conv_size_frame, from_=1, to=20,
                                   textvariable=conv_size_max, width=4,
                                   font=("微软雅黑", 10))
    conv_max_spinbox.pack(side=tk.LEFT, padx=(0, 10))

    # 提示信息
    ttk.Label(conv_frame, text="设置区间后，程序将依次处理该区间内所有大小的卷积核（如 3~6 则处理 3,4,5,6）",
              font=("微软雅黑", 8), foreground="gray").pack(anchor=tk.W, pady=(5, 0))

    # ========== 图像放大设置区域 ==========
    scale_frame = ttk.LabelFrame(main_frame, text="图像放大设置", padding="10")
    scale_frame.pack(fill=tk.X, pady=(10, 0))

    # 是否放大复选框
    scale_check_frame = ttk.Frame(scale_frame)
    scale_check_frame.pack(fill=tk.X, pady=(5, 0))

    ttk.Checkbutton(scale_check_frame, text="放大图像（在时间卷积前放大每帧）",
                    variable=enable_scale).pack(side=tk.LEFT)

    # 放大倍数选择（仅当启用放大时有效）
    scale_val_frame = ttk.Frame(scale_frame)
    scale_val_frame.pack(fill=tk.X, pady=(5, 0))

    ttk.Label(scale_val_frame, text="放大倍数：",
              font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=(0, 10))

    scale_options = [("2x", 2), ("3x", 3), ("4x", 4), ("5x", 5), ("6x", 6)]
    for text, val in scale_options:
        ttk.Radiobutton(scale_val_frame, text=text, variable=scale_factor,
                        value=val).pack(side=tk.LEFT, padx=(0, 5))

    ttk.Label(scale_frame, text="取消勾选则直接对原始尺寸进行时间卷积",
              font=("微软雅黑", 8), foreground="gray").pack(anchor=tk.W, pady=(5, 0))

    # ========== 帧间差分设置区域 ==========
    diff_frame = ttk.LabelFrame(main_frame, text="帧间差分设置", padding="10")
    diff_frame.pack(fill=tk.X, pady=(10, 0))

    ttk.Checkbutton(diff_frame, text="启用帧间差分",
                    variable=enable_diff).pack(anchor=tk.W, pady=(2, 0))

    ttk.Label(diff_frame, text="启用后输出帧间差分灰度视频，反映运动强度",
              font=("微软雅黑", 8), foreground="gray").pack(anchor=tk.W, pady=(5, 0))

    # 按钮框架
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill=tk.X, pady=(10, 0))

    def on_select():
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先选择一个视频文件")
            return
        idx = selection[0]
        selected[0] = videos[idx][0]
        selected[1] = videos[idx][1]
        root.quit()
        root.destroy()

    def on_cancel():
        root.quit()
        root.destroy()

    def on_double_click(event):
        on_select()

    def on_listbox_select(event):
        selection = listbox.curselection()
        if selection:
            idx = selection[0]
            name, path = videos[idx]
            size_mb = os.path.getsize(path) / (1024 * 1024)
            info_var.set(f"已选择: {name}  |  路径: {path}  |  大小: {size_mb:.1f} MB")

    listbox.bind('<<ListboxSelect>>', on_listbox_select)
    listbox.bind('<Double-Button-1>', on_double_click)

    select_btn = ttk.Button(btn_frame, text="确定选择", command=on_select)
    select_btn.pack(side=tk.RIGHT, padx=(5, 0))

    cancel_btn = ttk.Button(btn_frame, text="取消", command=on_cancel)
    cancel_btn.pack(side=tk.RIGHT)

    # 默认选中第一个
    if videos:
        listbox.selection_set(0)
        listbox.activate(0)
        on_listbox_select(None)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()

    if selected[0]:
        # 返回卷积核大小区间（最小值, 最大值）
        c_min = conv_size_min.get()
        c_max = conv_size_max.get()
        # 确保 min <= max
        if c_min > c_max:
            c_min, c_max = c_max, c_min
        return (selected[0], selected[1], c_min, c_max,
                enable_scale.get(), scale_factor.get(),
                enable_diff.get())
    return None, None, None, None, None, None, None


def select_video_simple(title="选择待处理的视频", default_conv_size=3, default_scale_factor=2) -> tuple:
    """
    简化版：如果只有一个视频则直接返回，多个视频则弹出 GUI。

    返回:
        (selected_name, selected_path, conv_size_min, conv_size_max,
         enable_scale, scale_factor, enable_diff)
    """
    videos = scan_video_files()
    if not videos:
        print("[错误] data/ 目录下没有找到任何视频文件！")
        return None, None, None, None, None, None, None
    if len(videos) == 1:
        name, path = videos[0]
        print(f"data/ 目录下仅有一个视频文件，自动选择: {name}")
        # 单视频时默认启用放大和帧间差分
        c_min = max(1, default_conv_size - 1)
        c_max = default_conv_size
        return name, path, c_min, c_max, True, default_scale_factor, True
    return select_video_gui(title, default_conv_size, default_scale_factor)


if __name__ == "__main__":
    # 测试 GUI 选择器
    result = select_video_gui()
    if result[0]:
        name, path, c_min, c_max, enable_s, s_factor, enable_diff = result
        print(f"已选择: {name}")
        print(f"路径: {path}")
        print(f"卷积核大小区间: {c_min} ~ {c_max}")
        print(f"帧间差分: {'启用' if enable_diff else '禁用'}")
        print(f"图像放大: {'启用' if enable_s else '禁用'} (倍数: {s_factor}x)")
    else:
        print("用户取消了选择")
