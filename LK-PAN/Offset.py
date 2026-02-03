import os
import cv2
import numpy as np
from pathlib import Path


def create_offset_images(input_folder, output_base_dir):
    """
    为输入文件夹中的每个图片创建40个偏移版本

    参数:
        input_folder: 输入文件夹路径
        output_base_dir: 输出基础目录路径
    """
    # 确保输出基础目录存在
    os.makedirs(output_base_dir, exist_ok=True)

    # 支持的图片格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # 定义所有偏移类型
    offset_types = [
        # 水平向左平移
        *[f"Offset-L-{i}" for i in range(1, 6)],
        # 水平向右平移
        *[f"Offset-R-{i}" for i in range(1, 6)],
        # 垂直向上平移
        *[f"Offset-U-{i}" for i in range(1, 6)],
        # 垂直向下平移
        *[f"Offset-D-{i}" for i in range(1, 6)],
        # 左上平移
        *[f"Offset-LU-{i}" for i in range(1, 6)],
        # 右上平移
        *[f"Offset-RU-{i}" for i in range(1, 6)],
        # 左下平移
        *[f"Offset-LD-{i}" for i in range(1, 6)],
        # 右下平移
        *[f"Offset-RD-{i}" for i in range(1, 6)]
    ]

    # 创建所有偏移类型的大文件夹
    for offset_type in offset_types:
        offset_dir = os.path.join(output_base_dir, offset_type)
        os.makedirs(offset_dir, exist_ok=True)

    # 统计信息
    total_images = 0
    total_generated = 0

    print(f"开始处理文件夹: {input_folder}")
    print("创建40个偏移文件夹，保持原始文件结构...")

    # 遍历输入文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(input_folder):
        # 计算相对于输入文件夹的路径
        relative_path = os.path.relpath(root, input_folder)
        if relative_path == '.':
            relative_path = ''

        # 处理当前文件夹中的图片文件
        for file in files:
            file_path = os.path.join(root, file)
            if Path(file).suffix.lower() in supported_formats:
                total_images += 1

                # 读取原始图片
                original_img = cv2.imread(file_path)
                if original_img is None:
                    print(f"无法读取图片: {file_path}")
                    continue

                # 获取原始图片尺寸
                original_height, original_width = original_img.shape[:2]

                # 验证原始图片尺寸
                if original_width != 512 or original_height != 512:
                    print(f"跳过 {file}: 图片尺寸必须为512×512像素，当前为{original_width}×{original_height}")
                    continue

                print(f"处理图片: {os.path.join(relative_path, file)}")

                # 新正方形的尺寸
                new_size = 384

                # 计算原始中心点
                center_x = original_width // 2
                center_y = original_height // 2

                # 计算初始裁剪区域的左上角坐标
                start_x = center_x - new_size // 2
                start_y = center_y - new_size // 2

                # 为每种偏移类型创建相应的图片
                for offset_type in offset_types:
                    # 解析偏移类型和步数
                    parts = offset_type.split('-')
                    direction = parts[1]
                    step = int(parts[2])

                    # 计算偏移量
                    offset_x = start_x
                    offset_y = start_y

                    if 'L' in direction:
                        offset_x -= step * 12
                    if 'R' in direction:
                        offset_x += step * 12
                    if 'U' in direction:
                        offset_y -= step * 12
                    if 'D' in direction:
                        offset_y += step * 12

                    # 创建输出子目录（保持原始文件夹结构）
                    output_subdir = os.path.join(output_base_dir, offset_type, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)

                    # 裁剪图片
                    cropped_img = original_img[offset_y:offset_y + new_size, offset_x:offset_x + new_size]

                    # 保存图片（保持原始文件名）
                    output_path = os.path.join(output_subdir, file)
                    cv2.imwrite(output_path, cropped_img)

                    total_generated += 1

                print(f"  └─ 生成40个偏移版本")

    print(f"\n处理完成！")
    print(f"处理图片总数: {total_images}")
    print(f"生成偏移图片总数: {total_generated}")
    print(f"输出基础目录: {output_base_dir}")

    # 显示输出文件夹结构
    print(f"\n输出文件夹结构:")
    for offset_type in offset_types[:5]:  # 只显示前5个偏移类型
        offset_dir = os.path.join(output_base_dir, offset_type)
        print(f"  {offset_type}/")
        # 显示该偏移类型下的文件结构
        for root, dirs, files in os.walk(offset_dir):
            level = root.replace(offset_dir, '').count(os.sep)
            indent = ' ' * 2 * (level + 1)
            if level == 0:
                # 根目录
                for file in files[:3]:  # 只显示前3个文件
                    print(f"{indent}{file}")
                if len(files) > 3:
                    print(f"{indent}... 和 {len(files) - 3} 个其他文件")
            break  # 只显示第一级目录


# 使用示例
if __name__ == "__main__":
    # 配置路径
    input_folder = "/data3/limian1/Documents/LPN-main/data/University-Release/train/rd4/0879"  # 替换为您的输入文件夹路径
    output_base_dir = "/data3/limian1/Documents/LPN-main/data/cs"  # 替换为您的输出基础目录路径

    # 处理整个文件夹
    create_offset_images(input_folder, output_base_dir)