import numpy as np
from PIL import Image
import os

def get_dominant_color(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((50,50))  # 缩小尺寸加速处理
    pixels = np.array(img)
    avg_color = np.mean(pixels, axis=(0,1))
    return avg_color

def is_light_color(rgb):
    # 计算亮度（CIE标准）
    brightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
    return brightness > 127

def get_contrast_color(bg_color):
    """根据背景色生成对比色"""
    brightness = 0.299*bg_color[0] + 0.587*bg_color[1] + 0.114*bg_color[2]
    return (0, 0, 0) if brightness > 127 else (255, 255, 255)

def generate_contrast_color(bg_rgb):
    if is_light_color(bg_rgb):
        # 深色标签（背景为浅色）
        return (
            max(0, bg_rgb[0] - 150),
            max(0, bg_rgb[1] - 150),
            max(0, bg_rgb[2] - 150)
        )
    else:
        # 浅色标签（背景为深色）
        return (
            min(255, bg_rgb[0] + 150),
            min(255, bg_rgb[1] + 150),
            min(255, bg_rgb[2] + 150)
        )


def get_label_txt(img_path):
    """获取YOLO标签文件路径，自动创建不存在的labels目录"""
    parent_dir = os.path.dirname(os.path.dirname(img_path))
    labels_dir = os.path.join(parent_dir, 'labels')
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 自动创建不存在的labels目录（递归创建）
    os.makedirs(labels_dir, exist_ok=True)
    
    return os.path.join(labels_dir, base_name + '.txt')