import os
import sys
import time
import random
import tempfile
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import imagehash
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QListWidget, QComboBox, QDialogButtonBox,
    QListWidgetItem, QMessageBox, QCheckBox, QFileDialog,
    QDialog, QSizePolicy, QSplitter, QSizeGrip,
    QInputDialog, QSlider,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QColor

from utils import get_label_txt

class PlatformUtils:
    @staticmethod
    def get_normalized_path(path):
        """Get platform-normalized path"""
        return Path(path).as_posix() if os.name != 'nt' else os.path.normpath(path)

    @staticmethod
    def get_config_dir():
        """Get platform-specific config directory"""
        if os.name == 'nt':  # Windows
            return os.path.join(os.environ.get('APPDATA'), 'ImageLabeler')
        elif sys.platform == 'darwin':  # MacOS
            return os.path.expanduser('~/Library/Application Support/ImageLabeler')
        else:  # Linux/Unix
            return os.path.expanduser('~/.config/imagelabeler')

    @staticmethod
    def get_temp_dir():
        """Get platform-specific temp directory with our app folder"""
        temp_dir = os.path.join(tempfile.gettempdir(), 'ImageLabeler')
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir


class AutoLabelingThread(QThread):
    progress_updated = pyqtSignal(int, int, str)  # current, total, image_path
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, image_paths, yolo_model, img_w, img_h, conf, iou):
        super().__init__()
        self.image_paths = image_paths
        self.yolo_model = yolo_model
        self.img_w = img_w
        self.img_h = img_h
        self.conf = conf
        self.iou = iou
        self.canceled = False

    def run(self):
        try:
            total = len(self.image_paths)
            for i, img_path in enumerate(self.image_paths, 1):
                if self.canceled:
                    break

                normalized_path = PlatformUtils.get_normalized_path(img_path)
                self.progress_updated.emit(i, total, normalized_path)

                try:
                    # 如果已存在标注，则跳过
                    txt_path = PlatformUtils.get_normalized_path(
                        os.path.splitext(img_path)[0] + '.txt'
                    )
                    if os.path.exists(txt_path):
                        continue

                    # 加载图像
                    img = Image.open(img_path)
                    img = img.resize((self.img_w, self.img_h))

                    # 获取预测
                    results = self.yolo_model.predict(
                        img,
                        verbose=False,
                        conf=self.conf,
                        iou=self.iou
                    )

                    # 以YOLO格式保存标签
                    with open(txt_path, 'w') as f:
                        for result in results:
                            for box in result.boxes:
                                # 获取YOLO格式的坐标
                                x_center = float(box.xywhn[0][0])
                                y_center = float(box.xywhn[0][1])
                                width = float(box.xywhn[0][2])
                                height = float(box.xywhn[0][3])
                                class_id = int(box.cls)

                                # 写入文件
                                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            if not self.canceled:
                self.finished.emit()

        except Exception as e:
            self.error_occurred.emit(f"Auto-labeling failed: {str(e)}")


class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self._image_path = ""
        self._click_enabled = True  # 用于防止双击的标志

    def setImagePath(self, path):
        self._image_path = path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._click_enabled:
            self._click_enabled = False
            QTimer.singleShot(300, lambda: setattr(self, '_click_enabled', True))  # 300毫秒后解锁
            self.clicked.emit(self._image_path)

        super().mousePressEvent(event)


class FullScreenImageDialog(QDialog):
    labels_changed = pyqtSignal()  # 添加信号
    def __init__(
            self,
            image_path, yolo_labels, all_colors, colors, parent=None, classes=None,
            yolo_model=None, yolo_img_w=640, yolo_img_h=640, yolo_conf=0.0, yolo_iou=0.0,
    ):
        super().__init__(parent)
        self.all_colors = all_colors if all_colors else [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红、绿、蓝
        Image.MAX_IMAGE_PIXELS = 2000000000

        # 设置对话框大小为父窗口的2/3
        if parent:
            parent_size = parent.size()
            self.resize(parent_size.width() * 2 // 3, parent_size.height() * 2 // 3)

        # 打标签需要用到的所有颜色, 与底色尽量相反
        self.all_colors = all_colors
        # 分类数组, 下标与打标软件下标对应
        self.classes = classes

        self.utils = PlatformUtils()
        normalized_path = self.utils.get_normalized_path(image_path)
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)
        self._is_closing = False  # 用于跟踪关闭状态的标志
        self.label_visibility = []
        self.setWindowTitle(f"Image Viewer - {os.path.basename(image_path)}")

        self.yolo_model: YOLO | None = yolo_model
        self.yolo_img_w = yolo_img_w
        self.yolo_img_h = yolo_img_h
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou

        self.image_path = normalized_path
        self.yolo_labels = yolo_labels
        self.colors = colors
        self.show_labels = True
        self.expand_image = False
        self._pixmap = None
        self._original_size = None

        self.drawing = False
        self.current_label = None
        self.start_point = None
        self.end_point = None

        # Labeling 2
        self.drawing_mode = ""
        self.first_click = None
        self.current_label = None
        self.temp_rect = None

        self.brightness_value = 100  # 100% - 原始亮度
        self.original_img = None  # 存储原始图像
        self.current_img = None    # 当前带调整的图像

        # 添加裁剪缩放等功能
        self.scale_factor = 1.0  # 当前缩放比例
        self.crop_size = (640, 640)  # 默认裁剪尺寸
        self.crop_rect = None  # 当前裁剪框位置
        self.dragging_crop = False  # 是否正在拖动裁剪框
        self.drag_offset = None  # 拖动偏移量

        self.init_ui()
        if not self.load_image():  # 如果加载失败
            self.close()  # 关闭对话框
        
        QTimer.singleShot(100, self.update_image)
        self.update_labels_list()


    def init_ui(self):
        # 主容器与分隔器
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(10)

        # 上半部分 - 可调整大小的图像
        self.image_container = QWidget()
        self.image_container.setMinimumHeight(300)
        image_layout = QVBoxLayout(self.image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.image_label)

        # 下半部分 - 控制和标签列表
        self.controls_container = QWidget()
        self.controls_container.setMinimumHeight(150)
        controls_layout = QVBoxLayout(self.controls_container)
        controls_layout.setContentsMargins(5, 5, 5, 5)

        # 控制面板与按钮
        controls_panel = QWidget()
        panel_layout = QHBoxLayout(controls_panel)
        panel_layout.setContentsMargins(10, 5, 10, 5)

        # 原有控件
        self.show_labels_check = QCheckBox("显示标记")
        self.show_labels_check.setChecked(True)
        self.show_labels_check.stateChanged.connect(self.toggle_labels_visibility)
        panel_layout.addWidget(self.show_labels_check, 10)

        self.add_label_btn = QPushButton("新增标记")
        self.add_label_btn.clicked.connect(self.start_labeling_mode)
        panel_layout.addWidget(self.add_label_btn, 10)

        # 缩放控制
        zoom_panel = QHBoxLayout()
        zoom_panel.setContentsMargins(50, 5, 50, 5)
        
        # 缩小按钮
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_panel.addWidget(self.zoom_out_btn)
        
        # 缩放滑块
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.zoom_changed)
        zoom_panel.addWidget(self.zoom_slider)
        
        # 放大按钮
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_panel.addWidget(self.zoom_in_btn)
        
        # 将紧凑的缩放面板添加到主控制面板
        # 先隐藏, 实现有问题...
        # panel_layout.addLayout(zoom_panel, 40)
        
        # 裁剪控制 - 创建独立面板
        crop_panel = QWidget()
        crop_layout = QHBoxLayout(crop_panel)
        crop_layout.setContentsMargins(30, 0, 30, 0)

        # 裁剪下拉框
        self.crop_combo = QComboBox()
        self.crop_combo.addItem("640x640", (640, 640))
        self.crop_combo.addItem("480x480", (480, 480))
        self.crop_combo.addItem("320x320", (320, 320))
        self.crop_combo.currentIndexChanged.connect(self.crop_size_changed)
        crop_layout.addWidget(self.crop_combo, 40)

        # 裁剪标签
        self.start_crop_btn = QPushButton("开始裁剪")
        self.start_crop_btn.clicked.connect(self.apply_start_crop)
        crop_layout.addWidget(self.start_crop_btn, 30)

        # 应用裁剪按钮
        self.apply_crop_btn = QPushButton("保存裁剪")
        self.apply_crop_btn.clicked.connect(self.apply_crop)
        crop_layout.addWidget(self.apply_crop_btn, 30)

        # 将裁剪面板添加到主控制面板
        panel_layout.addWidget(crop_panel, 30)

        # 保存按钮
        close_btn = QPushButton("保存并退出")
        close_btn.clicked.connect(self.close_and_save)
        panel_layout.addWidget(close_btn, 10)

        controls_layout.addWidget(controls_panel)

        # 可滚动标签列表
        self.labels_list = QListWidget()
        self.labels_list.setSelectionMode(QListWidget.MultiSelection)
        controls_layout.addWidget(self.labels_list)

        delete_layout = QHBoxLayout()
        delete_layout.addStretch()

        self.delete_selected_btn = QPushButton("删除选中")
        self.delete_selected_btn.clicked.connect(self.delete_selected_labels)
        delete_layout.addWidget(self.delete_selected_btn)
        
        delete_layout.addStretch()
        controls_layout.addLayout(delete_layout)

        # 将部分添加到分隔器
        self.splitter.addWidget(self.image_container)
        self.splitter.addWidget(self.controls_container)

        self.splitter.setStretchFactor(0, 9)
        self.splitter.setStretchFactor(1, 1)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self.splitter)

        # 添加用于调整窗口大小的SizeGrip
        self.size_grip = QSizeGrip(self)
        main_layout.addWidget(self.size_grip, 0, Qt.AlignBottom | Qt.AlignRight)

        self.setLayout(main_layout)

        self.setAttribute(Qt.WA_TranslucentBackground)  # 解决某些系统的绘制问题
        self.setMouseTracking(True)  # 启用鼠标跟踪


    def close_and_save(self):
        if self._is_closing:
            return
        self._is_closing = True

        # 仅在对话框尚未关闭时保存标签
        if hasattr(self, 'image_path') and hasattr(self, 'yolo_labels'):
            label_path = get_label_txt(self.image_path)
            # print(f"[ close_and_save ] image_path: {self.image_path}, label_path: {label_path}")
            try:
                with open(label_path, 'w') as file:
                    for tuple_item in self.yolo_labels:
                        line = ' '.join(map(str, tuple_item))
                        file.write(line + '\n')
            except Exception as e:
                print(f"Error saving labels: {e}")

        self.labels_changed.emit()
        self.close()

    def resizeEvent(self, event):
        """窗口大小改变事件处理程序"""
        super().resizeEvent(event)
        self.update_image()

    def toggle_expand_image(self, state):
        """切换图像拉伸模式"""
        self.expand_image = state == Qt.Checked
        self.update_image()

    def toggle_labels_visibility(self, state):
        """切换标签可见性"""
        self.show_labels = state == Qt.Checked
        self.update_image()

    def yolo_predict_old(self):
        if not os.path.exists(self.image_path):
            QMessageBox.warning(self, "错误", "文件图像未找到！")
            return
        if self.yolo_model is None:
            QMessageBox.warning(self, "错误", "未选择YOLO模型文件！")
            return
        img = Image.open(self.image_path)
        w, h = img.size
        pred_boxes = []
        pred_results = self.yolo_model.predict(img, verbose=False)
        for result in pred_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(x1 * w / 704)
                y1 = int(y1 * h / 704)
                x2 = int(x2 * w / 704)
                y2 = int(y2 * h / 704)
                pred_class = int(box.cls)
                pred_boxes.append((x1, y1, x2, y2, pred_class))
        draw = ImageDraw.Draw(img)
        for box in pred_boxes:
            x_min, y_min, x_max, y_max, class_id = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            draw.text((x_min, y_min), f"{class_id}", fill="red")

    def yolo_predict(self):
        if not os.path.exists(self.image_path):
            QMessageBox.warning(self, "错误", "文件图像未找到！")
            return
        if self.yolo_model is None:
            QMessageBox.warning(self, "错误", "未选择YOLO模型文件！")
            return

        try:
            # 在预测前清理内存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 打开图像
            img = Image.open(self.image_path)
            if max(img.size) > 2048:  # 限制最大尺寸
                img.thumbnail((2048, 2048))
            img = img.resize((self.yolo_img_w, self.yolo_img_h))

            # 获取预测并处理内存
            with torch.no_grad():
                pred_results = self.yolo_model.predict(img, verbose=False, conf=self.yolo_conf, iou=self.yolo_iou)

            # self.yolo_labels.clear()
            # self.colors.clear()

            # 处理结果
            for result in pred_results:
                for box in result.boxes:
                    # 获取YOLO格式的坐标（归一化）
                    x_center = float(box.xywhn[0][0])
                    y_center = float(box.xywhn[0][1])
                    width = float(box.xywhn[0][2])
                    height = float(box.xywhn[0][3])
                    class_id = int(box.cls)
                    conf = str(float(box.conf))[:4]

                    # 添加标签
                    self.yolo_labels.append((f"{class_id}::{conf}", x_center, y_center, width, height))
                    self.colors.append((255, 0, 0))  # 预测标签使用红色

            # 更新标签列表和图像
            self.update_labels_list()
            self.update_image()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测时出错：{str(e)}")
        finally:
            # 使用后清理内存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def delete_selected_labels(self):
        selected_items = self.labels_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No labels selected for deletion!")
            return

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete {len(selected_items)} selected labels?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            indices = sorted([self.labels_list.row(item) for item in selected_items], reverse=True)

            for i in indices:
                if i < len(self.yolo_labels):
                    del self.yolo_labels[i]
                    if i < len(self.colors):
                        del self.colors[i]

            self.update_labels_list()
            self.update_image()


    def adjust_brightness(self, value):
        """调整图像亮度"""
        if not hasattr(self, 'original_img') or self.original_img is None:
            return

        self.brightness_value = value
        self.brightness_value_label.setText(f"{value}%")

        if self.original_img is None:
            return

        # 应用亮度调整
        enhancer = ImageEnhance.Brightness(self.original_img)
        self.current_img = enhancer.enhance(value / 100.0)
        self.update_image()

    def reset_brightness(self):
        """将亮度重置为原始值"""
        self.brightness_slider.setValue(100)
        self.brightness_value_label.setText("100%")
        self.brightness_value = 100
        self.current_img = self.original_img.copy()
        self.update_image()

    def load_image(self):
        if not os.path.exists(self.image_path):
            QMessageBox.warning(self, "错误", "文件图像未找到！")
            return False

        try:
            with Image.open(self.utils.get_normalized_path(self.image_path)) as img:
                self.original_img = img.convert("RGB")
                self.current_img = self.original_img.copy()  # 用于调整的副本
                self._original_size = img.size

                self.label_visibility = [True] * len(self.yolo_labels) if self.yolo_labels else []

                # 设置分隔器的初始比例（70%图像，30%控制）
                total_height = self.height()
                self.splitter.setSizes([int(total_height * 0.7), int(total_height * 0.3)])

                self.update_image()
                return True

        except Exception as e:
            print(f"无法加载图像: {self.image_path}: {e}")
            QMessageBox.warning(self, "错误", f"无法加载图像：{str(e)}")
            return False

    def widget_to_image_coords(self, wx, wy):
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        img_width, img_height = self.original_img.size

        if self.expand_image:
            # 拉伸模式：无边距，直接按比例缩放
            displayed_width = label_width
            displayed_height = label_height
            left_margin = 0
            top_margin = 0
        else:
            # 未拉伸模式：保持纵横比，计算边距
            scale = min(label_width / float(img_width), label_height / float(img_height))
            displayed_width = img_width * scale
            displayed_height = img_height * scale
            left_margin = (label_width - displayed_width) / 2
            top_margin = (label_height - displayed_height) / 2

        # 映射窗口坐标到图像坐标
        px = wx - left_margin
        py = wy - top_margin

        if displayed_width > 0:
            ix = px * (img_width / displayed_width)
        else:
            ix = 0  # 避免除零（理论上不会发生）

        if displayed_height > 0:
            iy = py * (img_height / displayed_height)
        else:
            iy = 0

        return ix, iy

    def update_image(self):
        if not hasattr(self, 'original_img'):
            return

        try:
            img = self.current_img.copy() if self.current_img else self.original_img.copy()
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size

            # 1. 绘制临时矩形（从第一次点击到当前鼠标位置）
            if self.first_click and hasattr(self, 'temp_rect'):
                p1, p2 = self.current_rect
                ix1, iy1 = self.widget_to_image_coords(p1.x(), p1.y())
                ix2, iy2 = self.widget_to_image_coords(p2.x(), p2.y())

                x_min = min(ix1, ix2)
                y_min = min(iy1, iy2)
                x_max = max(ix1, ix2)
                y_max = max(iy1, iy2)

                # 此时新增的label还未保存, 因此len+1
                draw_color = (
                    self.all_colors[ (len(self.yolo_labels)+1) % len(self.all_colors) ]
                )
                draw.rectangle([x_min, y_min, x_max, y_max], outline=draw_color, width=1)

                # 添加临时标签
                class_name = (self.classes[self.current_label] 
                                if 0 <= self.current_label < len(self.classes) 
                                else str(self.current_label))
                
                if self.current_label is not None:
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                        draw.text((x_min, y_min), f"{class_name}", fill=draw_color, font=font)
                    except:
                        draw.text((x_min, y_min), f"{class_name}", fill=draw_color)

            # 2. 绘制所有保存的标签
            if self.show_labels and self.yolo_labels:
                for i, (label, color) in enumerate(zip(self.yolo_labels, self.colors)):
                    # 检查标签可见性
                    if not self.label_visibility[i]:
                        continue

                    class_id, x_center, y_center, box_width, box_height = label

                    # 将YOLO坐标转换为像素
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    box_width_px = box_width * img_width
                    box_height_px = box_height * img_height

                    # 计算矩形角点坐标
                    x1 = x_center_px - box_width_px / 2
                    y1 = y_center_px - box_height_px / 2
                    x2 = x_center_px + box_width_px / 2
                    y2 = y_center_px + box_height_px / 2

                    # Draw rectangle with class-specific color
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
                    # print(f"[ yolo_labels ] draw.rectangle: {[x1, y1, x2, y2]}")

                    # 获取class_name
                    class_name = (self.classes[class_id] 
                                if 0 <= class_id < len(self.classes) 
                                else str(class_id))

                    # 类别标签
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                        draw.text((x1, y1), class_name, fill=color, font=font)
                    except:
                        font = ImageFont.load_default()
                        draw.text((x1, y1), class_name, fill=color, font=font)



            # 应用zoom缩放按钮
            if self.scale_factor != 1.0:
                new_width = int(img.width * self.scale_factor)
                new_height = int(img.height * self.scale_factor)
                print(f"[scale_factor] new_width: {new_width}, new_height: {new_height}, self.scale_factor: {self.scale_factor}")
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 绘制裁剪框（如果有）
            if self.crop_rect:
                try:
                    # 计算矩形角点坐标
                    x1 = self.crop_rect['x'] - self.crop_rect['width'] / 2
                    y1 = self.crop_rect['y'] - self.crop_rect['height'] / 2
                    x2 = self.crop_rect['x'] + self.crop_rect['width'] / 2
                    y2 = self.crop_rect['y'] + self.crop_rect['height'] / 2
    
                    
                    # 确保坐标值都是数字类型
                    if all(isinstance(val, (int, float)) for val in [x1, y1, x2, y2]):
                        # 绘制裁剪框
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                        # print(f"[ crop_rect ] draw.rectangle: {[x1, y1, x2, y2]}")
                        
                        # 添加尺寸标签（黄色）
                        text = f"{self.crop_rect['width']}x{self.crop_rect['height']}"
                        font = ImageFont.load_default()
                        try:
                            # 尝试使用更大字体
                            font = ImageFont.truetype("arial.ttf", 20)
                        except:
                            pass
                            
                        # 确保文本位置在图像范围内
                        text_x = max(x1, 10)
                        text_y = max(y1 - 25, 10)
                        
                        draw.text((text_x, text_y), text, fill="red", font=font)
                    else:
                        raise ValueError(f"无效坐标类型: {[x1, y1, x2, y2]}")
                
                except Exception as e:
                    print(f"[绘制裁剪框出错] {e}")
                    # 删除无效的裁剪框
                    self.crop_rect = None


            # 画图要放到最后
            img = img.convert("RGBA")
            data = img.tobytes("raw", "RGBA")
            qim = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)

            self._pixmap = QPixmap.fromImage(qim)

            
            # 根据设置进行拉伸
            if self.expand_image:
                self.image_label.setPixmap(self._pixmap.scaled(
                    self.image_label.size(),
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation
                ))
            else:
                self.image_label.setPixmap(self._pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))

        except Exception as e:
            print(f"[update_image] exception: {e}")

    def toggle_single_label_visibility(self, label_idx, state):
        """切换单个标签的可见性"""
        self.label_visibility[label_idx] = state == Qt.Checked
        self.update_image()

    # Labeling
    def update_labels_list(self):
        self.labels_list.clear()
        self.label_visibility = []
        # print("[ update_labels_list ]test, labels: ", self.yolo_labels)

        for i, (label, color) in enumerate(zip(self.yolo_labels, self.colors)):
            # 为列表项创建控件
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(5, 2, 5, 2)

            # 可见性复选框
            visibility_check = QCheckBox()
            visibility_check.setChecked(True)
            visibility_check.stateChanged.connect(lambda state, idx=i: self.toggle_single_label_visibility(idx, state))
            item_layout.addWidget(visibility_check)

            # 标签文本
            class_id = label[0]
            class_name = (self.classes[class_id] 
              if 0 <= class_id < len(self.classes) 
              else "None")

            label_text = QLabel(
                f"■ 标签 {i + 1}: "
                f"类别<b> {class_name}</b> | "
                f"<span style='color:rgb({color[0]},{color[1]},{color[2]})'>■</span> "
                f"RGB: {color[0]}, {color[1]}, {color[2]} | "
                f"坐标: [{label[1]:.3f}, {label[2]:.3f}, {label[3]:.3f}, {label[4]:.3f}]"
            )
            
            label_text.setStyleSheet(f"color: rgb({color[0]}, {color[1]}, {color[2]});")
            item_layout.addWidget(label_text)
            item_layout.addStretch()

            # 删除按钮
            delete_btn = QPushButton("×")
            delete_btn.setFixedSize(20, 20)
            delete_btn.clicked.connect(lambda _, idx=i: self.delete_single_label(idx))
            item_layout.addWidget(delete_btn)

            # 创建QListWidgetItem
            item = QListWidgetItem()
            item.setSizeHint(item_widget.sizeHint())

            # 添加到列表
            self.labels_list.addItem(item)
            self.labels_list.setItemWidget(item, item_widget)

            # 保存可见性状态
            self.label_visibility.append(True)

    def delete_single_label(self, label_idx):
        if 0 <= label_idx < len(self.yolo_labels):
            del self.yolo_labels[label_idx]
            del self.colors[label_idx]
            self.update_labels_list()
            self.update_image()

    def start_labeling_mode(self):
        """激活添加标注模式(使用下拉框选择类别)"""
        if not hasattr(self, 'classes') or not self.classes:
            QMessageBox.warning(self, "错误", "未加载类别列表")
            return

        # 创建下拉框对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择类别")

        layout = QVBoxLayout(dialog)
        
        # 创建下拉框
        combo = QComboBox()
        combo.addItems(self.classes)  # 添加所有类别名称
        layout.addWidget(combo)
        
        # 创建按钮框
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.resize(300, 200)  # 直接调整对话框大小

        # 显示对话框并等待用户选择
        if dialog.exec_() == QDialog.Accepted:
            self.drawing_mode = "label"
            self.current_label = combo.currentIndex()  # 获取选中项的下标
            self.setCursor(Qt.CrossCursor)


    def mousePressEvent(self, event):
        if self.drawing_mode == "label" and event.button() == Qt.LeftButton:
            # 在点击时固定第一点
            self.first_click = event.pos()
            # print("[ mousePressEvent ] event.pos(): ", event.pos())

            self.current_rect = [self.first_click, self.first_click]  # 初始化矩形
            self.update_image()

        elif self.drawing_mode == "crop" and event.button() == Qt.LeftButton and self.crop_rect:
            self.dragging_crop = True  # 标记拖拽开始
            self.drag_offset = self.widget_to_image_coords(event.x(), event.y())

        else:
            super().mousePressEvent(event)


    def mouseMoveEvent(self, event):
        if self.drawing_mode == "label" and self.first_click:
            # 在鼠标移动时更新第二点
            self.current_rect[1] = event.pos()
            self.update_image()  # 使用更新的矩形重绘

        elif self.drawing_mode == "crop" and self.dragging_crop:
            # 更新坐标
            ix, iy = self.widget_to_image_coords(event.x(), event.y())

            # 限制坐标在图像范围内
            img_width, img_height = self.original_img.size
            ix = max(0, min(ix, img_width))
            iy = max(0, min(iy, img_height))
            
            # 计算偏移量
            dx = ix - self.drag_offset[0]
            dy = iy - self.drag_offset[1]


            # 重置塔伤
            self.drag_offset = ix, iy

            # 更新裁剪框位置（平移整个矩形）
            self.crop_rect = {
                "x": self.crop_rect["x"] + dx,
                "y": self.crop_rect["y"] + dy,
                "width": self.crop_rect["width"],
                "height": self.crop_rect["height"]
            }
            # print(f"self.crop_rect: {self.crop_rect}, dx: {dx}, dy: {dy}")

            self.update_image()

        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing_mode == "label" and event.button() == Qt.LeftButton and self.first_click:
            # 在释放时固定第二点
            second_point = event.pos()
            self.finish_labeling(second_point)
            self.first_click = None
            self.current_rect = None
            self.update_image()
        else:
            super().mouseReleaseEvent(event)

    def finish_labeling(self, second_point):
        if not self.first_click:
            return

        p1 = self.first_click
        p2 = second_point

        ix1, iy1 = self.widget_to_image_coords(p1.x(), p1.y())
        ix2, iy2 = self.widget_to_image_coords(p2.x(), p2.y())

        x_min = min(ix1, ix2)
        y_min = min(iy1, iy2)
        x_max = max(ix1, ix2)
        y_max = max(iy1, iy2)

        img_width, img_height = self.original_img.size
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        box_width = (x_max - x_min) / img_width
        box_height = (y_max - y_min) / img_height

        # 确保颜色列表非空
        if not hasattr(self, 'all_colors') or len(self.all_colors) == 0:
            self.all_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 默认颜色

        # 计算新标签的颜色索引
        color_index = len(self.yolo_labels) % len(self.all_colors)
        new_color = self.all_colors[color_index]

        new_label = (self.current_label, x_center, y_center, box_width, box_height)
        self.yolo_labels.append(new_label)

        new_color = (
            self.all_colors[ len(self.yolo_labels) % len(self.all_colors) ]
        )
        self.colors.append(new_color)

        self.update_labels_list()
        self.drawing_mode = ""
        self.setCursor(Qt.ArrowCursor)
        self.update_image()

    def keyPressEvent(self, event):
        """通过Esc取消绘制模式"""
        if event.key() == Qt.Key_Escape and self.drawing_mode != "":
            self.drawing_mode = ""
            self.first_click = None
            self.temp_rect = None
            self.setCursor(Qt.ArrowCursor)
            self.crop_rect = None
            
            self.dragging_crop = False  # 是否正在拖动裁剪框
            self.drag_offset = None  # 拖动偏移量

            self.update_image()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # 将亮度增加5%
            self.brightness_slider.setValue(min(self.brightness_value + 5, 200))
        elif event.key() == Qt.Key_Minus:
            # 将亮度减少5%
            self.brightness_slider.setValue(max(self.brightness_value - 5, 50))
        elif event.key() == Qt.Key_0:
            # 重置亮度
            self.reset_brightness()
        elif event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """关闭时重置状态"""
        if self._pixmap:
            self._pixmap = None
        if self._is_closing:
            event.accept()
            return
        self._is_closing = True
        self.drawing_mode = ""
        self.setCursor(Qt.ArrowCursor)
        super().closeEvent(event)



    # 缩放功能实现
    def zoom_in(self):
        """放大图片"""
        self.zoom_slider.setValue(min(self.zoom_slider.value() + 10, 200))

    
    def zoom_out(self):
        """缩小图片"""
        self.zoom_slider.setValue(max(self.zoom_slider.value() - 10, 50))

    def zoom_changed(self, value):
        """缩放滑块值变化处理"""
        self.scale_factor = value / 100.0
        self.update_image()


    # 裁剪尺寸选择
    def crop_size_changed(self, index):
        """裁剪尺寸变化处理"""
        self.crop_size = self.crop_combo.currentData()
        self.crop_rect = None  # 重置裁剪框
        self.drawing_mode = ""
        self.update_image()

    # 开始裁剪
    def apply_start_crop(self):
        """处理开始裁剪按钮点击事件"""
        # 1. 获取当前选中的裁剪尺寸
        self.current_crop_size = self.crop_combo.currentData()
        
        # 2. 计算图像中心位置
        center_x = self.image_label.width() // 2
        center_y = self.image_label.height() // 2

        x, y = self.widget_to_image_coords(center_x, center_y)

        # 3. 创建裁剪矩形框（居中显示）
        self.crop_rect = {
            "x": x,
            "y": y,
            "width": self.current_crop_size[0],
            "height": self.current_crop_size[1]
        }
        
        # 4. 切换到裁剪模式
        self.drawing_mode = "crop"
        self.setCursor(Qt.SizeAllCursor)  # 使用移动光标
        
        # 5. 更新界面显示
        self.update_image()


    # 应用裁剪
    def apply_crop(self):
        """应用裁剪操作"""
        if not self.crop_rect:
            QMessageBox.warning(self, "Warning", "没有可裁剪的区域")
            return
        
        # 暂不考虑缩放
        img_width, img_height = self.original_img.size


        # 计算实际裁剪区域
        x = int(self.crop_rect["x"])
        y = int(self.crop_rect["y"])
        width = int(self.crop_rect["width"])
        height = int(self.crop_rect["height"])
        

        x1 = x - width/2
        y1 = y - height/2

        x2 = x + width/2
        y2 = y + height/2
        
        # print(f"self.crop_rect: {self.crop_rect}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

        if x1 < 0 or y1 < 0 or x1 > img_width or y1 > img_height or x2 < 0 or y2 < 0 or x2 > img_width or y2 > img_height or x1 >= x2 or y1 >= y2:
            QMessageBox.warning(self, "Error", f"裁剪区域错误: x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            return

        # 执行裁剪, 但不修改原图, 只是保存到新文件即可
        # 如果需要对新文件进行标记, 自己重新load进来
        cropped_img = self.original_img.crop([x1, y1, x2, y2])
        
        # 重置状态
        self.crop_rect = None
        self.drawing_mode = ""
        # self.scale_factor = 1.0
        # self.zoom_slider.setValue(100)
        
        # 更新图像, 实际上更新是无效的, 不修改实际图片
        self.update_image()

        # 保存图片
        self.save_image_with_confirmation(cropped_img)  # 调用保存方法
    

    def save_image_with_confirmation(self, cropped_img):
        """带确认机制的图片保存方法"""
        base, ext = os.path.splitext(self.image_path)
        cropped_image_path = f"{base}_cropped{ext}"

        # 1. 创建保存对话框（默认使用原路径）
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存图片", 
            cropped_image_path,
            "图片文件 (*.png *.jpg *.jpeg);;所有文件 (*)",
            options=options
        )
        
        # 用户取消操作
        if not file_path:
            return
        
        # 2. 检查文件是否已存在 
        # 不需要, 发现windows会自动检查..
        # if os.path.exists(file_path):
        #     # 创建覆盖确认对话框
        #     reply = QMessageBox.question(
        #         self,
        #         "确认覆盖",
        #         f"文件leafan {os.path.basename(file_path)} 已存在，是否覆盖?",
        #         QMessageBox.Yes | QMessageBox.No,
        #         QMessageBox.No
        #     )
            
        #     if reply != QMessageBox.Yes:
        #         return  # 用户取消覆盖
        
        # 3. 执行保存操作
        try:
            # 获取图片格式, 与原图一致
            # 'PNG', 'BMP'格式无损
            file_format = os.path.splitext(file_path)[1][1:].upper()
            
            # 特殊处理JPEG的质量参数, jpeg格式是有损的, 尽量不要保存为这类格式
            save_kwargs = {'quality': 95} if file_format in ['JPG', 'JPEG'] else {}
            
            # 保存图片
            cropped_img.save(file_path, format=file_format, **save_kwargs)
            
            # 提示保存成功
            QMessageBox.information(self, "保存成功", f"图片已保存至: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存图片时出错: {str(e)}")


class ImageProcessingThread(QThread):
    progress_updated = pyqtSignal(int, str)  # 添加文本消息
    cluster_found = pyqtSignal(list)  # 新信号，用于找到的聚类
    finished_clustering = pyqtSignal()  # 完成信号

    def __init__(self, image_folder, threshold, skip_single, hash_method):
        super().__init__()
        self.image_folder = image_folder
        self.threshold = threshold
        self.skip_single = skip_single
        self.hash_method = hash_method
        self.canceled = False
        self.processed_images = 0
        self.total_images = 0

    def run(self):
        # 计算总图像数量
        self.total_images = sum(1 for _ in self._get_image_files())
        if self.total_images == 0:
            self.progress_updated.emit(0, "No images found")
            self.finished_clustering.emit()
            return

        # 阶段1：计算哈希值
        hashes = []
        for i, (img_path, hash_val) in enumerate(self._compute_hashes()):
            if self.canceled:
                return
            hashes.append((img_path, hash_val))
            progress = int((i + 1) / self.total_images * 50)
            self.progress_updated.emit(progress, f"Processing: {os.path.basename(img_path)}")

        # 阶段2：聚类
        used_indices = set()
        total_processed = 0

        for i, (path1, hash1) in enumerate(hashes):
            if i in used_indices:
                continue

            cluster = [path1]
            used_indices.add(i)

            for j, (path2, hash2) in enumerate(hashes[i + 1:], start=i + 1):
                if j in used_indices:
                    continue

                if hash1 - hash2 <= self.threshold:
                    cluster.append(path2)
                    used_indices.add(j)

            if not (self.skip_single and len(cluster) == 1):
                self.cluster_found.emit(cluster)  # 发送找到的聚类

            total_processed += len(cluster)
            progress = 50 + int(total_processed / self.total_images * 50)
            self.progress_updated.emit(progress, f"Clustering: {len(cluster)} images found")

        self.finished_clustering.emit()

    def _get_image_files(self):
        """图像路径生成器"""
        for root, _, files in os.walk(self.image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    yield PlatformUtils.get_normalized_path(
                        os.path.join(root, file)
                    )

    def _compute_hashes(self):
        """图像哈希值生成器"""
        for img_path in self._get_image_files():
            try:
                with Image.open(img_path) as img:
                    if self.hash_method == "average_hash":
                        hash_val = imagehash.average_hash(img)
                    elif self.hash_method == "phash":
                        hash_val = imagehash.phash(img)
                    elif self.hash_method == "dhash":
                        hash_val = imagehash.dhash(img)
                    else:
                        hash_val = imagehash.average_hash(img)
                    yield (img_path, hash_val)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


class DarkTheme:
    @staticmethod
    def apply(app):
        app.setStyle("Fusion")

        dark_palette = app.palette()
        dark_palette.setColor(dark_palette.Window, QColor(53, 53, 53))
        dark_palette.setColor(dark_palette.WindowText, Qt.white)
        dark_palette.setColor(dark_palette.Base, QColor(35, 35, 35))
        dark_palette.setColor(dark_palette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(dark_palette.ToolTipBase, Qt.white)
        dark_palette.setColor(dark_palette.ToolTipText, Qt.white)
        dark_palette.setColor(dark_palette.Text, Qt.white)
        dark_palette.setColor(dark_palette.Button, QColor(53, 53, 53))
        dark_palette.setColor(dark_palette.ButtonText, Qt.white)
        dark_palette.setColor(dark_palette.BrightText, Qt.red)
        dark_palette.setColor(dark_palette.Link, QColor(42, 130, 218))
        dark_palette.setColor(dark_palette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(dark_palette.HighlightedText, Qt.black)

        app.setPalette(dark_palette)
        app.setStyleSheet("""
            QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }
            QMenuBar::item:selected { background: #2a82da; }
            QTabBar::tab:selected { background: #2a82da; color: white; }
        """)