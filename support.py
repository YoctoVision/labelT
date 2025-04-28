import os
import sys
import random
import tempfile
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import imagehash
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QMessageBox, QCheckBox,
    QDialog, QSizePolicy, QSplitter, QSizeGrip,
    QInputDialog, QSlider,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QColor


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
                    # Пропускаем если уже есть разметка
                    txt_path = PlatformUtils.get_normalized_path(
                        os.path.splitext(img_path)[0] + '.txt'
                    )
                    if os.path.exists(txt_path):
                        continue

                    # Загружаем изображение
                    img = Image.open(img_path)
                    img = img.resize((self.img_w, self.img_h))

                    # Получаем предсказания
                    results = self.yolo_model.predict(
                        img,
                        verbose=False,
                        conf=self.conf,
                        iou=self.iou
                    )

                    # Сохраняем метки в YOLO формате
                    with open(txt_path, 'w') as f:
                        for result in results:
                            for box in result.boxes:
                                # Получаем координаты в YOLO формате
                                x_center = float(box.xywhn[0][0])
                                y_center = float(box.xywhn[0][1])
                                width = float(box.xywhn[0][2])
                                height = float(box.xywhn[0][3])
                                class_id = int(box.cls)

                                # Записываем в файл
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
        self._click_enabled = True  # Флаг для предотвращения двойных кликов

    def setImagePath(self, path):
        self._image_path = path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._click_enabled:
            self._click_enabled = False
            QTimer.singleShot(300, lambda: setattr(self, '_click_enabled', True))  # Разблокировка через 300 мс
            self.clicked.emit(self._image_path)
        super().mousePressEvent(event)


class FullScreenImageDialog(QDialog):
    labels_changed = pyqtSignal()  # Добавляем сигнал
    def __init__(
            self,
            image_path, yolo_labels, colors, parent=None,
            yolo_model=None, yolo_img_w=640, yolo_img_h=640, yolo_conf=0.0, yolo_iou=0.0,
    ):
        super().__init__(parent)
        self.utils = PlatformUtils()
        normalized_path = self.utils.get_normalized_path(image_path)
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)
        self._is_closing = False  # Флаг для отслеживания закрытия
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
        self.drawing_mode = False
        self.first_click = None
        self.current_label = None
        self.temp_rect = None

        self.brightness_value = 100  # 100% - исходная яркость
        self.original_img = None  # Будем хранить оригинальное изображение
        self.current_img = None    # Текущее изображение с настройками

        self.init_ui()
        if not self.load_image():  # Если загрузка не удалась
            self.close()  # Закрываем диалог
        self.update_labels_list()

    def init_ui(self):
        # Главный контейнер с разделителем
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(10)

        # Верхняя часть - изображение с возможностью изменения размера
        # The top part is a resizable image
        self.image_container = QWidget()
        self.image_container.setMinimumHeight(200)
        image_layout = QVBoxLayout(self.image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.image_label)

        # Нижняя часть - управление и список меток
        # Bottom part - control and tag list
        self.controls_container = QWidget()
        self.controls_container.setMinimumHeight(150)
        controls_layout = QVBoxLayout(self.controls_container)
        controls_layout.setContentsMargins(5, 5, 5, 5)

        # Панель управления с кнопками
        # Control panel with buttons
        controls_panel = QWidget()
        panel_layout = QHBoxLayout(controls_panel)
        panel_layout.setContentsMargins(0, 0, 0, 5)

        self.expand_check = QCheckBox("Stretch image")
        self.expand_check.stateChanged.connect(self.toggle_expand_image)
        panel_layout.addWidget(self.expand_check)

        self.show_labels_check = QCheckBox("Show markup")
        self.show_labels_check.setChecked(True)
        self.show_labels_check.stateChanged.connect(self.toggle_labels_visibility)
        panel_layout.addWidget(self.show_labels_check)

        self.delete_selected_btn = QPushButton("Delete selected")
        self.delete_selected_btn.clicked.connect(self.delete_selected_labels)
        panel_layout.addWidget(self.delete_selected_btn)

        self.add_label_btn = QPushButton("Add markup")
        self.add_label_btn.clicked.connect(self.start_labeling_mode)
        panel_layout.addWidget(self.add_label_btn)

        self.make_yolo_btn = QPushButton("Predict yolo model")
        self.make_yolo_btn.clicked.connect(self.yolo_predict)
        panel_layout.addWidget(self.make_yolo_btn)

        close_btn = QPushButton("Save and close")
        close_btn.clicked.connect(self.close_and_save)
        panel_layout.addWidget(close_btn)

        controls_layout.addWidget(controls_panel)

        # Добавим панель для регулировки яркости
        brightness_panel = QHBoxLayout()

        self.brightness_label = QLabel("Brightness:")
        brightness_panel.addWidget(self.brightness_label)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(50, 200)  # От 50% до 200%
        self.brightness_slider.setValue(100)  # 100% - исходная яркость
        self.brightness_slider.setTickInterval(10)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        brightness_panel.addWidget(self.brightness_slider)

        self.brightness_value_label = QLabel("100%")
        brightness_panel.addWidget(self.brightness_value_label)

        # Добавим кнопку сброса
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_brightness)
        brightness_panel.addWidget(reset_btn)

        # Вставим панель яркости перед списком меток
        controls_layout.insertLayout(1, brightness_panel)

        # Прокручиваемый список меток
        # Scrollable list of labels
        self.labels_list = QListWidget()
        self.labels_list.setSelectionMode(QListWidget.MultiSelection)
        controls_layout.addWidget(self.labels_list)

        # Добавьте детали в сепаратор
        # Add the parts to the separator
        self.splitter.addWidget(self.image_container)
        self.splitter.addWidget(self.controls_container)

        # Основной макет
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self.splitter)

        # Добавляем SizeGrip для изменения размера окна
        # Add a SizeGrip to resize the window
        self.size_grip = QSizeGrip(self)
        main_layout.addWidget(self.size_grip, 0, Qt.AlignBottom | Qt.AlignRight)

        self.setLayout(main_layout)

        # Настройки окна
        # Window settings
        self.setMinimumSize(600, 500)  # Минимальный размер окна
        self.resize(800, 600)  # Начальный размер окна

    def close_and_save(self):
        if self._is_closing:
            return
        self._is_closing = True

        # Сохраняем метки только если диалог еще не закрывается
        if hasattr(self, 'image_path') and hasattr(self, 'yolo_labels'):
            label_path = os.path.splitext(self.image_path)[0] + '.txt'
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
        """Обработчик изменения размера окна"""
        super().resizeEvent(event)
        self.update_image()

    def toggle_expand_image(self, state):
        """Переключает режим растягивания изображения"""
        self.expand_image = state == Qt.Checked
        self.update_image()

    def toggle_labels_visibility(self, state):
        """Переключает видимость меток"""
        self.show_labels = state == Qt.Checked
        self.update_image()

    def yolo_predict_old(self):
        if not os.path.exists(self.image_path):
            QMessageBox.warning(self, "Ошибка", "Файл изображения не найден!")
            return
        if self.yolo_model is None:
            QMessageBox.warning(self, "Ошибка", "Файл Yolo модели не выбран!")
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
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=5)
            draw.text((x_min, y_min), f"{class_id}", fill="red")

    def yolo_predict(self):
        if not os.path.exists(self.image_path):
            QMessageBox.warning(self, "Ошибка", "Файл изображения не найден!")
            return
        if self.yolo_model is None:
            QMessageBox.warning(self, "Ошибка", "Файл Yolo модели не выбран!")
            return

        try:
            # Очищаем память перед предсказанием
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Открываем изображение
            img = Image.open(self.image_path)
            if max(img.size) > 2048:  # Ограничиваем максимальный размер
                img.thumbnail((2048, 2048))
            img = img.resize((self.yolo_img_w,self.yolo_img_h))

            # Получаем предсказания с обработкой памяти
            with torch.no_grad():
                pred_results = self.yolo_model.predict(img, verbose=False, conf=self.yolo_conf, iou=self.yolo_iou)

            # Очищаем текущие метки
            # self.yolo_labels.clear()
            # self.colors.clear()

            # Обрабатываем результаты
            for result in pred_results:
                for box in result.boxes:
                    # Получаем координаты в формате YOLO (нормализованные)
                    x_center = float(box.xywhn[0][0])
                    y_center = float(box.xywhn[0][1])
                    width = float(box.xywhn[0][2])
                    height = float(box.xywhn[0][3])
                    class_id = int(box.cls)
                    conf = str(float(box.conf))[:4]

                    # Добавляем метку
                    self.yolo_labels.append((f"{class_id}::{conf}", x_center, y_center, width, height))
                    self.colors.append((255, 0, 0))  # Красный цвет для предсказаний

            # Обновляем список меток и изображение
            self.update_labels_list()
            self.update_image()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при предсказании: {str(e)}")
        finally:
            # Очищаем память после использования
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

            self.update_image()
            self.load_image()

    def adjust_brightness(self, value):
        """Изменяет яркость изображения"""
        if not hasattr(self, 'original_img') or self.original_img is None:
            return

        self.brightness_value = value
        self.brightness_value_label.setText(f"{value}%")

        if self.original_img is None:
            return

        # Применяем изменение яркости
        enhancer = ImageEnhance.Brightness(self.original_img)
        self.current_img = enhancer.enhance(value / 100.0)
        self.update_image()

    def reset_brightness(self):
        """Сбрасывает яркость к исходному значению"""
        self.brightness_slider.setValue(100)
        self.brightness_value_label.setText("100%")
        self.brightness_value = 100
        self.current_img = self.original_img.copy()
        self.update_image()

    def load_image(self):
        if not os.path.exists(self.image_path):
            QMessageBox.warning(self, "Ошибка", "Файл изображения не найден!")
            return False

        try:
            with Image.open(self.utils.get_normalized_path(self.image_path)) as img:
                self.original_img = img.convert("RGB")
                self.current_img = self.original_img.copy()  # Копия для изменений
                self._original_size = img.size

                self.label_visibility = [True] * len(self.yolo_labels) if self.yolo_labels else []

                # Устанавливаем начальное соотношение разделителя (70% изображение, 30% управление)
                total_height = self.height()
                self.splitter.setSizes([int(total_height * 0.7), int(total_height * 0.3)])

                self.update_image()

                # Заполняем список меток
                # self.labels_list.clear()
                # self.label_visibility = []  # Список для хранения состояния видимости меток
                for i, (label, color) in enumerate(zip(self.yolo_labels, self.colors)):
                    # Создаем виджет для элемента списка
                    item_widget = QWidget()
                    item_layout = QHBoxLayout(item_widget)
                    item_layout.setContentsMargins(5, 2, 5, 2)

                    # Чекбокс видимости
                    visibility_check = QCheckBox()
                    visibility_check.setChecked(True)  # По умолчанию метка видима
                    visibility_check.stateChanged.connect(
                        lambda state, idx=i: self.toggle_single_label_visibility(idx, state))
                    item_layout.addWidget(visibility_check)

                    # Текст метки
                    class_id = label[0]
                    label_text = QLabel(f"■ Метка {i + 1}: Класс {class_id} (RGB: {color[0]}, {color[1]}, {color[2]})")
                    label_text.setStyleSheet(f"color: rgb({color[0]}, {color[1]}, {color[2]});")
                    item_layout.addWidget(label_text)
                    item_layout.addStretch()

                    # Создаем QListWidgetItem
                    item = QListWidgetItem()
                    item.setSizeHint(item_widget.sizeHint())

                    # Добавляем в список
                    self.labels_list.addItem(item)
                    self.labels_list.setItemWidget(item, item_widget)

                    # Сохраняем состояние видимости
                    self.label_visibility.append(True)
                return True

        except Exception as e:
            print(f"Ошибка загрузки изображения {self.image_path}: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить изображение: {str(e)}")
            return False

    def update_image(self):
        if not hasattr(self, 'original_img'):
            return

        try:
            img = self.current_img.copy() if self.current_img else self.original_img.copy()
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size
            label_width = self.image_label.width()
            label_height = self.image_label.height()

            # Коэффициенты масштабирования
            scale_x = img_width / label_width
            scale_y = img_height / label_height

            # 1. Рисуем временный прямоугольник (между первым кликом и текущей позицией мыши)
            if self.first_click and hasattr(self, 'temp_rect'):
                p1, p2 = self.current_rect

                # Конвертируем координаты в пространство изображения
                x1 = p1.x() * scale_x
                y1 = p1.y() * scale_y
                x2 = p2.x() * scale_x
                y2 = p2.y() * scale_y

                # Убедимся, что x1 < x2 и y1 < y2
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])

                # Рисуем полупрозрачный прямоугольник
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                # Добавляем временную подпись
                if self.current_label is not None:
                    try:
                        font = ImageFont.load_default()
                        draw.text((x1, y1), f"Class {self.current_label}", fill="red", font=font)
                    except:
                        draw.text((x1, y1), f"Class {self.current_label}", fill="red")

            # 2. Рисуем все сохраненные метки
            if self.show_labels and self.yolo_labels:
                for i, (label, color) in enumerate(zip(self.yolo_labels, self.colors)):
                    # Проверяем видимость метки
                    if not self.label_visibility[i]:
                        continue

                    class_id, x_center, y_center, box_width, box_height = label

                    # Конвертируем координаты YOLO в пиксели
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    box_width_px = box_width * img_width
                    box_height_px = box_height * img_height

                    # Вычисляем координаты углов
                    x1 = x_center_px - box_width_px / 2
                    y1 = y_center_px - box_height_px / 2
                    x2 = x_center_px + box_width_px / 2
                    y2 = y_center_px + box_height_px / 2

                    # Draw rectangle with class-specific color
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    # Подпись класса
                    try:
                        font = ImageFont.load_default()
                        draw.text((x1, y1), str(class_id), fill=color, font=font)
                    except:
                        draw.text((x1, y1), str(class_id), fill=color)

            img = img.convert("RGBA")
            data = img.tobytes("raw", "RGBA")
            qim = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)

            self._pixmap = QPixmap.fromImage(qim)

            # Масштабируем согласно настройкам
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
            print(f"Ошибка обновления изображения: {e}")

    def toggle_single_label_visibility(self, label_idx, state):
        """Переключает видимость отдельной метки"""
        self.label_visibility[label_idx] = state == Qt.Checked
        self.update_image()

    # Labeling
    def update_labels_list(self):
        self.labels_list.clear()
        self.label_visibility = []

        for i, (label, color) in enumerate(zip(self.yolo_labels, self.colors)):
            # Создаем виджет для элемента списка
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(5, 2, 5, 2)

            # Чекбокс видимости
            visibility_check = QCheckBox()
            visibility_check.setChecked(True)
            visibility_check.stateChanged.connect(lambda state, idx=i: self.toggle_single_label_visibility(idx, state))
            item_layout.addWidget(visibility_check)

            # Текст метки
            class_id = label[0]
            label_text = QLabel(f"■ Метка {i + 1}: Класс {class_id} (RGB: {color[0]}, {color[1]}, {color[2]})")
            label_text.setStyleSheet(f"color: rgb({color[0]}, {color[1]}, {color[2]});")
            item_layout.addWidget(label_text)
            item_layout.addStretch()

            # Кнопка удаления
            delete_btn = QPushButton("×")
            delete_btn.setFixedSize(20, 20)
            delete_btn.clicked.connect(lambda _, idx=i: self.delete_single_label(idx))
            item_layout.addWidget(delete_btn)

            # Создаем QListWidgetItem
            item = QListWidgetItem()
            item.setSizeHint(item_widget.sizeHint())

            # Добавляем в список
            self.labels_list.addItem(item)
            self.labels_list.setItemWidget(item, item_widget)

            # Сохраняем состояние видимости
            self.label_visibility.append(True)

    def delete_single_label(self, label_idx):
        if 0 <= label_idx < len(self.yolo_labels):
            del self.yolo_labels[label_idx]
            del self.colors[label_idx]
            self.update_labels_list()
            self.update_image()

    def start_labeling_mode(self):
        """Активирует режим добавления разметки"""
        class_id, ok = QInputDialog.getInt(
            self,
            "Новый объект",
            "Введите номер класса:",
            min=0, max=100, value=0
        )

        if ok:
            self.drawing_mode = True
            self.current_label = class_id
            self.setCursor(Qt.CrossCursor)
            QMessageBox.information(
                self,
                "Создание разметки",
                "Кликните в левый верхний угол объекта, затем в правый нижний"
            )

    def mousePressEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            # Фиксируем первую точку при нажатии
            self.first_click = event.pos()
            self.current_rect = [self.first_click, self.first_click]  # Инициализируем прямоугольник
            self.update_image()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_mode and self.first_click:
            # Обновляем вторую точку при движении с зажатой кнопкой
            self.current_rect[1] = event.pos()
            self.update_image()  # Перерисовываем с обновленным прямоугольником
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton and self.first_click:
            # Фиксируем вторую точку при отпускании
            second_point = event.pos()
            self.finish_labeling(second_point)
            self.first_click = None
            self.current_rect = None
            self.update_image()
        else:
            super().mouseReleaseEvent(event)

    def finish_labeling(self, second_point):
        """Создает новую метку по двум точкам"""
        if not self.first_click:
            return

        # Получаем координаты изображения
        img_width, img_height = self.original_img.size
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # Получаем координаты прямоугольника
        x1 = min(self.first_click.x(), second_point.x())
        y1 = min(self.first_click.y(), second_point.y())
        x2 = max(self.first_click.x(), second_point.x())
        y2 = max(self.first_click.y(), second_point.y())

        # Преобразуем в YOLO формат
        x_center = ((x1 + x2) / 2) / label_width
        y_center = ((y1 + y2) / 2) / label_height
        width = (x2 - x1) / label_width
        height = (y2 - y1) / label_height

        # Добавляем новую метку
        new_label = (self.current_label, x_center, y_center, width, height)
        self.yolo_labels.append(new_label)

        # Генерируем цвет
        new_color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        self.colors.append(new_color)

        # Обновляем интерфейс
        self.update_labels_list()
        self.drawing_mode = False
        self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, event):
        """Отмена режима рисования по Esc"""
        if event.key() == Qt.Key_Escape and self.drawing_mode:
            self.drawing_mode = False
            self.first_click = None
            self.temp_rect = None
            self.setCursor(Qt.ArrowCursor)
            self.update_image()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # Увеличиваем яркость на 5%
            self.brightness_slider.setValue(min(self.brightness_value + 5, 200))
        elif event.key() == Qt.Key_Minus:
            # Уменьшаем яркость на 5%
            self.brightness_slider.setValue(max(self.brightness_value - 5, 50))
        elif event.key() == Qt.Key_0:
            # Сброс яркости
            self.reset_brightness()
        elif event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Сброс состояния при закрытии"""
        if self._pixmap:
            self._pixmap = None
        if self._is_closing:
            event.accept()
            return
        self._is_closing = True
        self.drawing_mode = False
        self.setCursor(Qt.ArrowCursor)
        super().closeEvent(event)


class ImageProcessingThread(QThread):
    progress_updated = pyqtSignal(int, str)  # Добавляем текстовое сообщение
    cluster_found = pyqtSignal(list)  # Новый сигнал для найденных кластеров
    finished_clustering = pyqtSignal()  # Сигнал завершения

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
        # Подсчет общего количества изображений
        self.total_images = sum(1 for _ in self._get_image_files())
        if self.total_images == 0:
            self.progress_updated.emit(0, "No images found")
            self.finished_clustering.emit()
            return

        # Этап 1: Вычисление хэшей
        hashes = []
        for i, (img_path, hash_val) in enumerate(self._compute_hashes()):
            if self.canceled:
                return
            hashes.append((img_path, hash_val))
            progress = int((i + 1) / self.total_images * 50)
            self.progress_updated.emit(progress, f"Processing: {os.path.basename(img_path)}")

        # Этап 2: Кластеризация
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
                self.cluster_found.emit(cluster)  # Отправляем найденный кластер

            total_processed += len(cluster)
            progress = 50 + int(total_processed / self.total_images * 50)
            self.progress_updated.emit(progress, f"Clustering: {len(cluster)} images found")

        self.finished_clustering.emit()

    def _get_image_files(self):
        """Генератор путей к изображениям"""
        for root, _, files in os.walk(self.image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    yield PlatformUtils.get_normalized_path(
                        os.path.join(root, file)
                    )

    def _compute_hashes(self):
        """Генератор хэшей изображений"""
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
