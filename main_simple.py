import os
import sys
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import imagehash
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QListWidget, QScrollArea, QGroupBox,
    QListWidgetItem, QMessageBox, QCheckBox, QProgressBar,
    QFrame, QComboBox, QDialog, QLineEdit, QSizePolicy, QSplitter, QSizeGrip, QInputDialog, QGridLayout, QSlider,
    QToolButton
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, pyqtSlot, QUrl
from PyQt5.QtGui import QPixmap, QImage, QFont, QIntValidator


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

                self.progress_updated.emit(i, total, img_path)

                try:
                    # Пропускаем если уже есть разметка
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
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

    def setImagePath(self, path):
        self._image_path = path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
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
        self.setWindowTitle(f"Image Viewer - {os.path.basename(image_path)}")

        self.yolo_model: YOLO | None = yolo_model
        self.yolo_img_w = yolo_img_w
        self.yolo_img_h = yolo_img_h
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou

        self.image_path = image_path
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

        close_btn = QPushButton("Close and save")
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
        label_path = os.path.splitext(self.image_path)[0] + '.txt'
        with open(label_path, 'w') as file:
            for tuple_item in self.yolo_labels:
                line = ' '.join(map(str, tuple_item))
                file.write(line + '\n')
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
            with Image.open(self.image_path) as img:
                self.original_img = img.convert("RGB")
                self.current_img = self.original_img.copy()  # Копия для изменений
                self._original_size = img.size

                # Устанавливаем начальное соотношение разделителя (70% изображение, 30% управление)
                total_height = self.height()
                self.splitter.setSizes([int(total_height * 0.7), int(total_height * 0.3)])

                self.update_image()

                # Заполняем список меток
                self.labels_list.clear()
                self.label_visibility = []  # Список для хранения состояния видимости меток
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

    def mouseMoveEvent(self, event):
        if self.drawing_mode and self.first_click:
            # Обновляем вторую точку при движении с зажатой кнопкой
            self.current_rect[1] = event.pos()
            self.update_image()  # Перерисовываем с обновленным прямоугольником

    def mouseReleaseEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton and self.first_click:
            # Фиксируем вторую точку при отпускании
            second_point = event.pos()
            self.finish_labeling(second_point)
            self.first_click = None
            self.current_rect = None
            self.update_image()

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
                    yield os.path.join(root, file)

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


class ImageClusterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Vision Labeler")
        self.setGeometry(100, 100, 1200, 800)

        self.image_folder = ""
        self.yolo_model = ""
        self.yolo_model_pt = None
        self.yolo_model_conf = 0.0
        self.yolo_model_iou = 0.5
        self.yolo_model_img_w = 640
        self.yolo_model_img_h = 640
        self.clusters = []
        self.current_cluster_index = -1
        self.processing_thread = None
        self.yolo_labels = {}
        self.label_colors = {}

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        main_widget = QWidget()
        self.main_layout = QHBoxLayout(main_widget)

        self.setup_left_panel()
        self.setup_right_panel()

        self.setCentralWidget(main_widget)

    def setup_left_panel(self):
        left_panel = QVBoxLayout()

        # YOLO Model Settings - collapsible section
        self.yolo_group = QGroupBox()
        self.yolo_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 3px; margin-top: 5px; }")

        # Main layout for the group
        yolo_main_layout = QVBoxLayout()
        yolo_main_layout.setContentsMargins(5, 5, 5, 5)

        # Toggle button
        self.yolo_toggle_btn = QPushButton("YOLO Model Settings ▼")
        self.yolo_toggle_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                font-weight: bold;
                border: none;
                padding: 5px;
            }
        """)
        self.yolo_toggle_btn.clicked.connect(self.toggle_yolo_settings)

        # Content widget that will be shown/hidden
        self.yolo_content = QWidget()
        yolo_content_layout = QVBoxLayout()
        yolo_content_layout.setContentsMargins(0, 0, 0, 0)

        # Model selection
        model_select_layout = QHBoxLayout()
        self.yolo_model_label = QLabel("No YOLO model selected")
        model_select_layout.addWidget(self.yolo_model_label)
        yolo_model_btn = QPushButton("Browse File")
        yolo_model_btn.setObjectName("yoloModelButton")
        yolo_model_btn.clicked.connect(self.browse_yolo_model_file)
        model_select_layout.addWidget(yolo_model_btn)
        yolo_content_layout.addLayout(model_select_layout)

        # Parameters grid
        params_layout = QGridLayout()

        # Confidence threshold
        self.conf_label = QLabel("Confidence (1-100):")
        self.conf_input = QLineEdit("55")
        self.conf_input.setValidator(QIntValidator(1, 100))
        params_layout.addWidget(self.conf_label, 0, 0)
        params_layout.addWidget(self.conf_input, 0, 1)

        # Image width
        self.img_w_label = QLabel("Image Width:")
        self.img_w_input = QLineEdit("640")
        self.img_w_input.setValidator(QIntValidator(1, 4096))
        params_layout.addWidget(self.img_w_label, 1, 0)
        params_layout.addWidget(self.img_w_input, 1, 1)

        # Image height
        self.img_h_label = QLabel("Image Height:")
        self.img_h_input = QLineEdit("640")
        self.img_h_input.setValidator(QIntValidator(1, 4096))
        params_layout.addWidget(self.img_h_label, 2, 0)
        params_layout.addWidget(self.img_h_input, 2, 1)

        # IOU threshold
        self.iou_label = QLabel("IOU Threshold (1-100):")
        self.iou_input = QLineEdit("45")
        self.iou_input.setValidator(QIntValidator(1, 100))
        params_layout.addWidget(self.iou_label, 3, 0)
        params_layout.addWidget(self.iou_input, 3, 1)

        yolo_content_layout.addLayout(params_layout)
        self.yolo_content.setLayout(yolo_content_layout)
        self.yolo_content.hide()

        # Add widgets to main group layout
        yolo_main_layout.addWidget(self.yolo_toggle_btn)
        yolo_main_layout.addWidget(self.yolo_content)
        self.yolo_group.setLayout(yolo_main_layout)

        left_panel.addWidget(self.yolo_group)

        # Folder Selection
        folder_group = QGroupBox("Folder Selection")
        folder_layout = QVBoxLayout()
        self.folder_label = QLabel("No folder selected")
        folder_layout.addWidget(self.folder_label)

        browse_btn = QPushButton("Browse Folder")
        browse_btn.setObjectName("browseButton")
        folder_layout.addWidget(browse_btn)

        # Similarity settings
        similarity_group = QGroupBox("Similarity Settings")
        similarity_layout = QVBoxLayout()

        # Hash method
        hash_layout = QHBoxLayout()
        hash_layout.addWidget(QLabel("Hash Method:"))
        self.hash_combo = QComboBox()
        self.hash_combo.addItems(["average_hash", "phash", "dhash"])
        hash_layout.addWidget(self.hash_combo)
        similarity_layout.addLayout(hash_layout)

        # Similarity threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Similarity Threshold (0-64):"))
        self.threshold_input = QLineEdit("5")
        self.threshold_input.setValidator(QIntValidator(0, 64))
        threshold_layout.addWidget(self.threshold_input)
        similarity_layout.addLayout(threshold_layout)

        # Advanced similarity options
        advanced_layout = QHBoxLayout()
        advanced_layout.addWidget(QLabel("Presets:"))
        self.similarity_preset = QComboBox()
        self.similarity_preset.addItems(["Strict (2)", "Normal (5)", "Loose (10)"])
        self.similarity_preset.setCurrentIndex(1)
        advanced_layout.addWidget(self.similarity_preset)
        similarity_layout.addLayout(advanced_layout)

        similarity_group.setLayout(similarity_layout)
        folder_layout.addWidget(similarity_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()

        self.skip_single_check = QCheckBox("Skip single-image clusters")
        self.skip_single_check.setChecked(True)
        options_layout.addWidget(self.skip_single_check)

        self.yolo_labeling_check = QCheckBox("Show YOLO labeling")
        options_layout.addWidget(self.yolo_labeling_check)

        options_group.setLayout(options_layout)
        folder_layout.addWidget(options_group)

        # Process button
        self.process_btn = QPushButton("Process Images")
        folder_layout.addWidget(self.process_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #5BC0DE;
                width: 10px;
                margin: 0.5px;
            }
        """)
        folder_layout.addWidget(self.progress_bar)

        # Auto labeling
        self.auto_label_btn = QPushButton("Auto Label with YOLO")
        self.auto_label_btn.setEnabled(False)
        folder_layout.addWidget(self.auto_label_btn)

        # Labeling progress
        self.labeling_progress = QProgressBar()
        self.labeling_progress.setVisible(False)
        folder_layout.addWidget(self.labeling_progress)

        folder_group.setLayout(folder_layout)
        left_panel.addWidget(folder_group)

        # Cluster list
        cluster_group = QGroupBox("Clusters")
        cluster_layout = QVBoxLayout()

        self.cluster_list = QListWidget()
        cluster_layout.addWidget(self.cluster_list)

        # Delete duplicates button
        self.delete_duplicates_btn = QPushButton("Delete All Duplicates")
        self.delete_duplicates_btn.clicked.connect(self.delete_all_duplicates)
        self.delete_duplicates_btn.setEnabled(False)
        cluster_layout.addWidget(self.delete_duplicates_btn)

        cluster_group.setLayout(cluster_layout)
        left_panel.addWidget(cluster_group)

        left_panel.addStretch()
        self.main_layout.addLayout(left_panel, 1)

    def setup_right_panel(self):
        right_panel = QVBoxLayout()

        self.cluster_images_area = QScrollArea()
        self.cluster_images_area.setWidgetResizable(True)

        self.cluster_images_widget = QWidget()
        self.cluster_images_layout = QVBoxLayout(self.cluster_images_widget)

        self.cluster_images_area.setWidget(self.cluster_images_widget)
        right_panel.addWidget(self.cluster_images_area)

        # Кнопки управления
        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All (A)")
        button_layout.addWidget(self.select_all_btn)

        self.delete_btn = QPushButton("Delete Selected (D)")
        button_layout.addWidget(self.delete_btn)

        # Новая кнопка для удаления дубликатов в кластере
        self.delete_cluster_duplicates_btn = QPushButton("Delete Cluster Duplicates (X)")
        self.delete_cluster_duplicates_btn.setToolTip("Keep only one image in current cluster")
        self.delete_cluster_duplicates_btn.clicked.connect(self.delete_current_cluster_duplicates)
        button_layout.addWidget(self.delete_cluster_duplicates_btn)

        right_panel.addLayout(button_layout)
        self.main_layout.addLayout(right_panel, 3)

        self.setStyleSheet("""
                QWidget#selected_image_widget {
                    border: 3px solid #2a82da;
                    border-radius: 5px;
                    background-color: #f0f0f0;
                }
            """)

    def setup_connections(self):
        self.findChild(QPushButton, "browseButton").clicked.connect(self.browse_folder)
        self.findChild(QPushButton, "yoloModelButton").clicked.connect(self.browse_yolo_model_file)
        self.process_btn.clicked.connect(self.process_images)
        self.cluster_list.itemClicked.connect(self.show_cluster_images)
        self.delete_btn.clicked.connect(self.delete_selected_images)
        self.select_all_btn.clicked.connect(self.toggle_all_images)
        self.similarity_preset.currentIndexChanged.connect(self.update_similarity_preset)
        # self.processing_thread.finished_clustering.connect(lambda: self.delete_duplicates_btn.setEnabled(True))
        self.auto_label_btn.clicked.connect(self.run_auto_labeling)
        # Подключаем сигнал завершения обработки изображений
        # self.processing_thread.finished_clustering.connect(self.check_yolo_model_ready)

    def toggle_yolo_settings(self):
        """Toggle YOLO settings visibility"""
        if self.yolo_content.isVisible():
            self.yolo_content.hide()
            self.yolo_toggle_btn.setText("YOLO Model Settings ▶")
        else:
            self.yolo_content.show()
            self.yolo_toggle_btn.setText("YOLO Model Settings ▼")

    def delete_current_cluster_duplicates(self):
        """Удаляет все дубликаты в текущем кластере, оставляя одно изображение"""
        if self.current_cluster_index == -1:
            QMessageBox.warning(self, "Warning", "No cluster selected!")
            return

        cluster = self.clusters[self.current_cluster_index]
        if len(cluster) <= 1:
            QMessageBox.information(self, "Information", "Cluster already contains only one image!")
            return

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"This will delete {len(cluster) - 1} images from this cluster,\n"
            "keeping only one. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        # Оставляем первое изображение, удаляем остальные
        image_to_keep = cluster[0]
        deleted_count = 0

        for img_path in cluster[1:]:
            try:
                # Удаляем файл изображения
                os.remove(img_path)

                # Удаляем соответствующий .txt файл если существует
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    os.remove(txt_path)

                # Удаляем из памяти
                if img_path in self.yolo_labels:
                    del self.yolo_labels[img_path]
                if img_path in self.label_colors:
                    del self.label_colors[img_path]

                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {img_path}: {e}")

        # Обновляем кластер - оставляем только одно изображение
        self.clusters[self.current_cluster_index] = [image_to_keep]

        # Обновляем интерфейс
        current_item = self.cluster_list.currentItem()
        current_item.setText(f"Cluster {self.current_cluster_index + 1} (1 image)")
        self.show_cluster_images(current_item)

        QMessageBox.information(
            self, "Operation Complete",
            f"Deleted {deleted_count} duplicate images.\n"
            f"Kept 1 unique image in cluster."
        )

    def check_yolo_model_ready(self):
        """Проверяет, можно ли активировать кнопку авторазметки"""
        # Проверяем что модель загружена и обработка завершена
        model_loaded = self.yolo_model_pt is not None
        processing_done = not (hasattr(self,
                                       'processing_thread') and self.processing_thread and self.processing_thread.isRunning())
        self.auto_label_btn.setEnabled(model_loaded and processing_done)

    def run_auto_labeling(self):
        """Запускает автоматическую разметку изображений без меток"""
        if not self.yolo_model_pt:
            QMessageBox.warning(self, "Warning", "YOLO model not loaded!")
            return

        # Находим все изображения без разметки
        unlabeled_images = []
        for cluster in self.clusters:
            for img_path in cluster:
                img_path: str
                if img_path.endswith(".txt"):
                    continue
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                if not os.path.exists(txt_path):
                    unlabeled_images.append(img_path)

        if not unlabeled_images:
            QMessageBox.information(self, "Information", "All images already have labels!")
            return

        reply = QMessageBox.question(
            self, "Confirm Auto Labeling",
            f"Found {len(unlabeled_images)} images without labels.\n"
            "Run YOLO model to automatically label them?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.start_auto_labeling(unlabeled_images)

    def start_auto_labeling(self, image_paths):
        """Запускает процесс автоматической разметки"""
        self.labeling_thread = AutoLabelingThread(
            image_paths,
            self.yolo_model_pt,
            int(self.img_w_input.text()),
            int(self.img_h_input.text()),
            int(self.conf_input.text()) / 100,
            int(self.iou_input.text()) / 100
        )

        self.labeling_thread.progress_updated.connect(self.update_labeling_progress)
        self.labeling_thread.finished.connect(self.on_labeling_finished)
        self.labeling_thread.error_occurred.connect(self.on_labeling_error)

        self.auto_label_btn.setEnabled(False)
        self.labeling_progress.setVisible(True)
        self.labeling_progress.setMaximum(len(image_paths))
        self.labeling_progress.setValue(0)
        self.labeling_progress.setFormat("Starting auto-labeling... %p%")

        self.labeling_thread.start()

    def update_labeling_progress(self, current, total, image_path):
        """Обновляет прогресс разметки"""
        self.labeling_progress.setValue(current)
        self.labeling_progress.setMaximum(total)
        self.labeling_progress.setFormat(f"Processing: {os.path.basename(image_path)}... {current}/{total}")

    def on_labeling_finished(self):
        """Действия по завершении разметки"""
        self.labeling_progress.setFormat("Labeling complete! %p%")
        self.auto_label_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "Auto-labeling completed successfully!")

        # Обновляем текущий кластер, если он есть
        if self.current_cluster_index != -1:
            current_item = self.cluster_list.currentItem()
            self.show_cluster_images(current_item)

    def on_labeling_error(self, error_msg):
        """Обработка ошибок при разметке"""
        self.labeling_progress.setFormat("Labeling failed! %p%")
        self.auto_label_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error_msg)

    def delete_all_duplicates(self):
        """Удаляет все дубликаты, оставляя по одному изображению из каждого кластера"""
        if not self.clusters:
            return

        reply = QMessageBox.question(
            self, "Confirm Mass Delete",
            "This will delete ALL duplicate images, keeping only one from each cluster.\n"
            "This action cannot be undone!\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        total_deleted = 0
        for cluster in self.clusters:
            # Оставляем первое изображение, удаляем остальные
            for img_path in cluster[1:]:
                try:
                    os.remove(img_path)
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                    total_deleted += 1
                except Exception as e:
                    print(f"Error deleting {img_path}: {e}")

        # Обновляем UI
        self.reset_ui()
        QMessageBox.information(
            self, "Operation Complete",
            f"Deleted {total_deleted} duplicate images.\n"
            f"Kept {len(self.clusters)} unique images."
        )

    def update_similarity_preset(self, index):
        presets = [2, 5, 10]
        self.threshold_input.setText(str(presets[index]))

    def toggle_all_images(self):
        """Инвертирует состояние выделения всех изображений в текущем кластере"""
        if self.current_cluster_index == -1:
            return

        # Проверяем, есть ли неотмеченные изображения
        has_unchecked = any(
            widget.findChild(QCheckBox, "image_checkbox").isChecked() == False
            for widget in self.get_image_widgets()
        )

        # Устанавливаем состояние в зависимости от наличия неотмеченных
        new_state = has_unchecked

        for widget in self.get_image_widgets():
            checkbox = widget.findChild(QCheckBox, "image_checkbox")
            checkbox.setChecked(new_state)

    def get_image_widgets(self):
        """Возвращает список виджетов изображений в текущем кластере"""
        widgets = []
        for i in range(self.cluster_images_layout.count()):
            widget = self.cluster_images_layout.itemAt(i).widget()
            if widget and widget.objectName() == "image_widget":
                widgets.append(widget)
        return widgets

    def browse_yolo_model_file(self):
        # Вариант 1: Используем getOpenFileName (рекомендуется)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model File",
            "",
            "YOLO Model Files (*.pt)"
        )

        # ИЛИ Вариант 2: Если нужно именно getOpenFileUrl
        # file_url = QFileDialog.getOpenFileUrl(self, "Select YOLO Model File")[0]
        # file_path = file_url.toLocalFile() if file_url.isValid() else ""

        if file_path:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)

                # Нормализуем путь (убираем дублирующиеся слеши и т.д.)
                normalized_path = os.path.normpath(file_path)

                print(f"Loading model from: {normalized_path}")  # Для отладки

                # Проверяем существование файла
                if not os.path.exists(normalized_path):
                    raise FileNotFoundError(f"Model file not found at: {normalized_path}")

                # Загружаем модель
                self.yolo_model_pt = YOLO(normalized_path)

                # Определяем устройство (GPU/CPU)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.yolo_model_pt.to(device)

                self.yolo_model = normalized_path
                self.yolo_model_label.setText(normalized_path)
                self.reset_ui()

            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"Failed to load YOLO model:\n{str(e)}\n"
                                     f"Path: {normalized_path}")
                self.yolo_model_pt = None
                self.yolo_model = ""
                self.yolo_model_label.setText("No YOLO model selected")

            finally:
                QApplication.restoreOverrideCursor()

            # self.yolo_model = file[0].path()
            # self.yolo_model_label.setText(file[0].path())
            # self.yolo_model_pt = YOLO(file[0].path())
            # self.reset_ui()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_folder = folder
            self.folder_label.setText(folder)
            self.reset_ui()

    def reset_ui(self):
        self.cluster_list.clear()
        self.clear_image_display()
        self.progress_bar.setValue(0)
        self.yolo_labels = {}
        self.label_colors = {}
        self.delete_duplicates_btn.setEnabled(False)

    def process_images(self):
        if not hasattr(self, 'image_folder') or not self.image_folder:
            QMessageBox.warning(self, "Warning", "Please select a folder first!")
            return

        try:
            threshold = int(self.threshold_input.text())
            if not 0 <= threshold <= 64:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid threshold (0-64)")
            return

        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.canceled = True
            self.processing_thread.wait()

        self.reset_ui()
        self.progress_bar.setFormat("Preparing... %p%")
        self.progress_bar.setValue(0)
        self.delete_duplicates_btn.setEnabled(False)
        self.auto_label_btn.setEnabled(False)  # Отключаем кнопку на время обработки

        self.processing_thread = ImageProcessingThread(
            self.image_folder,
            threshold,
            self.skip_single_check.isChecked(),
            self.hash_combo.currentText()
        )

        # Подключаем новые сигналы
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.cluster_found.connect(self.add_cluster)
        self.processing_thread.finished_clustering.connect(self.on_clustering_finished)
        self.processing_thread.finished_clustering.connect(lambda: self.delete_duplicates_btn.setEnabled(True))
        self.processing_thread.finished_clustering.connect(self.check_yolo_model_ready)
        self.processing_thread.start()

    def update_progress(self, value, message):
        """Обновление прогресс бара с дополнительной информацией"""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{message}... {value}%")

        # Анимация для индикации активности
        if value < 100:
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid grey;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #5BC0DE;
                    width: 10px;
                    margin: 0.5px;
                    animation: pulse 1s infinite;
                }
            """)
        else:
            self.progress_bar.setStyleSheet("")  # Сбрасываем стиль при завершении

    def add_cluster(self, cluster):
        # """Добавляет найденный кластер в список"""
        # item = QListWidgetItem(f"Cluster {self.cluster_list.count() + 1} ({len(cluster)} images)")
        # item.setData(Qt.UserRole, len(self.clusters))
        # self.clusters.append(cluster)
        # self.cluster_list.addItem(item)
        #
        # # Автоматически показываем первый кластер
        # if self.cluster_list.count() == 1:
        #     self.cluster_list.setCurrentItem(item)
        #     self.show_cluster_images(item)
        """Добавляет найденный кластер в список и сортирует кластеры по убыванию количества изображений"""
        # Добавляем кластер в список
        self.clusters.append(cluster)

        # Сортируем кластеры по убыванию количества изображений
        self.clusters.sort(key=lambda x: len(x), reverse=True)

        # Очищаем список и заполняем заново с учетом новой сортировки
        self.cluster_list.clear()

        for i, cluster in enumerate(self.clusters):
            item = QListWidgetItem(f"Cluster {i + 1} ({len(cluster)} images)")
            item.setData(Qt.UserRole, i)
            self.cluster_list.addItem(item)

        # Автоматически показываем первый кластер (самый большой)
        if self.cluster_list.count() > 0:
            self.cluster_list.setCurrentItem(self.cluster_list.item(0))
            self.show_cluster_images(self.cluster_list.item(0))

    def on_clustering_finished(self):
        """Действия по завершении кластеризации"""
        self.progress_bar.setFormat("Done! %p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar::chunk {
                background-color: #5CB85C;
            }
        """)
        if not self.clusters:
            QMessageBox.information(self, "Information", "No clusters found matching your criteria!")

    def handle_clusters_ready(self, clusters):
        self.clusters = clusters

        # Сортируем кластеры по убыванию количества изображений
        self.clusters.sort(key=lambda x: len(x), reverse=True)

        if not clusters:
            QMessageBox.information(self, "Information", "No clusters found matching your criteria!")
            return

        for i, cluster in enumerate(clusters):
            item = QListWidgetItem(f"Cluster {i + 1} ({len(cluster)} images)")
            item.setData(Qt.UserRole, i)
            self.cluster_list.addItem(item)

    def show_cluster_images(self, item):
        self.current_cluster_index = item.data(Qt.UserRole)
        cluster = self.clusters[self.current_cluster_index]

        self.clear_image_display()

        for img_path in cluster:
            img_widget = self.create_image_widget(img_path)
            self.cluster_images_layout.addWidget(img_widget)

        self.delete_btn.setEnabled(True)

    def create_image_widget(self, img_path):
        widget = QWidget()
        widget.setObjectName("image_widget")
        widget.setProperty("selected", False)  # Добавляем свойство для выделения
        layout = QHBoxLayout(widget)

        checkbox = QCheckBox()
        checkbox.setObjectName("image_checkbox")
        checkbox.setProperty("image_path", img_path)
        layout.addWidget(checkbox)

        thumbnail_label = ClickableLabel()
        thumbnail_label.setAlignment(Qt.AlignCenter)
        thumbnail_label.setImagePath(img_path)
        thumbnail_label.clicked.connect(self.show_fullscreen_image)

        try:
            pixmap = self.load_image_with_yolo_labels(img_path)
            if not pixmap.isNull():
                thumbnail_label.setPixmap(pixmap.scaled(
                    QSize(300, 300),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
            else:
                thumbnail_label.setText("Invalid image")
        except Exception as e:
            print(f"Error loading thumbnail: {e}")
            thumbnail_label.setText("Load error")

        layout.addWidget(thumbnail_label)

        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        file_name = os.path.basename(img_path)
        name_label = QLabel(file_name)
        info_layout.addWidget(name_label)

        path_label = QLabel(img_path)
        path_label.setWordWrap(True)
        info_layout.addWidget(path_label)

        if self.yolo_labeling_check.isChecked():
            labels = self.get_yolo_labels(img_path)
            if labels:
                labels_text = "\n".join([f"Class: {l[0]}" for l in labels])
                labels_label = QLabel(f"YOLO Labels ({len(labels)}):\n{labels_text}")
                labels_label.setWordWrap(True)
                info_layout.addWidget(labels_label)

        layout.addWidget(info_widget)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        self.cluster_images_layout.addWidget(separator)

        return widget

    def update_cluster_display(self, img_path):
        """Обновляет отображение кластера после изменения разметки"""
        current_item = self.cluster_list.currentItem()
        if current_item:
            # Очищаем текущее отображение
            self.clear_image_display()

            # Загружаем кластер заново
            # Reboot the cluster
            cluster = self.clusters[self.current_cluster_index]
            for img_path in cluster:
                img_widget = self.create_image_widget(img_path)
                self.cluster_images_layout.addWidget(img_widget)

            # Добавляем разделители
            # Add delimiters
            for i in range(self.cluster_images_layout.count()):
                if i % 2 != 0:
                    # Добавляем разделитель после каждого изображения
                    # Add a separator after each image
                    separator = QFrame()
                    separator.setFrameShape(QFrame.HLine)
                    self.cluster_images_layout.insertWidget(i, separator)

            # Принудительное обновление виджета
            # Forced widget update
            self.cluster_images_widget.update()
            self.cluster_images_area.viewport().update()

    @pyqtSlot(str)
    def show_fullscreen_image(self, img_path):
        if not os.path.exists(img_path):
            QMessageBox.warning(self, "Error", "Image file not found!")
            return

        labels = self.get_yolo_labels(img_path)

        if img_path not in self.label_colors:
            self.label_colors[img_path] = [
                (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                for _ in range(len(labels))
            ]

        try:
            dialog = FullScreenImageDialog(
                img_path,
                labels,
                self.label_colors[img_path],
                self,
                yolo_model=self.yolo_model_pt,
                yolo_img_w=int(self.img_w_input.text()), yolo_img_h=int(self.img_h_input.text()),
                yolo_conf=int(self.conf_input.text())/100, yolo_iou=int(self.iou_input.text())/100,
            )
            # Подключаем сигнал к слоту обновления
            dialog.labels_changed.connect(lambda: self.update_cluster_display(img_path))
            dialog.exec_()

            if labels != dialog.yolo_labels:
                self.save_yolo_labels(img_path, dialog.yolo_labels)
                self.yolo_labels[img_path] = dialog.yolo_labels
                self.label_colors[img_path] = dialog.colors

                # Обновляем отображение всего кластера
                # Update the display of the entire cluster
                current_item = self.cluster_list.currentItem()
                if current_item:
                    self.show_cluster_images(current_item)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot show image: {str(e)}")

    def get_yolo_labels(self, img_path):
        if img_path in self.yolo_labels:
            return self.yolo_labels[img_path]

        txt_path = os.path.splitext(img_path)[0] + '.txt'
        labels = []

        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                labels.append((
                                    int(parts[0]),
                                    float(parts[1]),
                                    float(parts[2]),
                                    float(parts[3]),
                                    float(parts[4])
                                ))
                            except ValueError:
                                continue
            except Exception as e:
                print(f"Error reading YOLO labels: {e}")

        self.yolo_labels[img_path] = labels
        return labels

    def save_yolo_labels(self, img_path, labels):
        txt_path = os.path.splitext(img_path)[0] + '.txt'

        if not labels:
            if os.path.exists(txt_path):
                try:
                    os.remove(txt_path)
                except Exception as e:
                    print(f"Error deleting label file: {e}")
            return

        try:
            with open(txt_path, 'w') as f:
                for label in labels:
                    f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")
        except Exception as e:
            print(f"Error saving YOLO labels: {e}")

    def load_image_with_yolo_labels(self, img_path):
        if not os.path.exists(img_path):
            return QPixmap()

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")

                if self.yolo_labeling_check.isChecked():
                    labels = self.get_yolo_labels(img_path)
                    if labels:
                        draw = ImageDraw.Draw(img)
                        width, height = img.size

                        if img_path not in self.label_colors:
                            self.label_colors[img_path] = [
                                (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                                for _ in range(len(labels))
                            ]

                        for i, label in enumerate(labels):
                            if i >= len(self.label_colors[img_path]):
                                color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                                self.label_colors[img_path].append(color)
                            else:
                                color = self.label_colors[img_path][i]

                            class_id, x_center, y_center, box_width, box_height = label

                            x_center *= width
                            y_center *= height
                            box_width *= width
                            box_height *= height

                            x1 = x_center - box_width / 2
                            y1 = y_center - box_height / 2
                            x2 = x_center + box_width / 2
                            y2 = y_center + box_height / 2

                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                            try:
                                font = ImageFont.load_default()
                                draw.text((x1, y1), str(class_id), fill=color, font=font)
                            except:
                                draw.text((x1, y1), str(class_id), fill=color)

                img = img.convert("RGBA")
                data = img.tobytes("raw", "RGBA")
                qim = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)
                return QPixmap.fromImage(qim)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return QPixmap()

    def clear_image_display(self):
        while self.cluster_images_layout.count():
            child = self.cluster_images_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def get_selected_images(self):
        selected = []
        for i in range(self.cluster_images_layout.count()):
            widget = self.cluster_images_layout.itemAt(i).widget()
            if widget and widget.objectName() == "image_widget":
                checkbox = widget.findChild(QCheckBox, "image_checkbox")
                if checkbox and checkbox.isChecked():
                    selected.append(checkbox.property("image_path"))
        return selected

    def next_image(self):
        """Перейти к следующему изображению в кластере с визуальным выделением"""
        widgets = self.get_image_widgets()
        if not widgets:
            return

        current_index = -1
        for i, widget in enumerate(widgets):
            if widget.property("selected"):
                widget.setProperty("selected", False)
                widget.setStyleSheet("")  # Сбрасываем стиль
                current_index = i
                break

        next_index = (current_index + 1) % len(widgets)
        widgets[next_index].setProperty("selected", True)
        widgets[next_index].setStyleSheet("""
            border: 3px solid #2a82da;
            border-radius: 5px;
            background-color: #f0f0f0;
        """)
        # Прокручиваем к выделенному изображению
        self.cluster_images_area.ensureWidgetVisible(widgets[next_index])

    def prev_image(self):
        """Перейти к предыдущему изображению в кластере с визуальным выделением"""
        widgets = self.get_image_widgets()
        if not widgets:
            return

        current_index = -1
        for i, widget in enumerate(widgets):
            if widget.property("selected"):
                widget.setProperty("selected", False)
                widget.setStyleSheet("")  # Сбрасываем стиль
                current_index = i
                break

        prev_index = (current_index - 1) % len(widgets)
        widgets[prev_index].setProperty("selected", True)
        widgets[prev_index].setStyleSheet("""
            border: 3px solid #2a82da;
            border-radius: 5px;
            background-color: #f0f0f0;
        """)
        # Прокручиваем к выделенному изображению
        self.cluster_images_area.ensureWidgetVisible(widgets[prev_index])

    def toggle_current_image(self):
        """Отметить/снять отметку с текущего изображения"""
        widgets = self.get_image_widgets()
        for widget in widgets:
            if widget.property("selected"):
                checkbox = widget.findChild(QCheckBox, "image_checkbox")
                checkbox.setChecked(not checkbox.isChecked())
                return

    def delete_selected_images(self):
        if self.current_cluster_index == -1:
            return

        selected_images = self.get_selected_images()
        if not selected_images:
            QMessageBox.warning(self, "Warning", "No images selected for deletion!")
            return

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete {len(selected_images)} selected images and their labels?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            cluster = self.clusters[self.current_cluster_index]

            for img_path in selected_images:
                try:
                    os.remove(img_path)

                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    if os.path.exists(txt_path):
                        os.remove(txt_path)

                    cluster.remove(img_path)

                    if img_path in self.yolo_labels:
                        del self.yolo_labels[img_path]
                    if img_path in self.label_colors:
                        del self.label_colors[img_path]

                except Exception as e:
                    print(f"Error deleting {img_path}: {e}")

            self.show_cluster_images(self.cluster_list.currentItem())

            current_item = self.cluster_list.currentItem()
            current_item.setText(f"Cluster {self.current_cluster_index + 1} ({len(cluster)} images)")

            if not cluster:
                self.cluster_list.takeItem(self.cluster_list.row(current_item))
                self.current_cluster_index = -1
                self.clear_image_display()
                self.delete_btn.setEnabled(False)

    def closeEvent(self, event):
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.canceled = True
            self.processing_thread.wait()
        event.accept()

    def get_current_selected_image(self):
        """Возвращает путь к текущему выделенному изображению (с синей обводкой)"""
        widgets = self.get_image_widgets()
        for widget in widgets:
            if widget.property("selected"):
                checkbox = widget.findChild(QCheckBox, "image_checkbox")
                return checkbox.property("image_path")
        return None

    def keyPressEvent(self, event):
        """Обработка нажатий клавиш"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Открываем выделенное изображение по Enter
            selected_image = self.get_current_selected_image()
            if selected_image:
                self.show_fullscreen_image(selected_image)
            return
        if event.key() == Qt.Key_S:  # Следующий кластер
            self.next_cluster()
        elif event.key() == Qt.Key_W:  # Предыдущий кластер
            self.prev_cluster()
        elif event.key() == Qt.Key_A:  # Инвертировать выделение всех
            self.toggle_all_images()
        elif event.key() == Qt.Key_D:  # Удалить выделенные
            self.delete_selected_images()
        elif event.key() == Qt.Key_L:  # Следующее изображение (выделение синей рамкой)
            self.next_image()
        elif event.key() == Qt.Key_O:  # Предыдущее изображение (выделение синей рамкой)
            self.prev_image()
        elif event.key() == Qt.Key_P:  # Переключить чекбокс текущего изображения
            self.toggle_current_image()
        elif event.key() == Qt.Key_X:
            self.delete_current_cluster_duplicates()
        else:
            super().keyPressEvent(event)

    def next_cluster(self):
        """Перейти к следующему кластеру"""
        if not self.clusters:
            print("no cluster")
            return

        current_row = self.cluster_list.currentRow()
        if current_row < self.cluster_list.count() - 1:
            next_item = self.cluster_list.item(current_row + 1)
            self.cluster_list.setCurrentItem(next_item)
            self.show_cluster_images(next_item)

    def prev_cluster(self):
        """Перейти к предыдущему кластеру"""
        if not self.clusters:
            return

        current_row = self.cluster_list.currentRow()
        if current_row > 0:
            prev_item = self.cluster_list.item(current_row - 1)
            self.cluster_list.setCurrentItem(prev_item)
            self.show_cluster_images(prev_item)


if __name__ == "__main__":
    # Проверяем доступность CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Установка шрифта для лучшей читаемости
    # Set the font for better readability
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    window = ImageClusterApp()
    window.show()
    sys.exit(app.exec_())
