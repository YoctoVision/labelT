import gc
import os
import platform
import sys
import random
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QListWidget, QScrollArea, QGroupBox,
    QListWidgetItem, QMessageBox, QCheckBox, QProgressBar,
    QFrame, QComboBox, QDialog, QLineEdit,
    QGridLayout, QTabWidget, QTreeView,
    QFileSystemModel, QStatusBar, QToolBar, QAction, QDockWidget, QMenu
)
from PyQt5.QtCore import Qt, QSize, pyqtSlot, QDir, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QIntValidator, QIcon

from support import ClickableLabel, FullScreenImageDialog, ImageProcessingThread, DarkTheme


class IDEMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Platform-specific initialization
        self._init_platform_settings()

        # Window setup
        self.setWindowTitle("YOLO Vision Labeler IDE")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize instance variables
        self._initialize_variables()

        # UI setup
        self.init_ui()
        self.setup_connections()

    def _init_platform_settings(self):
        """Initialize platform-specific settings"""
        self.os_name = platform.system()

        # Windows specific
        if self.os_name == "Windows":
            try:
                import ctypes
                # Set app ID for Windows taskbar
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('yolo.vision.labeler.ide')
            except:
                pass
            # Enable high DPI scaling
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

        # macOS specific
        elif self.os_name == "Darwin":
            # Enable retina display support
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_NoSystemBackground, False)
            # Enable unified toolbar style
            self.setUnifiedTitleAndToolBarOnMac(True)

        # Linux specific
        elif self.os_name == "Linux":
            # Additional Linux-specific settings if needed
            pass

    # --------------------------
    # Initialization Methods
    # --------------------------
    def _initialize_variables(self):
        """Initialize all instance variables"""
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
        self.current_image_index = -1
        self.images_per_page = 30  # Количество изображений для начальной загрузки
        self.load_batch_size = 20  # Сколько изображений подгружать при скролле
        self.current_loaded = 0    # Сколько изображений уже загружено
        self.current_page = 0
        self.is_loading = False
        self.scroll_connection = None  # Для хранения соединения сигнала скролла

    def init_ui(self):
        """Initialize all UI components"""
        # Устанавливаем центральный виджет (для отображения кластеров)
        self._create_central_display_area()

        # Create UI components
        self.create_toolbar()
        self.create_status_bar()
        self.create_dock_widgets()
        self.create_main_tab_dock()  # Переименованный метод для создания dock

        self.set_styles()

    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(16, 16))

        if self.os_name == "Darwin":
            toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        else:
            toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)

        actions = [
            ("document-open", "Open Folder", self.browse_folder),
            ("system-run", "Process Images", self.process_images),
            None,  # Separator
            ("applications-science", "Load YOLO Model", self.browse_yolo_model_file),
            ("document-edit", "Auto Label", self.run_auto_labeling),
        ]

        for action in actions:
            if action is None:
                toolbar.addSeparator()
            else:
                icon, text, callback = action
                act = QAction(QIcon.fromTheme(icon), text, self)
                act.triggered.connect(callback)
                toolbar.addAction(act)
        self.addToolBar(toolbar)

    def get_normalized_path(self, path):
        """Get platform-normalized path"""
        return Path(path).as_posix() if self.os_name != "Windows" else os.path.normpath(path)

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def create_dock_widgets(self):
        """Create dock widgets for file tree and clusters"""
        # File tree dock
        self.file_dock = QDockWidget("Project", self)
        self.file_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("")
        self.file_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files | QDir.Hidden)

        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(""))

        self.file_dock.setWidget(self.file_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)

        # Cluster list dock
        self.cluster_dock = QDockWidget("Clusters", self)
        self.cluster_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.cluster_list = QListWidget()

        self.cluster_dock.setWidget(self.cluster_list)
        self.addDockWidget(Qt.RightDockWidgetArea, self.cluster_dock)

    def create_main_tab_dock(self):
        """Create the main controls as a dock widget"""
        # Создаем dock виджет для mainTab
        main_dock = QDockWidget("Main Controls", self)
        main_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)

        main_tab = QWidget()
        main_tab.setObjectName("mainTab")
        layout = QVBoxLayout(main_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Add UI components (без _create_image_display_area)
        self._create_yolo_settings_group(layout)
        self._create_similarity_group(layout)
        self._create_options_group(layout)
        self._create_process_buttons(layout)
        self._create_progress_bars(layout)

        main_dock.setWidget(main_tab)
        self.addDockWidget(Qt.LeftDockWidgetArea, main_dock)

        # Размещаем dock в нижней части левой стороны
        self.splitDockWidget(self.file_dock, main_dock, Qt.Vertical)

    def _create_central_display_area(self):
        """Create the central image display area"""
        self.image_tabs = QTabWidget()
        self.image_tabs.setTabsClosable(True)
        self.image_tabs.tabCloseRequested.connect(self.close_image_tab)

        # Main cluster view tab
        self.cluster_tab = QWidget()
        self.cluster_tab_layout = QVBoxLayout(self.cluster_tab)
        self.cluster_tab_layout.setContentsMargins(0, 0, 0, 0)

        # Cluster images scroll area - с центрированием
        self.cluster_images_area = QScrollArea()
        self.cluster_images_area.setWidgetResizable(True)

        # Контейнер для центрирования
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addStretch()

        self.cluster_images_widget = QWidget()
        self.cluster_images_layout = QVBoxLayout(self.cluster_images_widget)
        self.cluster_images_layout.setAlignment(Qt.AlignTop)

        center_layout.addWidget(self.cluster_images_widget)
        center_layout.addStretch()

        self.cluster_images_area.setWidget(center_widget)
        self.cluster_tab_layout.addWidget(self.cluster_images_area)

        # Cluster control buttons
        cluster_btn_layout = QHBoxLayout()
        cluster_btn_layout.setContentsMargins(5, 5, 5, 5)

        self.select_all_btn = QPushButton("Select All (A)")
        cluster_btn_layout.addWidget(self.select_all_btn)

        self.delete_btn = QPushButton("Delete Selected (D)")
        cluster_btn_layout.addWidget(self.delete_btn)

        self.delete_cluster_duplicates_btn = QPushButton("Delete Cluster Duplicates (X)")
        cluster_btn_layout.addWidget(self.delete_cluster_duplicates_btn)

        self.cluster_tab_layout.addLayout(cluster_btn_layout)

        # Add tab to image tabs
        self.image_tabs.addTab(self.cluster_tab, "Clusters")

        # Устанавливаем image_tabs как центральный виджет
        self.setCentralWidget(self.image_tabs)

    def _create_yolo_settings_group(self, parent_layout):
        """Create YOLO settings group box"""
        self.yolo_settings_group = QGroupBox("YOLO Settings")
        self.yolo_settings_group.setCheckable(True)
        self.yolo_settings_group.setChecked(False)
        self.yolo_settings_group.toggled.connect(self.toggle_yolo_settings)

        layout = QGridLayout()
        layout.setContentsMargins(5, 15, 5, 5)

        # Model selection
        self.yolo_model_label = QLabel("No model selected")
        self.yolo_model_label.setWordWrap(True)
        yolo_model_btn = QPushButton("Browse...")
        yolo_model_btn.setObjectName("yoloModelButton")
        yolo_model_btn.setMaximumWidth(100)

        # Parameters
        self.conf_input = QLineEdit("55")
        self.conf_input.setValidator(QIntValidator(1, 100))
        self.conf_input.setMaximumWidth(60)

        self.img_w_input = QLineEdit("640")
        self.img_w_input.setValidator(QIntValidator(1, 4096))
        self.img_w_input.setMaximumWidth(60)

        self.img_h_input = QLineEdit("640")
        self.img_h_input.setValidator(QIntValidator(1, 4096))
        self.img_h_input.setMaximumWidth(60)

        self.iou_input = QLineEdit("45")
        self.iou_input.setValidator(QIntValidator(1, 100))
        self.iou_input.setMaximumWidth(60)

        # Add widgets to layout
        layout.addWidget(QLabel("Model:"), 0, 0)
        layout.addWidget(self.yolo_model_label, 0, 1)
        layout.addWidget(yolo_model_btn, 0, 2)

        layout.addWidget(QLabel("Confidence (%):"), 1, 0)
        layout.addWidget(self.conf_input, 1, 1)

        layout.addWidget(QLabel("Image Width:"), 2, 0)
        layout.addWidget(self.img_w_input, 2, 1)

        layout.addWidget(QLabel("Image Height:"), 3, 0)
        layout.addWidget(self.img_h_input, 3, 1)

        layout.addWidget(QLabel("IOU Threshold (%):"), 4, 0)
        layout.addWidget(self.iou_input, 4, 1)

        self.yolo_settings_group.setLayout(layout)
        parent_layout.addWidget(self.yolo_settings_group)

    def _create_similarity_group(self, parent_layout):
        """Create similarity settings group box"""
        self.similarity_group = QGroupBox("Similarity Settings")
        self.similarity_group.setCheckable(True)
        self.similarity_group.setChecked(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 15, 5, 5)

        # Hash method
        hash_layout = QHBoxLayout()
        self.hash_combo = QComboBox()
        self.hash_combo.addItems(["average_hash", "phash", "dhash"])
        hash_layout.addWidget(QLabel("Hash method:"))
        hash_layout.addWidget(self.hash_combo)
        layout.addLayout(hash_layout)

        # Threshold
        threshold_layout = QHBoxLayout()
        self.threshold_input = QLineEdit("5")
        self.threshold_input.setValidator(QIntValidator(0, 64))
        self.threshold_input.setMaximumWidth(40)
        threshold_layout.addWidget(QLabel("Threshold (0-64):"))
        threshold_layout.addWidget(self.threshold_input)
        layout.addLayout(threshold_layout)

        # Presets
        preset_layout = QHBoxLayout()
        self.similarity_preset = QComboBox()
        self.similarity_preset.addItems(["Strict (2)", "Normal (5)", "Loose (10)"])
        self.similarity_preset.setCurrentIndex(1)
        preset_layout.addWidget(QLabel("Presets:"))
        preset_layout.addWidget(self.similarity_preset)
        layout.addLayout(preset_layout)

        self.similarity_group.setLayout(layout)
        parent_layout.addWidget(self.similarity_group)

    def _create_options_group(self, parent_layout):
        """Create options group box"""
        self.options_group = QGroupBox("Options")
        self.options_group.setCheckable(True)
        self.options_group.setChecked(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 15, 5, 5)

        self.skip_single_check = QCheckBox("Skip single-image clusters")
        self.skip_single_check.setChecked(True)
        layout.addWidget(self.skip_single_check)

        self.yolo_labeling_check = QCheckBox("Show YOLO labeling")
        layout.addWidget(self.yolo_labeling_check)

        self.options_group.setLayout(layout)
        parent_layout.addWidget(self.options_group)

    def _create_process_buttons(self, parent_layout):
        """Create process buttons layout"""
        button_layout = QHBoxLayout()

        self.process_btn = QPushButton("Process Images")
        self.process_btn.setObjectName("processButton")
        button_layout.addWidget(self.process_btn)

        self.auto_label_btn = QPushButton("Auto Label")
        self.auto_label_btn.setObjectName("autoLabelButton")
        self.auto_label_btn.setEnabled(False)
        button_layout.addWidget(self.auto_label_btn)

        parent_layout.addLayout(button_layout)

    def _create_progress_bars(self, parent_layout):
        """Create progress bars"""
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #2a82da;
                width: 10px;
            }
        """)
        parent_layout.addWidget(self.progress_bar)

        self.labeling_progress = QProgressBar()
        self.labeling_progress.setVisible(False)
        self.labeling_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #5cb85c;
                width: 10px;
            }
        """)
        parent_layout.addWidget(self.labeling_progress)

    def _create_image_display_area(self, parent_layout):
        """Create image display area with tabs as central widget"""
        self.image_tabs = QTabWidget()
        self.image_tabs.setTabsClosable(True)
        self.image_tabs.tabCloseRequested.connect(self.close_image_tab)

        # Main cluster view tab
        self.cluster_tab = QWidget()
        self.cluster_tab_layout = QVBoxLayout(self.cluster_tab)
        self.cluster_tab_layout.setContentsMargins(0, 0, 0, 0)

        # Cluster images scroll area - теперь с центрированием
        self.cluster_images_area = QScrollArea()
        self.cluster_images_area.setWidgetResizable(True)

        # Контейнер для центрирования
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addStretch()

        self.cluster_images_widget = QWidget()
        self.cluster_images_layout = QVBoxLayout(self.cluster_images_widget)
        self.cluster_images_layout.setAlignment(Qt.AlignTop)

        center_layout.addWidget(self.cluster_images_widget)
        center_layout.addStretch()

        self.cluster_images_area.setWidget(center_widget)
        self.cluster_tab_layout.addWidget(self.cluster_images_area)

        # Cluster control buttons
        cluster_btn_layout = QHBoxLayout()
        cluster_btn_layout.setContentsMargins(5, 5, 5, 5)

        self.select_all_btn = QPushButton("Select All (A)")
        cluster_btn_layout.addWidget(self.select_all_btn)

        self.delete_btn = QPushButton("Delete Selected (D)")
        cluster_btn_layout.addWidget(self.delete_btn)

        self.delete_cluster_duplicates_btn = QPushButton("Delete Cluster Duplicates (X)")
        cluster_btn_layout.addWidget(self.delete_cluster_duplicates_btn)

        self.cluster_tab_layout.addLayout(cluster_btn_layout)

        # Add tab to image tabs
        self.image_tabs.addTab(self.cluster_tab, "Clusters")

        # Устанавливаем image_tabs как центральный виджет
        self.setCentralWidget(self.image_tabs)

    # --------------------------
    # UI Utility Methods
    # --------------------------
    def set_styles(self):
        """Set styles for UI elements"""
        style = """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 3px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                padding: 3px 10px;
                min-width: 80px;
            }
            QPushButton#processButton, QPushButton#autoLabelButton {
                padding: 5px 15px;
                font-weight: bold;
            }
            QLineEdit, QComboBox {
                padding: 3px;
                border: 1px solid #555;
                border-radius: 3px;
            }
        """
        self.setStyleSheet(style)

    def toggle_yolo_settings(self, checked):
        """Toggle YOLO settings panel visibility"""
        self.yolo_settings_group.setTitle(f"YOLO Settings {'▼' if checked else '▶'}")

    def reset_ui(self):
        """Reset UI to initial state"""
        if hasattr(self, 'cluster_list'):
            self.cluster_list.clear()

        self.clear_image_display()

        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)

        self.yolo_labels = {}
        self.label_colors = {}
        self.clusters = []
        self.current_cluster_index = -1

        if hasattr(self, 'delete_btn'):
            self.delete_btn.setEnabled(False)

        self.file_dock.setWindowTitle(f"{self.file_dock.windowTitle().split(':')[0]}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # --------------------------
    # Event Handlers
    # --------------------------
    def setup_connections(self):
        """Setup signal-slot connections"""
        # YOLO Settings
        yolo_model_btn = self.yolo_settings_group.findChild(QPushButton, "yoloModelButton")
        if yolo_model_btn:
            yolo_model_btn.clicked.connect(self.browse_yolo_model_file)

        # Main buttons
        self.process_btn.clicked.connect(self.process_images)
        self.auto_label_btn.clicked.connect(self.run_auto_labeling)

        # Cluster list
        self.cluster_list.itemClicked.connect(self.show_cluster_images)

        # Cluster control buttons
        self.select_all_btn.clicked.connect(self.toggle_all_images)
        self.delete_btn.clicked.connect(self.delete_selected_images)
        self.delete_cluster_duplicates_btn.clicked.connect(self.delete_current_cluster_duplicates)

        # Similarity preset combobox
        self.similarity_preset.currentIndexChanged.connect(self.update_similarity_preset)

        # File tree double click
        self.file_tree.doubleClicked.connect(self.on_file_double_clicked)
        self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.show_context_menu)

    def on_file_double_clicked(self, index):
        """Handle file double click in file tree"""
        file_path = self.file_model.filePath(index)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            self.show_fullscreen_image(file_path)

    def show_context_menu(self, position):
        # Получаем индекс элемента, на котором был сделан клик
        index = self.file_tree.indexAt(position)

        if index.isValid():
            # Создаем контекстное меню
            context_menu = QMenu(self)

            # Добавляем действия в меню
            action1 = context_menu.addAction("Открыть как датасет")
            action2 = context_menu.addAction("Добавить как YOLO модель")

            # Показываем меню в позиции клика
            action = context_menu.exec_(self.file_tree.viewport().mapToGlobal(position))
            file_path = self.file_model.filePath(index)
            # Обрабатываем выбранное действие
            if action == action1:
                self._open_folder(file_path)
            elif action == action2:
                self._open_yolo_model(file_path)

    def close_image_tab(self, index):
        """Close image tab (except main cluster tab)"""
        if index != 0:
            self.image_tabs.removeTab(index)

    def close_tab(self, index):
        """Close tab (except main tab)"""
        if index != 0:
            self.tab_widget.removeTab(index)

    # --------------------------
    # File Operations
    # --------------------------
    def browse_folder(self):
        """Browse for image folder"""
        try:
            # Clear previous data
            self.reset_ui()

            # Open folder dialog
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Image Folder",
                "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )

            if folder:
                self._open_folder(folder)
        except Exception as e:
            error_msg = f"Error loading folder: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _open_folder(self, folder_path):
        self.image_folder = self.get_normalized_path(folder_path)
        self.file_dock.setWindowTitle(f"{self.file_dock.windowTitle()}: {os.path.basename(self.image_folder)}")

        # Update file tree
        self.file_model.setRootPath(self.image_folder)
        self.file_tree.setRootIndex(self.file_model.index(self.image_folder))

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Enable process button
        self.process_btn.setEnabled(True)


    def browse_yolo_model_file(self):
        """Browse for YOLO model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model File",
            "",
            "YOLO Model Files (*.pt)"
        )

        if file_path:
            self._open_yolo_model(file_path)

    def _open_yolo_model(self, file_path):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            normalized_path = self.get_normalized_path(file_path)

            print(f"Loading model from: {normalized_path}")

            if not os.path.exists(normalized_path):
                raise FileNotFoundError(f"Model file not found at: {normalized_path}")

            # Load model
            self.yolo_model_pt = YOLO(normalized_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo_model_pt.to(device)

            self.yolo_model = normalized_path
            self.yolo_model_label.setText(os.path.basename(normalized_path))
            self.reset_ui()

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load YOLO model:\n{str(e)}\nPath: {normalized_path}"
            )
            self.yolo_model_pt = None
            self.yolo_model = ""
            self.yolo_model_label.setText("No YOLO model selected")

        finally:
            QApplication.restoreOverrideCursor()

    # --------------------------
    # Image Processing
    # --------------------------
    def process_images(self):
        """Process images for clustering"""
        try:
            # Validate folder
            if not hasattr(self, 'image_folder') or not self.image_folder:
                QMessageBox.warning(self, "Warning", "Please select a folder first!")
                return

            # Cleanup before processing
            self.cleanup_before_processing()

            # Validate threshold
            try:
                threshold = int(self.threshold_input.text())
                if not 0 <= threshold <= 64:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Warning", "Please enter a valid threshold (0-64)")
                return

            # Cancel previous processing
            if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.canceled = True
                self.processing_thread.wait()

            # Prepare UI
            self.reset_ui()
            self.progress_bar.setFormat("Preparing... %p%")
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("Starting image processing...")

            # Create processing thread
            self.processing_thread = ImageProcessingThread(
                self.image_folder,
                threshold,
                self.skip_single_check.isChecked(),
                self.hash_combo.currentText()
            )

            # Connect signals
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.cluster_found.connect(self.add_cluster)
            self.processing_thread.finished_clustering.connect(self.on_clustering_finished)
            self.processing_thread.finished_clustering.connect(self.check_yolo_model_ready)

            # Start thread
            self.processing_thread.start()

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            print(f"Processing error: {traceback.format_exc()}")
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(error_msg, 5000)

    def cleanup_before_processing(self):
        """Cleanup before processing"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Close open dialogs
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QDialog) and widget != self:
                widget.close()

        # Clear temporary data
        if hasattr(self, 'yolo_labels'):
            self.yolo_labels.clear()
        if hasattr(self, 'label_colors'):
            self.label_colors.clear()

    def update_progress(self, value, message):
        """Update progress bar and status"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)
            self.progress_bar.setFormat(f"{message}... {value}%")

        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(message)

    def check_yolo_model_ready(self):
        """Check if YOLO model is ready for auto-labeling"""
        model_loaded = self.yolo_model_pt is not None
        processing_done = not (hasattr(self, 'processing_thread') and
                               self.processing_thread and
                               self.processing_thread.isRunning())

        if hasattr(self, 'auto_label_btn'):
            self.auto_label_btn.setEnabled(model_loaded and processing_done)

    # --------------------------
    # Cluster Management
    # --------------------------
    def add_cluster(self, cluster):
        """Add new cluster to the list"""
        if not hasattr(self, 'clusters'):
            self.clusters = []

        self.clusters.append(cluster)
        self.clusters.sort(key=lambda x: len(x), reverse=True)

        if hasattr(self, 'cluster_list'):
            self.cluster_list.clear()
            for i, cluster in enumerate(self.clusters):
                item = QListWidgetItem(f"Cluster {i + 1} ({len(cluster)} images)")
                item.setData(Qt.UserRole, i)
                self.cluster_list.addItem(item)

            if self.cluster_list.count() > 0:
                self.cluster_list.setCurrentRow(0)

    def on_clustering_finished(self):
        """Actions after clustering is finished"""
        self.progress_bar.setFormat("Done! %p%")
        self.status_bar.showMessage("Clustering completed", 5000)

        if not self.clusters:
            QMessageBox.information(
                self, "Information",
                "No clusters found matching your criteria!"
            )

    def show_cluster_images(self, item):
        """Показываем изображения кластера с ленивой загрузкой"""
        if not item or not hasattr(self, 'clusters'):
            return

        try:
            # Отключаем предыдущий обработчик скролла
            if self.scroll_connection is not None:
                scroll_bar = self.cluster_images_area.verticalScrollBar()
                scroll_bar.valueChanged.disconnect(self.scroll_connection)
                self.scroll_connection = None

            # Сбрасываем состояние
            self._set_image_highlight(self.current_image_index, False)
            self.current_image_index = -1
            self.current_loaded = 0
            self.current_page = 0

            self.current_cluster_index = item.data(Qt.UserRole)
            self.current_cluster = self.clusters[self.current_cluster_index]

            self.clear_image_display()

            # Загружаем первую порцию изображений
            self._load_batch_of_images()

            # Настраиваем обработчик прокрутки
            self.scroll_connection = self.cluster_images_area.verticalScrollBar().valueChanged.connect(
                self._handle_scroll_event
            )
        except Exception as e:
            print(f"Error showing cluster: {e}")

    def _load_batch_of_images(self):
        """Загружает порцию изображений"""
        if not hasattr(self, 'current_cluster'):
            return

        start = self.current_loaded
        end = min(start + self.load_batch_size, len(self.current_cluster))

        for i in range(start, end):
            img_path = self.current_cluster[i]
            img_widget = self.create_image_widget(img_path)
            if hasattr(self, 'cluster_images_layout'):
                self.cluster_images_layout.addWidget(img_widget)

                # Добавляем разделитель, если не последний элемент
                if i < end - 1:
                    separator = QFrame()
                    separator.setFrameShape(QFrame.HLine)
                    self.cluster_images_layout.addWidget(separator)
        if hasattr(self, 'delete_btn'):
            self.delete_btn.setEnabled(True)
        self.current_loaded = end
        self.is_loading = False
        print(f"Loaded {end} of {len(self.current_cluster)} images")  # Отладочная информация

    def _handle_scroll_event(self):
        """Обрабатывает событие скролла для подгрузки новых изображений"""
        if self.is_loading:
            return
        scroll_bar = self.cluster_images_area.verticalScrollBar()
        if scroll_bar.value() > scroll_bar.maximum() * 0.8 and self.current_loaded < len(self.current_cluster):
            self._load_batch_of_images()

    def create_image_widget(self, img_path):
        """Create widget for displaying image in cluster"""
        widget = QWidget()
        widget.setObjectName("image_widget")
        widget.setProperty("selected", False)
        widget.setStyleSheet("""
                QWidget#image_widget {
                    background: transparent;
                    border: none;
                    margin: 2px;
                }
                QWidget#image_widget[selected="true"] {
                    border: 2px solid blue;
                    border-radius: 2px;
                }
            """)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)

        # Checkbox
        checkbox = QCheckBox()
        checkbox.setObjectName("image_checkbox")
        checkbox.setProperty("image_path", img_path)
        layout.addWidget(checkbox)

        # Thumbnail - теперь с ленивой загрузкой
        thumbnail_label = ClickableLabel()
        thumbnail_label.setAlignment(Qt.AlignCenter)
        thumbnail_label.setImagePath(img_path)
        thumbnail_label.clicked.connect(lambda: self.show_fullscreen_image(img_path))

        # Устанавливаем плейсхолдер перед загрузкой
        thumbnail_label.setMinimumSize(300, 300)
        thumbnail_label.setText("Loading...")
        layout.addWidget(thumbnail_label)

        # Info panel
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

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        self.cluster_images_layout.addWidget(separator)

        self._load_thumbnail_async(thumbnail_label, img_path)

        return widget

    def _load_thumbnail_async(self, label, img_path):
        """Load thumbnail in background with error handling"""

        def load_image():
            try:
                # Проверяем, существует ли еще виджет
                if not label or not label.parent():
                    return

                # Загружаем изображение
                pixmap = self.load_image_with_yolo_labels(img_path)

                # Проверяем, существует ли еще виджет после долгой загрузки
                if not label or not label.parent():
                    return

                if not pixmap.isNull():
                    label.setPixmap(pixmap.scaled(
                        QSize(300, 300),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    ))
                else:
                    label.setText("Invalid image")
            except Exception as e:
                if label and label.parent():
                    label.setText(f"Load error: {str(e)}")
                print(f"Error loading thumbnail: {e}")

        # Запускаем с небольшой задержкой для приоритизации видимых элементов
        QTimer.singleShot(100, load_image)

    def clear_image_display(self):
        """Очищает область отображения изображений"""
        # Отключаем обработчик скролла
        if self.scroll_connection is not None:
            try:
                scroll_bar = self.cluster_images_area.verticalScrollBar()
                scroll_bar.valueChanged.disconnect(self.scroll_connection)
            except:
                pass
            self.scroll_connection = None

        # Очищаем layout
        while self.cluster_images_layout.count():
            child = self.cluster_images_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.current_loaded = 0
        self.current_page = 0
        self.is_loading = False

    def get_selected_images(self):
        """Get list of selected images in current cluster"""
        selected = []
        for i in range(self.cluster_images_layout.count()):
            widget = self.cluster_images_layout.itemAt(i).widget()
            if widget and widget.objectName() == "image_widget":
                checkbox = widget.findChild(QCheckBox, "image_checkbox")
                if checkbox and checkbox.isChecked():
                    selected.append(checkbox.property("image_path"))
        return selected

    def toggle_all_images(self):
        """Toggle selection of all images in current cluster"""
        if self.current_cluster_index == -1:
            return

        # Check if any images are unchecked
        has_unchecked = any(
            widget.findChild(QCheckBox, "image_checkbox").isChecked() == False
            for widget in self.get_image_widgets()
        )

        # Set new state based on unchecked images
        new_state = has_unchecked

        for widget in self.get_image_widgets():
            checkbox = widget.findChild(QCheckBox, "image_checkbox")
            checkbox.setChecked(new_state)

    def get_image_widgets(self):
        """Get all image widgets in current cluster"""
        widgets = []
        for i in range(self.cluster_images_layout.count()):
            widget = self.cluster_images_layout.itemAt(i).widget()
            if widget and widget.objectName() == "image_widget":
                widgets.append(widget)
        return widgets

    def update_similarity_preset(self, index):
        """Update similarity threshold from preset"""
        presets = [2, 5, 10]
        self.threshold_input.setText(str(presets[index]))

    # --------------------------
    # Image Operations
    # --------------------------
    @pyqtSlot(str)
    def show_fullscreen_image(self, img_path):
        """Show image in fullscreen dialog"""
        try:
            if not os.path.exists(img_path):
                QMessageBox.warning(self, "Error", "Image file not found!")
                return

            self.cleanup_before_processing()

            labels = self.get_yolo_labels(img_path)

            if img_path not in self.label_colors:
                self.label_colors[img_path] = [
                    (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    for _ in range(len(labels))
                ]

            dialog = FullScreenImageDialog(
                img_path,
                labels,
                self.label_colors[img_path],
                self,
                yolo_model=self.yolo_model_pt,
                yolo_img_w=int(self.img_w_input.text()),
                yolo_img_h=int(self.img_h_input.text()),
                yolo_conf=int(self.conf_input.text()) / 100,
                yolo_iou=int(self.iou_input.text()) / 100,
            )

            dialog.labels_changed.connect(lambda: self.update_cluster_display(img_path))

            result = dialog.exec_()

            if result == QDialog.Accepted:
                self.save_yolo_labels(img_path, dialog.yolo_labels)
                self.yolo_labels[img_path] = dialog.yolo_labels
                self.label_colors[img_path] = dialog.colors
                self.update_cluster_display(img_path)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot show image: {str(e)}")
            print(f"Error showing image: {traceback.format_exc()}")

    def update_cluster_display(self, img_path):
        """Update cluster display after label changes"""
        current_item = self.cluster_list.currentItem()
        if current_item:
            self.clear_image_display()
            cluster = self.clusters[self.current_cluster_index]

            for img_path in cluster:
                img_widget = self.create_image_widget(img_path)
                self.cluster_images_layout.addWidget(img_widget)

            for i in range(self.cluster_images_layout.count()):
                if i % 2 != 0:
                    separator = QFrame()
                    separator.setFrameShape(QFrame.HLine)
                    self.cluster_images_layout.insertWidget(i, separator)

            self.cluster_images_widget.update()
            self.cluster_images_area.viewport().update()

    def load_image_with_yolo_labels(self, img_path):
        """Load image with YOLO labels drawn"""
        if not os.path.exists(img_path):
            return QPixmap()

        try:
            with Image.open(img_path) as img:
                if max(img.size) > 2048:
                    img.thumbnail((2048, 2048))
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

    def get_yolo_labels(self, img_path):
        """Get YOLO labels for image"""
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
        """Save YOLO labels to file"""
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

    # --------------------------
    # Delete Operations
    # --------------------------
    def delete_selected_images(self):
        """Delete selected images from current cluster"""
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

    def delete_current_cluster_duplicates(self):
        """Delete all duplicates in current cluster, keeping one image"""
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

        # Keep first image, delete others
        image_to_keep = cluster[0]
        deleted_count = 0

        for img_path in cluster[1:]:
            try:
                os.remove(img_path)

                txt_path = os.path.splitext(img_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    os.remove(txt_path)

                if img_path in self.yolo_labels:
                    del self.yolo_labels[img_path]
                if img_path in self.label_colors:
                    del self.label_colors[img_path]

                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {img_path}: {e}")

        # Update cluster - keep only one image
        self.clusters[self.current_cluster_index] = [image_to_keep]

        # Update UI
        current_item = self.cluster_list.currentItem()
        current_item.setText(f"Cluster {self.current_cluster_index + 1} (1 image)")
        self.show_cluster_images(current_item)

        QMessageBox.information(
            self, "Operation Complete",
            f"Deleted {deleted_count} duplicate images.\n"
            f"Kept 1 unique image in cluster."
        )

    def run_auto_labeling(self):
        """Run auto-labeling on unlabeled images"""
        if not self.yolo_model_pt:
            QMessageBox.warning(self, "Warning", "YOLO model not loaded!")
            return

        # Find unlabeled images
        unlabeled_images = []
        for cluster in self.clusters:
            for img_path in cluster:
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

    # --------------------------
    # Hot Keys Operations
    # --------------------------
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

    def get_current_selected_image(self):
        """Получить текущее выделенное изображение (с синей рамкой)"""
        if self.current_cluster_index == -1 or self.current_image_index == -1:
            return None

        try:
            cluster = self.clusters[self.current_cluster_index]
            if 0 <= self.current_image_index < len(cluster):
                return cluster[self.current_image_index]
        except:
            pass
        return None

    def next_cluster(self):
        """Перейти к следующему кластеру"""
        if not hasattr(self, 'cluster_list') or self.cluster_list.count() == 0:
            return

        current_row = self.cluster_list.currentRow()
        if current_row < self.cluster_list.count() - 1:
            self.cluster_list.setCurrentRow(current_row + 1)
            self.current_image_index = -1  # Сброс выделения изображения
            self.show_cluster_images(self.cluster_list.currentItem())

    def prev_cluster(self):
        """Перейти к предыдущему кластеру"""
        if not hasattr(self, 'cluster_list') or self.cluster_list.count() == 0:
            return

        current_row = self.cluster_list.currentRow()
        if current_row > 0:
            self.cluster_list.setCurrentRow(current_row - 1)
            self.current_image_index = -1  # Сброс выделения изображения
            self.show_cluster_images(self.cluster_list.currentItem())

    def next_image(self):
        """Выделить следующее изображение в текущем кластере"""
        if self.current_cluster_index == -1:
            return

        cluster = self.clusters[self.current_cluster_index]
        if not cluster:
            return

        # Снимаем выделение с текущего изображения
        self._set_image_highlight(self.current_image_index, False)

        # Переходим к следующему
        if self.current_image_index < len(cluster) - 1:
            self.current_image_index += 1
        else:
            self.current_image_index = 0

        # Устанавливаем выделение на новое изображение
        self._set_image_highlight(self.current_image_index, True)

        # Прокручиваем к выделенному изображению
        self._scroll_to_image(self.current_image_index)

    def prev_image(self):
        """Выделить предыдущее изображение в текущем кластере"""
        if self.current_cluster_index == -1:
            return

        cluster = self.clusters[self.current_cluster_index]
        if not cluster:
            return

        # Снимаем выделение с текущего изображения
        self._set_image_highlight(self.current_image_index, False)

        # Переходим к предыдущему
        if self.current_image_index > 0:
            self.current_image_index -= 1
        else:
            self.current_image_index = len(cluster) - 1

        # Устанавливаем выделение на новое изображение
        self._set_image_highlight(self.current_image_index, True)

        # Прокручиваем к выделенному изображению
        self._scroll_to_image(self.current_image_index)

    def toggle_current_image(self):
        """Переключить чекбокс текущего выделенного изображения"""
        if self.current_cluster_index == -1 or self.current_image_index == -1:
            return

        # Находим виджет изображения
        widget = self._get_image_widget_at_index(self.current_image_index)
        if widget:
            checkbox = widget.findChild(QCheckBox, "image_checkbox")
            if checkbox:
                checkbox.setChecked(not checkbox.isChecked())

    def _set_image_highlight(self, index, highlight):
        """Установить или снять выделение (синюю рамку) с изображения"""
        widget = self._get_image_widget_at_index(index)
        if widget:
            widget.setProperty("selected", highlight)
            widget.setStyle(widget.style())

    def _get_image_widget_at_index(self, index):
        """Получить виджет изображения по индексу"""
        if index == -1:
            return None

        # Считаем только виджеты изображений (пропускаем разделители)
        image_widgets = []
        for i in range(self.cluster_images_layout.count()):
            item = self.cluster_images_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget and widget.objectName() == "image_widget":
                    image_widgets.append(widget)

        if 0 <= index < len(image_widgets):
            return image_widgets[index]
        return None

    def _scroll_to_image(self, index):
        """Прокрутить к изображению с указанным индексом"""
        widget = self._get_image_widget_at_index(index)
        if widget:
            self.cluster_images_area.ensureWidgetVisible(widget)


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    app = QApplication(sys.argv)
    DarkTheme.apply(app)

    # Platform-specific font settings
    font = QFont()
    if platform.system() == "Windows":
        font.setFamily("Segoe UI")
    elif platform.system() == "Darwin":  # macOS
        font.setFamily("Helvetica")
    else:  # Linux
        font.setFamily("DejaVu Sans")

    font.setPointSize(10)
    app.setFont(font)

    window = IDEMainWindow()
    window.show()
    sys.exit(app.exec_())