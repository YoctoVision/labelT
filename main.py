import gc
import os
import platform
import sys
import signal
import random
import traceback
import yaml
from pathlib import Path

from PyQt5 import QtCore, QtGui

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QListWidget, QScrollArea, QGroupBox,
    QListWidgetItem, QMessageBox, QCheckBox, QProgressBar,
    QFrame, QComboBox, QDialog, QLineEdit, QDesktopWidget,
    QGridLayout, QTabWidget, QTreeView,
    QFileSystemModel, QStatusBar, QToolBar, QAction, QDockWidget, QMenu
)
from PyQt5.QtCore import Qt, QSize, pyqtSlot, QDir, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QIntValidator, QIcon

from support import ClickableLabel, FullScreenImageDialog, ImageProcessingThread, DarkTheme

from utils import get_dominant_color, get_contrast_color, get_label_txt


class IDEMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 平台特定初始化
        self._init_platform_settings()

        # 窗口设置
        self.setWindowTitle("YOLO视觉标注器IDE")
        self.set_window_geometry_by_ratio(width_ratio=0.75, height_ratio=0.8)

        # 初始化实例变量
        self._initialize_variables()

        # 界面设置
        self.init_ui()
        self.setup_connections()


    def set_window_geometry_by_ratio(self, width_ratio=0.75, height_ratio=0.8):
        """按屏幕比例设置窗口几何"""
        screen = QDesktopWidget().screenGeometry()
        width = int(screen.width() * width_ratio)
        height = int(screen.height() * height_ratio)
        x = (screen.width() - width) // 2  # 水平居中
        y = (screen.height() - height) // 2  # 垂直居中
        self.setGeometry(x, y, width, height)

    # --------------------------
    # 初始化方法
    # --------------------------
    def _initialize_variables(self):
        """初始化所有实例变量"""
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
        self.all_label_colors = {}
        self.current_image_index = -1
        self.images_per_page = 30  # 初始加载的图像数量
        self.load_batch_size = 20  # 滚动时加载的图像数量
        self.current_loaded = 0    # 已加载的图像数量
        self.current_page = 0
        self.is_loading = False
        self.scroll_connection = None  # 用于存储滚动信号的连接

    def init_ui(self):
        """初始化所有界面组件"""
        # 设置中心显示区域（用于显示聚类）
        self._create_central_display_area()

        # 创建界面组件
        self.create_toolbar()
        self.create_status_bar()
        self.create_dock_widgets()
        self.create_main_tab_dock()  # 创建dock的更名方法

        self.set_styles()

    def create_toolbar(self):
        """创建主工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(16, 16))

        if self.os_name == "Darwin":
            toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        else:
            toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)

        actions = [
            ("document-open", "打开文件夹", self.browse_folder),
            ("system-run", "处理图像", self.process_images),
            None,  # 分隔符
            ("applications-science", "加载YOLO模型", self.browse_yolo_model_file),
            ("document-edit", "自动标注", self.run_auto_labeling),
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
        """获取平台规范化的路径"""
        return Path(path).as_posix() if self.os_name != "Windows" else os.path.normpath(path)

    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def create_dock_widgets(self):
        """创建文件树和聚类列表的dock窗口"""
        # 文件树dock
        self.file_dock = QDockWidget("项目", self)
        self.file_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("")
        self.file_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files | QDir.Hidden)

        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(""))

        self.file_tree.header().setStretchLastSection(False)  # 禁用最后一列自动拉伸
        self.file_tree.setColumnWidth(0, int(self.width() * 0.8))
        self.file_tree.setColumnWidth(1, int(self.width() * 0.1))
        self.file_tree.setColumnWidth(3, int(self.width() * 0.1))

        self.file_dock.setWidget(self.file_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)
        self.file_dock.setMinimumHeight(300)

        # 聚类列表dock
        self.cluster_dock = QDockWidget("聚类", self)
        self.cluster_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.cluster_list = QListWidget()

        self.cluster_dock.setWidget(self.cluster_list)
        self.addDockWidget(Qt.RightDockWidgetArea, self.cluster_dock)

        self.resizeDocks(
            [self.file_dock, self.cluster_dock],
            [int(self.width() * 0.3), int(self.width() * 0.2)],
            Qt.Horizontal
        )

    def create_main_tab_dock(self):
        """创建主控件作为dock窗口"""
        # 创建mainTab的dock窗口
        main_dock = QDockWidget("主控件", self)
        main_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)

        main_tab = QWidget()
        main_tab.setObjectName("mainTab")
        layout = QVBoxLayout(main_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 添加界面组件（不包含_create_image_display_area）
        self._create_yolo_settings_group(layout)
        self._create_similarity_group(layout)
        self._create_options_group(layout)
        self._create_process_buttons(layout)
        self._create_progress_bars(layout)

        main_dock.setWidget(main_tab)
        self.addDockWidget(Qt.LeftDockWidgetArea, main_dock)

        # 将dock放置在左侧底部
        self.splitDockWidget(self.file_dock, main_dock, Qt.Vertical)
        self.resizeDocks(
            [self.file_dock, main_dock],
            [int(self.height() * 0.6), int(self.height() * 0.4)],  # 目标高度列表
            Qt.Vertical  # 调整方向
        )

    def _create_central_display_area(self):
        """创建中心图像显示区域"""
        self.image_tabs = QTabWidget()
        self.image_tabs.setTabsClosable(True)
        self.image_tabs.tabCloseRequested.connect(self.close_image_tab)

        # 主聚类视图选项卡
        self.cluster_tab = QWidget()
        self.cluster_tab_layout = QVBoxLayout(self.cluster_tab)
        self.cluster_tab_layout.setContentsMargins(0, 0, 0, 0)

        # 聚类图像滚动区域 - 居中显示
        self.cluster_images_area = QScrollArea()
        self.cluster_images_area.setWidgetResizable(True)

        # 居中容器
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

        # 聚类控制按钮
        cluster_btn_layout = QHBoxLayout()
        cluster_btn_layout.setContentsMargins(5, 5, 5, 5)

        self.select_all_btn = QPushButton("全选 (A)")
        cluster_btn_layout.addWidget(self.select_all_btn)

        self.delete_btn = QPushButton("删除选中 (D)")
        cluster_btn_layout.addWidget(self.delete_btn)

        self.delete_cluster_duplicates_btn = QPushButton("删除聚类重复 (X)")
        cluster_btn_layout.addWidget(self.delete_cluster_duplicates_btn)

        self.cluster_tab_layout.addLayout(cluster_btn_layout)

        # 添加选项卡到图像选项卡
        self.image_tabs.addTab(self.cluster_tab, "聚类")

        # 将image_tabs设置为中心窗口部件
        self.setCentralWidget(self.image_tabs)

    def _create_yolo_settings_group(self, parent_layout):
        """创建YOLO设置组框"""
        self.yolo_settings_group = QGroupBox("YOLO设置")
        self.yolo_settings_group.setCheckable(True)
        self.yolo_settings_group.setChecked(False)
        self.yolo_settings_group.toggled.connect(self.toggle_yolo_settings)

        layout = QGridLayout()
        layout.setContentsMargins(5, 15, 5, 5)

        # 模型选择
        self.yolo_model_label = QLabel("未选择模型")
        self.yolo_model_label.setWordWrap(True)
        yolo_model_btn = QPushButton("浏览...")
        yolo_model_btn.setObjectName("yoloModelButton")
        yolo_model_btn.setMaximumWidth(100)

        # 参数
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

        # 将控件添加到布局
        layout.addWidget(QLabel("模型："), 0, 0)
        layout.addWidget(self.yolo_model_label, 0, 1)
        layout.addWidget(yolo_model_btn, 0, 2)

        layout.addWidget(QLabel("置信度 (%)："), 1, 0)
        layout.addWidget(self.conf_input, 1, 1)

        layout.addWidget(QLabel("图像宽度："), 2, 0)
        layout.addWidget(self.img_w_input, 2, 1)

        layout.addWidget(QLabel("图像高度："), 3, 0)
        layout.addWidget(self.img_h_input, 3, 1)

        layout.addWidget(QLabel("IOU阈值 (%)："), 4, 0)
        layout.addWidget(self.iou_input, 4, 1)

        self.yolo_settings_group.setLayout(layout)
        parent_layout.addWidget(self.yolo_settings_group)

    def _create_similarity_group(self, parent_layout):
        """创建相似性设置组框"""
        self.similarity_group = QGroupBox("相似性设置")
        self.similarity_group.setCheckable(True)
        self.similarity_group.setChecked(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 15, 5, 5)

        # 哈希方法
        hash_layout = QHBoxLayout()
        self.hash_combo = QComboBox()
        self.hash_combo.addItems(["average_hash", "phash", "dhash"])
        hash_layout.addWidget(QLabel("哈希方法："))
        hash_layout.addWidget(self.hash_combo)
        layout.addLayout(hash_layout)

        # 阈值
        threshold_layout = QHBoxLayout()
        self.threshold_input = QLineEdit("5")
        self.threshold_input.setValidator(QIntValidator(0, 64))
        self.threshold_input.setMaximumWidth(40)
        threshold_layout.addWidget(QLabel("阈值 (0-64)："))
        threshold_layout.addWidget(self.threshold_input)
        # layout.addLayout(threshold_layout)

        # 预设
        preset_layout = QHBoxLayout()
        self.similarity_preset = QComboBox()
        self.similarity_preset.addItems(["严格 (2)", "正常 (5)", "宽松 (10)"])
        self.similarity_preset.setCurrentIndex(1)
        preset_layout.addWidget(QLabel("预设："))
        preset_layout.addWidget(self.similarity_preset)
        layout.addLayout(preset_layout)

        self.similarity_group.setLayout(layout)
        parent_layout.addWidget(self.similarity_group)

    def _create_options_group(self, parent_layout):
        """创建选项组框"""
        self.options_group = QGroupBox("选项")
        self.options_group.setCheckable(True)
        self.options_group.setChecked(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 15, 5, 5)

        self.skip_single_check = QCheckBox("跳过单图像聚类")
        self.skip_single_check.setChecked(False)
        # layout.addWidget(self.skip_single_check)

        self.yolo_labeling_check = QCheckBox("显示YOLO标注")
        layout.addWidget(self.yolo_labeling_check)

        self.options_group.setLayout(layout)
        parent_layout.addWidget(self.options_group)

    def _create_process_buttons(self, parent_layout):
        """创建处理按钮布局"""
        button_layout = QHBoxLayout()

        self.process_btn = QPushButton("处理图像")
        self.process_btn.setObjectName("processButton")
        button_layout.addWidget(self.process_btn)

        self.auto_label_btn = QPushButton("自动标注")
        self.auto_label_btn.setObjectName("autoLabelButton")
        self.auto_label_btn.setEnabled(False)
        # button_layout.addWidget(self.auto_label_btn)

        parent_layout.addLayout(button_layout)

    def _create_progress_bars(self, parent_layout):
        """创建进度条"""
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
        """创建图像显示区域，包含选项卡作为中心窗口部件"""
        self.image_tabs = QTabWidget()
        self.image_tabs.setTabsClosable(True)
        self.image_tabs.tabCloseRequested.connect(self.close_image_tab)

        # 主聚类视图选项卡
        self.cluster_tab = QWidget()
        self.cluster_tab_layout = QVBoxLayout(self.cluster_tab)
        self.cluster_tab_layout.setContentsMargins(0, 0, 0, 0)

        # 聚类图像滚动区域 - 居中显示
        self.cluster_images_area = QScrollArea()
        self.cluster_images_area.setWidgetResizable(True)

        # 居中容器
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

        # 聚类控制按钮
        cluster_btn_layout = QHBoxLayout()
        cluster_btn_layout.setContentsMargins(5, 5, 5, 5)

        self.select_all_btn = QPushButton("全选 (A)")
        cluster_btn_layout.addWidget(self.select_all_btn)

        self.delete_btn = QPushButton("删除选中 (D)")
        cluster_btn_layout.addWidget(self.delete_btn)

        self.delete_cluster_duplicates_btn = QPushButton("删除聚类重复 (X)")
        cluster_btn_layout.addWidget(self.delete_cluster_duplicates_btn)

        self.cluster_tab_layout.addLayout(cluster_btn_layout)

        # 添加选项卡到图像选项卡
        self.image_tabs.addTab(self.cluster_tab, "聚类")

        # 将image_tabs设置为中心窗口部件
        self.setCentralWidget(self.image_tabs)

    # --------------------------
    # 界面实用方法
    # --------------------------
    def set_styles(self):
        """设置界面元素的样式"""
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
        """切换YOLO设置面板的可见性"""
        self.yolo_settings_group.setTitle(f"YOLO设置 {'▼' if checked else '▶'}")

    def reset_ui(self):
        """将界面重置为初始状态"""
        if hasattr(self, 'cluster_list'):
            self.cluster_list.clear()

        self.clear_image_display()

        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)

        self.yolo_labels = {}
        self.label_colors = {}
        self.all_label_colors = {}
        self.clusters = []
        self.current_cluster_index = -1

        if hasattr(self, 'delete_btn'):
            self.delete_btn.setEnabled(False)

        self.file_dock.setWindowTitle(f"{self.file_dock.windowTitle().split(':')[0]}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # --------------------------
    # 事件处理程序
    # --------------------------
    def setup_connections(self):
        """设置信号-槽连接"""
        # YOLO设置
        yolo_model_btn = self.yolo_settings_group.findChild(QPushButton, "yoloModelButton")
        if yolo_model_btn:
            yolo_model_btn.clicked.connect(self.browse_yolo_model_file)

        # 主按钮
        self.process_btn.clicked.connect(self.process_images)
        self.auto_label_btn.clicked.connect(self.run_auto_labeling)

        # 聚类列表
        self.cluster_list.itemClicked.connect(self.show_cluster_images)

        # 聚类控制按钮
        self.select_all_btn.clicked.connect(self.toggle_all_images)
        self.delete_btn.clicked.connect(self.delete_selected_images)
        self.delete_cluster_duplicates_btn.clicked.connect(self.delete_current_cluster_duplicates)

        # 相似性预设下拉框
        self.similarity_preset.currentIndexChanged.connect(self.update_similarity_preset)

        # 文件树双击
        self.file_tree.doubleClicked.connect(self.on_file_double_clicked)
        self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.show_context_menu)

    def on_file_double_clicked(self, index):
        """处理文件树中的双击事件"""
        file_path = self.file_model.filePath(index)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            self.show_fullscreen_image(file_path)

    def show_context_menu(self, position):
        # 获取点击的元素索引
        index = self.file_tree.indexAt(position)

        if index.isValid():
            # 创建上下文菜单
            context_menu = QMenu(self)

            # 添加菜单项
            action1 = context_menu.addAction("作为数据集打开")
            action2 = context_menu.addAction("添加为YOLO模型")

            # 在点击位置显示菜单
            action = context_menu.exec_(self.file_tree.viewport().mapToGlobal(position))
            file_path = self.file_model.filePath(index)
            # 处理选中的菜单项
            if action == action1:
                self._open_folder(file_path)
            elif action == action2:
                self._open_yolo_model(file_path)

    def close_image_tab(self, index):
        """关闭图像选项卡（除了主聚类选项卡）"""
        if index != 0:
            self.image_tabs.removeTab(index)

    def close_tab(self, index):
        """关闭选项卡（除了主选项卡）"""
        if index != 0:
            self.tab_widget.removeTab(index)

    # --------------------------
    # 文件操作
    # --------------------------
    def browse_folder(self):
        """浏览图像文件夹"""
        try:
            # 清空之前的数据
            self.reset_ui()

            # 打开文件夹选择对话框
            folder = QFileDialog.getExistingDirectory(
                self,
                "选择图像文件夹",
                "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )

            if folder:
                self._open_folder(folder)
        except Exception as e:
            error_msg = f"加载文件夹出错：{str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "错误", error_msg)

    def dragEnterEvent(self, event):
        """仅接受文件夹拖入"""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):  # 关键判断：仅当拖入的是文件夹时才接受
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        """处理拖放事件"""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                self._open_folder(path)
            elif os.path.isfile(path) and path.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif')):
                # 如果是图片，则打开其所在文件夹
                self._open_folder(os.path.dirname(path))

    def _open_folder(self, folder_path):
        self.image_folder = self.get_normalized_path(folder_path)
        self.file_dock.setWindowTitle(f"{self.file_dock.windowTitle()}: {os.path.basename(self.image_folder)}")

        # 更新文件树
        self.file_model.setRootPath(self.image_folder)
        self.file_tree.setRootIndex(self.file_model.index(self.image_folder))

        # 清空内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 启用处理按钮
        self.process_btn.setEnabled(True)


    def browse_yolo_model_file(self):
        """浏览YOLO模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择YOLO模型文件",
            "",
            "YOLO模型文件 (*.pt)"
        )

        if file_path:
            self._open_yolo_model(file_path)

    def _open_yolo_model(self, file_path):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            normalized_path = self.get_normalized_path(file_path)

            print(f"正在从以下路径加载模型：{normalized_path}")

            if not os.path.exists(normalized_path):
                raise FileNotFoundError(f"在以下路径未找到模型文件：{normalized_path}")

            # 加载模型
            self.yolo_model_pt = YOLO(normalized_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo_model_pt.to(device)

            self.yolo_model = normalized_path
            self.yolo_model_label.setText(os.path.basename(normalized_path))
            self.reset_ui()

        except Exception as e:
            QMessageBox.critical(
                self, "错误",
                f"加载YOLO模型失败：\n{str(e)}\n路径：{normalized_path}"
            )
            self.yolo_model_pt = None
            self.yolo_model = ""
            self.yolo_model_label.setText("未选择YOLO模型")

        finally:
            QApplication.restoreOverrideCursor()

    # --------------------------
    # 图像处理
    # --------------------------
    def process_images(self):
        """处理图像进行聚类"""
        try:
            # 验证文件夹
            if not hasattr(self, 'image_folder') or not self.image_folder:
                QMessageBox.warning(self, "警告", "请先选择一个文件夹！")
                return

            # 处理前清理
            self.cleanup_before_processing()

            # 验证阈值
            try:
                threshold = int(self.threshold_input.text())
                if not 0 <= threshold <= 64:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "警告", "请输入有效的阈值（0-64）")
                return

            # 取消之前的处理
            if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.canceled = True
                self.processing_thread.wait()

            # 准备界面
            self.reset_ui()
            self.progress_bar.setFormat("准备中... %p%")
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("开始处理图像...")

            # 创建处理线程
            self.processing_thread = ImageProcessingThread(
                self.image_folder,
                threshold,
                self.skip_single_check.isChecked(),
                self.hash_combo.currentText()
            )

            # 连接信号
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.cluster_found.connect(self.add_cluster)
            self.processing_thread.finished_clustering.connect(self.on_clustering_finished)
            self.processing_thread.finished_clustering.connect(self.check_yolo_model_ready)

            # 启动线程
            self.processing_thread.start()

        except Exception as e:
            error_msg = f"处理失败：{str(e)}"
            QMessageBox.critical(self, "错误", error_msg)
            print(f"处理错误：{traceback.format_exc()}")
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(error_msg, 5000)

    def cleanup_before_processing(self):
        """处理前清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 关闭打开的对话框
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QDialog) and widget != self:
                widget.close()

        # 清空临时数据
        if hasattr(self, 'yolo_labels'):
            self.yolo_labels.clear()
        if hasattr(self, 'label_colors'):
            self.label_colors.clear()

    def update_progress(self, value, message):
        """更新进度条和状态"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)
            self.progress_bar.setFormat(f"{message}... {value}%")

        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(message)

    def check_yolo_model_ready(self):
        """检查YOLO模型是否准备好进行自动标注"""
        model_loaded = self.yolo_model_pt is not None
        processing_done = not (hasattr(self, 'processing_thread') and
                               self.processing_thread and
                               self.processing_thread.isRunning())

        if hasattr(self, 'auto_label_btn'):
            self.auto_label_btn.setEnabled(model_loaded and processing_done)

    # --------------------------
    # 聚类管理
    # --------------------------
    def add_cluster(self, cluster):
        """将新聚类添加到列表"""
        if not hasattr(self, 'clusters'):
            self.clusters = []

        self.clusters.append(cluster)
        self.clusters.sort(key=lambda x: len(x), reverse=True)

        if hasattr(self, 'cluster_list'):
            self.cluster_list.clear()
            for i, cluster in enumerate(self.clusters):
                item = QListWidgetItem(f"聚类 {i + 1} ({len(cluster)} 张图像)")
                item.setData(Qt.UserRole, i)
                self.cluster_list.addItem(item)

            if self.cluster_list.count() > 0:
                self.cluster_list.setCurrentRow(0)

    def on_clustering_finished(self):
        """聚类完成后的操作"""
        self.progress_bar.setFormat("完成！ %p%")
        self.status_bar.showMessage("聚类完成", 5000)

        if not self.clusters:
            QMessageBox.information(
                self, "信息",
                "未找到符合您标准的聚类！"
            )

    def show_cluster_images(self, item):
        """显示聚类中的图像，支持懒加载"""
        if not item or not hasattr(self, 'clusters'):
            return

        try:
            # 断开之前的滚动处理程序
            if self.scroll_connection is not None:
                scroll_bar = self.cluster_images_area.verticalScrollBar()
                scroll_bar.valueChanged.disconnect(self.scroll_connection)
                self.scroll_connection = None

            # 重置状态
            self._set_image_highlight(self.current_image_index, False)
            self.current_image_index = -1
            self.current_loaded = 0
            self.current_page = 0

            self.current_cluster_index = item.data(Qt.UserRole)
            self.current_cluster = self.clusters[self.current_cluster_index]

            self.clear_image_display()

            # 加载第一批图像
            self._load_batch_of_images()

            # 设置滚动事件处理程序
            self.scroll_connection = self.cluster_images_area.verticalScrollBar().valueChanged.connect(
                self._handle_scroll_event
            )
        except Exception as e:
            print(f"显示聚类时出错：{e}")

    def _load_batch_of_images(self):
        """加载一批图像"""
        if not hasattr(self, 'current_cluster'):
            return

        start = self.current_loaded
        end = min(start + self.load_batch_size, len(self.current_cluster))

        for i in range(start, end):
            img_path = self.current_cluster[i]
            img_widget = self.create_image_widget(img_path)
            if hasattr(self, 'cluster_images_layout'):
                self.cluster_images_layout.addWidget(img_widget)

                # 如果不是最后一个元素，添加分隔线
                if i < end - 1:
                    separator = QFrame()
                    separator.setFrameShape(QFrame.HLine)
                    self.cluster_images_layout.addWidget(separator)
        if hasattr(self, 'delete_btn'):
            self.delete_btn.setEnabled(True)
        self.current_loaded = end
        self.is_loading = False
        print(f"已加载 {end} 张图像，共 {len(self.current_cluster)} 张")  # 调试信息

    def _handle_scroll_event(self):
        """处理滚动事件以加载新图像"""
        if self.is_loading:
            return
        scroll_bar = self.cluster_images_area.verticalScrollBar()
        if scroll_bar.value() > scroll_bar.maximum() * 0.8 and self.current_loaded < len(self.current_cluster):
            self._load_batch_of_images()

    def create_image_widget(self, img_path):
        """为聚类中的图像创建显示控件"""
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

        # 复选框
        checkbox = QCheckBox()
        checkbox.setObjectName("image_checkbox")
        checkbox.setProperty("image_path", img_path)
        layout.addWidget(checkbox)

        # 缩略图 - 支持懒加载
        thumbnail_label = ClickableLabel()
        thumbnail_label.setAlignment(Qt.AlignCenter)
        thumbnail_label.setImagePath(img_path)
        thumbnail_label.clicked.connect(lambda: self.show_fullscreen_image(img_path))

        # 在加载前设置占位符
        thumbnail_label.setMinimumSize(300, 300)
        thumbnail_label.setText("加载中...")
        layout.addWidget(thumbnail_label)

        # 信息面板
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
                labels_text = "\n".join([f"类别：{l[0]}" for l in labels])
                labels_label = QLabel(f"YOLO标注 ({len(labels)})：\n{labels_text}")
                labels_label.setWordWrap(True)
                info_layout.addWidget(labels_label)

        layout.addWidget(info_widget)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        self.cluster_images_layout.addWidget(separator)

        self._load_thumbnail_async(thumbnail_label, img_path)

        return widget

    def _load_thumbnail_async(self, label, img_path):
        """在后台加载缩略图，包含错误处理"""

        def load_image():
            try:
                # 检查控件是否仍然存在
                if not label or not label.parent():
                    return

                # 加载图像
                pixmap = self.load_image_with_yolo_labels(img_path)

                # 检查长时间加载后控件是否仍然存在
                if not label or not label.parent():
                    return

                if not pixmap.isNull():
                    label.setPixmap(pixmap.scaled(
                        QSize(300, 300),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    ))
                else:
                    label.setText("无效图像")
            except Exception as e:
                if label and label.parent():
                    label.setText(f"加载错误：{str(e)}")
                print(f"加载缩略图出错：{e}")

        # 以短暂延迟启动以优先加载可见元素
        QTimer.singleShot(100, load_image)

    def clear_image_display(self):
        """清空图像显示区域"""
        # 断开滚动处理程序
        if self.scroll_connection is not None:
            try:
                scroll_bar = self.cluster_images_area.verticalScrollBar()
                scroll_bar.valueChanged.disconnect(self.scroll_connection)
            except:
                pass
            self.scroll_connection = None

        # 清空布局
        while self.cluster_images_layout.count():
            child = self.cluster_images_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.current_loaded = 0
        self.current_page = 0
        self.is_loading = False

    def get_selected_images(self):
        """获取当前聚类中选中的图像列表"""
        selected = []
        for i in range(self.cluster_images_layout.count()):
            widget = self.cluster_images_layout.itemAt(i).widget()
            if widget and widget.objectName() == "image_widget":
                checkbox = widget.findChild(QCheckBox, "image_checkbox")
                if checkbox and checkbox.isChecked():
                    selected.append(checkbox.property("image_path"))
        return selected

    def toggle_all_images(self):
        """切换当前聚类中所有图像的选中状态"""
        if self.current_cluster_index == -1:
            return

        # 检查是否有未选中的图像
        has_unchecked = any(
            widget.findChild(QCheckBox, "image_checkbox").isChecked() == False
            for widget in self.get_image_widgets()
        )

        # 根据未选中图像设置新状态
        new_state = has_unchecked

        for widget in self.get_image_widgets():
            checkbox = widget.findChild(QCheckBox, "image_checkbox")
            checkbox.setChecked(new_state)

    def get_image_widgets(self):
        """获取当前聚类中的所有图像控件"""
        widgets = []
        for i in range(self.cluster_images_layout.count()):
            widget = self.cluster_images_layout.itemAt(i).widget()
            if widget and widget.objectName() == "image_widget":
                widgets.append(widget)
        return widgets

    def update_similarity_preset(self, index):
        """从预设更新相似性阈值"""
        presets = [2, 5, 10]
        self.threshold_input.setText(str(presets[index]))

    # --------------------------
    # 图像操作, 我们要干的的都在这里
    # @modified by leafan @20250609.
    # --------------------------
    @pyqtSlot(str)
    def show_fullscreen_image(self, img_path):
        """在全屏对话框中显示图像"""
        try:
            if not os.path.exists(img_path):
                QMessageBox.warning(self, "错误", "图像文件未找到！")
                return

            self.cleanup_before_processing()

            labels = self.get_yolo_labels(img_path)
            classes = self.get_yolo_classes()

            # 设置标记颜色,尽量与背景色反着来
            bg_color = get_dominant_color(img_path)  # 需要实现获取主色函数
            base_colors = [
                (255, 0, 0),    # 🔴 红色
                (0, 255, 0),    # 🟢 绿色
                (0, 0, 255),    # 🔵 蓝色
                (255, 255, 0),  # 💛 黄色
                (255, 0, 255),  # 🟣 品红
                (255, 255, 255) # ⬜ 白色
            ]
            self.all_label_colors[img_path] = [
                # 第一个取背景色的反色, 如果多余1个标记, 从默认选项里面取
                get_contrast_color(bg_color) if i < 1 else base_colors[i % len(base_colors)]
                for i in range(len(classes)//2)# 一张图不需要全部标签
            ]

            self.label_colors[img_path] = [
                self.all_label_colors[img_path][i%len(self.all_label_colors[img_path])]
                for i in range(len(labels)) 
            ]

            # print(f"bg_color: {bg_color}, self.all_label_colors: {self.all_label_colors}, self.label_colors: {self.label_colors}")

            dialog = FullScreenImageDialog(
                img_path,
                labels,
                self.all_label_colors[img_path],
                self.label_colors[img_path],
                self,
                classes,
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
            QMessageBox.critical(self, "错误", f"无法显示图像：{str(e)}")
            print(f"显示图像出错：{traceback.format_exc()}")

    def update_cluster_display(self, img_path):
        """在标签更改后更新聚类显示"""
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
        """加载带有YOLO标签的图像"""
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
            print(f"加载图像 {img_path} 时出错：{e}")
            return QPixmap()

    def get_yolo_labels(self, img_path):
        """获取图像的YOLO标签"""
        if img_path in self.yolo_labels:
            return self.yolo_labels[img_path]

        txt_path = get_label_txt(img_path)
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
                print(f"读取YOLO标签出错：{e}")

        self.yolo_labels[img_path] = labels
        return labels
    

    def get_yolo_classes(self, classes_path="./data.yaml"):
        """获取YOLO的类别
        
        Args:
            classes_path (str): YAML配置文件路径，默认为"./data.yaml"
        Returns:
            list: 包含所有类别名称的列表，如['Stain', 'Scratch', ...]
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(classes_path):
                return []
            
            # 使用safe_load加载YAML文件
            with open(classes_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            # 检查names字段是否存在
            if 'names' not in data:
                return []
            
            # 返回names数组
            return data['names']
        
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading YAML classes: {str(e)}")




    def save_yolo_labels(self, img_path, labels):
        """将YOLO标签保存到文件"""
        txt_path = os.path.splitext(img_path)[0] + '.txt'

        if not labels:
            if os.path.exists(txt_path):
                try:
                    os.remove(txt_path)
                except Exception as e:
                    print(f"删除标签文件出错：{e}")
            return

        try:
            with open(txt_path, 'w') as f:
                for label in labels:
                    f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")
        except Exception as e:
            print(f"保存YOLO标签出错：{e}")

    # --------------------------
    # 删除操作
    # --------------------------
    def delete_selected_images(self):
        """从当前聚类中删除选中的图像"""
        if self.current_cluster_index == -1:
            return

        selected_images = self.get_selected_images()
        if not selected_images:
            QMessageBox.warning(self, "警告", "未选择要删除的图像！")
            return

        reply = QMessageBox.question(
            self, "确认删除",
            f"删除 {len(selected_images)} 张选中的图像及其标签？",
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
                    print(f"删除 {img_path} 时出错：{e}")

            self.show_cluster_images(self.cluster_list.currentItem())

            current_item = self.cluster_list.currentItem()
            current_item.setText(f"聚类 {self.current_cluster_index + 1} ({len(cluster)} 张图像)")

            if not cluster:
                self.cluster_list.takeItem(self.cluster_list.row(current_item))
                self.current_cluster_index = -1
                self.clear_image_display()
                self.delete_btn.setEnabled(False)

    def delete_current_cluster_duplicates(self):
        """删除当前聚类中的所有重复图像，仅保留一张"""
        if self.current_cluster_index == -1:
            QMessageBox.warning(self, "警告", "未选择聚类！")
            return

        cluster = self.clusters[self.current_cluster_index]
        if len(cluster) <= 1:
            QMessageBox.information(self, "信息", "聚类已仅包含一张图像！")
            return

        reply = QMessageBox.question(
            self, "确认删除",
            f"将从此聚类中删除 {len(cluster) - 1} 张图像，\n"
            "仅保留一张。继续？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        # 保留第一张图像，删除其他
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
                print(f"删除 {img_path} 时出错：{e}")

        # 更新聚类 - 仅保留一张图像
        self.clusters[self.current_cluster_index] = [image_to_keep]

        # 更新界面
        current_item = self.cluster_list.currentItem()
        current_item.setText(f"聚类 {self.current_cluster_index + 1} (1 张图像)")
        self.show_cluster_images(current_item)

        QMessageBox.information(
            self, "操作完成",
            f"已删除 {deleted_count} 张重复图像。\n"
            f"在聚类中保留了 1 张唯一图像。"
        )

    def run_auto_labeling(self):
        """对未标注的图像运行自动标注"""
        if not self.yolo_model_pt:
            QMessageBox.warning(self, "警告", "YOLO模型未加载！")
            return

        # 查找未标注的图像
        unlabeled_images = []
        for cluster in self.clusters:
            for img_path in cluster:
                if img_path.endswith(".txt"):
                    continue
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                if not os.path.exists(txt_path):
                    unlabeled_images.append(img_path)

        if not unlabeled_images:
            QMessageBox.information(self, "信息", "所有图像已具有标签！")
            return

        reply = QMessageBox.question(
            self, "确认自动标注",
            f"发现 {len(unlabeled_images)} 张没有标签的图像。\n"
            "运行YOLO模型以自动标注它们？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.start_auto_labeling(unlabeled_images)

    # --------------------------
    # 快捷键操作
    # --------------------------
    def keyPressEvent(self, event):
        """处理键盘按键事件"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # 通过Enter键打开选中的图像
            selected_image = self.get_current_selected_image()
            if selected_image:
                self.show_fullscreen_image(selected_image)
            return
        if event.key() == Qt.Key_S:  # 下一个聚类
            self.next_cluster()
        elif event.key() == Qt.Key_W:  # 上一个聚类
            self.prev_cluster()
        elif event.key() == Qt.Key_A:  # 反转所有图像的选中状态
            self.toggle_all_images()
        elif event.key() == Qt.Key_D:  # 删除选中的图像
            self.delete_selected_images()
        elif event.key() == Qt.Key_L:  # 下一个图像（蓝色边框选中）
            self.next_image()
        elif event.key() == Qt.Key_O:  # 上一个图像（蓝色边框选中）
            self.prev_image()
        elif event.key() == Qt.Key_P:  # 切换当前图像的复选框
            self.toggle_current_image()
        elif event.key() == Qt.Key_X:
            self.delete_current_cluster_duplicates()
        else:
            super().keyPressEvent(event)

    def get_current_selected_image(self):
        """获取当前选中的图像（带蓝色边框）"""
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
        """跳转到下一个聚类"""
        if not hasattr(self, 'cluster_list') or self.cluster_list.count() == 0:
            return

        current_row = self.cluster_list.currentRow()
        if current_row < self.cluster_list.count() - 1:
            self.cluster_list.setCurrentRow(current_row + 1)
            self.current_image_index = -1  # 重置图像选中状态
            self.show_cluster_images(self.cluster_list.currentItem())

    def prev_cluster(self):
        """跳转到上一个聚类"""
        if not hasattr(self, 'cluster_list') or self.cluster_list.count() == 0:
            return

        current_row = self.cluster_list.currentRow()
        if current_row > 0:
            self.cluster_list.setCurrentRow(current_row - 1)
            self.current_image_index = -1  # 重置图像选中状态
            self.show_cluster_images(self.cluster_list.currentItem())

    def next_image(self):
        """选中当前聚类中的下一个图像"""
        if self.current_cluster_index == -1:
            return

        cluster = self.clusters[self.current_cluster_index]
        if not cluster:
            return

        # 取消当前图像的选中状态
        self._set_image_highlight(self.current_image_index, False)

        # 移动到下一个
        if self.current_image_index < len(cluster) - 1:
            self.current_image_index += 1
        else:
            self.current_image_index = 0

        # 设置新图像的选中状态
        self._set_image_highlight(self.current_image_index, True)

        # 滚动到选中的图像
        self._scroll_to_image(self.current_image_index)

    def prev_image(self):
        """选中当前聚类中的上一个图像"""
        if self.current_cluster_index == -1:
            return

        cluster = self.clusters[self.current_cluster_index]
        if not cluster:
            return

        # 取消当前图像的选中状态
        self._set_image_highlight(self.current_image_index, False)

        # 移动到上一个
        if self.current_image_index > 0:
            self.current_image_index -= 1
        else:
            self.current_image_index = len(cluster) - 1

        # 设置新图像的选中状态
        self._set_image_highlight(self.current_image_index, True)

        # 滚动到选中的图像
        self._scroll_to_image(self.current_image_index)

    def toggle_current_image(self):
        """切换当前选中图像的复选框"""
        if self.current_cluster_index == -1 or self.current_image_index == -1:
            return

        # 查找图像控件
        widget = self._get_image_widget_at_index(self.current_image_index)
        if widget:
            checkbox = widget.findChild(QCheckBox, "image_checkbox")
            if checkbox:
                checkbox.setChecked(not checkbox.isChecked())

    def _set_image_highlight(self, index, highlight):
        """设置或取消图像的选中状态（蓝色边框）"""
        widget = self._get_image_widget_at_index(index)
        if widget:
            widget.setProperty("selected", highlight)
            widget.setStyle(widget.style())

    def _get_image_widget_at_index(self, index):
        """根据索引获取图像控件"""
        if index == -1:
            return None

        # 仅计数图像控件（跳过分隔线）
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
        """滚动到指定索引的图像"""
        widget = self._get_image_widget_at_index(index)
        if widget:
            self.cluster_images_area.ensureWidgetVisible(widget)

    def _init_platform_settings(self):
        """初始化平台特定的设置"""
        self.os_name = platform.system()

        # Windows特定设置
        if self.os_name == "Windows":
            try:
                import ctypes
                # 为Windows任务栏设置应用ID
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('yolo.vision.labeler.ide')
            except:
                pass

        # macOS特定设置
        elif self.os_name == "Darwin":
            # 启用Retina显示支持
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_NoSystemBackground, False)
            # 启用统一的工具栏样式
            self.setUnifiedTitleAndToolBarOnMac(True)

        # Linux特定设置
        elif self.os_name == "Linux":
            # 如果需要，添加Linux特定的设置
            pass

def init_qt_env():
    # 启用高DPI缩放, 但是鼠标漂移问题依然未解决
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    signal.signal(signal.SIGINT, lambda *_: QApplication.quit())
    print("init_qt_env finished..")


if __name__ == "__main__":
    print(f"PyTorch版本：{torch.__version__}")
    print(f"CUDA可用：{torch.cuda.is_available()}")

    init_qt_env()

    app = QApplication(sys.argv)
    DarkTheme.apply(app)

    # 平台特定的字体设置
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