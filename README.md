# YOLO Vision Labeler 🖼️🔍


[![English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![Русский](https://img.shields.io/badge/Язык-Русский-red)](README.ru.md)


### YOLO image clustering and markup tool

---

## 🔹 Описание
**YOLO Vision Labeler** — is a handy graphical application for:

✅ **Clustering of similar images** (hash-based)

✅ **Object markups in YOLO format** (rectangular bounding boxes)

✅ **Removing duplicate and trash photos**

Ideal for preparing datasets prior to training computer vision models.

---

## ✨ Features

### 1. Image clustering
- Support for hashing algorithms: **Average Hash, PHash, DHash**
- Setting similarity threshold (from 0 to 64)
- Skip single images (not in clusters)

### 2. Markup in YOLO format
- **Drawing bounding boxes** (two clicks: upper left corner → lower right corner)
- Specify **class number** for each object
- Editing, deleting and hiding existing labels
- Automatic saving to `.txt` (one file per image)

### 3. Image management
- View in **full screen mode** with zoom capability
- Deleting selected photos (along with markup)
- Support formats: **JPG, PNG, BMP, GIF**

### 4. Cross-platform
- Works on **Windows, macOS, Linux** (Debian/Ubuntu)

Translated with DeepL.com (free version)

---

## 🚀 Работа с YOLO моделью

### ⚙️ Customising the model
1. **Select Model** - specify the path to the `.pt` model file via the ‘Browse File’ button
2. **Detection Parameters**:
   - `Confidence` (1-100) - confidence threshold for detection
   - `Image Width/Height` - input image size for the model
   - `IOU Threshold` (1-100) - intersection threshold to suppress duplicate detections

### 🔍 Использование модели
| Function               |  Description                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| Predict           | The ‘Predict YOLO model’ button starts detection on the current image    |
| Automatic markup | The model automatically adds bounding boxes with classes               |
| Visualisation          |  Detections are displayed with coloured boxes with class signatures              |

### 🎨 Markup format
The model stores data in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where all coordinates are normalised relative to the image dimensions (0-1)

### 💡 Performance Tips
- For best quality, use models trained on your data
- Adjust Confidence and IOU Threshold to your task
- The image size should match the size on which the model was trained
- Predictions can be edited manually in full screen mode

---

## 🖼️ Auto-Labelling (Auto-Labelling)

#### 🔍 YOLO Auto-Labelling

The Auto-Labelling feature allows you to:
- Automatically generate YOLO markup for unlabelled images
- Use the loaded YOLO model to predict objects
- Save results to `.txt` files in YOLO format

**How to use:**
1. Load the YOLO model through the ‘YOLO Model Settings’ menu
2. Process the folder with the images (‘Process Images’)
3. click ‘Auto Label with YOLO’ button after the processing is finished.
4. Wait for the process to complete (progress is shown in the status bar)

**Features:**
- Works only with images that do not have corresponding `.txt` files
- Supports confidence threshold and IOU threshold setting
- Automatically detects CUDA availability for acceleration
- Preserves original images without modification

---
## 🌈 Adjusting the brightness of the image
#### 💡 Adjusting brightness in viewing mode

In full-screen viewing mode, tools are available to adjust brightness:

**Controls:**
- Brightness slider (50-200%)
- Digital display of the current value
- Reset button to reset the brightness to the original value

**Hotkeys:**
- `+` - Increase brightness by 5%
- `-` - Decrease brightness by 5%
- `0` - Reset brightness to 100%.

**Features:**
- Changes are applied only during playback
- Does not affect original image files
- Markup and bounding boxes remain visible at all brightness levels
- Supports smooth changes with real-time previews

---

## 🚀 Hotkeys

### Cluster Navigation
| Key | Action                           |
|-----|-----------------------------------|
| `W` | Previous cluster                |
| `S` | Next cluster                 |
| `X` | Keep one image, delete the others in the current cluster                 |

### Работа с изображениями
| Key  | Action                           |
|---------|-----------------------------------|
| `A`     | Invert all selection      |
| `D`     |  Delete selected images    |
| `P`     | Switch the current image   |

### Навигация в кластере
| Key  | Action                           |
|---------|-----------------------------------|
| `O`     | Next Image (→)         |
| `L`     | Previous image (←)        |

### Дополнительные функции
| Key  | Action                           |
|---------|-----------------------------------|
| `Esc`   | Exit markup mode          |
| `Enter` | Open the current image       |

> 💡 Hint: The current image is highlighted with a blue frame

---

## 🖥️ Download and Run (v0.0.1)

### [MacOS Build](https://github.com/aliensowo/YOLO-Vision-Labeler/releases/download/untagged-8f24ee15334a0b0e5dca/YOLO_VisionLabelerMacOS.app.zip)

### [MacOS_arm64 Build](https://github.com/aliensowo/YOLO-Vision-Labeler/releases/download/untagged-8f24ee15334a0b0e5dca/YOLO_VisionLabelerMacOS_arm64.app.zip)

### [Win64 Build](https://github.com/aliensowo/YOLO-Vision-Labeler/releases/download/untagged-8f24ee15334a0b0e5dca/YOLO_VisionLabelerWIN64.exe)

---

## 🖥️ Installing from source

### 1. Requirements
- Python 3.7+
- Libs:

```bash
pip install -r requirements.txt
```

### 2. RUN

```bash
python main.py
```


### 3. EXE build (optionally)

```bash
pyinstaller --onefile --windowed main.py
```

---

## 📜 License
MIT License — free use and modification.

---

## 💡 Support
Found a bug or have suggestions?
Contact the author.

---

🚀 With YOLO Vision Labeler, preparing datasets becomes easier!