
# YOLO Annotation Tool

这是一个基于 Python 和 PyQt5 的 YOLO 标注工具. 由于 label studio 过于庞大, 不方便修改, 因此 fork 了一份小的开源代码, 然后修正bug并增加我们自己的功能, 使其更方便使用.

## 依赖

- Python 3.6+
- 依赖库(见 `requirements.txt`)
- data.yaml(当前目录): 提供分类的id与名称, 用于标记的显示


## 安装

由于一些库可能比较老(未测试新版本是否有效), 所以建议先创建一个虚拟环境:
```bash
python -m venv .labelT  # 创建python虚拟环境
source .labelT/xxx/...  # 使用venv环境
```

再安装python包 `requirements.txt`:
```bash
pip install -r requirements.txt
python main.py
```
