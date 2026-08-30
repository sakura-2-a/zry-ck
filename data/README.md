# 数据说明

## 数据集来源
- 数据集名称：Kolektor Surface Defect Dataset
- 来源链接：https://www.kaggle.com/datasets/brumin/kolektor-surface-defect-dataset
- 数据集描述：包含真实工业环境下金属表面的缺陷图像
- 数据规模：约400张图像
- 缺陷类型：划痕（scratch）、裂纹（crack）
- 数据格式：JPEG图像 + XML标注（PASCAL VOC格式）

## 数据预处理说明
原始数据经过以下预处理：
1. 图像尺寸统一调整为 640×640 像素
2. 标注格式从 XML 转换为 YOLO 格式（txt）
3. 按 7:2:1 比例划分训练集、验证集、测试集
4. 应用数据增强（翻转、旋转、亮度调整）

## 文件结构
/data/
├── raw/               # 原始数据（需自行下载）
│   ├── images/
│   └── annotations/
├── processed/         # 预处理后数据
│   ├── train/
│   ├── val/
│   └── test/
└── README.md          # 本说明文件

## 数据集

本项目使用 Kolektor Surface Defect Dataset 公开数据集。

- 名称：Kolektor Surface Defect Dataset
- 来源：https://www.kaggle.com/datasets/brumin/kolektor-surface-defect-dataset
- 规模：约400张图像，包含划痕和裂纹两类缺陷
- 格式：JPEG + XML标注
