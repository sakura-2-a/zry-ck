#!/usr/bin/env python3
"""
数据预处理脚本
功能：将原始Kolektor数据集转换为YOLOv8训练格式
使用方法：python scripts/preprocess.py
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from tqdm import tqdm

# ============ 配置 ============
RAW_IMG_DIR = "data/raw/images"
RAW_ANNO_DIR = "data/raw/annotations"
OUTPUT_DIR = "data/processed"
TARGET_SIZE = (640, 640)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42

CLASS_MAPPING = {
    "scratch": 0,
    "crack": 1,
    "defect": 0,
}

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find('filename').text
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    objects = []
    for obj in root.findall('object'):
        label = obj.find('name').text.lower()
        if label not in CLASS_MAPPING:
            continue
        class_id = CLASS_MAPPING[label]
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        objects.append({
            'class_id': class_id,
            'x_center': max(0, min(1, x_center)),
            'y_center': max(0, min(1, y_center)),
            'width': max(0, min(1, width)),
            'height': max(0, min(1, height))
        })
    return filename, objects

def main():
    print("=" * 50)
    print("开始数据预处理...")
    print("=" * 50)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)
    
    xml_files = list(Path(RAW_ANNO_DIR).glob("*.xml"))
    print(f"找到 {len(xml_files)} 个标注文件")
    
    if len(xml_files) == 0:
        print("错误: 未找到XML标注文件，请检查原始数据路径！")
        return
    
    all_data = []
    for xml_path in tqdm(xml_files, desc="解析标注文件"):
        try:
            filename, objects = parse_xml(str(xml_path))
            img_path = os.path.join(RAW_IMG_DIR, filename)
            if not os.path.exists(img_path):
                continue
            all_data.append({
                'filename': filename,
                'image_path': img_path,
                'objects': objects
            })
        except Exception as e:
            continue
    
    print(f"成功解析 {len(all_data)} 个有效样本")
    
    if len(all_data) == 0:
        print("错误: 没有有效数据，请检查原始数据格式！")
        return
    
    train_data, temp_data = train_test_split(
        all_data,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED
    )
    val_data, test_data = train_test_split(
        temp_data,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED
    )
    
    print(f"数据划分完成: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")
    
    for split_name, data_list in [('train', train_data), ('val', val_data), ('test', test_data)]:
        print(f"正在处理 {split_name} 集...")
        for item in tqdm(data_list):
            img = cv2.imread(item['image_path'])
            if img is None:
                continue
            img_resized = cv2.resize(img, TARGET_SIZE)
            out_img_dir = os.path.join(OUTPUT_DIR, split_name, 'images')
            out_label_dir = os.path.join(OUTPUT_DIR, split_name, 'labels')
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_label_dir, exist_ok=True)
            
            img_path_out = os.path.join(out_img_dir, item['filename'])
            cv2.imwrite(img_path_out, img_resized)
            
            label_name = os.path.splitext(item['filename'])[0] + ".txt"
            label_path = os.path.join(out_label_dir, label_name)
            with open(label_path, 'w') as f:
                for bbox in item['objects']:
                    line = f"{bbox['class_id']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
                    f.write(line)
    
    print("=" * 50)
    print("数据预处理完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()
