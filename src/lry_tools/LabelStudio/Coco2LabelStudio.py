#!/usr/bin/env python3
import json
import os
import warnings
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
from collections import defaultdict

from unicodedata import category

from lry_tools.utils.debug import is_debugging
from lry_tools.BasicIMageProcess.get_image_size import get_image_size


def process_annotations(task, annotations, img_info):
    """处理单个图片的所有标注（合并重复代码）"""
    for ann in annotations:
        # Determine polygon label based on category_id
        # if ann['category_id'] == 0:
        #     categorylabels = ["天面正面角点"]
        # elif ann['category_id'] in [1, 2, 3]:
        #     categorylabels = ["天面侧面角点"]
        # elif ann['category_id'] in [4]:
        #     categorylabels = ["天面背面角点"]
        # else:
        #     categorylabels = ["其他角点"]

        if ann['category_id'] == 1:
            categorylabels = ["4G"]
        elif ann['category_id'] in [2]:
            categorylabels = ["5G大裙边"]
        elif ann['category_id'] in [3]:
            categorylabels = ["5G小裙边"]
        elif ann['category_id'] in [4]:
            categorylabels = ["5G异形"]
        elif ann['category_id'] in [5]:
            categorylabels = ["Front"]
        else:
            categorylabels = ["其他角点"]

        # Add bounding boxes
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            task['annotations'][0]['result'].append({
                "id": f"bbox_{ann['id']}",
                "type": "rectanglelabels",
                "value": {
                    "x": x / img_info['width'] * 100,
                    "y": y / img_info['height'] * 100,
                    "width": w / img_info['width'] * 100,
                    "height": h / img_info['height'] * 100,
                    "rotation": 0,
                    "rectanglelabels": categorylabels,
                    "original_width": img_info['width'],
                    "original_height": img_info['height']
                },
                "to_name": "image",
                "from_name": "rectLabel",
                "origin": "manual"
            })

        # Add keypoints/polygons
        if 'keypoints' in ann:
            points = []
            for i in range(0, len(ann['keypoints']), 3):
                x, y, v = ann['keypoints'][i:i + 3]
                if v > 0:  # Only include visible points
                    points.append([
                        x / img_info['width'] * 100,
                        y / img_info['height'] * 100
                    ])

            if points:
                task['annotations'][0]['result'].append({
                    "id": f"poly_{ann['id']}",
                    "type": "polygonlabels",
                    "value": {
                        "points": points,
                        "polygonlabels": categorylabels,
                        "original_width": img_info['width'],
                        "original_height": img_info['height']
                    },
                    "to_name": "image",
                    "from_name": "polyLabel",
                    "origin": "manual"
                })


def create_task_structure(img_info, image_path):
    """创建基础的 Label Studio 任务结构（合并重复代码）"""
    return {
        "data": {
            "image": f"/data/local-files/?d=SVC_251118/{Path(image_path).name}",
            "width": img_info['width'],
            "height": img_info['height']
        },
        "annotations": [{
            "result": [],
            "completed_by": 1,
            "was_cancelled": False,
            "ground_truth": False,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_created_at": datetime.now().isoformat(),
            "last_updated_at": datetime.now().isoformat()
        }]
    }


def coco_to_labelstudio(coco_json_path, image_dir, output_json_path, traverse_by="annotations"):
    """Convert COCO dataset to Label Studio JSON format with merged annotations

    Args:
        coco_json_path: Path to COCO annotations JSON file
        image_dir: Directory containing the images
        output_json_path: Path to save Label Studio JSON output
        traverse_by: "annotations" (default) or "images" - 决定遍历方式(是否导入无标注图片)
    """
    # Load COCO data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create mappings
    images = {img['id']: img for img in coco_data['images']}

    # Group annotations by image_id
    img_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        img_annotations[ann['image_id']].append(ann)

    # 预处理图像尺寸信息
    images = preprocess_image_sizes(images, image_dir)

    # Prepare Label Studio tasks (one per image)
    tasks = []
    if traverse_by == "annotations":
        print("遍历方式：有标注的图片")
        for img_id, annotations in tqdm(img_annotations.items(), desc="Processing images"):
            if img_id not in images:
                warnings.warn(f"Image {img_id} in annotations not found in images.")
                continue

            img_info = images[img_id]
            image_path = Path(image_dir) / Path(img_info['file_name']).name

            task = create_task_structure(img_info, image_path)

            # Process all annotations for this image
            process_annotations(task, annotations, img_info)

            tasks.append(task)

    elif traverse_by == "images":
        print("遍历方式：所有图片")
        for img_id, img_info in tqdm(images.items(), desc="Processing all images"):
            image_path = Path(image_dir) / Path(img_info['file_name']).name

            # Create Label Studio task structure
            task = create_task_structure(img_info, image_path)

            # 获取该图片的标注（如果没有则为空列表）
            annotations = img_annotations.get(img_id, [])

            # Process all annotations for this image
            process_annotations(task, annotations, img_info)

            tasks.append(task)
    else:
        raise ValueError(f"不支持的遍历方式: {traverse_by}。请使用 'annotations' 或 'images'")

    # 统计信息
    total_images = len(images)
    images_with_annotations = len(img_annotations)
    tasks_count = len(tasks)

    print(f"\n转换统计:")
    print(f"  总图片数: {total_images}")
    print(f"  有标注图片: {images_with_annotations}")
    print(f"  生成任务数: {tasks_count}")

    if traverse_by == "annotations":
        print(f"  无标注图片: {total_images - images_with_annotations} (未包含)")
    else:
        print(f"  无标注图片: {total_images - images_with_annotations} (已包含)")

    # Save Label Studio JSON
    with open(output_json_path, 'w') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(tasks)} tasks to {output_json_path}")


def preprocess_image_sizes(images, image_dir):
    """预处理图像尺寸信息，如果COCO数据中缺少尺寸则从图像文件读取"""
    for img_id, img_info in images.items():
        # 检查是否已有尺寸信息
        if 'width' not in img_info or 'height' not in img_info or img_info['width'] is None or img_info['height'] is None:
            # 尝试从图像文件读取尺寸
            image_path = Path(image_dir) / Path(img_info['file_name']).name
            if image_path.exists():
                img_size = get_image_size(str(image_path))
                img_info['width'] = img_size['width']
                img_info['height'] = img_size['width']
            else:
                print(f"警告: 图像文件不存在 {image_path}，使用默认尺寸 1920x1080")
    
    return images


if __name__ == '__main__':
    if is_debugging():
        print("🔧 调试模式激活，使用预设参数...")
        # 直接设置参数（用于VSCode调试）
        class Args:
            coco_json = "/data1/liruoyu/zhangzhikang_data/newData/EndTrain_lry.json"
            image_dir = "/data1/liruoyu/zhangzhikang_data/newData/EndTrain/"
            output = "/data1/liruoyu/zhangzhikang_data/newData/train_ls.json"
            traverse_by = "annotations"
        
        args = Args()
    else:
        # 使用命令行参数
        parser = argparse.ArgumentParser(description='Convert COCO dataset to Label Studio JSON format')
        parser.add_argument('--coco-json', required=True, help='Path to COCO annotations JSON file')
        parser.add_argument('--image-dir', required=True, help='Directory containing the images')
        parser.add_argument('--output', required=True, help='Output JSON file path for Label Studio')
        parser.add_argument('--traverse-by', choices=['annotations', 'images'], default='annotations',
                            help='遍历方式: annotations(仅含标注图片, 默认) 或 images(所有图片)')
        args = parser.parse_args()

    coco_to_labelstudio(args.coco_json, args.image_dir, args.output, args.traverse_by)
