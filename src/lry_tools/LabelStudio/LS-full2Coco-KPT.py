# -----------------------------------------------------------------------------
# Based on https://github.com/heartexlabs/label-studio-converter
# Original license: Copyright (c) Heartex, under the Apache 2.0 License.
# -----------------------------------------------------------------------------

import argparse
import io
import json
import logging
import pathlib
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

class LSConverter:

    def __init__(self, config: str):
        """Convert the Label Studio Format JSON file to COCO format JSON file

        Args:
        """
        # get label info from config file
        tree = ET.parse(config)
        root = tree.getroot()

        # 独立存储两类标签（完全不混合）
        self.rect_categories = []  # 矩形标签类别
        self.poly_categories = []  # 多边形标签类别
        self.rect_name_to_id = {}  # 矩形标签名称到ID映射
        self.poly_name_to_id = {}  # 多边形标签名称到ID映射

        # 提取矩形标签（BBox）
        rect_labels = root.findall('.//RectangleLabels/Label')
        for idx, label in enumerate(rect_labels, start=1):
            self.rect_categories.append({
                'id': idx,
                'name': label.get('value'),
                'color': label.get('background'),
                'label_type': 'rectangle'
            })
            self.rect_name_to_id[label.get('value')] = idx

        # 提取多边形标签（角点）
        poly_labels = root.findall('.//PolygonLabels/Label')
        for idx, label in enumerate(poly_labels, start=1):
            self.poly_categories.append({
                'id': idx,
                'name': label.get('value'),
                'color': label.get('background'),
                'hotkey': label.get('hotkey', None),
                'label_type': 'polygon'
            })
            self.poly_name_to_id[label.get('value')] = idx

    def convert_to_coco(self, input_json: str, output_json: str):
        """Convert `input_json` to COCO format and save in `output_json`.

        Args:
            input_json (str): The path of Label Studio format JSON file.
            output_json (str): The path of the output COCO JSON file.
        """

        output_path = pathlib.Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        images_coco = []
        annotations_coco = []

        with open(input_json, 'r', encoding='utf-8') as f:
            ann_list = json.load(f)

        for item_idx, item in enumerate(ann_list):
            orphan_keypoints_list = []  # 存储无parentID的关键点
            # each image is an item
            label_studio_image_id = item['id']
            image_name = item['data']['image']
            image_id = len(images_coco)
            has_draft = bool(item['drafts'])
            annotations_label_studio = item['annotations'][0]
            author_id = annotations_label_studio['completed_by']
            all_labels = annotations_label_studio['result']
            width, height = None, None
            annotations_dict = {}  # 存储bbox信息 {id: bbox_annotation}
            # 遍历标注，bbox与keypoints分开处理
            for label in all_labels:
                # 如果没有更新过height、width则更新height、width并更新image
                if not height or not width:
                    # 尝试从label的value中获取
                    if 'value' in label and all(k in label['value'] for k in ['original_width', 'original_height']):
                        width = label['value']['original_width']
                        height = label['value']['original_height']
                    # 尝试直接从label中获取
                    elif all(k in label for k in ['original_width', 'original_height']):
                        width = label['original_width']
                        height = label['original_height']
                    else:
                        warnings.warn(
                            f'original_width or original_height not found in {image_name}\n'
                            f'Available keys: {list(label.keys())}\n'
                            f'Value keys: {list(label.get("value", {}).keys())}'
                        )
                        continue

                    # 更新images列表
                    images_coco.append({
                        'id': image_id,
                        'width': width,
                        'height': height,
                        'file_name': image_name,
                        'label_studio_image_id': label_studio_image_id,
                    })

                # skip tasks without annotations
                if not item['annotations']:
                    logger.warning('No annotations found for item #' +
                                   str(item_idx) + 'Image ID: ' + str(item['id']))
                    continue

                if label['type'] == 'rectanglelabels':
                    label_studio_id = label['id']
                    value_keys = label['value'].keys()
                    if 'points' in value_keys:
                        warnings.warn(f'rectanglelabels contains key points, author:{author_id}, Image ID:{label_studio_image_id}, 请检查错误标签（是否存在未提交草稿：{has_draft}）')
                        continue
                    if label_studio_id in annotations_dict:
                        annotations_dict[label_studio_id]['bbox'] = label
                        annotations_dict[label_studio_id]['has_draft'] = has_draft
                    else:
                        annotations_dict[label_studio_id] = {
                            'image_id': image_id,
                            'bbox': label,
                            'keypoints': [],  # 初始化关键点存储
                            'has_draft': has_draft
                        }

                elif label['type'] == 'polygonlabels':
                    parent_id = label.get('parentID')
                    value_keys = label['value'].keys()
                    if 'x' in value_keys:
                        warnings.warn(f'polygonlabels contains key x, author:{author_id}, Image ID:{label_studio_image_id}, 请检查错误标签（是否存在未提交草稿：{has_draft}）')
                        continue
                    if parent_id:
                        if parent_id not in annotations_dict:
                            annotations_dict[parent_id] = {
                                'image_id': image_id,
                                'bbox': None,  # 占位，后续可能补充
                                'keypoints': [label],  # 直接放入当前关键点
                                'has_draft': has_draft
                            }
                        else:
                            annotations_dict[parent_id]['keypoints'].append(label)
                    else:
                        orphan_keypoints_list.append({
                            'image_id': image_id,
                            'label': label,
                            'width': width,
                            'height': height
                        })

                else:
                    logger.warning('Unknown label type or labels are empty')
                    continue

            # 处理有parentID的关键点
            for label_studio_bbox_id, data in annotations_dict.items():
                bbox_ann = data['bbox']
                if not bbox_ann:
                    # 错误标签引起的无法找到匹配项
                    continue
                keypoint_anns = data['keypoints']
                # 转换bbox
                bbox = self._convert_bbox(bbox_ann, width, height)
                x_min, y_min, w, h = bbox
                x_max, y_max = x_min + w, y_min + h
                # 转换关键点
                for keypoint_ann in keypoint_anns:
                    keypoints, num_keypoints = self._convert_single_keypoint(keypoint_ann, width, height)
                    if num_keypoints != 4:
                        warnings.warn(f"Number of keypoints is not 4: {num_keypoints}, author:{author_id}, Image ID:{label_studio_image_id}, labelID: {keypoint_ann['id']}, 请检查角点数量（是否存在未提交草稿：{has_draft}）")
                        continue
                    # 检查关键点是否在bbox内
                    abs_points = [(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 3)]
                    if not all(x_min <= x <= x_max and y_min <= y <= y_max for (x, y) in abs_points):
                        warnings.warn(f"kpt out of range, author:{author_id}, Image ID:{label_studio_image_id}, labelID: {keypoint_ann['id']}, 请检查角点超出bbox（是否存在未提交草稿：{has_draft}）")
                        continue
                    polygonlabels = keypoint_ann['value']['polygonlabels'][0]
                    category_id = self.poly_name_to_id[polygonlabels]
                    # if label_studio_image_id == 114305:
                    #     a = 1
                    if category_id == 1:  # 正面角点应逆时针
                        if not self._check_polygon_order(abs_points, expected_clockwise=False):
                            warnings.warn(
                                f"author:{author_id}, Image ID:{label_studio_image_id}, labelID: {keypoint_ann['id']}, 请检查正面角点标注顺序（是否存在未提交草稿：{has_draft}）")
                    elif category_id == 3:  # 背面角点应顺时针
                        if not self._check_polygon_order(abs_points, expected_clockwise=True):
                            warnings.warn(
                                f"author:{author_id}, Image ID:{label_studio_image_id}, labelID: {keypoint_ann['id']}, 请检查背面角点标注顺序（是否存在未提交草稿：{has_draft}）")
                    annotations_coco.append({
                        'id': len(annotations_coco),
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': bbox,
                        'area': bbox[2] * bbox[3],
                        'keypoints': keypoints,
                        'num_keypoints': num_keypoints,
                        'iscrowd': 0,
                        'bbox_label': bbox_ann['value']['rectanglelabels'],
                        'author_id': author_id,
                    })

            # 处理孤立关键点（尝试空间匹配）
            for orphan_keypoints in orphan_keypoints_list:
                matched = False
                keypoint_ann = orphan_keypoints['label']
                # 获取关键点坐标列表
                points = keypoint_ann['value'].get('points') # [[x1,y1], [x2,y2], ...]
                # 转换为绝对坐标
                abs_points = [(p[0] * width / 100, p[1] * height / 100) for p in points]
                # 检查所有bbox，寻找匹配
                for label_studio_bbox_id, data in annotations_dict.items():
                    if data['bbox'] is None:
                        continue
                    bbox_ann = data['bbox']
                    bbox = self._convert_bbox(bbox_ann, width, height)
                    x_min, y_min, w, h = bbox
                    x_max = x_min + w
                    y_max = y_min + h
                    if all(x_min <= x <= x_max and y_min <= y <= y_max for (x, y) in abs_points):
                        polygonlabels = keypoint_ann['value']['polygonlabels'][0]
                        category_id = self.poly_name_to_id[polygonlabels]
                        keypoints, num_keypoints = self._convert_single_keypoint(keypoint_ann, width, height)
                        annotations_coco.append({
                            'id': len(annotations_coco),
                            'image_id': image_id,
                            'category_id': category_id,
                            'bbox': bbox,
                            'area': bbox[2] * bbox[3],
                            'keypoints': keypoints,
                            'num_keypoints': num_keypoints,
                            'iscrowd': 0,
                            'bbox_label': bbox_ann['value']['rectanglelabels'],
                            'author_id': author_id,
                        })
                        matched = True
                        break

                if not matched:
                    logger.warning(f"Image {label_studio_image_id}, labelID: {keypoint_ann['id']}, author:{author_id} keypoints could not be matched to any bbox, keypoints: {abs_points},（是否存在未提交草稿：{has_draft}）")

        with io.open(output_json, mode='w', encoding='utf8') as fout:
            json.dump(
                {
                    'images': images_coco,
                    'categories': self._get_coco_categories(),
                    'annotations': annotations_coco,
                    'info': {
                        'year': datetime.now().year,
                        'version': '1.0',
                        'description': '',
                        'contributor': 'Label Studio',
                        'url': '',
                        'date_created': str(datetime.now()),
                    },
                },
                fout,
                indent=2,
            )

    @staticmethod
    def _convert_bbox(bbox_ann, width = None, height = None):
        """将百分比坐标转换为绝对坐标"""
        if not width or not height:
            width, height = bbox_ann['original_width'], bbox_ann['original_height']
        value = bbox_ann['value']
        if not isinstance(value, dict):
            print(value)
            breakpoint()
        x = value.get('x')
        y = value.get('y')
        w = value.get('width')
        h = value.get('height')
        if None in (x, y, w, h):
            print(value)
            breakpoint()
        return [
            bbox_ann['value']['x'] * width / 100,
            bbox_ann['value']['y'] * height / 100,
            bbox_ann['value']['width'] * width / 100,
            bbox_ann['value']['height'] * height / 100
        ]

    @staticmethod
    def _convert_keypoints(keypoint_anns, width, height):
        """转换多边形关键点"""
        keypoints = []
        for ann in keypoint_anns:
            for point in ann['value']['points']:
                x = point[0] * width / 100
                y = point[1] * height / 100
                keypoints.extend([x, y, 2])  # 2表示可见
        return keypoints, len(keypoint_anns)

    @staticmethod
    def _convert_single_keypoint(keypoint_ann, width = None, height = None):
        """转换单个关键点"""
        if not width or not height:
            width, height = keypoint_ann['original_width'], keypoint_ann['original_height']
        coords = []
        points = keypoint_ann['value']['points']
        for point in points:
            x = point[0] * width / 100
            y = point[1] * height / 100
            coords.extend([x, y, 2])
        num_keypoints = len(points)
        return coords, num_keypoints

    @staticmethod
    def _check_points_in_bbox(points, bbox, width, height):
        """检查所有点是否在bbox内"""
        x_min, y_min, w, h = bbox
        x_max = x_min + w
        y_max = y_min + h

        for point in points:
            x = point[0] * width / 100
            y = point[1] * height / 100
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                return False
        return True

    # 新增的检查方法
    @staticmethod
    def _check_polygon_order(points, expected_clockwise=False):
        """检查多边形顶点顺序
        :param points: 顶点列表 [(x1,y1), (x2,y2), ...]
        :param expected_clockwise: 期望是否为顺时针
        :return: True if order matches expectation
        """
        if len(points) < 3:  # 不是有效多边形
            return False

        # 计算带符号面积（鞋带公式）
        total = sum((points[(i + 1) % len(points)][0] - points[i][0]) *
                    (points[(i + 1) % len(points)][1] + points[i][1])
                    for i in range(len(points)))

        # 实际顺时针(面积<0) == 期望顺时针
        return (total < 0) == expected_clockwise

    def _get_coco_categories(self):
        """根据标注配置动态生成COCO categories"""
        categories = []

        # 多边形类别（关键点）
        for poly_name, poly_id in self.poly_name_to_id.items():
            # 默认关键点命名（可根据实际需求修改）
            keypoint_names = [f"kpt{i}" for i in range(4)]  # 假设都是4个关键点

            categories.append({
                'id': poly_id,
                'name': poly_name,
                'supercategory': 'tm',  # 可自定义或从配置读取
                'keypoints': keypoint_names,
                'skeleton': [[0, 1], [1, 2], [2, 3], [3, 0]]  # 四边形连接方式
            })

        return categories

def main():
    config = r"D:\MyProjects\ls.xml"
    input_json = r"D:\MyProjects\project-105-at-2025-10-29-02-37-cbad34d6.json"
    output_json = r"D:\MyProjects\output2025.json"
    converter = LSConverter(config)
    converter.convert_to_coco(input_json, output_json)


if __name__ == '__main__':
    main()
