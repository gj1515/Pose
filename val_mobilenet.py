import argparse
import cv2
import json
import math
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import pickle
from datetime import datetime

import torch

from models.mobilenet.with_mobilenet import PoseEstimationWithMobileNet
from modules.mobilenet.keypoints import extract_keypoints, group_keypoints, single_keypoints
from modules.load_state import load_state

from tqdm import tqdm

from datasets.coco.single.coco_single import CocoDataset_SinglePerson
from datasets.Robot.dataset_provider import create_dataset_loader

from config.skeleton import Skeleton
from concurrent.futures import ThreadPoolExecutor

from config.config_omnipose import ConfigOmniPose
from models.omnipose.omnipose import OmniPose, get_omnipose

from datasets.Robot.helper import denormalize_image

from datasets.Robot.dataset_provider import create_dataset



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def process_item(idx_item):
    idx, item = idx_item
    result = {'images': None, 'annotations': None}

    if 'img_file' in item:
        img_id = int(item['img_file'].split('.')[0])
        result['images'] = {
            'id': img_id,
            'file_name': item['img_file']
        }

        if 'keypoints' in item:
            num_keypoints = np.sum(item['keypoints'][:, 2] > 0)
            result['annotations'] = {
                'id': idx,
                'image_id': img_id,
                'category_id': 1,
                'keypoints': item['keypoints'].flatten().tolist(),
                'bbox': item['obj_bbox'],
                'area': item['obj_bbox'][2] * item['obj_bbox'][3],
                'num_keypoints': int(num_keypoints),
                'iscrowd': 0
            }

    return result



def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))
    if gt_file_path.endswith('.pkl'):
        import pickle
        with open(gt_file_path, 'rb') as f:
            gt_data = pickle.load(f)

        skeleton = Skeleton('coco17')

        coco_format = {
            "images": [],
            "annotations": gt_data,
            "categories": [{
                "id": 1,
                "name": "person",
                "keypoints": skeleton.names,
                "skeleton": skeleton.bones
            }]
        }

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(process_item, enumerate(gt_data)),
                total=len(gt_data),
                desc="Processing annotations"
            ))

        coco_format['images'] = [r['images'] for r in results if r['images'] is not None]
        coco_format['annotations'] = [r['annotations'] for r in results if r['annotations'] is not None]

        gt_json_path = gt_file_path.replace('.pkl', '.json')
        with open(gt_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, cls=NumpyEncoder)
        gt_file_path = gt_json_path

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def infer(model, img, scales, base_height, stride, pad_value=(0, 0, 0)):
    normed_img = img
    print('normed_img shape: ', normed_img.shape)
    height, width, _ = normed_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
    avg_pafs = np.zeros((height, width, 38), dtype=np.float32)

    for ratio in scales_ratios:
        scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled_img.shape[1], base_height)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        print('tensor_img shape: ', tensor_img.shape)
        stages_output = model(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_pafs = avg_pafs + pafs / len(scales_ratios)

    return avg_heatmaps, avg_pafs


def evaluate(config, labels, output_name, model, multiscale=False, visualize=False):
    model = model.cuda().eval()
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    _, dataset = create_dataset_loader(config)
    coco_result = []
    for sample in dataset:
        file_name = sample['file_name']
        ori_img = sample['ori_img']
        ori_img = ori_img.squeeze(0).permute(0, 1, 2).cpu().numpy()
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)

        avg_heatmaps, avg_pafs = infer(model, ori_img, scales, base_height, stride)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

        file_name = file_name[0]
        image_id = int(file_name.split('.')[0])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    cv2.circle(ori_img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imshow('keypoints', ori_img)
            key = cv2.waitKey()
            if key == 27:  # esc
                return

    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    run_coco_eval(labels, output_name)

if __name__ == '__main__':

    from config.config_mobilenet import ConfigMobilenet

    args = ConfigMobilenet().parse()

    model = PoseEstimationWithMobileNet(num_refinement_stages=3)
    checkpoint = torch.load(args.checkpoint_path)
    load_state(model, checkpoint)

    evaluate(args, args.val_labels, args.val_output_name, model, False, args.visualize)