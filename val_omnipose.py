import cv2
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch

from modules.load_state import load_state

from tqdm import tqdm

from datasets.Robot.dataset_provider import create_dataset_loader

from config.skeleton import Skeleton
from concurrent.futures import ThreadPoolExecutor

from config.config_omnipose import ConfigOmniPose
from models.omnipose.omnipose import get_omnipose
from config.config_omnipose_model import _C as cfg

from testing.decode_net import resize_hm
from testing.post_heatmap import decode_pose_single_person
from testing.eval_util import  get_keypoints_ori_coco

import matplotlib.pyplot as plt

from modules.loss import AverageMeter
from utils.omni_utils.evaluate import accuracy
import time

from utils.visualize import visualize_keypoints, ColorStyle


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



def validate_coco(criterion, val_loader, model, val_pbar=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model = model.cuda().eval()

    test_loader = val_loader

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(total=len(test_loader), desc='Testing', dynamic_ncols=True)
        for i, sample in enumerate(test_loader):
            # b. image preprocessing
            input_img = sample['image']
            input_img = input_img.float().cuda()

            heat_maps_gt = sample['heat_maps'].cuda()
            target_weight = sample['target_weight'].cuda()

            # c. model inference
            heatmap_outs = model(input_img)  # torch.Size([1, 17, 48, 64])

            if isinstance(heatmap_outs, list):
                output = heatmap_outs[-1]
            else:
                output = heatmap_outs

            loss = criterion(output, heat_maps_gt, target_weight)
            losses.update(loss.item(), input_img.size(0))

            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             heat_maps_gt.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if val_pbar is not None:
                val_pbar.update(input_img.size(0))
            pbar.update(1)

    pbar.close()

    # Return metrics dictionary
    metrics = {
        'Acc': acc.avg,
        'heatmap_loss': losses.avg
    }

    return metrics


def test(labels, output_name, net, config, visualize=False, val_pbar=None):
    net = net.cuda().eval()
    Thresh = 0.1

    _, test_loader = create_dataset_loader(config)

    coco_result = []

    pbar = tqdm(total=len(test_loader), desc='Validation', dynamic_ncols=True)
    for i, sample in enumerate(test_loader):
        with torch.no_grad():
            # a. get origin image
            image_ori = test_loader.dataset.get_item_image_ori(i)
            image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
            file_name = sample['file_name']
            file_name = file_name[0]

            # b. image preprocessing
            tensor_img = sample['image']
            tensor_img = tensor_img.float().cuda()

            # c. model inference
            heatmap_outs = net(tensor_img)                              # torch.Size([1, 17, 48, 64])
            heatmap_out = heatmap_outs[0].squeeze().cpu().data.numpy()  # (17, 64, 64)

            # tensor_img
            input_img = tensor_img[0].cpu().numpy()  # CHW format

            # denormalize: reverse of normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            input_img = (input_img * std[:, None, None]) + mean[:, None, None]
            input_img = np.clip(input_img * 255.0, 0, 255)

            input_img = input_img.transpose(1, 2, 0)
            input_img = input_img.astype(np.uint8)

            # viz heatmap output
            # fig = visualize_heatmaps(image_ori, input_img, heatmap_out)
            # plt.close(fig)

            # d. heatmap processing and extract keypoints
            shape_big = (sample['image'].shape[-2], sample['image'].shape[-1])
            heatmaps = resize_hm(heatmap_out, shape_big)                # (17, 256, 256)

            background = np.zeros(heatmaps[0].shape)
            heatmaps = np.vstack((heatmaps, background[None, ...]))     # (18, 256, 256)

            # keypoints decoding
            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            keypoints = decode_pose_single_person(heatmaps, param) # (17, 3)


            # keypoints normalization and covert to origin coordinate
            normalized_coord = np.zeros_like(keypoints[:, :2])
            normalized_coord[:, 0] = (keypoints[:, 0] / heatmaps[0].shape[1] - 0.5) * 2.0  # -1 ~ 1 width
            normalized_coord[:, 1] = (keypoints[:, 1] / heatmaps[0].shape[0] - 0.5) * 2.0  # -1 ~ 1 height
            keypoints = get_keypoints_ori_coco(input_img, normalized_coord, keypoints, sample)

            # Threshold
            for j in range(keypoints.shape[0]):
                if keypoints[j, -1] > Thresh:
                    keypoints[j, -1] = 1
                else:
                    keypoints[j, -1] = 0

            # e. save result into COCO format
            # extract (COCO: XXXXXXXXXXXX.jpg -> XXXXXXXXXXXX)
            image_id = int(file_name.split('.')[0])
            key = keypoints.flatten().tolist()
            scores = float(np.mean(keypoints[:, 2]))

            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': key,
                'score': scores
            })


            if val_pbar is not None:
                val_pbar.update(1)
            pbar.update(1)

            if visualize:
                color_style = ColorStyle()
                fig = visualize_keypoints(image_ori, keypoints, color_style)
                plt.show()
                key = cv2.waitKey()
                if key == 27:  # esc
                    return
                plt.close()

    pbar.close()

    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    run_coco_eval(labels, output_name)


if __name__ == '__main__':
    args = ConfigOmniPose().parse()
    model = get_omnipose(cfg, is_train=True)

    checkpoint = torch.load(args.checkpoint_path)
    load_state(model, checkpoint)


    test(args.val_labels, args.val_output_name, model, args, args.visualize)
