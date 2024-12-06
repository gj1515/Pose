from datasets.Robot.helper import normalize_image, denormalize_keypoints, resize_keypoints
from testing.decode_net import *
import cv2
import torch

"""
def get_keypoints_ori(image_ori, coord_out, keypoints):
    height_ori, width_ori, _ = image_ori.shape
    is_horizontal_image = width_ori > height_ori
    pad_ori = abs(width_ori - height_ori) // 2

    if is_horizontal_image:
        keypoints[:, :2] = resize_keypoints(coord_out, width_ori, width_ori)  # 1280 x 1280
        keypoints[:, 1] = keypoints[:, 1] - pad_ori
    else:
        keypoints[:, :2] = resize_keypoints(coord_out, height_ori, height_ori)  # 1280 x 1280
        keypoints[:, 0] = keypoints[:, 0] - pad_ori

    return keypoints
"""

def get_keypoints_ori(image_ori, coord_out, keypoints):
    height_ori, width_ori, _ = image_ori.shape

    keypoints[:, :2] = resize_keypoints(coord_out, width_ori, height_ori)
    return keypoints


def get_keypoints_ori_coco(image_ori, coord_out, keypoints, sample):
    height_ori, width_ori, _ = image_ori.shape

    # 1. normalize
    keypoints_ori = keypoints.copy()
    keypoints_ori[:, :2] = resize_keypoints(coord_out, width_ori, height_ori)

    # 3. ObjCrop
    offset_left = float(sample['label']['offset_left'].cpu().numpy())
    offset_up = float(sample['label']['offset_up'].cpu().numpy())

    keypoints_ori[:, 0] -= offset_left
    keypoints_ori[:, 1] -= offset_up

    # 2. ObjScale_coco
    scale = float(sample['label']['scale'].cpu().numpy())
    keypoints_ori[:, 0:2] /= scale

    return keypoints_ori



def run_regress_draw(config, model, dataset, index):
    stage_idx = -1
    netOutSize = dataset.config.netSize // dataset.config.netOutScale

    Thresh = 0.1
    with torch.no_grad():
        image_ori = dataset.get_item_image_ori(index)
        # ------------------------
        keypoints_gt, category_gt, bbox_gt, area_gt, image_path = dataset.get_item_gt(index)

        sample = dataset.get_item_raw(index)
        nor_image = sample['image'].copy()
        nor_image = normalize_image(nor_image) # ch first

        tensor_img = torch.from_numpy(nor_image).unsqueeze(0).float() # torch.Size([1, 3, 256, 256])
        tensor_img = tensor_img.cuda()

        coords_outs, heatmap_nors, _ = model(tensor_img)

        coord_out = coords_outs[stage_idx].squeeze().cpu().data.numpy() # (13, 2)
        heat_nor = heatmap_nors[stage_idx].squeeze().cpu().data.numpy() # (13, 64, 64)
        heat_coord = denormalize_keypoints(coord_out, netOutSize)
        keypoints = decode_regress(heat_coord, heat_nor, thresh=Thresh)

        keypoints = get_keypoints_ori(image_ori, coord_out, keypoints)

        img_key = draw_keypoints_gt(image_ori, config.skeleton.bones, keypoints_gt, 0.499, (0, 255, 0), (0, 255, 0))
        img_key = draw_bbox(img_key, bbox_gt)

        img_key = draw_keypoints(img_key, config.skeleton.bones, keypoints, 0.499, (0, 255, 255), config.skeleton.bone_colors)


        cv2.imshow('keypoints', img_key)
        cv2.waitKey(0)
