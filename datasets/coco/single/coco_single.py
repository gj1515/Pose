from .transform_coco_single import ObjCrop, ObjScale_coco, BoneFlip, ImageRotate
from datasets.Robot.transform_robot import ImageColorCh
from .preprocess_single_person_coco import preprocess_single_person_coco
from datasets.Robot.helper import normalize_image, binary_viz, normalize_keypoints, add_gaussian, set_paf, _split_dataset


from torchvision import transforms
from torchvision.transforms import Normalize, ToTensor
import os
import numpy as np
import cv2
import copy
from torch.utils.data import Dataset

import pycocotools



class CocoDataset_SinglePerson(Dataset):
    def __init__(self, config, type='train', is_regress=False, need_paf=False, transform=None):
        self.config = config
        self.transform = transform
        self.labels = preprocess_single_person_coco(config, type)
        # labels = {
        #     'img_file': images_info[annotation['image_id']]['file_name'],
        #     'img_width': images_info[annotation['image_id']]['width'],
        #     'img_height': images_info[annotation['image_id']]['height'],
        #     'obj_pos': person_center,
        #     'obj_bbox': annotation['bbox']
        #     'keypoints': np.zeros((18, 3), dtype='float32')
        # }

        self.img_dir = os.path.join(config.data, type + str(2017))
        print('Loaded {:,} person images for {}'.format(len(self.labels), type))

        self._paf_thickness = 1
        self._heat_sigma = 3

        self._is_regress = is_regress
        self._need_paf = need_paf

        # new(omnipose)
        self.num_joints = self.config.skeleton.num_joints
        self.sigma = self._heat_sigma
        self.heatmap_size = [
            self.config.netSize[0] // self.config.netOutScale,
            self.config.netSize[1] // self.config.netOutScale
        ]
        self.use_different_joints_weight = False
        self.joints_weight = np.ones((self.num_joints, 1), dtype=np.float32)

    def get_item_raw(self, index):
        label = copy.deepcopy(self.labels[index])  # label modified in transform
        img_path = os.path.join(self.img_dir, label['img_file'])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)    #uint8, bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # RGB

        sample = {'label': label, 'image': img}

        if self.transform:
            sample = self.transform(sample)

        if not self._is_regress:
            joints = sample['label']['keypoints'][:, :2]
            joints_vis = (sample['label']['keypoints'][:, 2:] >= 0.5).astype(np.float32)
            sample['heat_maps'], sample['target_weight'] = self.generate_target(joints, joints_vis)

        if self._need_paf:
            sample['paf_maps'] = self._generate_paf_maps(sample)

        return sample
    """
    def _generate_keypoint_maps(self, sample):
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(self.config.skeleton.num_joints,
                                        n_rows // self.config.netOutScale, n_cols // self.config.netOutScale), dtype=np.float32)
        label = sample['label']
        for keypoint_idx in range(self.config.skeleton.num_joints):
            keypoint = label['keypoints'][keypoint_idx]
            if keypoint[-1] >= 0.5:
                keypoint_maps[keypoint_idx] = add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self.config.netOutScale, self._heat_sigma)
        return keypoint_maps
    """

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]


        target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

        tmp_size = self.sigma * 3

        scale_x = self.heatmap_size[0] / self.config.netSize[0]
        scale_y = self.heatmap_size[1] / self.config.netSize[1]

        for joint_id in range(self.num_joints):
            target_weight[joint_id] = \
                    self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)

            if target_weight[joint_id] == 0:
                continue

            mu_x = joints[joint_id][0] * scale_x
            mu_y = joints[joint_id][1] * scale_y

            x = np.arange(0, self.heatmap_size[0], 1, np.float32)
            y = np.arange(0, self.heatmap_size[1], 1, np.float32)
            y = y[:, np.newaxis]

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

        return target, target_weight

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        scale_x = self.heatmap_size[0] / self.config.netSize[0]
        scale_y = self.heatmap_size[1] / self.config.netSize[1]

        mu_x = joint[0] * scale_x
        mu_y = joint[1] * scale_y
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight

    def _generate_paf_maps(self, sample):
        n_pafs = self.config.skeleton.num_bones  # 17

        n_rows, n_cols, _ = sample['image'].shape
        paf_maps = np.zeros(shape=(n_pafs * 2, n_rows // self.config.netOutScale, n_cols // self.config.netOutScale),
                            dtype=np.float32)
        label = sample['label']
        for paf_idx in range(n_pafs):
            keypoint_a = label['keypoints'][self.config.skeleton.bones[paf_idx][0]]
            keypoint_b = label['keypoints'][self.config.skeleton.bones[paf_idx][1]]

            if keypoint_a[-1] > 0 and keypoint_b[-1] > 0:
                paf_maps[paf_idx * 2:paf_idx * 2 + 2] = set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                                                                keypoint_a[0], keypoint_a[1], keypoint_b[0],
                                                                keypoint_b[1],
                                                                self.config.netOutScale, self._paf_thickness)
        return paf_maps

    def split_dataset(self, split_ratio):
        self.labels = _split_dataset(self.labels, split_ratio)

    def __getitem__(self, index):
        sample = self.get_item_raw(index)
        sample['label']['keypoints'] = binary_viz(sample['label']['keypoints'])  # [0 or 1] viz for regression
        sample['file_name'] = sample['label']['img_file']

        # Normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(), normalize, ])
        sample['image'] = transform(sample['image'])
        # sample['image'] = normalize_image(sample['image'])


        if self._is_regress:
            sample['label']['keypoints'] = normalize_keypoints(sample['label']['keypoints'], self.config.netSize)
            # [-1,1] for regression

        return sample

    def __len__(self):
        return len(self.labels)

    # ------- only for eval data ! ----------------
    def get_item_image_ori(self, index):
        label = self.labels[index]
        img_path = os.path.join(self.img_dir, label['img_file'])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # uint8, bgr
        return img

def create_coco_dataset(config):
    # -----------------------------------------------------------------
    need_paf = config.need_paf # currently not using
    # -----------------------------------------------------------------

    if config.dataset == 'coco_eval':
        trans_val = transforms.Compose([
            ObjScale_coco(prob=0., target_size=[config.netSize[0], config.netSize[1]], target_dist=1.0, min_scale=1., max_scale=1.),
            ObjCrop(crop_size=[config.netSize[0], config.netSize[1]], center_perterb_max=0)])
        val_set = CocoDataset_SinglePerson(config, 'train', is_regress=config.is_regress, need_paf=need_paf,
                                           transform=trans_val)
        return val_set, val_set
    else:
        trans_train = transforms.Compose([
            ImageColorCh(prob=0.9, contrast=0.2, brightness=20),
            ObjScale_coco(prob=1.0, target_size=[config.netSize[0], config.netSize[1]], target_dist=1.0, min_scale=0.5, max_scale=1.1),
            ImageRotate(max_rotate_degree=40),
            ObjCrop(crop_size=[config.netSize[0], config.netSize[1]], center_perterb_max=20), #translation
            BoneFlip(flipPairList=config.skeleton.flipIndices)])

        train_set = CocoDataset_SinglePerson(config, 'train', is_regress=config.is_regress, need_paf=need_paf, transform=trans_train)
        train_set.split_dataset(config.train_split)

        trans_val = transforms.Compose([
            ObjScale_coco(prob=0., target_size=[config.netSize[0], config.netSize[1]], target_dist=1.0, min_scale=1., max_scale=1.),
            ObjCrop(crop_size=[config.netSize[0], config.netSize[1]], center_perterb_max=0)])

        val_set = CocoDataset_SinglePerson(config, 'val', is_regress=config.is_regress, need_paf=need_paf, transform=trans_val)
        val_set.split_dataset(config.val_split)
        return train_set, val_set

