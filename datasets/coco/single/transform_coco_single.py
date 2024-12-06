import random
import cv2
import numpy as np


class ObjCrop:
    def __init__(self, crop_size=368, center_perterb_max=40, pad=(128, 128, 128)):
        if isinstance(crop_size, (list, tuple)):
            self._crop_width = crop_size[0]
            self._crop_height = crop_size[1]
            self._crop_half_w = self._crop_width / 2
            self._crop_half_h = self._crop_height / 2
        else:
            self._crop_width = crop_size
            self._crop_height = crop_size
            self._crop_half_w = crop_size / 2
            self._crop_half_h = crop_size / 2
        self._center_perterb_max = center_perterb_max
        self._pad = pad

    def __call__(self, sample):
        prob_x = random.random()
        prob_y = random.random()

        offset_x = int((prob_x - 0.5) * 2 * self._center_perterb_max)
        offset_y = int((prob_y - 0.5) * 2 * self._center_perterb_max)
        label = sample['label']
        shifted_center = (label['obj_pos'][0] + offset_x, label['obj_pos'][1] + offset_y)
        offset_left = -int(shifted_center[0] - self._crop_half_w)
        offset_up = -int(shifted_center[1] - self._crop_half_h)

        label['offset_left'] = offset_left
        label['offset_up'] = offset_up

        cropped_image = np.empty(shape=(self._crop_height, self._crop_width, 3), dtype=np.uint8)
        for i in range(3):
            cropped_image[:, :, i].fill(self._pad[i])

        image_x_start = int(shifted_center[0] - self._crop_half_w)
        image_y_start = int(shifted_center[1] - self._crop_half_h)
        image_x_finish = image_x_start + self._crop_width
        image_y_finish = image_y_start + self._crop_height
        crop_x_start = 0
        crop_y_start = 0
        crop_x_finish = self._crop_width
        crop_y_finish = self._crop_height

        w, h = label['img_width'], label['img_height']
        should_crop = True
        if image_x_start < 0:  # Adjust crop area
            crop_x_start -= image_x_start
            image_x_start = 0
        if image_x_start >= w:
            should_crop = False

        if image_y_start < 0:
            crop_y_start -= image_y_start
            image_y_start = 0
        if image_y_start >= w:
            should_crop = False

        if image_x_finish > w:
            diff = image_x_finish - w
            image_x_finish -= diff
            crop_x_finish -= diff
        if image_x_finish < 0:
            should_crop = False

        if image_y_finish > h:
            diff = image_y_finish - h
            image_y_finish -= diff
            crop_y_finish -= diff
        if image_y_finish < 0:
            should_crop = False


        if should_crop:
            cropped_image[crop_y_start:crop_y_finish, crop_x_start:crop_x_finish, :] =\
                sample['image'][image_y_start:image_y_finish, image_x_start:image_x_finish, :]

        sample['image'] = cropped_image
        label['img_width'] = self._crop_width
        label['img_height'] = self._crop_height

        label['obj_pos'][0] += offset_left
        label['obj_pos'][1] += offset_up

        keypoints = label['keypoints']
        valid_keys = keypoints[:, 2] > 0.5
        keypoints[valid_keys, 0] += offset_left
        keypoints[valid_keys, 1] += offset_up
        label['keypoints'] = keypoints

        return sample


class ObjScale_coco:
    def __init__(self, prob=1, min_scale=0.5, max_scale=1.1, target_size=368, target_dist=0.6):
        self._prob = prob
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._target_dist = target_dist
        if isinstance(target_size, (list, tuple)):
            self._target_width = target_size[0]
            self._target_height = target_size[1]
        else:
            self._target_width = target_size
            self._target_height = target_size

    def __call__(self, sample):
        prob = random.random()
        scale_multiplier = 1
        if prob <= self._prob:
            prob = random.random()
            scale_multiplier = (self._max_scale - self._min_scale) * prob + self._min_scale

        label = sample['label']
        obj_scale = sample['label']['obj_bbox'][3] / self._target_height
        scale_abs = self._target_dist / obj_scale

        scale = scale_abs * scale_multiplier

        label['scale'] = scale

        sample['image'] = cv2.resize(sample['image'], dsize=(0, 0), fx=scale, fy=scale)
        label['img_height'], label['img_width'], _ = sample['image'].shape

        label['obj_pos'][0] *= scale
        label['obj_pos'][1] *= scale

        keypoints = label['keypoints'].astype(np.float32)
        keypoints[:, 0:2] *= scale
        label['keypoints'] = keypoints

        return sample


class ObjScale_mpii:
    def __init__(self, prob=1, min_scale=0.5, max_scale=1.1, target_size=368, target_dist=0.6):
        self._prob = prob
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._target_size = target_size
        self._target_dist = target_dist

    def __call__(self, sample):
        prob = random.random()
        scale_multiplier = 1
        if prob <= self._prob:
            prob = random.random()
            scale_multiplier = (self._max_scale - self._min_scale) * prob + self._min_scale

        label = sample['label']
        obj_scale = sample['label']['obj_scale'] * 200. / self._target_size
        scale_abs = self._target_dist / obj_scale

        scale = scale_abs * scale_multiplier

        sample['image'] = cv2.resize(sample['image'], dsize=(0, 0), fx=scale, fy=scale)
        label['img_height'], label['img_width'], _ = sample['image'].shape

        label['obj_pos'][0] *= scale
        label['obj_pos'][1] *= scale

        keypoints = label['keypoints']
        keypoints[:, 0:2] *= scale
        label['keypoints'] = keypoints

        return sample


class BoneFlip:
    def __init__(self, flipPairList, prob=0.5):
        self._prob = prob
        self._flipIndices = flipPairList

    def __call__(self, sample):
        prob = random.random()
        do_flip = prob <= self._prob
        if not do_flip:
            return sample

        sample['image'] = cv2.flip(sample['image'], 1)

        label = sample['label']
        h, w, _ = sample['image'].shape
        label['obj_pos'][0] = w - 1 - label['obj_pos'][0]

        keypoints = label['keypoints']
        keypoints[:, 0] = w - 1 - keypoints[:, 0]
        label['keypoints'] = self._swap_left_right(keypoints)

        return sample

    def _swap_left_right(self, keypoints):
        for pair in self._flipIndices: # (r, l)
            temp_key = keypoints[pair[1]].copy()
            keypoints[pair[1]] = keypoints[pair[0]]
            keypoints[pair[0]] = temp_key
        return keypoints


class UpsideDown: # always happens
    def __init__(self):
        pass

    def __call__(self, sample):
        h, w, _ = sample['image'].shape
        sample['image'] = cv2.flip(sample['image'], -1) # both flip
        label = sample['label']

        label['obj_pos'][0] = w - 1 - label['obj_pos'][0]
        label['obj_pos'][1] = h - 1 - label['obj_pos'][1]

        keypoints = label['keypoints']
        keypoints[:, 0] = w - 1 - keypoints[:, 0]
        keypoints[:, 1] = h - 1 - keypoints[:, 1]
        label['keypoints'] = keypoints
        return sample


class ImageRotate:
    def __init__(self, max_rotate_degree=40, pad=(128, 128, 128)):
        self._pad = pad
        self._max_rotate_degree = max_rotate_degree

    def __call__(self, sample):
        prob = random.random()
        degree = (prob - 0.5) * 2 * self._max_rotate_degree

        h, w, _ = sample['image'].shape
        img_center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(img_center, degree, 1)

        abs_cos = abs(R[0, 0])
        abs_sin = abs(R[0, 1])

        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        dsize = (bound_w, bound_h) # image size changed here !!

        R[0, 2] += dsize[0] / 2 - img_center[0]
        R[1, 2] += dsize[1] / 2 - img_center[1]
        sample['image'] = cv2.warpAffine(sample['image'], R, dsize=dsize,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=self._pad)
        sample['label']['img_height'], sample['label']['img_width'], _ = sample['image'].shape

        label = sample['label']
        label['obj_pos'] = self._rotate(label['obj_pos'], R)

        keypoints = label['keypoints']
        keypoints[:, :2] = np.matmul(R[:, :2], keypoints[:, :2].transpose()).transpose() + R[:, 2]
        label['keypoints'] = keypoints
        return sample

    def _rotate(self, point, R):
        return [R[0, 0] * point[0] + R[0, 1] * point[1] + R[0, 2],
                R[1, 0] * point[0] + R[1, 1] * point[1] + R[1, 2]]