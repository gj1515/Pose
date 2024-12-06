import random
import cv2
import numpy
import numpy as np


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


class FlipSelected:
    def __init__(self, flipPairList):
        self._flipIndices = flipPairList

    def __call__(self, sample):
        label = sample['label']
        do_flip = label['flip'] > 0
        if not do_flip:
            return sample

        sample['image'] = cv2.flip(sample['image'], 1)


        h, w, _ = sample['image'].shape

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

        sample['image'] = cv2.warpAffine(sample['image'], R, dsize=(sample['image'].shape[:2]),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=self._pad)

        label = sample['label']
        keypoints = label['keypoints']

        keypoints[:, :2] = np.matmul(R[:, :2], keypoints[:, :2].transpose()).transpose() + R[:, 2]
        label['keypoints'] = keypoints
        return sample


class ImageColorAug:
    def __init__(self, prob=0.5, contrast=0.2, brightness=30):
        self._contrast = contrast
        self._brightness = brightness
        self._prob = prob

    def __call__(self, sample):
        prob = random.random()
        do_aug = prob <= self._prob
        if not do_aug:
            return sample

        alpha = 1 + (random.random() - 0.5) * 2 * self._contrast
        beta = (random.random() - 0.5) * 2 * self._brightness
        sample['image'] = (np.clip(sample['image'] * alpha + beta, 0, 255)).astype('uint8') # uint8 -> float64 -> uint8

        return sample


class ImageColorCh: # each channel differently
    def __init__(self, prob=0.5, contrast=0.2, brightness=30):
        self._contrast = contrast
        self._brightness = brightness
        self._prob = prob

    def __call__(self, sample):
        prob = random.random()
        do_aug = prob <= self._prob
        if not do_aug:
            return sample

        beta = (random.random() - 0.5) * 2 * self._brightness
        alpha1 = 1 + (random.random() - 0.5) * 2 * self._contrast
        alpha2 = 1 + (random.random() - 0.5) * 2 * self._contrast
        alpha3 = 1 + (random.random() - 0.5) * 2 * self._contrast
        sample['image'][:, :, 0] = np.clip(sample['image'][:, :, 0] * alpha1 + beta, 0, 255).astype('uint8')
        sample['image'][:, :, 1] = np.clip(sample['image'][:, :, 1] * alpha2 + beta, 0, 255).astype('uint8')
        sample['image'][:, :, 2] = np.clip(sample['image'][:, :, 2] * alpha3 + beta, 0, 255).astype('uint8')
        return sample


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


# https://github.com/JimmySuen/integral-human-pose/ - Integral pose estimation,
# This paper has very good results on single person pose
def gen_trans_from_patch_sr(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot_rad, dx, dy):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center + [dx, dy]
    dst[1, :] = dst_center + dst_downdir + [dx, dy]
    dst[2, :] = dst_center + dst_rightdir + [dx, dy]
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def gen_trans_from_patch_srt(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot_rad, sx, sy, dx, dy):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center + [sx, sy]
    src[1, :] = src_center + src_downdir + [sx, sy]
    src[2, :] = src_center + src_rightdir + [sx, sy]

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center + [dx, dy]
    dst[1, :] = dst_center + dst_downdir + [dx, dy]
    dst[2, :] = dst_center + dst_rightdir + [dx, dy]
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


# scale & rotation
class ScaleRot:
    def __init__(self, prob_scale=0, prob_rot=0, min_scale=0.5, max_scale=1.1, max_rotate_degree=40, target_size=368, interp_method=cv2.INTER_CUBIC):
        self._prob_scale = prob_scale
        self._prob_rot = prob_rot
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._max_rotate_rad = max_rotate_degree * np.pi / 180.
        self._target_size = target_size
        self._target_shape = (target_size, target_size)
        self._interp_method = interp_method # INTER_LINEAR, INTER_CUBIC

    def __call__(self, sample):
        prob = random.random()
        h, w, _ = sample['image'].shape
        c_y = h / 2
        c_x = w / 2

        scale = 1
        if prob <= self._prob_scale:
            prob = random.random()
            scale = (self._max_scale - self._min_scale) * prob + self._min_scale

        prob = random.random()
        angle_rad = 0
        if prob <= self._prob_rot:
            prob = random.random()
            angle_rad = (prob - 0.5) * 2 * self._max_rotate_rad

        # image size scale added here..
        # Aspect ratio Wrong !
        #transform = gen_trans_from_patch_sr(c_x, c_y, w, h, self._target_size, self._target_size, scale, angle_rad)

        # keep aspect ratio !
        if w > h:
            len_src = w
        else:
            len_src = h

        scale_transform = self._target_size / len_src
        sw = w * scale_transform
        sh = h * scale_transform

        dest_x = 0
        dest_y = 0
        if w > h:
            pad = (sw - sh) * 0.5
            dest_y += pad
        else:
            pad = (sh - sw) * 0.5
            dest_x += pad
        transform = gen_trans_from_patch_sr(c_x, c_y, w, h, sw, sh, scale, angle_rad, dest_x, dest_y)


        sample['image'] = cv2.warpAffine(sample['image'], transform, self._target_shape, flags=self._interp_method,
                                         borderValue=(128, 128, 128))

        label = sample['label']
        keypoints = label['keypoints']
        keypoints[:, :2] = np.matmul(transform[:, :2],
                                               keypoints[:, :2].transpose()).transpose() + transform[:, 2]

        label['keypoints'] = keypoints

        return sample


# scale & rotation & translation
class ScaleRotTrans:
    def __init__(self, prob_scale=0, prob_rot=0, prob_trans=0,
                 min_scale=0.5, max_scale=1.1, max_rotate_degree=40, max_trans_pix=40,
                 target_size=368, interp_method=cv2.INTER_CUBIC):
        self._prob_scale = prob_scale
        self._prob_rot = prob_rot
        self._prob_trans = prob_trans
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._max_rotate_rad = max_rotate_degree * np.pi / 180.
        self._max_trans_pix = max_trans_pix
        self._target_size = target_size
        self._target_shape = (target_size, target_size)
        self._interp_method = interp_method # INTER_LINEAR, INTER_CUBIC

    def __call__(self, sample):
        prob = random.random()
        h, w, _ = sample['image'].shape
        c_y = h / 2
        c_x = w / 2

        scale = 1
        if prob <= self._prob_scale:
            prob = random.random()
            scale = (self._max_scale - self._min_scale) * prob + self._min_scale
            scale = 1./scale
            # The scale will be multiplied with the src, hence it should be reciprocal

        prob = random.random()
        angle_rad = 0
        if prob <= self._prob_rot:
            prob = random.random()
            angle_rad = (prob - 0.5) * 2 * self._max_rotate_rad

        prob = random.random()
        dx=0.
        dy=0.
        if prob <= self._prob_trans:
            prob = random.random()
            dx = (prob - 0.5) * 2 * self._max_trans_pix
            prob = random.random()
            dy = (prob - 0.5) * 2 * self._max_trans_pix

        # Aspect ratio Wrong !
        #transform = gen_trans_from_patch_srt(c_x, c_y, w, h, self._target_size, self._target_size, scale, angle_rad, dx, dy)


        # keep aspect ratio !
        if w > h: len_src = w
        else: len_src = h

        scale_transform = self._target_size / len_src
        sw = w * scale_transform
        sh = h * scale_transform

        dest_x = 0
        dest_y = 0
        if w > h:
            pad = (sw - sh) * 0.5
            dest_y += pad
        else:
            pad = (sh - sw) * 0.5
            dest_x += pad

        transform = gen_trans_from_patch_srt(c_x, c_y, w, h, sw, sh, scale, angle_rad, dx, dy, dest_x, dest_y)

        sample['image'] = cv2.warpAffine(sample['image'], transform, self._target_shape, flags=self._interp_method,
                                         borderValue=(128, 128, 128))

        label = sample['label']
        keypoints = label['keypoints']
        keypoints[:, :2] = np.matmul(transform[:, :2],
                                               keypoints[:, :2].transpose()).transpose() + transform[:, 2]

        label['keypoints'] = keypoints

        return sample


# check out of image after transform
class CheckOutOfImage:
    def __init__(self):
        pass

    def __call__(self, sample):
        h, w, _ = sample['image'].shape
        label = sample['label']
        keypoints = label['keypoints']

        for i in range(keypoints.shape[0]):
            keypoints[i] = self.check_key4d(keypoints[i], w, h)

        label['keypoints'] = keypoints
        return sample

    def check_key4d(self, keypoint, w, h):
        if keypoint[0] < 0 or keypoint[0] >= w or keypoint[1] < 0 or keypoint[1] >= h:
            keypoint[3] = 0
        else:
            keypoint[3] = 1

        return keypoint


class ImageBlurNoise:
    def __init__(self, prob_blur=0, prob_noise=0,
                 blur_max_sigma=2, noise_max_intensity=25):
        self._prob_blur = prob_blur
        self._prob_noise = prob_noise
        self._blur_max_sigma = blur_max_sigma
        self._noise_max_intensity = noise_max_intensity

    def __call__(self, sample):

        prob = random.random()
        if prob <= self._prob_blur:
            blur_sigma = random.random() * self._blur_max_sigma
            sample['image'] = cv2.GaussianBlur(sample['image'], (0, 0), blur_sigma)
        #     print('bluerred!' + str(blur_sigma))
        # else:
        #     print('no bluerred!')

        prob = random.random()
        if prob <= self._prob_noise:
            nose_intensity = random.random() * self._noise_max_intensity
            noise = np.random.randint(-nose_intensity, nose_intensity + 1, sample['image'].shape) # random int noise
            #noise = np.random.normal(0, nose_intensity, sample['image'].shape).astype('uint8') # Gaussian noise bad result !
            sample['image'] = np.clip(sample['image'] + noise, 0, 255).astype('uint8')
        #     print('noised!' + str(nose_intensity))
        # else:
        #     print('no noised!')

        return sample




class ScaleRotTransObjSize:
    def __init__(self, prob_scale=0, prob_rot=0, prob_trans=0,
                 min_scale=0.5, max_scale=1.1, max_rotate_degree=40, max_trans_ratio=0.1,
                 target_size=368, interp_method=cv2.INTER_CUBIC):
        self._prob_scale = prob_scale
        self._prob_rot = prob_rot
        self._prob_trans = prob_trans
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._max_rotate_rad = max_rotate_degree * np.pi / 180.
        self._max_trans_ratio = max_trans_ratio
        self._target_size = target_size
        self._target_shape = (target_size, target_size)
        self._interp_method = interp_method # INTER_LINEAR, INTER_CUBIC

        self._target_scale = 0.6 # the portion of the object on image

    def __call__(self, sample):
        h, w, _ = sample['image'].shape
        c_y = h / 2
        c_x = w / 2

        obj_scale, obj_center = self.find_obj_scale_center(sample) # image center coordinates

        prob = random.random()
        scale = 1
        if prob <= self._prob_scale:
            scale_abs = self._target_scale / obj_scale

            prob = random.random()
            scale_multiplier = (self._max_scale - self._min_scale) * prob + self._min_scale
            scale = scale_abs * scale_multiplier

            #print('obj_scale:', obj_scale, 'scale:', scale_abs)

            scale = 1./scale
            # The scale will be multiplied with the src, hence it should be reciprocal


        prob = random.random()
        angle_rad = 0
        if prob <= self._prob_rot:
            prob = random.random()
            angle_rad = (prob - 0.5) * 2 * self._max_rotate_rad

        prob = random.random()
        dx=0.
        dy=0.
        if prob <= self._prob_trans:
            trans_dir = -obj_center/abs(obj_center)

            max_trans_pix = self._max_trans_ratio * min(h, w)

            prob = random.random()
            dx = prob * max_trans_pix * trans_dir[0]
            prob = random.random()
            dy = prob * max_trans_pix * trans_dir[1]

        # Aspect ratio Wrong !
        #transform = gen_trans_from_patch_srt(c_x, c_y, w, h, self._target_size, self._target_size, scale, angle_rad, dx, dy)


        # keep aspect ratio !
        if w > h: len_src = w
        else: len_src = h

        scale_transform = self._target_size / len_src
        sw = w * scale_transform
        sh = h * scale_transform

        dest_x = 0
        dest_y = 0
        if w > h:
            pad = (sw - sh) * 0.5
            dest_y += pad
        else:
            pad = (sh - sw) * 0.5
            dest_x += pad

        transform = gen_trans_from_patch_srt(c_x, c_y, w, h, sw, sh, scale, angle_rad, dx, dy, dest_x, dest_y)

        sample['image'] = cv2.warpAffine(sample['image'], transform, self._target_shape, flags=self._interp_method,
                                         borderValue=(128, 128, 128))

        label = sample['label']
        keypoints = label['keypoints']
        keypoints[:, :2] = np.matmul(transform[:, :2],
                                               keypoints[:, :2].transpose()).transpose() + transform[:, 2]

        label['keypoints'] = keypoints

        return sample

    def find_obj_scale_center(self, sample):
        h, w, _ = sample['image'].shape
        center = np.array([w/2.0, h/2.0])

        keypoints = sample['label']['keypoints']
        num_keypoints, _ = keypoints.shape

        valid_keypoints = []
        for keypoint in keypoints:
            if keypoint[-1] >= 0.5:
                valid_keypoints.append(keypoint[:-1])

        if len(valid_keypoints) < 2:
            obj_scale = self._target_scale
            obj_center = valid_keypoints[0] - center  # image center coordinates
            return obj_scale, obj_center

        arr = np.array(valid_keypoints)
        obj_len = arr.max(axis=0) - arr.min(axis=0)
        obj_scale_pix = obj_len / np.array([w, h])

        obj_center = arr.mean(axis=0) - center # image center coordinates
        obj_scale = obj_scale_pix.max()
        return obj_scale, obj_center