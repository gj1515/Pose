import numpy as np
import math
import torchvision.transforms as transforms



def depth_sigma(net_size):
    # sigma = 1.6  # in 32x32 == depth_sigma_ori 12.8 in 256x256 (0.05)
    # sigma = 1.8 # in == 40x40 depth_sigma_ori 14.4 in 320x320 (0.045)
    # sigma = 2.0 # in 46x46 == depth_sigma_ori 16 in 368x368 (0.434)
    sigma = 0
    if net_size == 368:
        sigma = 2.0
    elif net_size == 320:
        sigma = 1.8
    elif net_size == 256:
        sigma = 1.6
    else:
        if net_size < 368 and net_size > 320:
            sigma = 1.9
        elif net_size < 320 and net_size > 256:
            sigma = 1.7
        else:
            sigma = 1.8

        print('  -Warning: unknown net_size for depth_sigma:', net_size)
    return sigma


def normalize_image(img_ori):
    # BGR !!!
    #img_ori = img_ori[:, :, ::-1]  # reverse_channels -> RGB
    img_nor = img_ori.astype(np.float32)
    img_nor = (img_nor - 128) / 256 # [-0.5 +0.5]
    # H x W x C -> C x H x W
    return img_nor.transpose(2, 0, 1) # channel first


def denormalize_image(img_nor):
    # C x H x W  ->  H x W x C
    #img_nor = img_nor[:, :, ::-1]  # reverse_channels -> BGR
    img_ori = img_nor.transpose(1, 2, 0) # channel last
    img_ori = img_ori * 256 + 128
    return img_ori # BGR !!!


def binary_viz(keypoints):
    keypoints[:, -1] = (keypoints[:, -1] >= 0.49).astype('float32')
    return keypoints


def normalize_keypoints(keypoints, netSize):  # (x, y)
    keypoints[:, :2] = (keypoints[:, :2] * 2 + 1) / netSize - 1
    return keypoints


def denormalize_keypoints(keypoints, netSize):
    keypoints = ((keypoints[:, :2] + 1) * netSize - 1) * 0.5
    return keypoints # not overwrite !


def resize_keypoints(keypoints, w, h):
    keypoints[:, 0] = ((keypoints[:, 0] + 1) * w - 1) * 0.5
    keypoints[:, 1] = ((keypoints[:, 1] + 1) * h - 1) * 0.5
    return keypoints


def add_gaussian(keypoint_map, x, y, stride, sigma):
    n_sigma = 4
    tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
    tl[0] = max(tl[0], 0)
    tl[1] = max(tl[1], 0)

    br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
    map_h, map_w = keypoint_map.shape
    br[0] = min(br[0], map_w * stride)
    br[1] = min(br[1], map_h * stride)

    shift = stride / 2 - 0.5
    for map_y in range(tl[1] // stride, br[1] // stride):
        for map_x in range(tl[0] // stride, br[0] // stride):
            d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                 (map_y * stride + shift - y) * (map_y * stride + shift - y)
            exponent = d2 / 2 / sigma / sigma
            if exponent > 4.6052:  # threshold, ln(100), ~0.01
                continue
            keypoint_map[map_y, map_x] += math.exp(-exponent)
            if keypoint_map[map_y, map_x] > 1:
                keypoint_map[map_y, map_x] = 1

    return keypoint_map


def set_paf(paf_map, x_a, y_a, x_b, y_b, stride, thickness):
    x_a /= stride
    y_a /= stride
    x_b /= stride
    y_b /= stride
    x_ba = x_b - x_a
    y_ba = y_b - y_a
    _, h_map, w_map = paf_map.shape
    x_min = int(max(min(x_a, x_b) - thickness, 0))
    x_max = int(min(max(x_a, x_b) + thickness, w_map))
    y_min = int(max(min(y_a, y_b) - thickness, 0))
    y_max = int(min(max(y_a, y_b) + thickness, h_map))
    norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
    if norm_ba < 1e-7:  # Same points, no paf
        return
    x_ba /= norm_ba
    y_ba /= norm_ba

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            x_ca = x - x_a
            y_ca = y - y_a
            d = math.fabs(x_ca * y_ba - y_ca * x_ba)
            if d <= thickness:
                paf_map[0, y, x] = x_ba
                paf_map[1, y, x] = y_ba

    return paf_map


def normalize_depth(depth):  # inverse depth: 0-2 meter -> 1-0
    return (2. - depth) / 2.


def denormalize_depth(depth):  # inv_dep to meter : 0-1 -> 2-0 meter
    return 2. - depth * 2.


def _split_dataset(labels, split_ratio):
    num_ori = len(labels)
    num_split = int(split_ratio * num_ori)

    split_indices = np.random.choice(num_ori, num_split, replace=False)  # replace=False: Unique
    split_data = []
    split_data.extend([labels[i] for i in split_indices])
    print('split data: {} -> {}'.format(num_ori, len(split_data)))
    return split_data