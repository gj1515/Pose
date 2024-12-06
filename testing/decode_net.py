from datasets.Robot.helper import denormalize_depth
import numpy as np
import cv2


def read_subpix_bilinear(imArr, posX, posY):  # must be gray image
    # Get integer and fractional parts of numbers
    modXi = int(posX)
    modYi = int(posY)
    modXf = posX - modXi
    modYf = posY - modYi
    modXiPlusOneLim = min(modXi + 1, imArr.shape[1] - 1)
    modYiPlusOneLim = min(modYi + 1, imArr.shape[0] - 1)

    # Get pixels in four corners
    bl = imArr[modYi, modXi]
    br = imArr[modYi, modXiPlusOneLim]
    tl = imArr[modYiPlusOneLim, modXi]
    tr = imArr[modYiPlusOneLim, modXiPlusOneLim]

    # Calculate interpolation
    b = modXf * br + (1. - modXf) * bl
    t = modXf * tr + (1. - modXf) * tl
    pxf = modYf * t + (1. - modYf) * b
    return pxf


def read_subpix_max(imArr, posX, posY):  # must be gray image
    modXi = int(posX)
    modYi = int(posY)
    modXiPlusOneLim = min(modXi + 1, imArr.shape[1] - 1)
    modYiPlusOneLim = min(modYi + 1, imArr.shape[0] - 1)

    # Get pixels in four corners
    bl = imArr[modYi, modXi]
    br = imArr[modYi, modXiPlusOneLim]
    tl = imArr[modYiPlusOneLim, modXi]
    tr = imArr[modYiPlusOneLim, modXiPlusOneLim]

    return max([bl, br, tl, tr])


def decode_regress(coors_pred, heatmap_nors, thresh=0.005, keep_conf=False):
    keypoints_pred = np.zeros([coors_pred.shape[0], 4])
    keypoints_pred[:, :2] = coors_pred
    #thresh = 0.05 # # 46x46 : 0.0004725897920604915
    #thresh = 0.005 # 64x64 : 0.000244
    for i in range(coors_pred.shape[0]):
        coord = coors_pred[i, :]
        keypoints_pred[i, -1] = read_subpix_bilinear(heatmap_nors[i], coord[0], coord[1])

    mask = (keypoints_pred[:, -1] >= thresh).astype('float32')
    keypoints_pred[:, 0] = keypoints_pred[:, 0] * mask
    keypoints_pred[:, 1] = keypoints_pred[:, 1] * mask

    if not keep_conf:
        keypoints_pred[:, -1] = mask
    return keypoints_pred


def decode_depth_cm(depthmaps, keypoints, thresh=0):
    depth_out = np.zeros((keypoints.shape[0]))

    for i in range(keypoints.shape[0]):
        jnt_1 = keypoints[i]

        if jnt_1[3] < thresh:
            continue

        depth = read_subpix_max(depthmaps[i, :, :], jnt_1[0], jnt_1[1])
        #depth = read_subpix_bilinear(depthmaps[i, :, :], jnt_1[0], jnt_1[1])
        depth = denormalize_depth(depth)
        depth_out[i] = depth * 100 # meter to cm

    return depth_out


def draw_keypoints(img, body_parts, keypoints, thresh=0,
                   jnt_color=(0, 1, 0), bone_colors=None, joint_thick=-1, bone_thick=2):    # img = img.copy must be done in advance

    for i in range(keypoints.shape[0]):
        if keypoints[i, -1] > thresh:
            cv2.circle(img, tuple(keypoints[i,:2].astype(int)), 3, jnt_color, joint_thick)

    for i, part in enumerate(body_parts):
        keypoint_1 = keypoints[part[0]]
        keypoint_2 = keypoints[part[1]]
        if keypoint_1[-1] > thresh and keypoint_2[-1] > thresh:
            cv2.line(img, tuple(keypoint_1[:2].astype(int)), tuple(keypoint_2[:2].astype(int)), bone_colors[i], bone_thick)
    return img

def draw_keypoints_gt(img, body_parts, keypoints, thresh=0,
                   jnt_color=(0, 1, 0), bone_color=None,):    # img = img.copy must be done in advance

    for i in range(keypoints.shape[0]):
        if keypoints[i, -1] > thresh:
            cv2.circle(img, tuple(keypoints[i,:2].astype(int)), 2, jnt_color, -1)

    for i, part in enumerate(body_parts):
        keypoint_1 = keypoints[part[0]]
        keypoint_2 = keypoints[part[1]]
        if keypoint_1[-1] > thresh and keypoint_2[-1] > thresh:
            cv2.line(img, tuple(keypoint_1[:2].astype(int)), tuple(keypoint_2[:2].astype(int)), bone_color, 1)
    return img

def draw_bbox(img, bbox):
    tl = (int(bbox[0]), int(bbox[1]))
    br = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
    cv2.rectangle(img, tl, br, color=(0,255,0), thickness=2) # green
    return img

def resize_hm(heatmap, hm_size):
    if np.isscalar(hm_size):
        hm_size = (hm_size, hm_size)
    heatmap = cv2.resize(heatmap.transpose(1, 2, 0), hm_size, interpolation=cv2.INTER_CUBIC)
    return heatmap.transpose(2, 0, 1)


def reshape_paf(paf):
    n_bones = int(paf.shape[0]/2)
    rows = paf.shape[1]
    cols = paf.shape[2]
    paf = paf.reshape(n_bones, 2, rows, cols)
    return paf


def draw_heatmap(img, heat_maps):
    heat_map_max = heat_maps.max(axis=0)
    heat_map_max = (heat_map_max * 255.).astype('uint8')
    img = img.copy()

    colored = cv2.applyColorMap(heat_map_max, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    return img