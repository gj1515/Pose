# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
#                                    OmniPose                                    #
#      Rochester Institute of Technology - Vision and Image Processing Lab       #
#                      Bruno Artacho (bmartacho@mail.rit.edu)                    #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from tqdm import tqdm

from config.config_omnipose_model import _C as cfg
from config.config_omnipose import ConfigOmniPose

from modules.load_state import load_state

from utils.omni_utils.inference import get_final_preds_no_transform

from models.omnipose.omnipose import get_omnipose
from datasets.Robot.helper import resize_keypoints
from testing.decode_net import resize_hm
from testing.post_heatmap import decode_pose_single_person
from testing.eval_util import  get_keypoints_ori
from val_omnipose import visualize_keypoints




class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = [pair[:] for pair in link_pairs]
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i]) / 255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i]) / 255.))

        # Red    = (240,  2,127)
        # Yellow = (255,255,  0)
        # Green  = (169,209,142)
        # Pink   = (252,176,243)
        # Blue   = (0,176,240)
        color_ids = [(0, 176, 240), (252, 176, 243), (169, 209, 142), (255, 255, 0), (240, 2, 127)]

        self.color_ids = []
        for i in range(len(color_ids)):
            self.color_ids.append(tuple(np.array(color_ids[i]) / 255.))


color = [(252, 176, 243), (252, 176, 243), (252, 176, 243),
         (0, 176, 240), (0, 176, 240), (0, 176, 240),
         (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127),
         (255, 255, 0), (255, 255, 0), (169, 209, 142),
         (169, 209, 142), (169, 209, 142)]

link_pairs = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], \
              [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], \
              [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]]

point_color = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
               (240, 2, 127), (240, 2, 127),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (252, 176, 243), (0, 176, 240), (252, 176, 243),
               (0, 176, 240), (252, 176, 243), (0, 176, 240),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142)]

artacho_style = ColorStyle(color, link_pairs, point_color)



def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict


def plot_COCO_image(preds, img_path, save_path, link_pairs, ring_color, color_ids, save=True):
    # Read Images
    data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    h = data_numpy.shape[0]
    w = data_numpy.shape[1]

    # Plot
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.subplot(1, 1, 1)
    bk = plt.imshow(data_numpy[:, :, ::-1])
    bk.set_zorder(-1)
    joints_dict = map_joint_dict(preds[0])

    # stick
    for k, link_pair in enumerate(link_pairs):
        color_idx = k % len(color_ids)
        lw = 2
        line = mlines.Line2D(
            np.array([joints_dict[link_pair[0]][0],
                      joints_dict[link_pair[1]][0]]),
            np.array([joints_dict[link_pair[0]][1],
                      joints_dict[link_pair[1]][1]]),
            ls='-', lw=lw, alpha=1, color=color_ids[color_idx], )
        line.set_zorder(0)
        ax.add_line(line)
    # black ring
    for k in range(preds.shape[1]):
        if preds[0, k, 0] > w or preds[0, k, 1] > h:
            continue
        radius = 2

        circle = mpatches.Circle(tuple(preds[0, k, :2]),
                                 radius=radius,
                                 ec='black',
                                 fc=ring_color[k],
                                 alpha=1,
                                 linewidth=1)
        circle.set_zorder(1)
        ax.add_patch(circle)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    print(save_path)
    plt.savefig(save_path, format='jpg', bbox_inches='tight', dpi=100)
    plt.close()


def draw_pose_on_frame(frame, keypoints, colorstyle):
    """Draw pose estimation on a single frame"""
    h, w = frame.shape[:2]
    joints_dict = map_joint_dict(keypoints[0])

    # Draw skeleton lines
    for k, link_pair in enumerate(colorstyle.link_pairs):
        color_idx = k % len(colorstyle.color_ids)
        pt1 = joints_dict[link_pair[0]]
        pt2 = joints_dict[link_pair[1]]
        cv2.line(frame, pt1, pt2,
                 (int(colorstyle.color_ids[color_idx][2] * 255),
                  int(colorstyle.color_ids[color_idx][1] * 255),
                  int(colorstyle.color_ids[color_idx][0] * 255)), 2)

    # Draw keypoints
    for k in range(keypoints.shape[1]):
        if keypoints[0, k, 0] > w or keypoints[0, k, 1] > h:
            continue
        x, y = int(keypoints[0, k, 0]), int(keypoints[0, k, 1])
        cv2.circle(frame, (x, y), 2,
                   (int(colorstyle.ring_color[k][2] * 255),
                    int(colorstyle.ring_color[k][1] * 255),
                    int(colorstyle.ring_color[k][0] * 255)), -1)
        cv2.circle(frame, (x, y), 2, (0, 0, 0), 1)

    return frame


def process_frame(frame, model, transform, model_height, model_width):
    """Process a single frame and return keypoints"""
    origin_img = frame.copy()

    # Resize and preprocess
    input_img = cv2.resize(frame, (model_width, model_height), interpolation=cv2.INTER_AREA)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = transform(input_img)

    input = torch.zeros((1, 3, input_img.shape[1], input_img.shape[2]))
    input[0] = input_img
    input = input.cuda()

    # Get model output
    outputs = model(input)
    output = outputs[0].squeeze().cpu().data.numpy()

    # Process heatmap and extract keypoints
    shape_big = (model_height, model_width)
    heatmaps = resize_hm(output, shape_big)
    background = np.zeros(heatmaps[0].shape)
    heatmaps = np.vstack((heatmaps, background[None, ...]))

    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    keypoints = decode_pose_single_person(heatmaps, param)

    # Normalize and convert coordinates
    normalized_coord = np.zeros_like(keypoints[:, :2])
    normalized_coord[:, 0] = (keypoints[:, 0] / heatmaps[0].shape[1] - 0.5) * 2.0
    normalized_coord[:, 1] = (keypoints[:, 1] / heatmaps[0].shape[0] - 0.5) * 2.0
    keypoints = get_keypoints_ori(origin_img, normalized_coord, keypoints)

    return np.expand_dims(keypoints, axis=0)



def main(model, file_path):
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = model.cuda()
    model.eval()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize, ])

    img_path = os.path.join(file_path)
    image = os.path.basename(img_path)

    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    # Model input size
    model_height, model_width = 256, 192

    # Check if input is video or image
    is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.wmv'))

    if is_video:
        # Video processing
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {file_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video writer
        output_path = os.path.join('results', 'output_' + os.path.basename(file_path))
        os.makedirs('results', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Initialize tqdm progress bar
        pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            keypoints = process_frame(frame, model, transform, model_height, model_width)

            # Draw pose on frame
            frame_with_pose = draw_pose_on_frame(frame, keypoints, artacho_style)

            # Write frame
            out.write(frame_with_pose)

            # Update progress bar
            pbar.update(1)

            # Close progress bar
        pbar.close()

        # Release resources
        cap.release()
        out.release()
        print(f"\nVideo processing complete. Output saved to: {output_path}")
    else:
        # Image processing (original functionality)
        input_img = cv2.imread(file_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        keypoints = process_frame(input_img, model, transform, model_height, model_width)

        # Save output image
        output_path = os.path.join('results', os.path.basename(file_path))
        os.makedirs('results', exist_ok=True)
        plot_COCO_image(keypoints, file_path, output_path, artacho_style.link_pairs,
                        artacho_style.ring_color, artacho_style.color_ids)
        print(f"Image processing complete. Output saved to: {output_path}")


if __name__ == '__main__':
    args = ConfigOmniPose().parse()
    model = get_omnipose(cfg, is_train=True)

    checkpoint = torch.load(args.checkpoint_path)
    load_state(model, checkpoint)

    # Example usage
    file_path = "D:/Dev/Dataset/inputs/solo_dance.mp4"
    main(model, file_path)