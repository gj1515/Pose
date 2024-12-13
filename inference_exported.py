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
import numpy as np
from tqdm import tqdm
import onnxruntime as ort

import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from config.config_omnipose_model import _C as cfg
from utils.omni_utils.inference import get_final_preds_no_transform



class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i]) / 255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i]) / 255.))

        # RED    = (240,  2,127)
        # Yellow = (255,255,  0)
        # Green  = (169,209,142)
        # Pink   = (252,176,243)
        # BLUE   = (0,176,240)
        color_ids = [(0, 176, 240), (252, 176, 243), (169, 209, 142), (255, 255, 0), (240, 2, 127)]

        self.color_ids = []
        for i in range(len(color_ids)):
            self.color_ids.append(tuple(np.array(color_ids[i]) / 255.))

#color = [(252, 176, 243), (252, 176, 243), (252, 176, 243),
#         (0, 176, 240), (0, 176, 240), (0, 176, 240),
#         (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127),
#         (255, 255, 0), (255, 255, 0), (169, 209, 142),
#         (169, 209, 142), (169, 209, 142)]

RED = (127, 2, 240)
BLUE = (240, 176, 0)
GREEN  = (142,209,169)

color = [RED,RED,BLUE,BLUE,GREEN,RED,BLUE,GREEN,RED,BLUE,RED,BLUE,GREEN,RED,BLUE,RED, BLUE, RED, BLUE]

link_pairs = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], \
              [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], \
              [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]]

point_color = [RED,RED,BLUE,RED,BLUE,RED,BLUE,RED,BLUE,RED,BLUE,RED,BLUE,RED,BLUE,RED,BLUE]

artacho_style = ColorStyle(color, link_pairs, point_color)


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict


def plot_COCO_image(preds, img_path, save_path, link_pairs, ring_color, model_size):
    # Read Images
    model_width, model_height = model_size
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Failed to load image: {img_path}")
        return

    # frame = cv2.resize(frame, (model_width, model_height), interpolation=cv2.INTER_AREA)

    # Scale predictions to match image size
    h, w = frame.shape[:2]
    preds[0, :, 0] = preds[0, :, 0] * w / model_width
    preds[0, :, 1] = preds[0, :, 1] * h / model_height

    # Draw skeleton and joints using OpenCV
    joints_dict = map_joint_dict(preds[0])

    # Draw lines
    for k, link_pair in enumerate(link_pairs):
        pt1 = tuple(map(int, [joints_dict[link_pair[0]][0], joints_dict[link_pair[0]][1]]))
        pt2 = tuple(map(int, [joints_dict[link_pair[1]][0], joints_dict[link_pair[1]][1]]))
        color = tuple(int(c * 255) for c in link_pair[2])  # Convert normalized RGB to 0-255
        cv2.line(frame, pt1, pt2, color, 2)

    # Draw joints
    for k in range(preds.shape[1]):
        x, y = int(preds[0, k, 0]), int(preds[0, k, 1])
        if 0 <= x < w and 0 <= y < h:  # Check if point is within image bounds
            color = tuple(int(c * 255) for c in ring_color[k])  # Convert normalized RGB to 0-255
            cv2.circle(frame, (x, y), 3, color, -1)  # Filled circle
            cv2.circle(frame, (x, y), 3, (0, 0, 0), 1)  # Black outline

    # Save the result
    print(save_path)
    cv2.imwrite(save_path, frame)


def process_image(ort_session, transform, img_path, file_name, colorstyle, model_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return

    model_width, model_height = model_size

    img = cv2.resize(img, (model_width, model_height), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = transform(img).unsqueeze(0)

    onnx_input = img.cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: onnx_input}
    ort_outs = ort_session.run(None, ort_inputs)

    preds, maxvals = get_final_preds_no_transform(cfg, ort_outs[0])

    save_path = os.path.join('results', f'{file_name}_result.jpg')
    plot_COCO_image(4 * preds, img_path, save_path, colorstyle.link_pairs, colorstyle.ring_color,
                    model_size=model_size)


def process_video(ort_session, transform, video_path, file_name, colorstyle, model_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join('results', f'{file_name}_result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    person_colors = {}
    next_color_index = 0

    for _ in tqdm(range(total_frames), desc=f"Processing {file_name}"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model_width, model_height = model_size
        frame_resized = cv2.resize(frame_rgb, (model_width, model_height), interpolation=cv2.INTER_AREA)

        frame_input = transform(frame_resized).unsqueeze(0)

        onnx_input = frame_input.cpu().numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: onnx_input}
        ort_outs = ort_session.run(None, ort_inputs)

        preds, _ = get_final_preds_no_transform(cfg, ort_outs[0])
        preds *= 4

        preds[0, :, 0] = preds[0, :, 0] * width / model_width
        preds[0, :, 1] = preds[0, :, 1] * height / model_height

        #for person_idx in range(preds.shape[0]):
        #    if person_idx not in person_colors:
        #        person_colors[person_idx] = colorstyle.color_ids[next_color_index % len(colorstyle.color_ids)]
        #        next_color_index += 1

        #    person_color = person_colors[person_idx]
        #    frame = draw_pose_on_frame(frame, preds[person_idx], colorstyle, person_color)

        for person_idx in range(preds.shape[0]):
            frame = draw_pose_on_frame(frame, preds[person_idx], colorstyle)

        out.write(frame)

    cap.release()
    out.release()


def draw_pose_on_frame(frame, pred, colorstyle):
    # Draw skeleton lines
    for k, link_pair in enumerate(colorstyle.link_pairs):
        pt1 = tuple(map(int, pred[link_pair[0], :2]))
        pt2 = tuple(map(int, pred[link_pair[1], :2]))
        cv2.line(frame, pt1, pt2, tuple(int(c * 255) for c in link_pair[2]), 2)

    # Draw joint points
    for k in range(pred.shape[0]):
        x, y = int(pred[k, 0]), int(pred[k, 1])
        cv2.circle(frame, (x, y), 3, tuple(int(c * 255) for c in colorstyle.ring_color[k]), -1)
        cv2.circle(frame, (x, y), 3, (0, 0, 0), 1)  # Black outline

    return frame


def main(args):
    global artacho_style
    colorstyle = artacho_style

    model_height = cfg.MODEL.IMAGE_SIZE[1]
    model_width = cfg.MODEL.IMAGE_SIZE[0]
    model_size = (model_width, model_height)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize, ])

    ort_session = ort.InferenceSession(args.onnx_name)  # load model here

    files_loc = args.fileDir
    files = os.listdir(files_loc)

    for file in files:
        file_path = os.path.join(files_loc, file)
        file_name, file_ext = os.path.splitext(file)

        if file_ext.lower() in ['.jpg', '.jpeg', '.png']:
            process_image(ort_session, transform, file_path, file_name, colorstyle, model_size)
        elif file_ext.lower() == '.mp4':
            process_video(ort_session, transform, file_path, file_name, colorstyle, model_size)
        else:
            print(f"Unsupported file format: {file}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--fileDir', help='model directory', type=str, default='D:/Dev/Dataset/inputs/videos/single_person')
    parser.add_argument('--onnx-name', help='model directory', type=str, default='results/export_omnipose_256_model_best.onnx')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = parse_args()
    main(arg)