# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

from utils.omni_utils.evaluate import accuracy
from utils.omni_utils.inference import get_final_preds
from utils.omni_utils.transforms import flip_back
from utils.omni_utils.vis import save_debug_images
from utils.omni_utils.vis import save_images

# Miscellaneous Imports
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def visualize_pose_data(input_tensor, target_tensor, target_weight_tensor, meta):
    """
    Visualize pose estimation data including input image, joint heatmaps, and target weights.

    Args:
        input_tensor (torch.Tensor): Input image tensor of shape [B, C, H, W]
        target_tensor (torch.Tensor): Target heatmaps tensor of shape [B, num_joints, H, W]
        target_weight_tensor (torch.Tensor): Target weights tensor of shape [B, num_joints]
        meta (dict): Metadata dictionary containing image information
    """
    # Print basic information
    print('input shape: ', input_tensor.shape)
    print('target shape: ', target_tensor.shape)
    print('image filename: ', meta['image'])

    def _denormalize_image(img_tensor):
        """Convert normalized tensor to displayable image."""
        # Convert from CHW to HWC format and move to CPU
        image = img_tensor.cpu().permute(1, 2, 0).numpy()

        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Convert to uint8
        image = (image * 255).astype(np.uint8)
        return image

    def _visualize_joint_heatmaps(target_heatmaps):
        """Display individual joint heatmaps."""
        num_joints = target_heatmaps.shape[0]
        plt.figure(figsize=(20, 4))

        for joint_idx in range(num_joints):
            plt.subplot(1, num_joints, joint_idx + 1)
            plt.imshow(target_heatmaps[joint_idx], cmap='jet')
            plt.colorbar()
            plt.title(f'Joint {joint_idx}')
            plt.xticks([])
            plt.yticks([])

        plt.suptitle('Joint Heatmaps')
        plt.tight_layout()
        plt.show()

    def _visualize_combined_heatmap(image, target_heatmaps):
        """Display combined heatmap overlay on the original image."""
        plt.figure(figsize=(10, 10))

        # Show original image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Overlay combined heatmap
        combined_heatmap = np.max(target_heatmaps, axis=0)
        plt.imshow(combined_heatmap, cmap='jet', alpha=0.3)

        plt.title('Combined Heatmap Overlay')
        plt.axis('off')
        plt.show()

    def _visualize_target_weights(target_weights):
        """Display target weights for each joint."""
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(target_weights)), target_weights.flatten())
        plt.title('Target Weights per Joint')
        plt.xlabel('Joint Index')
        plt.ylabel('Weight')
        plt.show()

    # Get the first sample's image and convert it
    image = _denormalize_image(input_tensor[0])

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get the first sample's target heatmaps
    target_heatmaps = target_tensor[0].cpu().numpy()

    # Get the first sample's target weights
    target_weights = target_weight_tensor[0].cpu().numpy()

    # Visualize all components
    _visualize_joint_heatmaps(target_heatmaps)
    _visualize_combined_heatmap(image, target_heatmaps)
    _visualize_target_weights(target_weights)

    # Display image using OpenCV
    cv2.imshow('Input Image', image_bgr)
    key = cv2.waitKey(0)
    if key == 27:  # ESC key
        cv2.destroyAllWindows()


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir):  # , writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    tbar = tqdm(train_loader)

    print("Epoch ", str(epoch), ":")

    for i, (input, target, target_weight, meta) in enumerate(tbar):
        # measure data loading time
        data_time.update(time.time() - end)

        # visualize_pose_data(input, target, target_weight, meta)   # for viz heatmap_gt

        input = input.cuda()
        target = target.cuda()
        target_weight = target_weight.cuda()

        # compute output
        # print(model)
        outputs = model(input)

        # quit()

        # target = target.cuda(non_blocking=True)
        # target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            # print(outputs[0].shape, target.shape, target_weight.shape)
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                # print(output.shape, target.shape, target_weight.shape)
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # quit()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            # msg = 'Epoch: [{0}][{1}/{2}]\t' \
            #       'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #       'Speed {speed:.1f} samples/s\t' \
            #       'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #       'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
            #       'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
            #           epoch, i, len(train_loader), batch_time=batch_time,
            #           speed=input.size(0)/batch_time.val,
            #           data_time=data_time, loss=losses, acc=acc)
            # logger.info(msg)

            # writer = writer_dict['writer']
            # global_steps = writer_dict['train_global_steps']
            # writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            # writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input, meta, target, pred*4, output,
            #                   prefix)

        tbar.set_description('Train Acc: %.6f' % acc.avg)


def validate(config, val_loader, val_dataset, dataset_name, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()

        tbar = tqdm(val_loader)

        for i, (input, target, target_weight, meta) in enumerate(tbar):
            input = input.cuda()
            target = target.cuda()
            target_weight = target_weight.cuda()

            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output = (output + output_flipped) * 0.5

            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                # msg = 'Test: [{0}/{1}]\t' \
                #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                #       'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                #           i, len(val_loader), batch_time=batch_time,
                #           loss=losses, acc=acc)
                # logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                # save_debug_images(config, input, meta, target, pred*4, output,
                #                   prefix)

            # print(meta)

            # if i == 10:
            #     break
            #
            save_images(input, pred * 4, meta['joints_vis'], output_dir, meta, i)

            tbar.set_description('Val   Acc: %.6f' % acc.avg)

        if dataset_name == 'mpii':
            name_values, perf_indicator, perf_indicator_01 = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path,
                filenames, imgnums)

        elif dataset_name == 'coco':
            name_values, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path,
                filenames, imgnums)

        elif dataset_name == 'posetrack':
            name_values, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path,
                filenames, imgnums)

            print(name_values, perf_indicator)

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    if dataset_name == 'mpii':
        return perf_indicator, perf_indicator_01
    elif dataset_name == 'coco' or dataset_name == 'posetrack':
        return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '|   Arch   ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|------' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
