import argparse
import cv2
import os
import numpy as np

import torch
from torch.nn import DataParallel
import torch.optim as optim

from modules.loss import JointsMSELoss, AverageMeter
from modules.load_state import load_state

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from datasets.Robot.dataset_provider import create_dataset_loader

from config.config_omnipose import ConfigOmniPose
from config.config_omnipose_model import *
from models.omnipose.omnipose import get_omnipose
from config.config_omnipose_model import _C as cfg
from utils.omni_utils.utils import get_optimizer
from val_omnipose import validate_coco

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
import time
from modules.evaluate import accuracy


def visualize_training_data(
        images: Union[torch.Tensor, np.ndarray],
        heat_maps_gt: Optional[Union[torch.Tensor, np.ndarray]] = None,
        wait_key: bool = True
) -> None:
    """
    Visualize training images and their corresponding heatmaps if available.

    Args:
        images: Input images tensor/array of shape [B, C, H, W]
        heat_maps_gt: Optional ground truth heatmaps tensor/array of shape [B, num_keypoints, H, W]
        wait_key: Whether to wait for key press before closing windows
    """
    # Print input shape and value range
    if isinstance(images, torch.Tensor):
        print('Input shape:', images.shape)
        # Get the first image and convert it
        image = images[0].cpu()  # Shape: [3, H, W]
        image = image.permute(1, 2, 0).numpy()  # Shape: [H, W, 3]
    else:
        print('Input shape:', images.shape)
        image = images[0]

    # Normalize if needed (assuming input range is [0,1] or [-1,1])
    if image.min() < 0:  # if normalized to [-1,1], convert to [0,1]
        image = (image + 1) / 2


    # Denormalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std[None, None, :] * image + mean[None, None, :]

    # Convert to uint8 format (0-255)
    image = (image * 255).clip(0, 255).astype(np.uint8)

    # Convert RGB to BGR for cv2
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display image
    cv2.imshow('Input Image', image_bgr)

    # Visualize heatmaps if they exist
    if heat_maps_gt is not None:
        if isinstance(heat_maps_gt, torch.Tensor):
            heatmaps = heat_maps_gt[0].cpu().detach().numpy()
        else:
            heatmaps = heat_maps_gt[0]

        num_keypoints = heatmaps.shape[0]

        # Create figure for individual heatmaps
        plt.figure(figsize=(20, 4))
        for joint_idx in range(num_keypoints):
            plt.subplot(1, num_keypoints, joint_idx + 1)

            # Plot heatmap
            plt.imshow(heatmaps[joint_idx], cmap='jet')
            plt.colorbar()
            plt.title(f'Joint {joint_idx}')
            plt.xticks([])
            plt.yticks([])

        plt.suptitle('Joint Heatmaps')
        plt.tight_layout()
        plt.show()

        # Create figure for heatmap overlays
        plt.figure(figsize=(60, 16))
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        for joint_idx in range(num_keypoints):
            plt.subplot(1, num_keypoints, joint_idx + 1)

            # Resize heatmap to match image dimensions
            resized_heatmap = cv2.resize(
                heatmaps[joint_idx],
                (rgb_image.shape[1], rgb_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # Show original image with heatmap overlay
            plt.imshow(rgb_image)
            plt.imshow(resized_heatmap, cmap='jet', alpha=0.5)
            plt.colorbar()
            plt.title(f'Joint {joint_idx} Overlay')
            plt.xticks([])
            plt.yticks([])

        plt.suptitle('Joint Heatmap Overlays')
        plt.tight_layout()
        plt.show()

    if wait_key:
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            cv2.destroyAllWindows()



cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


def train(model, train_loader, val_loader, checkpoint_path, weights_only, checkpoints_folder):
    losses = AverageMeter()
    batch_time = AverageMeter()
    acc = AverageMeter()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        torch.cuda.empty_cache()
    else:
        print("GPU is not available, using CPU instead")

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tensorboard_dir = os.path.join(checkpoints_folder, current_time)
    writer = SummaryWriter(tensorboard_dir)

    dummy_input = torch.randn(1, 3, config.netSize[0], config.netSize[0])
    writer.add_graph(model, dummy_input)

    num_iter = 0
    len_epoch = 300

    optimizer = get_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=-1
    )
    criterion = JointsMSELoss(use_target_weight=False).cuda()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        load_state(model, checkpoint)
        if not weights_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            num_iter = checkpoint['iter']
            current_epoch = checkpoint['current_epoch']

    model = DataParallel(model).cuda()
    model.train()

    end = time.time()

    epoch_pbar = tqdm(range(0, len_epoch), desc='Training Epochs', position=0)
    for epochId in epoch_pbar:
        scheduler.step()

        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epochId)

        running_loss = 0.0

        total_batches = len(train_loader)
        batch_pbar = tqdm(enumerate(train_loader),
                          desc=f'Epoch {epochId + 1}',
                          total=total_batches,
                          leave=False,
                          position=1,
                          dynamic_ncols=True)

        validation_done = False

        for batch_idx, batch_data in batch_pbar:
            images = batch_data['image'].cuda()
            heat_maps_gt = batch_data['heat_maps'].cuda() if 'heat_maps' in batch_data else None
            target_weight = batch_data['target_weight'].cuda()

            # visualize_training_data(images, heat_maps_gt) # viz input image and heatmap_gt

            # model input
            heatmaps = model(images)

            if isinstance(heatmaps, list):
                # print(outputs[0].shape, target.shape, target_weight.shape)
                loss = criterion(heatmaps[0], heat_maps_gt, target_weight)
                for output in heatmaps[1:]:
                    # print(output.shape, target.shape, target_weight.shape)
                    loss += criterion(output, heat_maps_gt, target_weight)
            else:
                output = heatmaps
                loss = criterion(output, heat_maps_gt, target_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             heat_maps_gt.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)

            batch_time.update(time.time() - end)
            end = time.time()

            batch_pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'Train Acc': f'{acc.avg: .1f}'
            }, refresh=True)


        snapshot_name = '{}/checkpoint_epoch_{}.pth'.format(checkpoints_folder, epochId)
        torch.save({'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'current_epoch': epochId},
                        snapshot_name)


        print('\nValidation...')
        metrics = validate_coco(criterion, val_loader, model)
        writer.add_scalar('Validation/Acc', metrics['Acc'], epochId)
        writer.add_scalar('Validation/HeatmapLoss', metrics['heatmap_loss'], epochId)


        model.train()

        batch_pbar.close()

        epoch_pbar.set_postfix({
            'Epoch': f'{epochId + 1}/{len_epoch}',
            'Loss': f'{avg_loss:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        }, refresh=True)

    writer.close()
    epoch_pbar.close()


if __name__ == '__main__':
    config = ConfigOmniPose().parse()
    model = get_omnipose(cfg, is_train=True)
    train_loader, val_loader = create_dataset_loader(config)

    exp_root = 'exp'
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)

    checkpoints_folder = os.path.join(exp_root, config.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    train(model, train_loader, val_loader, config.checkpoint_path, config.weights_only, checkpoints_folder)
