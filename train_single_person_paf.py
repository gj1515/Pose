import cv2
import os
from tqdm import tqdm
from datetime import datetime
import time

import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from modules.loss import JointsMSELoss, AverageMeter, l2_loss
from modules.load_state import load_state
from modules.evaluate import accuracy
from datasets.Robot.dataset_provider import create_dataset_loader
from val_omnipose import validate_coco

from config.config_mobilenet import ConfigMobilenet
from models.mobilenet.with_mobilenet import PoseEstimationWithMobileNet
from modules.mobilenet.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


def train(model, train_loader, val_loader, checkpoint_path, weights_only, checkpoints_folder, num_refinement_stages):
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
    drop_after_epoch = [100, 200, 260]
    base_lr = config.base_lr

    optimizer = optim.Adam([
        {'params': get_parameters_conv(model.model, 'weight')},
        {'params': get_parameters_conv_depthwise(model.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(model.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(model.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(model.cpm, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(model.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(model.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(model.initial_stage, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(model.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(model.refinement_stages, 'weight'), 'lr': base_lr * 4},
        {'params': get_parameters_conv(model.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
        {'params': get_parameters_bn(model.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(model.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
    ], lr=base_lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)

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

        total_losses = [0, 0] * (num_refinement_stages + 1) # heatmap loss, paf loss per stage

        total_batches = len(train_loader)
        batch_pbar = tqdm(enumerate(train_loader),
                          desc=f'Epoch {epochId + 1}',
                          total=total_batches,
                          leave=False,
                          position=1,
                          dynamic_ncols=True)

        validation_done = False
        running_heatmap_loss = 0.0
        running_paf_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in batch_pbar:
            images = batch_data['image'].cuda()
            heat_maps_gt = batch_data['heat_maps'].cuda() if 'heat_maps' in batch_data else None
            target_weight = batch_data['target_weight'].cuda()
            paf_maps = batch_data['paf_maps'].cuda()
            keypoint_masks = batch_data['keypoint_mask'].cuda()
            paf_masks = batch_data['paf_mask'].cuda()

            # visualize_training_data(images, heat_maps_gt) # viz input image and heatmap_gt

            # model input (heatmap, paf map)
            stages_output = model(images)

            losses = []
            for loss_idx in range(len(total_losses) // 2):
                losses.append(l2_loss(stages_output[loss_idx * 2], heat_maps_gt, keypoint_masks, images.shape[0]))
                losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
                total_losses[loss_idx * 2] += losses[-2].item()
                total_losses[loss_idx * 2 + 1] += losses[-1].item()

            loss = losses[0]
            for loss_idx in range(1, len(losses)):
                loss += losses[loss_idx]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))

            _, avg_acc, cnt, pred = accuracy(stages_output[-2].detach().cpu().numpy(),
                                             heat_maps_gt.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            last_stage = (len(total_losses) // 2) - 1
            current_heatmap_loss = total_losses[last_stage * 2]
            current_paf_loss = total_losses[last_stage * 2 + 1]

            running_heatmap_loss += current_heatmap_loss
            running_paf_loss += current_paf_loss
            num_batches += 1

            batch_pbar.set_postfix({
                'Heatmap Loss': f'{current_heatmap_loss:.4f}',
                'PAF Loss': f'{current_paf_loss:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'Train Acc': f'{acc.avg:.1f}'
            }, refresh=True)

            for loss_idx in range(len(total_losses)):
                total_losses[loss_idx] = 0

        # Log average metrics for the epoch
        writer.add_scalar('Training/Epoch_Heatmap_Loss', running_heatmap_loss / num_batches, epochId)
        writer.add_scalar('Training/Epoch_PAF_Loss', running_paf_loss / num_batches, epochId)
        writer.add_scalar('Training/Epoch_Accuracy', acc.avg, epochId)

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
            'Loss': f'{losses.avg:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        }, refresh=True)

    writer.close()
    epoch_pbar.close()



if __name__ == '__main__':
    config = ConfigMobilenet().parse()
    model = PoseEstimationWithMobileNet(config.num_refinement_stages, num_heatmaps=17, num_pafs=30)
    train_loader, val_loader = create_dataset_loader(config)

    exp_root = 'exp'
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)

    checkpoints_folder = os.path.join(exp_root, config.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    train(model, train_loader, val_loader, config.checkpoint_path, config.weights_only, checkpoints_folder, config.num_refinement_stages)
