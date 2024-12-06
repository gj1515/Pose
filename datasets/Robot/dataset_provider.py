from datasets.coco.single.coco_single import create_coco_dataset

from torch.utils.data import DataLoader
import numpy as np


def create_dataset(config):
    dataset = {}

    if config.dataset == 'coco':
        train_set, val_set = create_coco_dataset(config)

    elif config.dataset == 'coco_eval':
        train_set, val_set = create_coco_dataset(config)
    else:
        raise ValueError('Dataset ' + config.dataset + ' not available.')

    return train_set, val_set


def create_dataset_loader(config):
    train_set, val_set = create_dataset(config)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers
    )  # drop_last: drop the last incomplete batch, if the dataset size is not divisible by the batch size.

    multiple = config.num_workers / config.batch_size
    val_n_batch = int(np.max([int(config.batch_size * 0.5), 1]))
    val_n_thread = int(np.max([int(val_n_batch * multiple), 1]))
    test_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=val_n_thread
    )

    return train_loader, test_loader
