import argparse
import json
import os
from .skeleton import Skeleton
from .config_omnipose_model import _C as cfg_default


class ConfigOmniPose:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--netSize', type=int, default=[256, 192], help='network input image size')
        self.parser.add_argument('--batch-size', type=int, default=1, help='batch size')
        self.parser.add_argument('--num-workers', type=int, default=1, help='number of workers')
        self.parser.add_argument('--checkpoint-path', type=str, default='weights/omnipose_256_model_best.pth', help='path to the checkpoint to continue training from')
        self.parser.add_argument('--experiment-name', type=str, default='1203_omnipose', help='experiment name to create folder for checkpoints')

        self.parser.add_argument('--dataset', default='coco', help='Dataset: coco, coco_eval')  ##-coco, robot22 implemented
        self.parser.add_argument('--need-paf', default=False, help='need_paf')
        self.parser.add_argument('--num-keypoints', type=int, default=17, help='number of keypoints')
        self.parser.add_argument('--skeleton-type', default='coco17', help='joint type: opose14, coco17')
        self.parser.add_argument('--is_regress', default=False, help='is_regress')
        self.parser.add_argument('--netOutScale', type=int, default=4, help='network output scale factor')
        self.parser.add_argument('--data', type=str, default='D:/Dev/Dataset/coco', help='path to COCO dataset')
        self.parser.add_argument('--train-split', type=float, default=0.8, help='training data split ratio')
        self.parser.add_argument('--val-split', type=float, default=1.0, help='validation data split ratio')

        # Additional Training Arguments
        self.parser.add_argument('--prepared-train-labels', type=str, default="D:/Dev/Dataset/coco/annotations/train_label.pkl", help='path to the file with prepared annotations')
        self.parser.add_argument('--train-images-folder', type=str, default="D:/Dev/Dataset/coco/train2017", help='path to COCO train images folder')

        self.parser.add_argument('--val-labels', type=str, default="D:/Dev/Dataset/coco/annotations/person_keypoints_val2017.json", help='path to json with keypoints val labels')
        self.parser.add_argument('--val-images-folder', type=str, default="D:/Dev/Dataset/coco/val2017", help='path to COCO val images folder')
        self.parser.add_argument('--val-output-name', type=str, default='results/output_omnipose_1209_checkpoint_epoch_96_best.json', help='name of output json file with detected keypoints')

        self.parser.add_argument('--weights-only', action='store_true', help='just initialize layers with pre-trained weights and start training from the beginning')
        self.parser.add_argument('--visualize', action='store_true', help='show keypoints')
        self.parser.add_argument('--pretrained', type=bool, default=False)

    def parse(self):
        self.init()
        self.config = self.parser.parse_args()

        self.config.netSize = self.config.netSize
        self.config.is_regress = self.config.is_regress
        self.config.netOutScale = self.config.netOutScale

        cfg = cfg_default.clone()
        cfg.defrost()

        # Model
        cfg.MODEL.NUM_JOINTS = self.config.num_keypoints
        cfg.MODEL.IMAGE_SIZE = [self.config.netSize, self.config.netSize]
        cfg.MODEL.PRETRAINED = self.config.pretrained
        cfg.MODEL.INIT_WEIGHTS = True
        cfg.MODEL.NAME = 'omnipose'
        cfg.MODEL.TAG_PER_JOINT = True
        cfg.MODEL.TARGET_TYPE = 'gaussian'
        cfg.MODEL.HEATMAP_SIZE = [self.config.netSize[0] // self.config.netOutScale,
                                  self.config.netSize[1] // self.config.netOutScale]

        # High Resolution Net
        # STAGE2
        cfg.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
        cfg.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
        cfg.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
        cfg.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [32, 64]
        cfg.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
        cfg.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

        # STAGE3
        cfg.MODEL.EXTRA.STAGE3.NUM_MODULES = 1
        cfg.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
        cfg.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
        cfg.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [32, 64, 128]
        cfg.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
        cfg.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

        # STAGE4
        cfg.MODEL.EXTRA.STAGE4.NUM_MODULES = 1
        cfg.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
        cfg.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
        cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
        cfg.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
        cfg.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

        # Training 설정
        cfg.TRAIN.LR = self.config.base_lr if hasattr(self.config, 'base_lr') else 0.001
        cfg.TRAIN.LR_FACTOR = 0.1
        cfg.TRAIN.LR_STEP = [90, 110]
        cfg.TRAIN.OPTIMIZER = 'adam'
        cfg.TRAIN.MOMENTUM = 0.9
        cfg.TRAIN.WD = 0.0001
        cfg.TRAIN.NESTEROV = False
        cfg.TRAIN.GAMMA1 = 0.99
        cfg.TRAIN.GAMMA2 = 0.0

        # CUDNN 설정
        cfg.CUDNN.BENCHMARK = True
        cfg.CUDNN.DETERMINISTIC = False
        cfg.CUDNN.ENABLED = True

        cfg.freeze()

        self.config.MODEL = cfg.MODEL
        self.config.TRAIN = cfg.TRAIN
        self.config.CUDNN = cfg.CUDNN

        # Create experiment directories if needed
        exp_root = 'exp'
        if not os.path.exists(exp_root):
            os.makedirs(exp_root)

        self.config.saveDir = os.path.join(exp_root, self.config.experiment_name)
        if not os.path.exists(self.config.saveDir):
            os.makedirs(self.config.saveDir)

        # Handle training vs evaluation mode
        if hasattr(self.config, 'train') and self.config.train == 0:
            # Load existing config for evaluation
            config_file = os.path.join(self.config.saveDir, 'config.txt')
            self.load_config_from_file(config_file)
            self.init_dataset(set=set)
        else:
            # Initialize dataset for training
            self.init_dataset(set=set)

            # Save configuration
            config_file = os.path.join(self.config.saveDir, 'config.txt')
            args = dict((name, getattr(self.config, name)) for name in dir(self.config)
                        if not name.startswith('_'))
            self.save_config(config_file, args)

        # Initialize skeleton configuration
        self.config.skeleton = Skeleton(self.config.skeleton_type)

        return self.config

    def save_config(self, file_name, args):
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  {}: {}\n'.format(str(k), str(v)))

    def read_user_config(self, fname):
        with open(fname) as f:
            loaded_json = json.load(f)
            for key, value in loaded_json.items():
                setattr(self.config, key, value)

    def load_config_from_file(self, file_name):
        with open(file_name, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                if line.startswith("==> Args:"):
                    continue
                key, value = line.strip().split(': ', 1)
                try:
                    value = eval(value)
                except:
                    pass
                setattr(self.config, key.strip(), value)

    def init_dataset_coco(self, set=True):
        if set:
            if self.config.dataset == 'coco':
                self.config.train_split = 1.0
                self.config.val_split = 1.0
            elif self.config.dataset == 'coco_eval':
                self.config.train_split = 1.0
                self.config.val_split = 1.0
                self.config.val_labels = self.config.prepared_train_labels
                self.config.val_images_folder = self.config.train_images_folder
        print('Dataset << COCO >> : train_split({:.1f}), val_split({:.1f})'.format(self.config.train_split, self.config.val_split))
        return True

    def init_dataset(self, set=True):
        if self.config.dataset == 'coco':
            if self.init_dataset_coco(set=set):
                return
        elif self.config.dataset == 'coco_eval':
            if self.init_dataset_coco(set=set):
                return
        raise ValueError('Dataset ' + self.config.dataset + ' not available.')