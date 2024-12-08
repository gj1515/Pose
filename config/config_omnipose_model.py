import os
from yacs.config import CfgNode as CN


_C = CN()
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True
_C.MODEL = CN()
_C.MODEL.NAME = 'omnipose'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [192, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [48, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 3

_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.EXTRA.PRETRAINED_LAYERS = ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2', 'transition2', 'stage3', 'transition3', 'stage4']
_C.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
_C.MODEL.EXTRA.STAGE2 = CN()
_C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
_C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
_C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE3 = CN()
_C.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
_C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
_C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
_C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE4 = CN()
_C.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
_C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
_C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
_C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'


# train
_C.TRAIN = CN()
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [170, 200]
_C.TRAIN.WD = 0.0001
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NESTEROV = False


#======for later==============
_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False


# testing
_C.TEST = CN()
_C.TEST.BBOX_THRE = 1.0
_C.TEST.IMAGE_THRE = 0.0
_C.TEST.IN_VIS_THRE = 0.2
_C.TEST.MODEL_FILE = ''
_C.TEST.NMS_THRE = 1.0
_C.TEST.OKS_THRE = 0.9
_C.TEST.USE_GT_BBOX = True
_C.TEST.FLIP_TEST = True
_C.TEST.POST_PROCESS = True
_C.TEST.BLUR_KERNEL = 11
_C.TEST.SOFT_NMS = False



def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()

if __name__ == '__main__':
    import sys