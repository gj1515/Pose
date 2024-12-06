import torch
import os
import collections
from ptflops import get_model_complexity_info
import re
from models.vitpose.vitpose import ViTPose


def create_model(config):
    num_jnt = config.skeleton.num_joints

    if config.modelType == 'vitpose':
        model = ViTPose(config.model)
    else:
        raise ValueError('Model ' + config.modelType + ' not available.')
    return model


def parse_criterion(criterion, is_sum=True):
    if is_sum:
        if criterion == 'l1':
            return torch.nn.L1Loss(reduction='sum')
        elif criterion == 'mse':
            return torch.nn.MSELoss(reduction='sum')
        else:
            raise ValueError('Criterion ' + criterion + ' not supported')
    else:
        if criterion == 'l1':
            return torch.nn.L1Loss(reduction='mean')
        elif criterion == 'mse':
            return torch.nn.MSELoss(reduction='mean')
        else:
            raise ValueError('Criterion ' + criterion + ' not supported')



def load_checkpoint(config, model, scheduler, optimizer):
    if config.loadModel == 'none' or config.loadModel == '':
        print('--------- Start From Scratch ----------')
    else:
        model = load_model(config, model)

        if config.train == 2: # continue
            if config.loadModel.find('model_') == 0: # if no path specified
                path = config.saveDir
                fname = config.loadModel
            else:
                path, fname = os.path.split(config.loadModel)

            epoch = int(fname[6:-4])
            ck_name = os.path.join(path, 'checkpoint_{}.pth'.format(epoch))
            checkpoint = torch.load(ck_name)

            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            config.epoch_first = epoch + 1
            print('--------- Continue at Epoch ({}) ----------'.format(config.epoch_first))
        else:
            print('--------- Start Over with ({}) ----------'.format(config.loadModel))

    return config, model, scheduler, optimizer


def load_model(config, model):
    if not config.loadModel == 'none':
        if config.loadModel.find('model_') == 0:
            fname = os.path.join(config.saveDir, config.loadModel)
        else:
            fname = os.path.join(config.loadModel)
        state_dict = torch.load(fname)

        # -----------------------------------------------------------------
        # Handling nn.DataParallel
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        load_state_from_other_state(model, new_state_dict)
        print('Loaded model from ' + fname)
    else:
        print('  -Err: load model not specified in config.loadModel=', config.loadModel)

    return model


def to_gpu(model, optimizer, criterions):
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
        if torch.cuda.device_count() > 1:
            num = torch.cuda.device_count()
            print("Let's use", num, "GPUs!")

            ids = []
            for i in range(num):
                ids.append(i)

            model = torch.nn.DataParallel(model, device_ids=ids)
        else:
            print("Let's use single GPU..")
    else:
        print("In CPU mode...")

    model = model.to(device)

    ## -----------------------------------------------------------------
    ## optimizer.step() --> exp_avg.mul_(beta1).add_(1 - beta1, grad) --> exp_avg -> cpu, Not cuda 0 --> error
    ## --> manually moving to gpu
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    ## -----------------------------------------------------------------
    criterions_cuda = []
    for loss_func in criterions:
        criterions_cuda.append(loss_func.cuda())

    return model, optimizer, criterions_cuda


def load_state_from_other_state(target, source_state):
    target_state = target.state_dict()
    new_target_state = collections.OrderedDict()

    num_suc = 0
    num_fail = 0
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
            num_suc += 1
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))
            num_fail += 1

    num_total = num_suc + num_fail
    print('  -Loading Model State: <{}> % success.. Not loaded: {}'.format(float(num_suc)/float(num_total)*100., num_fail))
    target.load_state_dict(new_target_state)
    return target


def load_state_from_other(target, source):
    source_state = source.state_dict()
    target_state = target.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    target.load_state_dict(new_target_state)
    return target


def get_num_model_parameters(model):
    ## -- Check Num Params
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Number of parameters: {total_params}, {trainable_params}")
    return trainable_params


def get_gflops(config, model):
    input_size = (3, config.netSize, config.netSize)

    macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                             print_per_layer_stat=True, verbose=False)
    # Extract the numerical value
    flops = eval(re.findall(r'([\d.]+)', macs)[0]) * 2
    # Extract the unit
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

    print('Computational complexity: {:<8}'.format(macs))
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))



