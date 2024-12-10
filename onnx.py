import onnx
import torch
import onnxruntime as ort
import os
import numpy as np

from config.config_omnipose import ConfigOmniPose
from config.config_omnipose_model import _C as cfg
from models.omnipose.omnipose import get_omnipose

from modules.load_state import load_state


def main():
    config = ConfigOmniPose().parse()
    model = get_omnipose(cfg, is_train=True)
    checkpoint = torch.load(config.checkpoint_path)
    load_state(model, checkpoint)
    model = model.cuda().eval()
    model_name = os.path.splitext(os.path.basename(config.checkpoint_path))[0]

    with torch.no_grad():
        example = torch.rand(1, 3, config.netSize[1], config.netSize[0])
        example = example.cuda()

        heatmaps_output =  model(example)

        onnx_name = os.path.join('results', 'export_' + model_name + '.onnx')

        input_names = ['images']
        output_names = ['heat_maps']
        dynamic_inout = {'images': {0: 'batch'}, 'heat_maps': {0: 'batch'}}

        ## --------------------------------------------------------
        print('// -------------------------------------')
        print('  --save:', onnx_name)
        print('// -------------------------------------')
        ## --------------------------------------------------------

        # model export
        torch.onnx.export(model,
                          example,
                          onnx_name,
                          input_names=input_names, output_names=output_names, dynamic_axes=dynamic_inout)

        # load model
        onnx_model = onnx.load(onnx_name)
        onnx.checker.check_model(onnx_model)

        # inference
        gt_out = heatmaps_output.cpu().numpy()
        ort_session = ort.InferenceSession(onnx_name) # load model here

        onnx_example = example.cpu().numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: onnx_example}
        ort_outs = ort_session.run(None, ort_inputs)

        err = sum(sum(sum(abs(ort_outs[0] - gt_out))))
        print('recon err: ', err)

    print('Done')


if __name__ == '__main__':
    main()