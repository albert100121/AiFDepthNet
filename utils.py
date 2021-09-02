import importlib
import torch
import torch.nn as nn


def load_trained_model(path):
    checkpoint = torch.load(path)
    args_dict = checkpoint['args_dict']
    args_model = checkpoint['args_model']

    Model_module_path = '.'.join(
        args_model['model_setting']['model'].split('.')[:-1])
    Model_name = args_model['model_setting']['model'].split('.')[-1]
    Model = getattr(importlib.import_module(Model_module_path), Model_name)
    net = Model(**args_model['model_kwargs'])
    net = nn.DataParallel(net)
    net.load_state_dict(checkpoint['state_dict'])

    return net, args_dict, args_model
