import torch
from importlib import import_module
from .lle import MobileIELLENet, MobileIELLENetS
from .lle_6channel import MobileIELLENet as MobileIELLENet6, MobileIELLENetS as MobileIELLENetS6
from .isp import MobileIEISPNet, MobileIEISPNetS

__all__ = {
    'MobileIELLENet',
    'MobileIELLENetS',
    'MobileIELLENet6',
    'MobileIELLENetS6',
    'MobileIEISPNet', 
    'MobileIEISPNetS',
    'import_model'
}

def import_model(opt):
    model_name = 'MobileIE'+opt.model_task.upper()
    kwargs = {'channels': opt.config['model']['channels']}

    if opt.config['model']['type'] == 're-parameterized':
        model_name += 'NetS'
    elif opt.config['model']['type'] == 'original':
        model_name += 'Net'
        kwargs['rep_scale'] = opt.config['model']['rep_scale']
    else:
        raise ValueError('unknown model type, please choose from [original, re-parameterized]')

    if hasattr(opt, 'use_6channel') and opt.use_6channel:
        model_name += '6'

    model = getattr(import_module('model'), model_name)(**kwargs)
    model = model.to(opt.device)

    if opt.config['model']['pretrained']:
        model.load_state_dict(torch.load(opt.config['model']['pretrained']), strict=False)

    if opt.config['model']['type'] == 'original' and opt.config['model']['need_slim'] is True:
        model = model.slim().to(opt.device)
    return model
