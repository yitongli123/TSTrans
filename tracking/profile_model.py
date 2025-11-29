import os
import sys
prj_path = os.path.join(os.getcwd())
if prj_path not in sys.path:
    sys.path.append(prj_path)
print('System paths:', sys.path)

import argparse
import torch
from thop import profile
from thop.utils import clever_format
import time
import importlib


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='tstrans', choices=['tstrans'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='vitb_256_mae_ce_ep600', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate_vit(model, template, search, pattern, num_temporal_templates, ce_template_mask):
    macs1, params1 = profile(model, inputs=(template, search, pattern, num_temporal_templates, ce_template_mask),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)   


if __name__ == "__main__":
    device = "cuda:0"  # set the available device here
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our TSTrans model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    if args.script == "tstrans":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_tstrans
        model = model_constructor(cfg, training=False)
        # get the template and search
        template = torch.randn((cfg.DATA.TEMPLATE.L+1), bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        mask_dim = int((cfg.DATA.TEMPLATE.L+1)*((z_sz / cfg.MODEL.BACKBONE.STRIDE)**2))
        ce_template_mask = torch.randn(bs, mask_dim).to(torch.bool)
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)

        evaluate_vit(model, template, search, 'tracking', cfg.DATA.TEMPLATE.L, ce_template_mask)
    else:
        raise NotImplementedError

