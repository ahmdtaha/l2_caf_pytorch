import sys
import torch
import common
import numpy as np
import torch.nn as nn
import os.path as osp
from utils import lr_utils
import torch.optim as optim
from utils import heatmap_utils
from configs.base_config import Config
from constraint_attention_filter import L2_CAF




def main(cfg):
    cfg.logger.info(cfg)
    output_dir = cfg.output_dir
    img_name_ext = cfg.input_img

    img_name, _ = osp.splitext(img_name_ext)
    rgb_img , pytorch_img = common.load_img(img_name_ext)
    pytorch_img = pytorch_img.cuda()

    arch_name = cfg.arch #'resnet50'
    model,last_conv_feature_maps,post_conv_subnet = common.load_architecture(arch_name)
    NT = model(pytorch_img)
    A = last_conv_feature_maps[-1] ## Last conv layer feature maps extracted using a PyTorch hook


    l2_caf = L2_CAF(A.shape[-1]).cuda()

    max_iter = cfg.max_iter
    initial_lr = cfg.lr


    l2loss = nn.MSELoss() ## || NT - FT ||^2

    optimizer = optim.SGD(l2_caf.parameters(),lr=initial_lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_utils.polynomial_lr_decay(step,
                                                                                      init_learning_rate=initial_lr,
                                                                                      max_iter=max_iter) / initial_lr)
    MAX_INT = np.iinfo(np.int16).max
    prev_loss = torch.tensor(MAX_INT).cuda()
    iteration = 0
    while iteration < max_iter:

        FT = post_conv_subnet(l2_caf(A))
        loss = l2loss(FT,NT)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if iteration % 50 == 0:
            if torch.abs(loss.item() - prev_loss) < cfg.min_error:
                break
            prev_loss = loss

        iteration += 1

    cfg.logger.info('Done after {} iterations'.format(iteration))
    ## Save result filter
    frame_mask = common.normalize_filter(l2_caf.filter.detach().cpu().numpy())
    heatmap_utils.apply_heatmap(rgb_img, frame_mask, alpha=0.6,
                                save=output_dir + img_name + '_cls_oblivious_{}.png'.format(arch_name),
                                axis='off', cmap='bwr')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Loading the default parameters')
        default_args = [
            '--output_dir', './output_heatmaps/',
            '--max_iter', '1000',
            "--lr", '0.5',
            "--arch",'densenet169',
            # '--input_img','dog_ball.jpg',
            '--input_img', 'dog_butterfly.jpg',
        ]
        cfg = Config().parse(default_args)
    else:
        print('Loading parameters from cmd line')
        cfg = Config().parse(None)

    main(cfg)

