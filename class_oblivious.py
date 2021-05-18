import torch
import common
import numpy as np
import torch.nn as nn
import os.path as osp
from utils import lr_utils
import torch.optim as optim
from utils import heatmap_utils
from constraint_attention_filter import L2_CAF




def main():
    output_dir = './output_heatmaps/'

    # img_name_ext = 'cute_dog.jpg'
    # img_name_ext = 'ILSVRC2012_val_00000009.JPEG'
    # img_name_ext = 'ILSVRC2012_val_00000021.JPEG'
    # img_name_ext = 'ILSVRC2012_val_00000012.JPEG'
    # img_name_ext = 'bike_dog.jpg'
    img_name_ext = 'dog_ball.jpg'
    # img_name_ext = 'dog_butterfly.jpg'

    img_name, _ = osp.splitext(img_name_ext)
    rgb_img , pytorch_img = common.load_img(img_name_ext)
    pytorch_img = pytorch_img.cuda()

    arch_name = 'googlenet'
    model,feature_maps,post_conv_subnet = common.load_architecture(arch_name)
    NT = model(pytorch_img)
    A = feature_maps[-1] ## Last conv layer feature maps extracted using a PyTorch hook


    l2_caf = L2_CAF(A.shape[-1]).cuda()


    max_iter = 500
    initial_lr = 0.5


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
        # print(loss.item(),'->',optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if iteration % 50 == 0:
            if torch.abs(loss.item() - prev_loss) < 10e-7:
                break
            prev_loss = loss

        iteration += 1

    print('Done after {} iterations'.format(iteration))
    ## Save result filter
    frame_mask = common.normalize_filter(l2_caf.filter.detach().cpu().numpy())
    heatmap_utils.apply_heatmap(rgb_img, frame_mask, alpha=0.6,
                                save=output_dir + img_name + '_cls_oblivious_{}.png'.format(arch_name),
                                axis='off', cmap='bwr')

if __name__ == '__main__':
    main()

