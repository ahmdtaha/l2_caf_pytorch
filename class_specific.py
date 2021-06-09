import sys
import torch
import common
import numpy as np
import os.path as osp
from utils import lr_utils
import torch.optim as optim
from utils import heatmap_utils
from configs.base_config import Config
from utils.imagenet_lbls import imagenet_lbls
from constraint_attention_filter import L2_CAF




def main(cfg):
    cfg.logger.info(cfg)

    output_dir = cfg.output_dir

    # img_name_ext = 'dog_butterfly.jpg';top_k = [207, 323];k = 2
    img_name_ext = cfg.input_img; top_k = cfg.cls_logits;k=len(cfg.cls_logits)


    img_name, _ = osp.splitext(img_name_ext)
    rgb_img , pytorch_img = common.load_img(img_name_ext)
    pytorch_img = pytorch_img.cuda()

    arch_name = cfg.arch #'resnet50'
    model, feature_maps, post_conv_subnet = common.load_architecture(arch_name)
    NT = model(pytorch_img)
    A = feature_maps[-1] ## Last conv layer feature maps extracted using a PyTorch hook

    ## Feel Free to override the classes I manually picked
    # k = 5
    # top_k = np.argsort(np.squeeze(NT.cpu().numpy()))[::-1][:k]
    cfg.logger.info('Top K={} {}'.format(k, [imagenet_lbls[i] for i in top_k]))




    max_iter = cfg.max_iter
    initial_lr = cfg.lr

    def cls_specific_loss(output_vec,per_class_logits_ph):
        loss = torch.sum(output_vec * per_class_logits_ph)  ## -FT_cls[top_i] + FT_cls[!top_i]
        return loss

    for top_i in top_k:

        l2_caf = L2_CAF(A.shape[-1]).cuda()
        per_class_logits_ph = torch.ones_like(NT)
        per_class_logits_ph[0,top_i] = -1 # -> Maximize top_i cls logit and minimize other logits



        optimizer = optim.SGD(l2_caf.parameters(),lr=initial_lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_utils.polynomial_lr_decay(step,
                                                                                          init_learning_rate=initial_lr,
                                                                                          max_iter=max_iter) / initial_lr)
        MAX_INT = np.iinfo(np.int16).max
        prev_loss = torch.tensor(MAX_INT).cuda()
        iteration = 0
        while iteration < max_iter:

            FT = post_conv_subnet(l2_caf(A))
            loss = cls_specific_loss(FT,per_class_logits_ph)
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
                                    save=output_dir + img_name + '_cls_specific_{}_{}_{}.png'.format(top_i,imagenet_lbls[top_i].replace(',',''),arch_name),
                                    axis='off', cmap='bwr')

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('Loading the default parameters')
        default_args = [
            '--output_dir','./output_heatmaps/',
            '--max_iter','1000',
            "--lr",'0.5',

            # '--input_img','dog_ball.jpg','--cls_logits','242,852'
            '--input_img','dog_butterfly.jpg','--cls_logits','207, 323'
        ]
        cfg = Config().parse(default_args)
    else:
        print('Loading parameters from cmd line')
        cfg = Config().parse(None)

    main(cfg)



