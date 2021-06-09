import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.models as models
from torchvision import transforms as T

def load_img(img_name_ext,datasets_dir = './input_imgs'):

    test_img = Image.open('{}/{}'.format(datasets_dir, img_name_ext))
    resize_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

    normalize_transform = T.Compose([

        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    raw_img = resize_transform(test_img)
    normalized_img = normalize_transform(raw_img)
    normalized_img = normalized_img.unsqueeze(0)  # Add batch dimension
    if torch.cuda.is_available():
        normalized_img = normalized_img.cuda()

    return raw_img.permute(1,2,0), normalized_img



def normalize_filter(_atten_var,filter_type='l2norm'):
    if filter_type == 'l2norm':
        frame_mask = np.reshape(np.abs(_atten_var), (_atten_var.shape[0], _atten_var.shape[1]))
        frame_mask = frame_mask / np.linalg.norm(frame_mask)
    else:
        raise NotImplementedError('Invalid filter type {}'.format(filter_type))

    return frame_mask

def get_activation(feature_maps):  ## forward hook to catch the feature maps at a certain layer
    def hook(model, input, output):
        feature_maps.append( output.detach())
    return hook

def load_architecture(arch_name):
    feature_maps = []
    if arch_name in ['resnet50']:
        model = models.resnet50(pretrained=True)
        if torch.cuda.is_available():
            model = model.cuda()

        model.layer4.register_forward_hook(get_activation(feature_maps)) ## Layer4 provides the last conv in resnet50
        ## Replicate the layers after the last conv layer
        post_conv_subnet = nn.Sequential(
            model.avgpool,
            nn.Flatten(),
            model.fc,
        )
        post_conv_subnet.eval()
    elif arch_name in ['googlenet']:
        model = models.googlenet(pretrained=True)
        if torch.cuda.is_available():
            model = model.cuda()

        model.inception5b.register_forward_hook(get_activation(feature_maps)) ## inception5b provides the last conv in googlenet
        ## Replicate the layers after the last conv layer
        post_conv_subnet = nn.Sequential(
            model.avgpool,
            nn.Flatten(),
            model.fc,
        )
    elif arch_name in ['densenet169']:
        model = models.densenet169(pretrained=True)
        if torch.cuda.is_available():
            model = model.cuda()

        model.features.register_forward_hook(get_activation(feature_maps)) ## features provides the last conv in densenet169

        ## Replicate the layers after the last conv layer
        post_conv_subnet = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            model.classifier,
        )
    else:
        raise NotImplementedError('Invalid arch_name {}'.format(arch_name))


    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    post_conv_subnet.eval()

    return model, feature_maps, post_conv_subnet
