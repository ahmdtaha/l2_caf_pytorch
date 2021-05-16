# L2-CAF: A Debugger Neural Network
(ECCV 2020) Official PyTorch implementation of **A Generic Visualization Approach for Convolutional Neural Networks**

[Paper](https://arxiv.org/abs/2007.09748) | [1 Min Video](https://youtu.be/W4xaKQlPEl0) | [10 Mins Video](https://youtu.be/Wpw3ewSvnFE)

The goal of this PyTorch implementation is to provide a simple, readable, and fast implementation for L2-CAF. If you have an idea to make this simpler, please let me know!

L2-CAF does not require any finetuning or specific layers. It is easy to integrate L2-CAF in various network architectures. 

## Usage example
The `class_oblivious.py` presents the class oblivious variant of L2-CAF, while `class_specific.py` to presents the class specific variant of L2-CAF.

All hyperparameters are hard-coded. Just pick an image from `input_imgs` dir (e.g., [class_oblivious](https://github.com/ahmdtaha/l2_caf_pytorch/blob/448f6e8d71c60006edbd069a3b0025b1eab6a1f5/class_oblivious.py#L25)), and the output visualization maps will be saved inside the `output_heatmaps` dir.



### MISC Notes
* The L2-CAF is originally implemented in [Tensorflow](https://github.com/ahmdtaha/constrained_attention_filter). I hope this PyTorch implementation raises awareness of L2-CAF, this cool visualization tool  :)
* I noticed that visualization heatmaps are consistent across different Tensorflow models. In contrast, the output heatmaps from different PyTorch models seems inconsistent.  
* This PyTorch implementation visualizes attention of the last conv layer. While L2-CAF can visualize attention in intermediate layers, I did not support this feature to keep the code simple. Yet, please note that the Tensorflow implementation supports intermediate layer visualization. 

### TODO LIST
* Support more architectures

## Release History
* 1.0.0
    * First commit Vanilla L2-CAF on DenseNet169, GoogleNet, and ResNet50 on 16 May 2021


### Citation
```
@inproceedings{taha2020generic,
title={A Generic Visualization Approach for Convolutional Neural Networks},
author={Taha, Ahmed and Yang, Xitong and Shrivastava, Abhinav and Davis, Larry},
booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
year={2020}
}
```