# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import importlib
import numpy as np
import os
from model.SAFA import FrequencySelection

import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import shutil
# savepath = r'/data3/limian1/Documents/LPN-main/data/University-Release/relitu/dh'
# if not os.path.exists(savepath):
#     os.mkdir(savepath)
# def draw_features(width, height, x, savename=None):
#     tic = time.time()
#     if savename is None:
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         millis=int((time.time()%1)*1000)
#         filename = f"heatmap_{timestamp}_{millis:03d}.png"
#         savename =  os.path.join(savename,filename)
#     original_image_path = get_original_image_path()
#     if original_image_path:
#         save_original_image_to_path(original_image_path, os.path.dirname(savename))
#     # 获取实际的通道数
#     channels = x.shape[1]
#
#     # 计算实际可以绘制的通道数
#     total_plots = min(width * height, channels)
#
#     # 如果实际通道数少于需要的子图数量，调整布局
#     if total_plots < width * height:
#         # 重新计算合适的网格布局
#         new_width = int(np.ceil(np.sqrt(total_plots)))
#         new_height = int(np.ceil(total_plots / new_width))
#         print(f"Adjusting layout from {width}x{height} to {new_width}x{new_height}")
#         width, height = new_width, new_height
#
#     fig = plt.figure(figsize=(16, 16))
#     fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
#
#     for i in range(total_plots):
#         plt.subplot(height, width, i + 1)
#         plt.axis('off')
#         img = x[0, i, :, :]  # 现在 i 不会超出通道数范围
#         pmin = np.min(img)
#         pmax = np.max(img)
#         img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
#         img = img.astype(np.uint8)
#         img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
#         img = img[:, :, ::-1]
#         plt.imshow(img)
#         print("{}/{}".format(i + 1, total_plots))
#
#     fig.savefig(savename, dpi=200)
#     fig.clf()
#     plt.close()
#     print("time:{}".format(time.time() - tic))



class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        if num_classes:
            self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        if num_classes:
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        fs_cfg = {
            'k_list': [1],
            'fs_feat': 'feat',
            'lowfreq_att': True,
            'lp_type': 'freq',
            'act': 'softmax',
            'spatial': 'conv',
            'spatial_group': 1,
            'spatial_kernel': 3,
            'init': 'zero',
            'global_selection': True,
        }
        self.frequency_selection = FrequencySelection(in_channels=3, **fs_cfg)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        B, C, H, W = x.shape

       #draw_features(H, W, x.detach().cpu().numpy(), "{}/None".format(savepath))
        x = self.frequency_selection(x)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x  # global average pooling, (N, C, H, W) -> (N, C)
    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x.mean([-2, -1]))
        if hasattr(self, "head"):
            x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {

    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k_1k_384": 'https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth',
    "convnext_small_22k_1k_224": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_base_22k_1k_384": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth",
    "convnext_base_22k_1k_224": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def convnext_tiny(pretrained=True, down_from_url=True, model_folder="~/.cache/torch/hub/checkpoints/",
                  weight_name='22k', **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        assert weight_name in ['1k', '22k', '22k_1k_384',
                               '22k_1k_224'], "weight_name must be either '1k' or '22k' or '22k_1k_384'"
        url_name = 'convnext_tiny_' + weight_name
        if down_from_url:
            try:
                checkpoint = torch.load(os.path.join(model_folder, url_name + '.pth'), map_location="cpu")
                model.load_state_dict(checkpoint["model"], strict=False)
            except:
                url = model_urls[url_name]
                checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
                model.load_state_dict(checkpoint["model"], strict=False)
        else:
            checkpoint = torch.load(os.path.join(model_folder, url_name + '.pth'), map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
    return model


def convnext_small(pretrained=True, model_folder="~/.cache/torch/hub/checkpoints/", weight_name='22k_1k_384', **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        assert weight_name in ['1k', '22k', '22k_1k_384',
                               '22k_1k_224'], "weight_name must be either '1k' or '22k' or '22k_1k_384'"
        url_name = 'convnext_small_' + weight_name
        try:
            checkpoint = torch.load(os.path.join(model_folder, url_name + '.pth'), map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
        except:
            url = model_urls[url_name]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)

    return model


def convnext_base(pretrained=True, model_folder="/home/hello/cjl/UCVGL-main/", weight_name='22k_1k_384', **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        assert weight_name in ['1k', '22k', '22k_1k_384',
                               '22k_1k_224'], "weight_name must be either '1k' or '22k' or '22k_1k_384'"
        url_name = 'convnext_base_' + weight_name
        try:
            checkpoint = torch.load(os.path.join(model_folder, url_name + '.pth'), map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
        except:
            url = model_urls[url_name]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)

    return model


def convnext_large(pretrained=False, down_from_url=True, model_folder="~/.cache/torch/hub/checkpoints/",
                   weight_name='1k', **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        assert weight_name in ['1k', '22k', '22k_1k_384',
                               '22k_1k_224'], "weight_name must be either '1k' or '22k' or '22k_1k_384'"
        url_name = 'convnext_large_' + weight_name
        if down_from_url:
            try:
                checkpoint = torch.load(os.path.join(model_folder, url_name + '.pth'), map_location="cpu")
                model.load_state_dict(checkpoint["model"], strict=False)
            except:
                url = model_urls[url_name]
                checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
                model.load_state_dict(checkpoint["model"])
        else:
            checkpoint = torch.load(os.path.join(model_folder, url_name + '.pth'), map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)

    return model


def convnext_xlarge(pretrained=False, down_from_url=True, model_folder="~/.cache/torch/hub/checkpoints/",
                    weight_name='1k', **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert weight_name == '22k', "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url_name = 'convnext_xlarge_22k'
        url = model_urls[url_name]
        if down_from_url:
            try:
                checkpoint = torch.load(os.path.join(model_folder, url_name + '.pth'), map_location="cpu")
                model.load_state_dict(checkpoint["model"], strict=False)
            except:
                url = model_urls[url_name]
                checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
                model.load_state_dict(checkpoint["model"])
        else:
            checkpoint = torch.load(os.path.join(model_folder, url_name + '.pth'), map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
    return model


class model(nn.Module):

    def __init__(self,
                 model_name="convnext_small",
                 pretrained=True,
                 weight_name='22k_1k_224',
                 model_folder='~/.cache/torch/hub/checkpoints/',
                 **kwargs
                 ):
        super().__init__()

        if "convnext_base" in model_name:
            self.model = convnext_base(pretrained=pretrained, weight_name=weight_name, model_folder=model_folder,
                                       **kwargs)
        elif "convnext_small" in model_name:
            self.model = convnext_small(pretrained=pretrained, weight_name=weight_name, model_folder=model_folder,
                                        **kwargs)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, img1, img2=None):

        if img2 is not None:

            image_features1 = self.model(img1)
            image_features2 = self.model(img2)

            return image_features1, image_features2

        else:
            image_features = self.model(img1)

            return image_features


if __name__ == '__main__':
    kwags = {'num_classes': None, 'drop_path_rate': 0.2}
    model = model('convnext_base', down_from_url=False, pretrained=True, weight_name='22k_1k_384', **kwags)
