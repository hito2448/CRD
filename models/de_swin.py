import torch
import timm
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

class de_PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: nn.Module = nn.LayerNorm
    ):
        """
        Args:
            dim (int): Number of input channels.
            out_dim (int): Number of output channels (or 2 * dim if None)
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        # self.dim = dim
        # self.out_dim = out_dim or 2 * dim
        # self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        # self.norm = norm_layer(self.out_dim)

        self.dim = dim
        self.out_dim = out_dim or dim // 2
        self.deconv2x2 = nn.ConvTranspose2d(self.dim, self.out_dim, kernel_size=2, stride=2, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, H, W, C = x.shape
        #
        # pad_values = (0, 0, 0, H % 2, 0, W % 2)
        # x = nn.functional.pad(x, pad_values)
        # _, H, W, _ = x.shape
        #
        # x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        # x = self.reduction(x)
        # x = self.norm(x)

        # B, H, W, C
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.deconv2x2(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)

        return x


class de_swinv2(torch.nn.Module):
    def __init__(self, backbone='swinv2_tiny_window16_256'):
        super(de_swinv2, self).__init__()

        self.swindecoder = self._get_decoder(backbone)

        self.apply(self._init_weights)
        for bly in self.swindecoder:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            timm.layers.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_decoder(self, backbone):
        backbone_layers = timm.create_model(backbone).layers
        layer2 = backbone_layers[2]
        layer2.downsample = de_PatchMerging(dim=768)
        layer1 = backbone_layers[1]
        layer1.downsample = de_PatchMerging(dim=384)
        layer0 = backbone_layers[0]
        layer0.downsample = de_PatchMerging(dim=192)
        layers = [layer2, layer1, layer0]

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1).contiguous()
        x = self.swindecoder[0](x)
        feature_a = x.permute(0, 3, 1, 2).contiguous()
        x = self.swindecoder[1](x)
        feature_b = x.permute(0, 3, 1, 2).contiguous()
        x = self.swindecoder[2](x)
        feature_c = x.permute(0, 3, 1, 2).contiguous()

        return [feature_c, feature_b, feature_a]


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: nn.Module = nn.LayerNorm
    ):
        """
        Args:
            dim (int): Number of input channels.
            out_dim (int): Number of output channels (or 2 * dim if None)
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        pad_values = (0, 0, 0, H % 2, 0, W % 2)
        x = nn.functional.pad(x, pad_values)
        _, H, W, _ = x.shape

        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.reduction(x)
        x = self.norm(x)
        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BN_layer_swin(nn.Module):
    def __init__(self,
                 backbone='swinv2_tiny_window16_256',
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super(BN_layer_swin, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.LayerNorm
        self.bn_layer = self._make_layer(backbone)

        self.conv1 = conv3x3(96, 192, 2)
        self.bn1 = norm_layer(192)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(192, 384, 2)
        self.bn2 = norm_layer(384)
        self.conv3 = conv3x3(192, 384, 2)
        self.bn3 = norm_layer(384)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, backbone):
        template = timm.create_model(backbone)
        bn_layer = template.layers[3]
        # bn_layer.downsample.reduction.in_features = 4608
        # bn_layer.downsample = timm.models.PatchMerging(dim=1152, out_dim=768)
        bn_layer.downsample = PatchMerging(dim=1152, out_dim=768)

        return bn_layer

    def _forward_impl(self, x: Tensor) -> Tensor:
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)
        # print(feature.shape)
        feature = feature.permute(0, 2, 3, 1).contiguous()
        # print(self.bn_layer)
        output = self.bn_layer(feature)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
