import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils.parametrizations import spectral_norm
from models.networks.normalization import FADE
from models.networks.ppad import PeriodicPad2d


# ResNet block that uses FADE.
# It differs from the ResNet block of SPADE in that
# it takes in the feature map as input, learns the skip connection if necessary.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability
# and https://github.com/NVlabs/SPADE.
class FADEResnetBlockPeriodic(nn.Module):
    def __init__(self, fin, fout, params):
        super().__init__()
        # attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = fin
        self.ppad = params.use_periodic_padding

        # create conv layers
        self.pad_0 = PeriodicPad2d(1) if self.ppad else nn.Identity()
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0)
        self.pad_1 = PeriodicPad2d(1) if self.ppad else nn.Identity()
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in params.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        fade_config_str = params.norm_G.replace('spectral', '')
        self.norm_0 = FADE(fade_config_str, fin, fin, ppad=self.ppad)
        self.norm_1 = FADE(fade_config_str, fmiddle, fmiddle, ppad=self.ppad)
        if self.learned_shortcut:
            self.norm_s = FADE(fade_config_str, fin, fin, self.ppad)

    # Note the resnet block with FADE also takes in |feat|,
    # the feature map as input
    def forward(self, x, feat):
        x_s = self.shortcut(x, feat)

        dx = self.conv_0(self.pad_0(self.actvn(self.norm_0(x, feat))))
        dx = self.conv_1(self.pad_1(self.actvn(self.norm_1(dx, feat))))

        out = x_s + dx

        return out

    def shortcut(self, x, feat):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, feat))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class StreamResnetBlockPeriodic(nn.Module):
    def __init__(self, fin, fout, params):
        super().__init__()
        # attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = fin

        # create conv layers
        self.pad_0 = PeriodicPad2d(1)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0)
        self.pad_1 = PeriodicPad2d(1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in params.norm_S:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        subnorm_type = params.norm_S.replace('spectral', '')
        if subnorm_type == 'batch':
            self.norm_layer_in = nn.BatchNorm2d(fin, affine=True)
            self.norm_layer_out= nn.BatchNorm2d(fout, affine=True)
            if self.learned_shortcut:
                self.norm_layer_s = nn.BatchNorm2d(fout, affine=True)
        elif subnorm_type == 'syncbatch':
            self.norm_layer_in = SynchronizedBatchNorm2d(fin, affine=True)
            self.norm_layer_out= SynchronizedBatchNorm2d(fout, affine=True)
            if self.learned_shortcut:
                self.norm_layer_s = SynchronizedBatchNorm2d(fout, affine=True)
        elif subnorm_type == 'instance':
            self.norm_layer_in = nn.InstanceNorm2d(fin, affine=False)
            self.norm_layer_out= nn.InstanceNorm2d(fout, affine=False)
            if self.learned_shortcut:
                self.norm_layer_s = nn.InstanceNorm2d(fout, affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

    def forward(self, x):
        x_s = self.shortcut(x)

        dx = self.actvn(self.norm_layer_in(self.conv_0(self.pad_0(x))))
        dx = self.actvn(self.norm_layer_out(self.conv_1(self.pad_1(dx))))

        out = x_s + dx

        return out

    def shortcut(self,x):
        if self.learned_shortcut:
            x_s = self.actvn(self.norm_layer_s(self.conv_s(x)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
