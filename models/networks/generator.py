import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import FADEResnetBlock as FADEResnetBlock
from models.networks.architecture_periodic import FADEResnetBlockPeriodic
from models.networks.stream import Stream as Stream
from models.networks.stream import NoiseStream as NoiseStream
from models.networks.AdaIN.function import adaptive_instance_normalization as FAdaIN
from models.networks.ppad import PeriodicPad2d
from utils.img_utils import compute_latent_vector_size

class TSITGenerator(BaseNetwork):

    def __init__(self, params):
        super().__init__()
        self.params = params
        nf0 = params.ngf0
        nf = params.ngf
        self.ppad = self.params.use_periodic_padding
        self.FADEResnetBlock = FADEResnetBlockPeriodic if self.ppad else FADEResnetBlock

        self.sw, self.sh, self.n_stages = compute_latent_vector_size(params)
        self.content_stream = Stream(self.params)  # , reshape_size=(self.sh*(2**self.n_stages), self.sw*(2**self.n_stages)))
        self.style_stream = Stream(self.params) if not self.params.no_ss else None
        self.noise_stream = NoiseStream(self.params) if self.params.additive_noise else None

        if params.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(params.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled input instead of random z, unless params.downsamp is False
            self.nz = self.params.z_dim if not self.params.downsamp else self.params.input_nc
            padder = PeriodicPad2d(1) if self.ppad else nn.Identity()
            convpad = 0 if self.ppad else 1
            self.fc = nn.Sequential(padder, nn.Conv2d(self.nz, 16 * nf, 3, padding=convpad))

        self.head_0 = self.FADEResnetBlock(16 * nf, 16 * nf, params) if self.params.num_upsampling_blocks == 8 else None
        self.head_1 = self.FADEResnetBlock(16 * nf, 16 * nf, params)

        self.G_middle_0 = self.FADEResnetBlock(16 * nf, 16 * nf, params)
        self.G_middle_1 = self.FADEResnetBlock(16 * nf, 16 * nf0, params)

        self.up_0 = self.FADEResnetBlock(16 * nf0, 8 * nf0, params)
        self.up_1 = self.FADEResnetBlock(8  * nf0, 4 * nf0, params)
        self.up_2 = self.FADEResnetBlock(4  * nf0, 2 * nf0, params)
        self.up_3 = self.FADEResnetBlock(2  * nf0, 1 * nf0, params)

        final_nc = nf0

        if self.ppad:
            self.conv_img = nn.Sequential(PeriodicPad2d(1), nn.Conv2d(final_nc, self.params.output_nc, 3, padding=0))
        else:
            self.conv_img = nn.Conv2d(final_nc, self.params.output_nc, 3, padding=1)

    def up(self, input, size=None):
        if size is None:
            return F.interpolate(input, scale_factor=2.)
        return F.interpolate(input, size=self.params.img_size)

    def fadain_alpha(self, content_feat, style_feat, alpha=1.0, c_mask=None, s_mask=None):
        # FAdaIN performs AdaIN on the multi-scale feature representations
        assert 0 <= alpha <= 1
        t = FAdaIN(content_feat, style_feat, c_mask, s_mask)
        t = alpha * t + (1 - alpha) * content_feat
        return t

    def forward(self, input, real, z=None):
        content = input
        style =  real
        ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8 = self.content_stream(content)
        sft0, sft1, sft2, sft3, sft4, sft5, sft6, sft7, sft8 = self.style_stream(style) if not self.params.no_ss else [None] * 9
        nft0, nft1, nft2, nft3, nft4, nft5, nft6, nft7, nft8 = self.noise_stream(style) if self.params.additive_noise else [None] * 9
        if self.params.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(content.size(0), self.params.z_dim,
                                dtype=torch.float32, device=content.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.params.ngf, self.sw, self.sh)
        else:
            if self.params.downsamp:
                # following SPADE, downsample segmap and run convolution for SIS
                x = F.interpolate(content, size=(self.sw, self.sh))
            else:
                # sample random noise
                x = torch.randn(content.size(0), self.nz, self.sw, self.sh,
                                dtype=torch.float32, device=content.get_device())
            x = self.fc(x) # (n, 16 * nf, 4, 8), assuming num_upsampling_blocks == 8

        if self.params.num_upsampling_blocks == 8:
            x = self.fadain_alpha(x, sft8, alpha=self.params.alpha) if not self.params.no_ss else x
            x = x + nft8 if self.params.additive_noise else x
            x = self.head_0(x, ft8) # (n, 16 * nf, 4, 8)
            x = self.up(x)

        x = self.fadain_alpha(x, sft7, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft7 if self.params.additive_noise else x
        x = self.head_1(x, ft7) # (n, 16 * nf, 8, 16)

        x = self.up(x)
        x = self.fadain_alpha(x, sft6, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft6 if self.params.additive_noise else x
        x = self.G_middle_0(x, ft6) # (n, 16 * nf, 16, 32)

        x = self.up(x)
        x = self.fadain_alpha(x, sft5, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft5 if self.params.additive_noise else x
        x = self.G_middle_1(x, ft5) # (n, 16 * nf0, 32, 64)

        x = self.up(x)
        x = self.fadain_alpha(x, sft4, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft4 if self.params.additive_noise else x
        x = self.up_0(x, ft4) # (n, 8 * nf0, 64, 128)

        x = self.up(x)
        x = self.fadain_alpha(x, sft3, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft3 if self.params.additive_noise else x
        x = self.up_1(x, ft3) # (n, 4 * nf0, 128, 256)

        x = self.up(x)
        x = self.fadain_alpha(x, sft2, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft2 if self.params.additive_noise else x
        x = self.up_2(x, ft2) # (n, 2 * nf0, 256, 512)

        x = self.up(x)
        x = self.fadain_alpha(x, sft1, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft1 if self.params.additive_noise else x
        x = self.up_3(x, ft1) # (n, 1 * nf0, 512, 1024)

        if self.params.DEBUG:
            assert x.shape[-2] == 512 and x.shape[-1] == 1024, \
                f'unexepcted img shape before final upsampling: {x.shape}'

        x = self.up(x, size=self.params.img_size)
        x = self.conv_img(F.leaky_relu(x, 2e-1)) # (n, input_nc, 720, 1440)
        x = F.relu(x)
        return x


class Pix2PixHDGenerator(BaseNetwork):
    '''Not tested.'''

    def __init__(self, params):
        super().__init__()
        input_nc = params.label_nc + (1 if params.contain_dontcare_label else 0) + (0 if params.no_instance else 1)

        norm_layer = get_norm_layer(params, params.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(params.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, params.ngf,
                                       kernel_size=params.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(params.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(params.ngf * mult, params.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(params.resnet_n_blocks):
            model += [ResnetBlock(params.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=params.resnet_kernel_size)]

        # upsample
        for i in range(params.resnet_n_downsample):
            nc_in = int(params.ngf * mult)
            nc_out = int((params.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, params.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
