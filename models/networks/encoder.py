import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_norm_layer
from utils.img_utils import compute_latent_vector_size

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, params):
        super().__init__()
        self.params = params
        sw, sh, _ = compute_latent_vector_size(params)
        self.img_size_log2 = 2**int(np.log2(params.img_size[0])), \
                2**int(np.log2(params.img_size[1]))
        if self.img_size_log2 != params.img_size:
            self.n_layers = params.num_upsampling_blocks - 1
        else:
            self.n_layers = params.num_upsampling_blocks
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = params.nef
        norm_layer = get_norm_layer(params, params.norm_E)

        if params.vae_full_res_start:
            self.layer0 = norm_layer(nn.Conv2d(self.params.output_nc, ndf, kernel_size=kw, padding=1))
            self.layer1 = norm_layer(nn.Conv2d(ndf, ndf * 2, kw, stride=2, padding=pw))
            ndf = ndf * 2
        else:
            self.layer0 = None
            self.layer1 = norm_layer(nn.Conv2d(self.params.output_nc, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if self.n_layers == 7:
            self.layer7 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * sw * sh, self.params.z_dim)
        self.fc_var = nn.Linear(ndf * 8 * sw * sh, self.params.z_dim)

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, x):

        if self.params.vae_full_res_start:
            x = self.layer0(x)

        if x.size(2) != self.img_size_log2[0] or x.size(3) != self.img_size_log2[1]:
            x = F.interpolate(x, size=self.img_size_log2, mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.layer6(self.actvn(x))
        if self.n_layers == 7:
            x = self.layer7(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar
