import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.architecture import StreamResnetBlock
from models.networks.architecture_periodic import StreamResnetBlockPeriodic


# Content/style stream.
# The two streams are symmetrical with the same network structure,
# aiming at extracting corresponding feature representations in different levels.
class Stream(BaseNetwork):
    def __init__(self, params, reshape_size=None):
        super().__init__()
        self.params = params
        self.ppad = self.params.use_periodic_padding
        self.StreamResnetBlock = StreamResnetBlockPeriodic if self.ppad else StreamResnetBlock
        nf = params.ngf
        
        self.res_0 = self.StreamResnetBlock(params.input_nc, 1 * nf, params)  # 64-ch feature
        self.res_1 = self.StreamResnetBlock(1  * nf, 2  * nf, params)   # 128-ch  feature
        self.res_2 = self.StreamResnetBlock(2  * nf, 4  * nf, params)   # 256-ch  feature
        self.res_3 = self.StreamResnetBlock(4  * nf, 8  * nf, params)   # 512-ch  feature
        self.res_4 = self.StreamResnetBlock(8  * nf, 16 * nf, params)   # 1024-ch feature
        self.res_5 = self.StreamResnetBlock(16 * nf, 16 * nf, params)   # 1024-ch feature
        self.res_6 = self.StreamResnetBlock(16 * nf, 16 * nf, params)   # 1024-ch feature
        self.res_7 = self.StreamResnetBlock(16 * nf, 16 * nf, params) if params.num_upsampling_blocks != 6 else None   # 1024-ch feature

    def down(self, input):
        return F.interpolate(input, scale_factor=0.5)

    def forward(self,input):
        # assume that input shape is (n,c,256,512)

        x0 = self.res_0(input) # (n,64,256,512)
        x1 = self.down(x0)
        x1 = self.res_1(x1)    # (n,128,128,256)

        x2 = self.down(x1)
        x2 = self.res_2(x2)    # (n,256,64,128)

        x3 = self.down(x2)
        x3 = self.res_3(x3)    # (n,512,32,64)

        x4 = self.down(x3)
        x4 = self.res_4(x4)    # (n,1024,16,32)

        x5 = self.down(x4)
        x5 = self.res_5(x5)    # (n,1024,8,16)

        x6 = self.down(x5)
        x6 = self.res_6(x6)    # (n,1024,4,8)

        if self.params.num_upsampling_blocks != 6:
            x7 = self.down(x6)
            x7 = self.res_7(x7)    # (n,1024,2,4)
        else:
            x7 = None

        return [x0, x1, x2, x3, x4, x5, x6, x7]


# Additive noise stream inspired by StyleGAN.
# The two streams are symmetrical with the same network structure,
# aiming at extracting corresponding feature representations in different levels.
class NoiseStream(BaseNetwork):
    def __init__(self, params):
        super().__init__()
        self.params = params
        nf = params.ngf

        iloc, isc = 1., 0.05
        scalers = []
        for i in range(8):
            val = torch.from_numpy(np.random.normal(loc=iloc, scale=isc, size=(1,nf,1,1)).astype(np.float32))
            scalers.append(nn.Parameter(val, requires_grad=True))
            nf = min(nf*2, self.params.ngf*16)
        self.featmult = nn.ParameterList(scalers)

    def forward(self,input):
        # assume that input shape is (n,c,h,w)
        n,h,w = input.shape[0], input.shape[2], input.shape[3]

        nf = self.params.ngf
        out = []
        for i in range(8):
            noise = 2.*torch.randn((n, 1, h, w), device=input.device)
            out.append(noise*self.featmult[i])
            nf = min(nf*2, self.params.ngf*16)
            h //= 2
            w //= 2

        return out

