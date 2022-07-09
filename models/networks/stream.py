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

        nf0 = params.ngf0
        nf = params.ngf

        self.res_0 = self.StreamResnetBlock(params.input_nc, 1 * nf0, params)  # 64-ch feature (default)
        self.res_1 = self.StreamResnetBlock(1  * nf0, 2  * nf0, params)   # 128-ch  feature
        self.res_2 = self.StreamResnetBlock(2  * nf0, 4  * nf0, params)   # 256-ch  feature
        self.res_3 = self.StreamResnetBlock(4  * nf0, 8  * nf0, params)   # 512-ch  feature
        self.res_4 = self.StreamResnetBlock(8  * nf0, 16 * nf0, params)    # 1024-ch feature
        self.res_5 = self.StreamResnetBlock(16 * nf0,  16 * nf, params)    # 1024-ch feature
        self.res_6 = self.StreamResnetBlock(16 * nf,   16 * nf, params)   # 1024-ch feature
        self.res_7 = self.StreamResnetBlock(16 * nf,   16 * nf, params)   # 1024-ch feature
        self.res_8 = self.StreamResnetBlock(16 * nf,   16 * nf, params) if params.num_upsampling_blocks == 8 else None # 1024b-ch feature


    def down(self, input, size=None):
        if size is None:
            return F.interpolate(input, scale_factor=0.5)
        return F.interpolate(input, size=size)


    def forward(self, input):
        # assume that input shape is (n,c,256,512) # (n,c,720,1440)

        x0 = self.res_0(input)  # (n,64,256,512) # (n, 1 * nf0, 720, 1440)

        img_size_log2 = 2**int(np.log2(self.params.img_size[0])), \
            2**int(np.log2(self.params.img_size[1]))

        if img_size_log2 != self.params.img_size:
            x1 = self.down(x0, size=img_size_log2)
        else:
            x1 = self.down(x0)
        x1 = self.res_1(x1)    # (n,128,128,256) # (n, 2 * nf0, 512, 1024)

        x2 = self.down(x1)
        x2 = self.res_2(x2)    # (n,256,64,128) # (n, 4 * nf0, 256, 512)

        x3 = self.down(x2)
        x3 = self.res_3(x3)    # (n,512,32,64) # (n, 8 * nf0, 128, 256)

        x4 = self.down(x3)
        x4 = self.res_4(x4)    # (n,1024,16,32) # (n, 16 * nf0, 64, 128)

        x5 = self.down(x4)
        x5 = self.res_5(x5)    # (n,1024,8,16) # (n, 16 * nf, 32, 64)

        x6 = self.down(x5)
        x6 = self.res_6(x6)    # (n,1024,4,8) # (n, 16 * nf, 16, 32)

        x7 = self.down(x6)
        x7 = self.res_7(x7)    # (n,1024,2,4) # (n, 16 * nf, 8, 16)

        if self.params.num_upsampling_blocks == 7:
            return [x0, x1, x2, x3, x4, x5, x6, x7, None]

        x8 = self.down(x7)
        x8 = self.res_8(x8)  # (n, 16 * nf, 4, 8)

        return [x0, x1, x2, x3, x4, x5, x6, x7, x8]


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

