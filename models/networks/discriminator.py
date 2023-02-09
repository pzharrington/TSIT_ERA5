import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_norm_layer
from models.networks.ppad import PeriodicPad2d


class MultiscaleDiscriminator(BaseNetwork):
    
    def __init__(self, params):
        super().__init__()
        self.params = params

        for i in range(params.num_D):
            subnetD = NLayerDiscriminator(params)
            self.add_module('discriminator_%d' % i, subnetD)


    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=2,
                            stride=2, padding=[0, 0],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size params.num_D x params.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.params.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.ppad = self.params.use_periodic_padding

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        convpad = 0 if self.ppad else padw
        nf = params.ndf
        input_nc = self.compute_D_input_nc(params)

        norm_layer = get_norm_layer(params, params.norm_D)
        Padder = PeriodicPad2d if self.ppad else nn.Identity
        sequence = [[Padder(padw),
                     nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=convpad),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, params.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == (params.n_layers_D - 1) else 2
            sequence += [[Padder(padw),
                          norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=convpad)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[Padder(padw),
                      nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=convpad)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, params):
        input_nc = params.output_nc
        if params.cat_inp:
            input_nc += params.input_nc
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.params.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
