import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.spectra_metrics import fl_weights


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, params=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.params = params
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class FFLoss(nn.Module):

    def __init__(self, num_freq: int, num_lat: int,
                 freq_weighting: bool=True, lat_weighting: bool=True,
                 clamp_weights: bool=True, min: float=0.1, max: float=1.,
                 device: torch.device = torch.device('cpu')):
        self.device = device
        if freq_weighting or lat_weighting:
            w = fl_weights(num_freq, num_lat,
                           freq_weighting, lat_weighting,
                           clamp_weights, min, max,
                           self.device)
            self.w = torch.stack([w, w], -1).to(self.device).detach()
        else:
            self.w = None

    def __call__(self, pred, target,
                 alpha: float=1.,
                 log_matrix: bool=False,
                 batch_matrix: bool=False):

        pred_fft = torch.fft.rfft(pred, dim=-1, norm='ortho')
        tar_fft = torch.fft.rfft(target, dim=-1, norm='ortho')

        pred_freq = torch.stack([pred_fft.real, pred_fft.imag], -1)
        tar_freq = torch.stack([tar_fft.real, tar_fft.imag], -1)

        if self.w is None:
            matrix_tmp = (pred_freq - tar_freq) ** 2
        else:
            if self.w.device != pred.device:
                self.w = w.to(pred.device).detach()
            w = self.w.clone().detach()
            matrix_tmp = w * (pred_freq - tar_freq) ** 2

        # if the matrix is calculated online: continuous, dynamic, based on
        # current Euclidean distance
        matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** alpha

        # whether to adjust the spectrum weight matrix by logarithm
        if log_matrix:
            matrix_tmp = torch.log(matrix_tmp + 1.0)

        # whether to calculate the spectrum weight matrix using batch-based statistics
        if batch_matrix:
            matrix_tmp = matrix_tmp / matrix_tmp.max()
        else:
            matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, None, None]

        matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
        matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
        weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0. and weight_matrix.max().item() <= 1., (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (pred_freq - tar_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        # return torch.mean(loss)
        return loss.mean()
