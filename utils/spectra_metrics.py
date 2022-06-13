import numpy as np
import torch
from typing import Dict
from utils.weighted_acc_rmse import unweighted_acc_torch_channels, \
    unweighted_rmse_torch_channels

@torch.jit.script
def ffl_torch(pred_fft: torch.Tensor, target_fft: torch.Tensor, # matrix: torch.Tensor=None,
              alpha: float=1., log_matrix: bool=False, batch_matrix: bool=False) -> torch.Tensor:
    """Focal Frequency Loss
    See https://github.com/EndlessSora/focal-frequency-loss

    Omits patching step (e.g. FocalFrequencyLoss.tensor2freq).
    """
    pred_freq = torch.stack([pred_fft.real, pred_fft.imag], -1)
    target_freq = torch.stack([target_fft.real, target_fft.imag], -1)

    # spectrum weight matrix
    # if matrix is not None:
    #     # if the matrix is predefined
    #     weight_matrix = matrix.detach()
    # else:

    # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
    matrix_tmp = (pred_freq - target_freq) ** 2
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

    assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
        'The values of spectrum weight matrix should be in the range [0, 1], '
        'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

    # frequency distance using (squared) Euclidean distance
    tmp = (pred_freq - target_freq) ** 2
    freq_distance = tmp[..., 0] + tmp[..., 1]

    # dynamic spectrum weighting (Hadamard product)
    loss = weight_matrix * freq_distance
    # return torch.mean(loss)
    return loss

@torch.jit.script
def fcl_torch(pred_amp: torch.Tensor, pred_phase: torch.Tensor,
              target_amp: torch.Tensor, target_phase: torch.Tensor) -> torch.Tensor:
    """Fourier Coefficient Loss
    See https://github.com/juglab/FourierImageTransformer
    """
    amp_loss = 1 + torch.pow(pred_amp - target_amp, 2)
    phi_loss = 2 - torch.cos(pred_phase - target_phase)
    return amp_loss * phi_loss

@torch.jit.script
def spectra_metrics_fft_input(pred_fft: torch.Tensor, target_fft: torch.Tensor,
                              alpha: float=1., log_matrix: bool=False, batch_matrix: bool=False) -> Dict[str, torch.Tensor]:
    pred_amp = pred_fft.abs()
    pred_phase = pred_fft.angle()
    tar_amp = target_fft.abs()
    tar_phase = target_fft.angle()

    out_dict = {
        'acc_amp': torch.mean(unweighted_acc_torch_channels(pred_amp, tar_amp)),
        'rmse_amp': torch.mean(unweighted_rmse_torch_channels(pred_amp, tar_amp)),
        'acc_phase': torch.mean(unweighted_acc_torch_channels(pred_phase, tar_phase)),
        'rmse_phase': torch.mean(unweighted_rmse_torch_channels(pred_phase, tar_phase)),
        'ffl': torch.mean(ffl_torch(pred_fft, target_fft, alpha, log_matrix, batch_matrix)),
        'fcl': torch.mean(fcl_torch(pred_amp, pred_phase, tar_amp, tar_phase)),
        'pred_fft': pred_fft,
        'tar_fft': target_fft,
    }
    return out_dict

@torch.jit.script
def spectra_metrics_rfft2(pred: torch.Tensor, target: torch.Tensor, eps: float=1.,
                          alpha: float=1., log_matrix: bool=False, batch_matrix: bool=False) -> Dict[str, torch.Tensor]:
    pred_fft = torch.fft.rfft2(pred, norm='ortho') / eps
    tar_fft = torch.fft.rfft2(target, norm='ortho') / eps
    return spectra_metrics_fft_input(pred_fft, tar_fft, alpha, log_matrix, batch_matrix)

@torch.jit.script
def spectra_metrics_rfft(pred: torch.Tensor, target: torch.Tensor, dim: int=-1, eps: float=1.,
                         alpha: float=1., log_matrix: bool=False, batch_matrix: bool=False) -> Dict[str, torch.Tensor]:

    pred_fft = torch.fft.rfft(pred, dim=dim, norm='ortho') / eps
    tar_fft = torch.fft.rfft(target, dim=dim, norm='ortho') / eps
    return spectra_metrics_fft_input(pred_fft, tar_fft, alpha, log_matrix, batch_matrix)

