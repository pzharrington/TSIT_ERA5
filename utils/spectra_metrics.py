import torch
from typing import Dict, List, Optional
from utils.weighted_acc_rmse import lat, latitude_weighting_factor_torch

@torch.jit.script
def rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]=None, dim: Optional[List[int]]=None) -> torch.Tensor:
    if dim is None:
        dim = [-1, -2]
    if weight is None:
        weight = torch.ones_like(pred)
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=dim))
    return result

@torch.jit.script
def acc_torch_channels(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]=None, dim: Optional[List[int]]=None) -> torch.Tensor:
    if dim is None:
        dim = [-1, -2]
    if weight is None:
        weight = torch.ones_like(pred)
    result = torch.sum(weight * pred * target, dim=dim) / torch.sqrt(torch.sum(weight * pred * pred, dim=dim) * torch.sum(weight * target * target, dim=dim))
    return result

@torch.jit.script
def freq_weights(num_freq: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    freq_t = torch.arange(start=num_freq - 1, end=-1, step=-1, device=device)
    lam = 1. / (num_freq / 10.)
    freq_weights = lam * torch.exp(-lam*freq_t)
    freq_weights = freq_weights / freq_weights.max()
    return torch.reshape(freq_weights, (1, 1, 1, -1))

@torch.jit.script
def fl_weights(num_freq: int, num_lat: int,
               freq_weighting: bool=False, lat_weighting: bool=False,
               clamp: bool=True, min: float=0., max: float=1.,
               device: torch.device = torch.device('cpu')) -> torch.Tensor:

    if freq_weighting:
        w = freq_weights(num_freq, device=device)
    else:
        w = torch.ones(1, 1, 1, num_freq, device=device)

    if lat_weighting:
        lat_t = torch.arange(start=0, end=num_lat, device=device)
        s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
        lat_w = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s),
                              (1, 1, -1, 1))
        if freq_weighting:
            # lat_w = num_freq * torch.outer(lat_w.squeeze(), w.squeeze())
            lat_w = torch.outer(lat_w.squeeze(), w.squeeze())

        w = lat_w

    w = w.expand(1, 1, num_lat, num_freq)

    if clamp:
        w = torch.clamp(w, min=min, max=max)

    return w

@torch.jit.script
def ffl_torch(pred_fft: torch.Tensor, target_fft: torch.Tensor,
              alpha: float=1., log_matrix: bool=False, batch_matrix: bool=False,
              weight: Optional[torch.Tensor]=None) -> torch.Tensor:
    """Focal Frequency Loss
    See https://github.com/EndlessSora/focal-frequency-loss

    Omits patching step (i.e., FocalFrequencyLoss.tensor2freq).
    """
    pred_freq = torch.stack([pred_fft.real, pred_fft.imag], -1)
    target_freq = torch.stack([target_fft.real, target_fft.imag], -1)

    # spectrum weight matrix
    # if matrix is not None:
    #     # if the matrix is predefined
    #     weight_matrix = matrix.detach()
    # else:

    if weight is None:
        w = torch.ones_like(pred_freq)
    else:
        w = torch.stack([weight, weight], -1)

    # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
    matrix_tmp = w * (pred_freq - target_freq) ** 2
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
              target_amp: torch.Tensor, target_phase: torch.Tensor,
              weight: Optional[torch.Tensor]=None) -> torch.Tensor:
    """Fourier Coefficient Lossprecip_eps
    See https://github.com/juglab/FourierImageTransformer
    """
    amp_loss = 1 + torch.pow(pred_amp - target_amp, 2)
    phi_loss = 2 - torch.cos(pred_phase - target_phase)
    if weight is None:
        weight = torch.ones_like(pred_amp)
    return weight * amp_loss * phi_loss

@torch.jit.script
def spectra_metrics_fft_input(pred_fft: torch.Tensor,
                              target_fft: torch.Tensor,
                              dim_mean: Optional[List[int]]=None,
                              alpha: float=1.,
                              log_ffl: bool=False,
                              batch_ffl: bool=False,
                              freq_weighting: bool=False,
                              lat_weighting: bool=False) -> Dict[str, torch.Tensor]:

    num_lat = pred_fft.shape[-2]
    num_freq = pred_fft.shape[-1]

    w = fl_weights(num_freq, num_lat, freq_weighting, lat_weighting).to(pred_fft.device)

    assert w.shape[-1] == pred_fft.shape[-1], \
        f'unexpected weights shape: {w.shape}'
    assert w.shape[-2] == pred_fft.shape[-2], \
        f'unexpected weights shape: {w.shape}'

    pred_amp = pred_fft.abs()
    pred_phase = torch.sin(pred_fft.angle())

    tar_amp = target_fft.abs()
    tar_phase = torch.sin(target_fft.angle())

    if dim_mean is None:
        dim_mean = [-2, -1]

    pam = pred_amp.mean(dim=dim_mean, keepdim=True)
    ppm = pred_phase.mean(dim=dim_mean, keepdim=True)
    tam = tar_amp.mean(dim=dim_mean, keepdim=True)
    tpm = tar_phase.mean(dim=dim_mean, keepdim=True)

    out_dict = {
        'corr_amp': torch.nan_to_num(acc_torch_channels(pred_amp - pam, tar_amp - tam, weight=w, dim=dim_mean)),
        'corr_phase': torch.nan_to_num(acc_torch_channels(pred_phase - ppm, tar_phase - tpm, weight=w, dim=dim_mean)),
        'rmse_amp': rmse_torch_channels(pred_amp, tar_amp, weight=w, dim=dim_mean),
        'rmse_phase': rmse_torch_channels(pred_phase, tar_phase, weight=w, dim=dim_mean),
        'ffl': torch.mean(ffl_torch(pred_fft, target_fft, alpha, log_ffl, batch_ffl, weight=w), dim=dim_mean),
        'fcl': torch.mean(fcl_torch(pred_amp, pred_phase, tar_amp, tar_phase, weight=w), dim=dim_mean),
    }

    return out_dict

@torch.jit.script
def spectra_metrics_rfft2(pred: torch.Tensor,
                          target: torch.Tensor,
                          dim_mean: Optional[List[int]]= None,
                          alpha: float=1.,
                          log_ffl: bool=False,
                          batch_ffl: bool=False,
                          freq_weighting: bool=False,
                          lat_weighting: bool=False) -> Dict[str, torch.Tensor]:
    if dim_mean is None:
        dim_mean = [-2, -1]

    pred_fft = torch.fft.rfft2(pred, norm='ortho')
    tar_fft = torch.fft.rfft2(target, norm='ortho')
    out_dict = spectra_metrics_fft_input(pred_fft, tar_fft, dim_mean, alpha, log_ffl, batch_ffl)
    out_dict.update({'pred_fft': pred_fft, 'tar_fft': tar_fft})
    return out_dict

@torch.jit.script
def spectra_metrics_rfft(pred: torch.Tensor,
                         target: torch.Tensor,
                         dim: int=-1,
                         dim_mean: Optional[List[int]]=None,
                         alpha: float=1.,
                         log_ffl: bool=False,
                         batch_ffl: bool=False,
                         freq_weighting: bool=False,
                         lat_weighting: bool=False) -> Dict[str, torch.Tensor]:
    if dim_mean is None:
        dim_mean = [-2, -1]

    pred_fft = torch.fft.rfft(pred, dim=dim, norm='ortho')
    tar_fft = torch.fft.rfft(target, dim=dim, norm='ortho')
    out_dict = spectra_metrics_fft_input(pred_fft, tar_fft, dim_mean, alpha, log_ffl, batch_ffl)
    out_dict.update({'pred_fft': pred_fft, 'tar_fft': tar_fft})
    return out_dict
