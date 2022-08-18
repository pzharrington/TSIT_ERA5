import torch
from typing import Tuple, Optional


@torch.jit.script
def precip_histc(field: torch.Tensor, bins: int=300, max: float=11.) -> torch.Tensor:
    hists = []
    field = field.float()
    for i in range(field.shape[0]):
        hists.append(torch.unsqueeze(torch.histc(field[i].float(), bins=bins, min=0., max=max), dim=0))
    hists = torch.cat(hists, dim=0)
    return hists


@torch.jit.script
def precip_histc2(pred: torch.Tensor, target: torch.Tensor, bins: int=300, max: float=11.) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_hists = []
    tar_hists = []
    pred = pred.float()
    target = target.float()
    for i in range(target.shape[0]):
        pred_hists.append(torch.unsqueeze(torch.histc(pred[i].float(), bins=bins, min=0., max=max), dim=0))
        tar_hists.append(torch.unsqueeze(torch.histc(target[i].float(), bins=bins, min=0., max=max), dim=0))
    pred_hists = torch.cat(pred_hists, dim=0)
    tar_hists = torch.cat(tar_hists, dim=0)
    return pred_hists, tar_hists


@torch.jit.script
def precip_histc3(pred: torch.Tensor, target: torch.Tensor, afno: torch.Tensor, bins: int=300, max: float=11.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_hists = []
    tar_hists = []
    afno_hists = []
    pred = pred.float()
    target = target.float()
    for i in range(target.shape[0]):
        pred_hists.append(torch.unsqueeze(torch.histc(pred[i].float(), bins=bins, min=0., max=max), dim=0))
        tar_hists.append(torch.unsqueeze(torch.histc(target[i].float(), bins=bins, min=0., max=max), dim=0))
        afno_hists.append(torch.unsqueeze(torch.histc(afno[i].float(), bins=bins, min=0., max=max), dim=0))
    pred_hists = torch.cat(pred_hists, dim=0)
    tar_hists = torch.cat(tar_hists, dim=0)
    afno_hists = torch.cat(afno_hists, dim=0)
    return pred_hists, tar_hists, afno_hists


@torch.jit.script
def binned_precip_log_l1(pred: torch.Tensor, target: torch.Tensor,
                         pred_hist: torch.Tensor, target_hist: torch.Tensor,
                         bins: int=300, max: float=11.) -> torch.Tensor:
    if pred_hist is None:
        pred_hist = precip_histc(pred, bins, max)
    if target_hist is None:
        target_hist = precip_histc(target, bins, max)
    err = torch.log1p(torch.abs(pred_hist - target_hist)).sum(dim=1)
    pred_counts = torch.sum(pred > max, dim=[1, 2, 3])
    tar_counts = torch.sum(pred > max, dim=[1, 2, 3])
    err += (pred_counts - tar_counts).abs().log1p()
    return err / (bins + 1)

