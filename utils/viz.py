import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from utils.weighted_acc_rmse import unlog_tp


def viz_fields(flist):
    pred, tar, afno = flist
    pred = pred[0]
    tar = tar[0]
    afno = afno[0]
    sc = tar.max()

    rows = 3
    if afno is not None:
        rows = 6

    f = plt.figure(figsize=(18,12*rows))

    plt.subplot(rows,1,1)
    plt.imshow(tar, cmap='Blues', norm=Normalize(0., sc))
    plt.colorbar(shrink=0.4, location='bottom', pad=0.05)
    plt.title(f'Truth (max={tar.max():.4f})')

    plt.subplot(rows,1,2)
    plt.imshow(pred, cmap='Blues', norm=Normalize(0., sc))
    plt.colorbar(shrink=0.4, location='bottom', pad=0.05, extend='max')
    plt.title(f'TSIT (max={pred.max():.4f}), bounded by Truth max')

    plt.subplot(rows,1,3)
    err = pred - tar

    if afno is not None:
        afno_err = afno - tar
        err_min = afno_err.min()
        err_max = afno_err.max()
    else:
        err_min = err.min()
        err_max = err.max()

    if err_min < 0. and err_max > 0.:
        plt.imshow(err, norm=TwoSlopeNorm(0., err_min, err_max), cmap='bwr')
        plt.colorbar(shrink=0.4, location='bottom', pad=0.05, extend='both')
        err_title = f'TSIT error (max absolute err={np.abs(err).max():.4f})'
        if afno is not None:
             err_title += ', bounded by AFNO error range'
        plt.title(err_title)

        if afno is not None:
            plt.subplot(rows,1,4)
            plt.imshow(afno, cmap='Blues', norm=Normalize(0., sc))
            plt.colorbar(shrink=0.4, location='bottom', pad=0.05, extend='max')
            plt.title(f'AFNO (max: {afno.max():.4f}), bounded by Truth max')

            plt.subplot(rows,1,5)
            afno_err = afno - tar
            plt.imshow(afno_err,
                       norm=TwoSlopeNorm(0., err_min, err_max), cmap='bwr')
            plt.colorbar(shrink=0.4, location='bottom', pad=0.05)
            plt.title(f'AFNO error (max absolute err={np.abs(afno_err).max():.4f})')

            abs_err_diff = np.abs(err) - np.abs(afno_err)
            if abs_err_diff.min() < 0. and abs_err_diff.max() > 0.:
                plt.subplot(rows,1,6)
                plt.imshow(abs_err_diff, norm=TwoSlopeNorm(0., abs_err_diff.min(), abs_err_diff.max()), cmap='bwr')
                plt.colorbar(shrink=0.4, location='bottom', pad=0.05)
                plt.title(f'Difference in absolute error (blue = TSIT better): min={abs_err_diff.min():.4f}, max={abs_err_diff.max():.4f})')

    plt.tight_layout()
    return f


def viz_ens(fields):
    tar, afno, mean_field, std_field, hists = fields

    f = plt.figure(figsize=(32, 24))

    err = mean_field - tar
    if afno is not None:
        afno_err = afno - tar
        err_min = afno_err.min()
        err_max = afno_err.max()
    else:
        err_min = err.min()
        err_max = err.max()

    plt.subplot(2,2,1)
    # sc = np.quantile(std_field, 0.999)
    plt.imshow(std_field, cmap='inferno')
    plt.colorbar(shrink=0.4, location='bottom', pad=0.05)
    plt.title(f'Standard deviation field (max={std_field.max():.4g})')

    plt.subplot(2,2,2)
    plt.imshow(mean_field, norm=Normalize(0., np.max(tar)), cmap='Blues')
    plt.colorbar(shrink=0.4, location='bottom', pad=0.05)
    plt.title('Mean field, bounded by Truth max')

    plt.subplot(2,2,3)
    plt.imshow(err, norm=TwoSlopeNorm(0., err.min(), err.max()), cmap='bwr')
    plt.colorbar(shrink=0.4, location='bottom', pad=0.05, extend='both')
    err_title = f'Mean field error (max absolute err={np.abs(err).max():.4f})'
    if afno is not None:
        err_title += ', bounded by AFNO error range'
    plt.title(err_title)

    ens = hists['ens']
    ens_mean = ens.mean(axis=0)
    ens_std = ens.std(axis=0) # / np.sqrt(ens.shape[0])
    ens_memb = hists['ens_memb']
    ens_memb_mean = ens_memb.mean(axis=0)
    ens_memb_std = ens_memb.std(axis=0) # / np.sqrt(ens.shape[0])
    tar = hists['target']
    tar_mean = tar.mean(axis=0)
    tar_std = tar.std(axis=0) # / np.sqrt(ens.shape[0])

    bins = 300
    max = 11.
    bin_edges = np.linspace(0., max, bins + 1)

    # xlim, ylim = (0., 11.), (0.0000001, 0.5)
    xlim, ylim = (8., 11.1), (0.0000001, 0.01)

    plt.subplot(2,2,4, adjustable='box', aspect=0.3)

    # era5
    plt.hist(bin_edges[:-1], bin_edges, weights=tar_mean, density=True, histtype='step', log=True,
             color='k', linestyle='--', label='era5')
    plt.hist(bin_edges[:-1], bin_edges, weights=tar_mean - tar_std, density=True, histtype='step', log=True,
             color='k', linestyle='--', alpha=0.3)
    plt.hist(bin_edges[:-1], bin_edges, weights=tar_mean + tar_std, density=True, histtype='step', log=True,
             color='k', linestyle='--', alpha=0.3)

    # ens mean
    plt.hist(bin_edges[:-1], bin_edges, weights=ens_mean, density=True, histtype='step', log=True,
             color='g', linestyle='-', label='ens mean')
    plt.hist(bin_edges[:-1], bin_edges, weights=ens_mean - ens_std, density=True, histtype='step', log=True,
             color='g', linestyle='-', alpha=0.3)
    plt.hist(bin_edges[:-1], bin_edges, weights=ens_mean + ens_std, density=True, histtype='step', log=True,
             color='g', linestyle='-', alpha=0.3)

    # ens member
    plt.hist(bin_edges[:-1], bin_edges, weights=ens_memb_mean, density=True, histtype='step', log=True,
             color='darkorange', linestyle='--', label='mean of ens. membs.')
    plt.hist(bin_edges[:-1], bin_edges, weights=ens_memb_mean - ens_memb_std, density=True, histtype='step', log=True,
             color='darkorange', linestyle='--', alpha=0.3)
    plt.hist(bin_edges[:-1], bin_edges, weights=ens_mean + ens_memb_std, density=True, histtype='step', log=True,
             color='darkorange', linestyle='--', alpha=0.3)

    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('log(1 + TP / 1e-5)')
    plt.ylabel('log density')
    plt.title(f'mean (+/- std) of per-image TP distribution from {unlog_tp(xlim[0]):.4f} to {unlog_tp(xlim[1]):.4f} m (n={tar.shape[0]})')

    plt.suptitle('Ensemble')
    plt.tight_layout()
    return f


def viz_density(precip_hists: dict):
    pred = precip_hists['pred']
    pred_mean = pred.mean(axis=0)
    pred_stderr = pred.std(axis=0) # / np.sqrt(pred.shape[0])
    tar = precip_hists['target']
    tar_mean = tar.mean(axis=0)
    tar_stderr = tar.std(axis=0) # / np.sqrt(tar.shape[0])
    afno = precip_hists.get('afno')
    if afno is not None:
        afno_mean = precip_hists['afno'].mean(axis=0)
        afno_stderr = precip_hists['afno'].std(axis=0) # / np.sqrt(afno.shape[0])

    bins = 300
    max = 11.
    bin_edges = np.linspace(0., max, bins + 1)

    f = plt.figure(figsize=(20, 10))

    xylims = [((0., 11.), (0.0000001, 0.5)),
              ((0., 4.), (0.075, 0.325)),
              ((4., 6.), (0.04, 0.15)),
              ((6., 11.), (0.0000001, 0.04))]

    for i, (xlim, ylim) in enumerate(xylims):
        plt.subplot(2, 2, i+1)
        # era5
        plt.hist(bin_edges[:-1], bin_edges, weights=tar_mean, density=True, histtype='step', log=True,
                 color='k', linestyle='--', label='era5')
        plt.hist(bin_edges[:-1], bin_edges, weights=tar_mean - tar_stderr, density=True, histtype='step', log=True,
                 color='k', linestyle='--', alpha=0.3)
        plt.hist(bin_edges[:-1], bin_edges, weights=tar_mean + tar_stderr, density=True, histtype='step', log=True,
                 color='k', linestyle='--', alpha=0.3)

        # tsit
        plt.hist(bin_edges[:-1], bin_edges, weights=pred_mean, density=True, histtype='step', log=True,
                 color='g', linestyle='-', label='tsit')
        plt.hist(bin_edges[:-1], bin_edges, weights=pred_mean - pred_stderr, density=True, histtype='step', log=True,
                 color='g', linestyle='-', alpha=0.3)
        plt.hist(bin_edges[:-1], bin_edges, weights=pred_mean + pred_stderr, density=True, histtype='step', log=True,
                 color='g', linestyle='-', alpha=0.3)

        if afno is not None:
            plt.hist(bin_edges[:-1], bin_edges, weights=afno_mean, density=True, histtype='step', log=True,
                     color='r', linestyle='-', label='era5')
            plt.hist(bin_edges[:-1], bin_edges, weights=afno_mean - afno_stderr, density=True, histtype='step', log=True,
                     color='r', linestyle='-', alpha=0.3)
            plt.hist(bin_edges[:-1], bin_edges, weights=afno_mean + afno_stderr, density=True, histtype='step', log=True,
                     color='r', linestyle='-', alpha=0.3)

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(f'{unlog_tp(xlim[0]):.4f} to {unlog_tp(xlim[1]):.4f} m')
        plt.legend()
        if i in [1, 3]:
            plt.ylabel('log density')
        if i in [3, 4]:
            plt.xlabel('log(1 + TP / 1e-5)')
    plt.suptitle(f'mean (+/- std) of per-image TP distribution (n={pred.shape[0]})')
    plt.tight_layout()
    return f


def viz_spectra(spectra):
    spec_mean, spec_std, n = spectra

    plt_fmt = {
        'tsit': 'g-',
        'era5': 'k--',
        'afno': 'r-',
    }

    f = plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    for key in spec_mean.keys():
        mean = spec_mean[key]
        std = spec_std[key]
        wavenum = np.arange(start=0, stop=mean.shape[-1])
        plt.semilogy(wavenum, mean, plt_fmt[key], label=key)
        plt.fill_between(wavenum, mean - std, mean + std,
                         color=plt_fmt[key][0], alpha=0.2)

    plt.xlabel('wave number')
    plt.ylabel('amplitude')
    plt.legend()
    plt.title(f'Power spectra +/- STD (n = {n})')

    plt.subplot(2, 1, 2)
    for key in spec_mean.keys():
        mean = spec_mean[key][600:]
        stderr = spec_std[key][600:] / np.sqrt(n)
        wavenum = np.arange(start=600, stop=spec_mean[key].shape[-1])
        plt.semilogy(wavenum, mean, plt_fmt[key], label=key)
        plt.fill_between(wavenum, mean - stderr, mean + stderr,
                         color=plt_fmt[key][0], alpha=0.2)

    plt.xlabel('wave number')
    plt.ylabel('amplitude')
    plt.legend()
    plt.title(f'Power spectra +/- std. err. (n = {n})')

    return f
