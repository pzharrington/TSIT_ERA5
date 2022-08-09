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
    plt.title('Truth')

    plt.subplot(rows,1,2)
    plt.imshow(pred, cmap='Blues', norm=Normalize(0., sc))
    plt.title('TSIT')

    plt.subplot(rows,1,3)
    err = (pred - tar) / (tar + 1.)

    if err.min() < 0. and err.max() > 0.:
        plt.imshow(err, norm=TwoSlopeNorm(0., err.min(), err.max()), cmap='bwr')
        plt.title('TSIT relative error')

        if afno is not None:
            plt.subplot(rows,1,4)
            plt.imshow(afno, cmap='Blues', norm=Normalize(0., sc))
            plt.title('AFNO')

            plt.subplot(rows,1,5)
            afno_err = (afno - tar) / (tar + 1.)
            plt.imshow(afno_err,
                       norm=TwoSlopeNorm(0., err.min(), err.max()), cmap='bwr')
            plt.title('AFNO relative error')

            abs_err_diff = np.abs(err) - np.abs(afno_err)
            if abs_err_diff.min() < 0. and abs_err_diff.max() > 0.:
                plt.subplot(rows,1,6)
                plt.imshow(abs_err_diff, norm=TwoSlopeNorm(0., abs_err_diff.min(), abs_err_diff.max()), cmap='bwr')
                plt.title('Difference in absolute relative error (blue = TSIT better)')

    plt.tight_layout()
    return f


def viz_std_field(std_field):
    std_field = std_field[0]
    sc = np.quantile(std_field, 0.99)
    f = plt.figure(figsize=(18, 12))
    plt.imshow(std_field, cmap='inferno', norm=Normalize(0., sc))
    plt.colorbar(shrink=0.6, extend='max')
    plt.title('Mean ensemble STD')
    return f


def viz_density(flist):
    pred, tar, afno = flist
    pred = pred[0]
    tar = tar[0]
    afno = afno[0] if afno is not None else None

    bins = 200
    log = True
    cumulative = False

    plt_fmt = {
        'tsit': 'g-',
        'era5': 'k--',
        'afno': 'r-',
    }

    f = plt.figure(figsize=(20, 10))

    xylims = [((0., 9.75), (0.00001, 0.5)),
              ((0., 4.), (0.075, 0.325)),
              ((4., 6.), (0.04, 0.15)),
              ((6., 9.75), (0.00001, 0.04))]

    for i, (xlim, ylim) in enumerate(xylims):
        plt.subplot(2, 2, i+1)
        plt.hist(tar.ravel(), bins=bins, log=log, cumulative=cumulative, histtype='step', density=True,
                 color='k', linestyle='--', label='era5')
        plt.hist(pred.ravel(), bins=bins, log=log, cumulative=cumulative, histtype='step', density=True,
                 color='g', linestyle='-', label='tsit')
        if afno is not None:
            plt.hist(afno.ravel(), bins=bins, log=log, cumulative=cumulative, histtype='step', density=True,
                     color='r', linestyle='-', label='afno')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(f'{unlog_tp(xlim[0]):.4f} to {unlog_tp(xlim[1]):.4f} m')
        plt.legend()
        if i in [1, 3]:
            plt.ylabel(f'{"log " if log else ""}density')
        if i in [3, 4]:
            plt.xlabel('log(1 + TP / 1e-5)')
    plt.suptitle(f'Precip {"C" if cumulative else "P"}DFs')
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
