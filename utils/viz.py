import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


def viz_fields(flist):
    pred, tar, afno = flist
    pred = pred[0]
    tar = tar[0]
    afno = afno[0]
    sc = tar.max()

    rows = 3
    if afno is not None:
        rows = 5

    f = plt.figure(figsize=(18,12*rows))

    plt.subplot(rows,1,1)
    plt.imshow(tar, cmap='Blues', norm=Normalize(0., sc))
    plt.title('Truth')

    plt.subplot(rows,1,2)
    plt.imshow(pred, cmap='Blues', norm=Normalize(0., sc))
    plt.title('TSIT')

    plt.subplot(rows,1,3)
    err = (pred - tar) / (tar + 1)
    plt.imshow(err, norm=TwoSlopeNorm(0., err.min(), err.max()), cmap='bwr')
    plt.title('TSIT relative error')

    if afno is not None:
        plt.subplot(rows,1,4)
        plt.imshow(afno, cmap='Blues', norm=Normalize(0., sc))
        plt.title('AFNO')

        plt.subplot(rows,1,5)
        plt.imshow((afno - tar) / (tar + 1),
                   norm=TwoSlopeNorm(0., err.min(), err.max()), cmap='bwr')
        plt.title('AFNO relative error')

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
    plt.title(f'Power spectra +/- std (n = {n})')

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
    plt.title(f'Power spectra +/- std err (n = {n})')

    return f
