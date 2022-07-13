import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def viz_fields(flist):
    pred, tar, afno = flist
    pred = pred[0]
    tar = tar[0]
    afno = afno[0]
    sc = tar.max()
    f = plt.figure(figsize=(18,12))

    plt.subplot(2,2,1)
    plt.imshow(tar, cmap='Blues', norm=Normalize(0., sc))
    plt.title('Truth')

    if afno is not None:
        plt.subplot(2,2,2)
        plt.imshow(afno, cmap='Blues', norm=Normalize(0., sc))
        plt.title('AFNO')

    plt.subplot(2,2,3)
    plt.imshow(pred, cmap='Blues', norm=Normalize(0., sc))
    plt.title('TSIT')

    plt.subplot(2,2,4)
    plt.imshow((pred - tar) / np.abs(tar + 1), cmap='bwr')
    plt.title('TSIT relative error')

    plt.tight_layout()
    return f


def viz_spectra(spectra):
    spec_mean, spec_stderr = spectra

    plt_fmt = {
        'tsit': 'g-',
        'era5': 'k--',
        'afno': 'r-',
    }

    f = plt.figure(figsize=(10, 5))

    for key in spec_mean.keys():
        mean = spec_mean[key]
        stderr = spec_stderr[key]
        wavenum = np.arange(start=0, stop=mean.shape[-1])
        plt.semilogy(wavenum, mean, plt_fmt[key], label=key)
        plt.fill_between(wavenum, mean - stderr, mean + stderr,
                         color=plt_fmt[key][0], alpha=0.2)

    plt.xlabel('wave number')
    plt.ylabel('amplitude')
    plt.legend()

    return f
