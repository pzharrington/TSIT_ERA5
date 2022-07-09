import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def viz_fields(flist):
    pred, tar, inp, afno = flist
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
    plt.imshow((pred - tar) / (tar + 1), cmap='bwr')
    plt.title('TSIT relative error')

    plt.tight_layout()
    return f


def viz_spectra(spectra):
    pred_fft, tar_fft, afno_fft = spectra
    pred, tar, afno = np.abs(pred_fft)[0], np.abs(tar_fft)[0], None

    plt_params = {
        'tsit': [pred, 'g-'],
        'era5': [tar, 'k--'],
    }

    if afno_fft is not None:
        afno = np.abs(afno_fft)[0]
        plt_params['afno'] = [afno, 'r-']

    wavenum = np.arange(start=0, stop=pred_fft.shape[-1])

    f = plt.figure(figsize=(10, 5))

    for label, (amp, fmt) in plt_params.items():
        plt.semilogy(wavenum, np.mean(amp, axis=-2), fmt, label=label)

    plt.xlabel('wave number')
    plt.ylabel('amplitude')
    plt.legend()

    return f
