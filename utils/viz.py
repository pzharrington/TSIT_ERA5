import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from utils.weighted_acc_rmse import unlog_tp


def viz_fields(flist):
    pred, tar, _ = flist
    pred = pred[0]
    tar = tar[0]
    sc = tar.max()
    f = plt.figure(figsize=(18,6))
    plt.subplot(1,3,1)
    plt.imshow(pred, cmap='Blues', norm=Normalize(0., sc))
    plt.title('Generated')
    plt.subplot(1,3,2)
    plt.imshow(tar, cmap='Blues', norm=Normalize(0., sc))
    plt.title('Truth')
    plt.subplot(1,3,3)
    plt.imshow(pred - tar, cmap='bwr')
    plt.title('Error')

    plt.tight_layout()
    return f


def viz_spectra(spectra):
    pred_fft, tar_fft = spectra

    pred, tar = np.abs(pred_fft)[0], np.abs(tar_fft)[0]

    wavenum = np.arange(start=0, stop=pred_fft.shape[-1])

    plt_params = {
        # 'afno': [afno, 'r-'],
        'tsit': [pred, 'g-'],
        'era5': [tar, 'k--'],
    }

    f = plt.figure(figsize=(10, 5))

    for label, (amp, fmt) in plt_params.items():
        plt.semilogy(wavenum, np.mean(amp, axis=-2), fmt, label=label)

    plt.xlabel('wave number')
    plt.ylabel('amplitude')
    plt.legend()

    return f
