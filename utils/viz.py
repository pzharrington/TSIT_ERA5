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
    f = plt.figure(figsize=(15,9))
    pred, tar = np.log1p(np.abs(pred_fft)), np.log1p(np.abs(tar_fft))
    # pred, tar = np.abs(pred_fft), np.abs(tar_fft)
    cmap = 'Reds'
    sc = tar.max()
    plt.subplot(2,3,1)
    plt.imshow(pred, cmap=cmap, norm=Normalize(0., sc))
    plt.title('Generated log1p(amplitude)')
    plt.subplot(2,3,2)
    plt.imshow(tar, cmap=cmap)
    plt.title('Truth log1p(amplitude)', norm=Normalize(0., sc))
    plt.subplot(2,3,3)
    # TODO: errors in unlogged space
    plt.imshow(pred - tar, cmap='bwr')
    plt.title('Error')
    
    pred, tar = np.angle(pred_fft), np.angle(tar_fft)
    sc_min, sc_max = tar.min(), tar.max()
    plt.subplot(2,3,4)
    plt.imshow(pred, cmap='bwr', norm=Normalize(sc_min, sc_max))
    plt.title('Generated phase')
    plt.subplot(2,3,5)
    plt.imshow(tar, cmap='bwr', norm=Normalize(sc_min, sc_max))
    plt.title('Truth phase')
    plt.subplot(2,3,6)
    plt.imshow(pred - tar, cmap='bwr')
    plt.title('Error')

    plt.tight_layout()
    return f
