import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def viz_fields(flist):
    pred, tar = flist
    pred = pred[0]
    tar = tar[0]
    sc = tar.max()
    f = plt.figure(figsize=(12,6))    
    plt.subplot(1,2,1)
    plt.imshow(pred, cmap='Blues', norm=Normalize(0., sc))
    plt.title('Generated')
    plt.subplot(1,2,2)
    plt.imshow(tar, cmap='Blues', norm=Normalize(0., sc))
    plt.title('Truth')
        
    plt.tight_layout()
    return f
