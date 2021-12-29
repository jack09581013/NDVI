import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import torch
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, LinearStretch
import mpl_scatter_density

def version_code(file):
    a = file.index('-')
    b = file.index('.')
    return int(file[a + 1:b])


def get_latest_version(file_path):
    version_codes = [version_code(x) for x in os.listdir(file_path)]
    if len(version_codes) > 1:
        version_codes.sort()
        return version_codes[-1]
    else:
        return None


def timespan_str(timespan):
    total = timespan.seconds
    second = total % 60 + timespan.microseconds / 1e+06
    total //= 60
    minute = int(total % 60)
    total //= 60
    return f'{minute:02d}:{second:05.2f}'


def rgb2bgr(x):
    y = np.zeros(x.shape, dtype=np.uint8)
    y[..., 0], y[..., 1], y[..., 2] = x[..., 2], x[..., 1], x[..., 0]
    return y


def plot_image_disparity(X: torch.Tensor, Y: torch.Tensor, predict: torch.Tensor, mse_loss: float, save_file=None):
    assert X.dim() == 3
    assert Y.dim() == 2

    X = (X * 255).data.cpu().numpy().astype('uint8')
    X = X.swapaxes(1, 2).swapaxes(0, 2)
    Y = Y.data.cpu().numpy()
    predict = predict.data.cpu().numpy()

    error_map = np.abs(predict - Y)

    fig = plt.figure(figsize=(16, 10))

    plt.subplot(231)
    plt.title('Input Image')
    plt.imshow(X[:, :, 0:3])

    plt.subplot(232)
    plt.title('Ground True')
    plt.imshow(Y, vmin=-1, vmax=1, cmap='jet')

    plt.subplot(233)
    plt.title(f'Predict, MSE = {mse_loss:.3f}')
    plt.imshow(predict, vmin=-1, vmax=1, cmap='jet')

    plt.subplot(234)
    plt.title('Error Map')
    plt.imshow(error_map, vmin=0, vmax=2, cmap='jet')

    # plt.subplot(235)
    # plt.title('Fit line')
    # plt.scatter(predict.reshape(-1), Y.reshape(-1), color='k', marker='.')
    # plt.xlabel('Predict')
    # plt.ylabel('NDVI')

    norm = ImageNormalize(vmin=0, vmax=100, stretch=LogStretch())
    distribution = fig.add_subplot(235, projection='scatter_density')
    distribution.scatter_density(predict.reshape(-1), Y.reshape(-1), cmap=plt.cm.Blues, norm=norm)
    distribution.plot([-1, 1], [-1, 1], 'r')
    distribution.set_xlim(-1, 1)
    distribution.set_ylim(-1, 1)
    distribution.set_xlabel("Predict")
    distribution.set_ylabel("NDVI")

    if save_file is None:
        plt.show()
        plt.close(fig)
    else:
        plt.savefig(save_file, box_inches='tight')
        plt.close(fig)
