import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# generates an image of a Gaussian distribution
def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


if __name__ == '__main__':
    img = make_gaussian(256, 70)
    print(img)
    # plot the image
    plt.imshow(img, cmap='gray')
    plt.show()

    # save the array as a grayscale image
    Image.fromarray(img * 255).convert('L').save('../data/gaussian_reference.png')
