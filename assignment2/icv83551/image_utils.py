# from future import standard_library

# standard_library.install_aliases()
import io
import os
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from builtins import range

import numpy as np
from imageio import imread  # Assuming you're using imageio or similar for imread
from PIL import Image

"""
Utility functions used for viewing and processing images.
"""


def blur_image(X):
    """
    A very gentle image blurring operation, to be used as a regularizer for
    image generation.

    Inputs:
    - X: Image data of shape (N, 3, H, W)

    Returns:
    - X_blur: Blurred version of X, of shape (N, 3, H, W)
    """
    from .fast_layers import conv_forward_fast

    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {"stride": 1, "pad": 1}
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
    w_blur /= 200.0
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img):
    """Preprocess an image for squeezenet.

    Subtracts the pixel mean and divides by the standard deviation.
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """Undo preprocessing on an image and convert back to uint8."""
    img = img * SQUEEZENET_STD + SQUEEZENET_MEAN
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url_old(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        f = urllib.request.urlopen(req)
        _, fname = tempfile.mkstemp()
        with open(fname, "wb") as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print("URL Error: ", e.reason, url)
    except urllib.error.HTTPError as e:
        print("HTTP Error: ", e.code, url)


def image_from_url(url):
    """
    Read an image from a URL directly into memory.
    Uses an unverified SSL context to bypass certificate errors.
    """
    # Create a context that doesn't check certificates
    context = ssl._create_unverified_context()

    try:
        req = urllib.request.Request(str(url), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=context) as response:
            # Read the data into a BytesIO buffer instead of a temp file
            img_data = io.BytesIO(response.read())
            return imread(img_data)

    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None


def load_image(filename, size=None):
    """Load and resize an image from disk.

    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        # TODO: the width, height values are currently flipped here, and we should
        # change the resampling method to BILINEAR to match the torch implementation
        img = np.array(Image.fromarray(img).resize(new_shape, resample=Image.NEAREST))
    return img
