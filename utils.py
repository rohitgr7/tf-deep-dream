import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path, factor=None):
    image = Image.open(image_path)
    image = np.float32(image)
    if factor is not None:
        image = resize_image(image, factor=factor)

    return image


def save_image(image_path, image):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)

    image = Image.fromarray(image)
    with open(image_path, 'wb') as f:
        image.save(f, 'jpeg')


def resize_image(image, shape=None, factor=None):
    # Get the height and width
    new_size = np.array(image.shape[:2])

    # Check for conditions
    if factor is not None:
        new_size = new_size * factor
    elif shape is not None:
        new_size = shape[:2]

    # Make new size to (W, H)
    new_size = np.int32(new_size[::-1])

    # Preprocess the image
    img = np.clip(image, 0.0, 255.0)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    new_image = img.resize(new_size, Image.LANCZOS)
    new_image = np.float32(new_image)

    return new_image


def plot_image(image):
    image = np.clip(image / 255., 0., 1.)
    plt.imshow(image, interpolation='lanczos')
    plt.show()


def _normalize_grads(grads):
    # Get min and max values
    g_min = grads.min()
    g_max = grads.max()

    # Normalize and return
    return (grads - g_min) / (g_max - g_min)


def plot_gradients(gradients):
    # Normalize to get the values b/w 0 and 1
    grads = _normalize_grads(gradients)

    plt.imshow(grads, interpolation='spline36')
    plt.show()
