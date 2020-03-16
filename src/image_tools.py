import numpy as np
import skimage, skimage.color
import matplotlib.pyplot as plt
import os

def translate(value, leftMin, leftMax, rightMin, rightMax):
    if leftMin == leftMax:
        value.fill(0.0)
        return value
    else:
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        valueScaled = (value - leftMin) / leftSpan
        return rightMin + (valueScaled * rightSpan)

def map_image_to_rgb(image):
    if image.ndim == 3 and image.shape[2] == 1: # depth buffer
        image = translate(image, np.min(image), np.max(image), 0.0, 255.0)
        image.shape = (image.shape[0], image.shape[1])
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    if image.shape[-1] == 4 or image.shape[-1] == 5:
        image = image[..., :3]
    return image

def image2rgb(img, imin = -1.0, imax = 1.0):
    img = translate(img, imin, imax, 0.0, 255.0)
    img = np.clip(img.round(), 0, 255).astype(np.uint8)
    channels = img.shape[2]
    if channels == 1:
        img = np.concatenate((img, img, img), axis = 2)
    if channels == 4:
        img = img[:, :, :3]
    return img

def save_image(path, name, img, convert = True):
    _, ax = plt.subplots(1, figsize = (16, 16))
    os.makedirs(path, exist_ok = True)
    if convert:
        img = img.astype(np.uint32).copy()
        img = img.copy().astype(np.uint8)
    height, width = img.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.imshow(img)
    os.makedirs(path, exist_ok = True)
    plt.savefig(os.path.join(path, name))