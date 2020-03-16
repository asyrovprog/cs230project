from skimage import io
import matplotlib.pyplot as plt
import argparse
import os, glob
from pathlib import Path
import skimage.io
import skimage.external.tifffile as skitiff
import glob
import random
import shutil, matplotlib
import numpy as np

BINS_RGB = 256
BINS_DEPTH = 1024
MAX_FILE  = 10
HOME_PATH = str(Path.home())
DATA_ROOT = os.getenv("DATASET_ROOT", f"{HOME_PATH}/data")

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default = f'{DATA_ROOT}/spacenet-dataset/urban3d/train/inputs', help=f"Image Directory")
# parser.add_argument('--img_dir', default = f'{DATA_ROOT}/spacenet/urban3d/01-Provisional_Train/Inputs', help=f"Image Directory")
parser.add_argument('--type', default = "rgb", help=f"Image info type")

def image_info(image):
    print("Min", np.min(image))
    print("Max", np.max(image))
    print("Mean", np.average(image))
    print("Variance", np.var(image))
    print("Number Unique", np.unique(image).size)



def rgb_hist(image):
    image = io.imread(image)
    _ = plt.hist(image.ravel(), bins=BINS_RGB, color='orange', )
    _ = plt.hist(image[:, :, 0].ravel(), bins=BINS_RGB, color='red', alpha=0.5)
    _ = plt.hist(image[:, :, 1].ravel(), bins=BINS_RGB, color='Green', alpha=0.5)
    _ = plt.hist(image[:, :, 2].ravel(), bins=BINS_RGB, color='Blue', alpha=0.5)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['sum', 'red', 'green', 'blue'])
    plt.show()
    image_info(image)

def dsm_hist(image):
    image = io.imread(image)
    ax = plt.hist(image.ravel(), bins = BINS_DEPTH)
    plt.show()
    image_info(image)

if __name__ == "__main__":
    args = parser.parse_args()
    pattern = '*DSM*.tif' if not args.type == "rgb" else '*RGB*.tif'
    fullpath = os.path.join(args.img_dir, pattern)

    all_files = glob.glob(fullpath)
    for i in range(0, len(all_files)):
        if args.type == "rgb":
            rgb_hist(all_files[i])
        else:
            dsm_hist(all_files[i])

        if i > MAX_FILE:
            break


