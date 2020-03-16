"""Split the Urban 3D dataset into train/dev/test and splits each image to 256x256 sub-tiles.

Original images have size (2048, 2048).

We do not resize images, but split them into smaller images i.e. each input image divided into 64 256x256 images
"""

import argparse
import os
from pathlib import Path
import skimage.io
import skimage.external.tifffile as skitiff
import glob
import random
import shutil
import numpy as np
import src.image_tools as it

HOME_PATH   = str(Path.home())
DS_NAME     = "SpaceNet Urban3D"
SIZE        = 256
IMG_PER_DIM = 2048 // SIZE

DATA_ROOT = os.getenv("DATASET_ROOT", f"{HOME_PATH}/data")

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=f'{DATA_ROOT}/spacenet/urban3d', help=f"Directory with {DS_NAME} images, that has 01-Provisional_Train and other folders")
parser.add_argument('--output_dir', default=f'{DATA_ROOT}/spacenet-dataset/urban3d', help="Where we create dataset with of 256x256 images divided into train/dev/test")
parser.add_argument('--seed', default=-1, help="Random generator seed.")

def data_input_files(image_dirs, im_type = "rgb"):
    """
    Returns list of files of specified type
        im_type: should be RGB, or depth bufers DSM or DTM
    """
    im_type = im_type.upper()
    assert im_type in {"RGB", "DSM", "DTM", "GTI"}
    pattern = os.path.join(image_dirs, f"*_Tile_*_{im_type}.tif")
    yield from glob.glob(pattern)

def savetiff(name, data):
    fdir = os.path.dirname(name)
    if not os.path.isdir(fdir):
        os.makedirs(fdir, exist_ok = True)
    skitiff.imsave(name, data)

def build_dataset_image(src_image, out_folder, func_post_process = None):
    src_image_basename = os.path.basename(src_image)
    img = skimage.io.imread(src_image)
    for r in range(IMG_PER_DIM):
        for c in range(IMG_PER_DIM):
            row_col_file = src_image_basename.replace(".tif", f"_{r}_{c}.tif")
            row_col_file = os.path.join(out_folder, row_col_file)
            x1 = c * SIZE
            x2 = x1 + SIZE
            y1 = r * SIZE
            y2 = y1 + SIZE
            cropped = img[x1:x2, y1:y2]
            if not func_post_process is None:
                cropped = func_post_process(cropped)
            savetiff(row_col_file, cropped)

def normalize_depth_buffer(image):
    MISSING_VALUE = -32767
    mask_missing = (image == MISSING_VALUE)
    count_missing = np.sum(mask_missing)

    if count_missing == SIZE * SIZE:
        translated = image
        translated.fill(0.0)

    else:
        mask_has_value = (image != MISSING_VALUE)
        vmin = np.min(image[mask_has_value])
        vmax = np.max(image[mask_has_value])
        translated = it.translate(image, vmin, vmax, 0.11, 0.99)
        translated[mask_missing] = 0.0

    return translated

def build_dataset_input_image(rgb_image, out_folder):
    assert("_RGB." in rgb_image)
    build_dataset_image(rgb_image, out_folder)
    depth_image = rgb_image.replace("_RGB.", "_DSM.")
    build_dataset_image(depth_image, out_folder, normalize_depth_buffer)
    terrain_image = rgb_image.replace("_RGB.", "_DTM.")
    build_dataset_image(terrain_image, out_folder, normalize_depth_buffer)

def normalize_instance_indexes(img_instances):
    instance_ids, i = np.unique(img_instances), 0
    instance_ids = sorted(instance_ids)
    for iid in instance_ids:
        if iid != i:
            img_instances[img_instances == iid] = i
        i += 1
    return img_instances

def get_file_index(img_filename):
    img_filename = os.path.basename(img_filename)
    parts = img_filename.split("_")
    iid = parts[2]
    assert(len(iid) == 3 and iid.isdigit())
    return parts[0], iid


def build_dataset_images(src_images, dest_dir, name = "train set"):
    print(f"\nBuilding {name} files. Output folder {dest_dir}")
    i = 0
    for f in src_images:
        # if this is RGB image, then we also preprocess DSM
        if "_RGB.tif" in f:
            build_dataset_input_image(f, dest_dir)
        else:
            build_dataset_image(f, dest_dir, normalize_instance_indexes)
        i += 1
        if i % 2 == 0:
            print(".", end = "")
    print()
    print(f"\n{len(src_images)} files processed. {len(src_images) * IMG_PER_DIM**2} files created.")


def build_dataset_label_images(input_files, src_folder, output_folder, title):
    ids = [get_file_index(f) for f in input_files]
    src_files = [os.path.join(src_folder, f"{t}_Tile_{iid}_GTI.tif") for t, iid in ids]
    build_dataset_images(src_files, output_folder, title)

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the oritinal spacent urgan3d at {}. Please download as defined in doc/datasets.md".format(args.data_dir)

    # init random number generator
    if args.seed != -1:
        random.seed(args.seed)

    # cleanup output folder
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    # TRAIN
    # writing training files (X)
    pattern = os.path.join(args.data_dir, "01-Provisional_Train", "Inputs", "*_RGB.tif")
    input_files = glob.glob(pattern)
    build_dataset_images(input_files, os.path.join(args.output_dir, "train", "inputs"), "train set")
    # writing training files (Y labels)
    src = os.path.join(args.data_dir, "01-Provisional_Train", "GT")
    dest = os.path.join(args.output_dir, "train", "target")
    build_dataset_label_images(input_files, src, dest, "train target set")
    train_files, src, dest = None, None, None

    # test files (we will split them into 2 equal dev and test parts randomly)
    input_test_dir = os.path.join(args.data_dir, "02-Provisional_Test", "Inputs")
    devtest_files  = list(data_input_files(input_test_dir))
    random.shuffle(devtest_files)
    mid = len(devtest_files) // 2
    dev_files, test_files = devtest_files[:mid], devtest_files[mid:]

    # DEV
    # writing dev set files (X)
    dev_dir = os.path.join(args.output_dir, "dev", "inputs")
    build_dataset_images(dev_files, dev_dir, "dev set")
    # writing dev files (Y)
    src = os.path.join(args.data_dir, "02-Provisional_Test", "GT")
    dest = os.path.join(args.output_dir, "dev", "target")
    build_dataset_label_images(dev_files, src, dest, "dev target set")
    dev_files, dev_dir, src, dest = None, None, None, None

    # TEST
    # writing test set files (X)
    test_dir = os.path.join(args.output_dir, "test", "inputs")
    build_dataset_images(test_files, test_dir, "test set")
    # writing test files (Y)
    src = os.path.join(args.data_dir, "02-Provisional_Test", "GT")
    dest = os.path.join(args.output_dir, "test", "target")
    build_dataset_label_images(test_files, src, dest, "test target set")

