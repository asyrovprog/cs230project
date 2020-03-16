import warnings, os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
import sys
import time
import numpy as np
import skimage.io
import skimage.external.tifffile as skitiff
import matplotlib.pyplot as plt
import imgaug
import glob
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
import urllib.request
import shutil
from pathlib import Path
import src.dataset_urban3d as ds
from src.metrics import *

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
DATA_ROOT = os.getenv("DATASET_ROOT", os.path.join(str(Path.home()), "data"))

# path to coco pre-trained model
COCO_MODEL_PATH = os.path.join(DATA_ROOT, "models", "mask_rcnn_coco.h5")
# Path to trained weights file
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# fixing: breaking change between versions of numpy
try:
    np.random.bit_generator = np.random._bit_generator
except:
    pass

# fixing: file "/home/ubuntu/anaconda3/envs/cs230/lib/python3.7/site-packages/keras/layers/convolutional.py", line 132,
# ValueError: The channel dimension of the inputs should be defined. Found `None`.
K = keras.backend.backend()
if K == 'tensorflow':
    keras.backend.set_image_dim_ordering('tf')