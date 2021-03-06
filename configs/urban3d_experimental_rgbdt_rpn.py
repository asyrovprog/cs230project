from mrcnn.config import Config
from .constants import *

class Urban3dExperimentalRGBDTRPN(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "urban3d_experimental_rgbdt_rpn"

    BACKBONE = 'resnet50'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = GLOB_IMAGES_PER_GPU

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + building footprints

    BATCH_SIZE = GLOB_BATCH_SIZE

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 128

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = GLOB_DETECTION_MIN_CONFIDENCE

    WORKING_SIZE = GLOB_IMAGE_SIZE
    IMAGE_MIN_DIM = WORKING_SIZE
    IMAGE_MAX_DIM = WORKING_SIZE

    TRAIN_ROIS_PER_IMAGE = 100
    MAX_GT_INSTANCES = 50

    # Random crops of size IMAGE_MAX_DIM
    IMAGE_RESIZE_MODE = "crop"

    LEARNING_RATE = 0.001
    LEARNING_RATES = [LEARNING_RATE, LEARNING_RATE / 10]
    LEARNING_LAYERS = ["heads", "all"]
    LEARNING_EPOCHS = [5, 150]

    IMAGE_TYPE = "RGBDT"
    IMAGE_CHANNEL_COUNT = 5
    MEAN_PIXEL = 5
    TRAIN_CONV1 = True

    EXT_USE_AUGMENTATION = GLOB_USE_AUGMENTATION

    #RPN experiments:
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256 * 16
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]


class Urban3dExperimentalRGBDTRPNInference(Urban3dExperimentalRGBDTRPN):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1