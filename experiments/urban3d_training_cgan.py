"""
    Conditional Adversarial Network, training with RGB(D,T) Urban3d dataset.
    This script must run from project root folder as "python -m experiments.urban3d_training_cgan.py"
"""

from src.cgan_urban3d import *
import argparse

def init_config(itype, epochs, train = "train", eval = "dev"):
    """
    Initialize training configuration or evaluation. 'train' parameter may be None if intent is to load existing
    model weights.
    """
    assert(itype in {"rgb", "rgbd", "rgbdt"})

    config = Urban3dCondGANConfig(train, eval, itype)

    config.NAME = f"urban3d_cond_gan_{itype}"
    if itype == "rgb":
        config.COND_SHAPE = (256, 256, 3)
    elif itype == "rgbd":
        config.COND_SHAPE = (256, 256, 4)
    else: # rgbdt
        config.COND_SHAPE = (256, 256, 5)

    config.IMAGE_FOLDER = os.path.join("logs", config.NAME, "images")
    config.MODEL_FOLDER = os.path.join("logs", config.NAME, "models")
    config.EPOCHS = epochs

    return config

if __name__ == "__main__":

    # this script must run from project root folder as "python -m experiments.urban3d_training_cgan.py"
    assert (os.path.isfile("LICENSE"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--itype', default = 'rgbd', help=f"Image type (rgb, rgbd, rgbdt)")
    parser.add_argument('--epochs', default = 2, help=f"Number of epochs to train")
    args = parser.parse_args()

    assert(args.itype in {"rgb", "rgbd", "rgbdt"})
    config = init_config(args.itype, int(args.epochs))

    cgan = CondImageGAN(config)
    cgan.train()
