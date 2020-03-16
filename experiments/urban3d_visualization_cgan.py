from src.cgan_urban3d import *
import argparse
import numpy as np
import experiments.urban3d_training_cgan as gt
from src.cgan_metrics import *

if __name__ == "__main__":

    # this script must run from project root folder as "python -m experiments.urban3d_validation_cgan.py"
    assert(os.path.isfile("LICENSE"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dev', help = f"Type of dataset (normally 'dev' or 'test')")
    parser.add_argument('--itype', default='rgbd', help=f"Image type (rgb, rgbd, rgbdt)")
    args = parser.parse_args()

    config = gt.init_config(args.itype, 1, None, args.dataset)
    # final models stored (copied manually) in 'models' folder instead of 'logs' folder
    config.MODEL_FOLDER = os.path.join("models", config.NAME, "models")
    config.IMAGE_FOLDER = os.path.join("models", config.NAME, "images")

    cgan = CondImageGAN(config)

    # load trained models
    cgan.load_models(weights_only = False)

    # save 10 sample predictions into 'models/<NAME>/images
    cgan.sample_images(10)

    print(f"\nResults images are saved in: \n{config.IMAGE_FOLDER}")
