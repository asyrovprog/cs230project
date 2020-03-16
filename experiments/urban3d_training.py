"""
    Train with RGB(D,T) Urban3d dataset. This program should be called from root folder of the
    project (cs230project) as follows:

        python -m experiments.urban3d_training --config "easy_debug"

    List of configs located at configs/config_factory.py
"""

from src.mrcnn_imports import *
from configs.config_factory import *
from src.training import *
import argparse

def urban3d_train(config, dataset_train, dataset_val, init_with = "coco"):
    config.display()

    # Create model in training mode
    model_dir = os.path.join(MODEL_DIR, config.NAME)
    os.makedirs(model_dir, exist_ok = True)
    model = modellib.MaskRCNN(mode="training", config = config, model_dir = model_dir)

    print(f"INFO: using config {config.NAME}")

    exclude = []
    # if this is not RGB we must exclude "conv1" layer, because it has different shape
    if config.IMAGE_CHANNEL_COUNT != 3 or config.TRAIN_CONV1:
        print("INFO: Excluding conv1.")
        exclude = ["conv1"]

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name = True, exclude = exclude)

    elif init_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name = True,
                           exclude = exclude + ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    elif init_with == "last":
        model.load_weights(model.find_last(), by_name = True)

    # Training
    multi_train(config, dataset_train, dataset_val, model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='easy_rgbd', help = f"Name of config. For complete list see configs/config_factory.py")
    parser.add_argument('--init', default='coco', help = f"Initialization mode (imagenet, coco or last)")
    args = parser.parse_args()

    # this script must run from project root folder as "python -m experiment.urban3d_training.py"
    assert(os.path.isfile("LICENSE"))

    # config
    config = create_config(args.config)

    # Training dataset
    dataset_train = ds.create_urban3d_model("train", config.IMAGE_TYPE)

    # Validation dataset
    dataset_val = ds.create_urban3d_model("dev", config.IMAGE_TYPE)

    urban3d_train(config, dataset_train, dataset_val, args.init)
