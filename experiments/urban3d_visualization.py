"""
    Visualize results of model run. This program should be called from root folder of the
    project (cs230project) as follows:

        python -m experiments.urban3d_visualization --config "easy_debug"

    List of configs located at configs/config_factory.py

    You also need to install Keras-vis if it's not installed already:
        pip install git+https://github.com/raghakot/keras-vis.git -U
"""

from src.mrcnn_imports import *
from configs.config_factory import *
from src.visualization_occlusions import *
from src.visualization_saliency import *
import argparse

def visualize(inference_config, dataset_val):
    # Recreate the model in inference mode
    model_dir = os.path.join(MODEL_DIR, inference_config.NAME)
    model = modellib.MaskRCNN(mode="inference", config = inference_config, model_dir = model_dir)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name = True)

    # visualize performance of the model using occlusions
    visualize_occlusions(dataset_val, inference_config, model)

    # visualize effects of pixels to layers activations with saliency maps
    visualize_saliency_maps(dataset_val, inference_config, model)

if __name__ == "__main__":
    # this script must run from project root folder as "python -m experiment.urban3d_vizualization.py"
    assert(os.path.isfile("LICENSE"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='easy_rgbd',help = f"Name of config. For complete list see configs/config_factory.py")
    parser.add_argument('--dataset', default='dev',help = f"Dataset type, valid values are train, dev and test")
    args = parser.parse_args()

    inference_config = create_config(args.config, inference=True)
    dataset_val = ds.create_urban3d_model(args.dataset, inference_config.IMAGE_TYPE)

    visualize(inference_config, dataset_val)