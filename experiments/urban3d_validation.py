"""
    Evaluate RGB Urban3d dataset. This program should be called from root folder of the
    project (cs230project) as follows:

        python -m experiments.urban3d_validation --config "easy_debug"

    List of configs located at configs/config_factory.py
"""

from src.mrcnn_imports import *
from configs.config_factory import *
from src.evaluation import *
import argparse
CALCULATE_mAP = False

def evaluate(inference_config, dataset_val):
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

    # evaluate several random samples from dev set
    random_detect(dataset_val, inference_config, model)

    # gather results
    precision, recall, f1, iou = compute_f1(model, inference_config, dataset_val, print_each = 20)

    if CALCULATE_mAP:
        mAP = compute_mAP(dataset_val, inference_config, model)
    else:
        mAP = 0

    print("-----------------------------------------------------")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}, mAP {mAP:.4f}")

if __name__ == "__main__":
    # this script must run from project root folder as "python -m experiment.urban3d_training.py"
    assert(os.path.isfile("LICENSE"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='easy_rgbd',help = f"Name of config. For complete list see configs/config_factory.py")
    parser.add_argument('--dataset', default='dev',help = f"Dataset type, valid values are train, dev and test")
    args = parser.parse_args()

    inference_config = create_config(args.config, inference=True)
    dataset_val = ds.create_urban3d_model(args.dataset, inference_config.IMAGE_TYPE)

    evaluate(inference_config, dataset_val)