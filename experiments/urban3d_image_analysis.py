from src.mrcnn_imports import *
from configs.config_factory import *
from src.evaluation import *
import src.dataset as ds
import src.dataset_urban3d as dsu
import argparse
import shutil

MAX_DISPLAY_COUNT = 30

def compute_image_metrics(model, dataset, config, image_id):
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask = False)

    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]

    # Compute AP
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])

    return AP, precisions, recalls, overlaps

def create_images(model_left, config_left, model_right, config_right, output_folder, ds_type, img_count):

    dataset_left = dsu.create_urban3d_model(ds_type, config_left.IMAGE_TYPE)
    dataset_right = dsu.create_urban3d_model(ds_type, config_right.IMAGE_TYPE)

    results = []

    image_ids = dataset_left.image_ids
    for i, image_id in enumerate(image_ids):

        AP_left, precisions_left, recalls_left, overlaps_left = compute_image_metrics(model_left, dataset_left, config_left, image_id)

        image_file = dataset_left.image_info[image_id]["path"]
        image_id_right = dataset_right.get_image_id(image_file)

        if image_id_right == -1:
            continue

        AP_right, precisions_right, recalls_right, overlaps_right = compute_image_metrics(model_right, dataset_right, config_right, image_id_right)

        results.append((AP_left - AP_right, image_file))
        print(".", end = "")

    print()
    print("*******************************************")
    print(f"Top {MAX_DISPLAY_COUNT} where {config_left.NAME} is better than {config_right.NAME}")
    print("*******************************************")
    results = sorted(results, reverse = True)
    for i in range(MAX_DISPLAY_COUNT):
        print(f"Diff: {abs(results[i][0])}")
        print(f"{results[i][1]}\n")

    print("*******************************************")
    print(f"Top {MAX_DISPLAY_COUNT} where {config_right.NAME} is better than {config_left.NAME}")
    print("*******************************************")
    results = sorted(results)
    for i in range(MAX_DISPLAY_COUNT):
        print(f"Diff: {abs(results[i][0])}")
        print(f"{results[i][1]}\n")


def load_model(model_name):
    inference_config = create_config(model_name, True)

    model_dir = os.path.join(MODEL_DIR, inference_config.NAME)
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=model_dir)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name = True)

    return inference_config, model


if __name__ == "__main__":
    # this script must run from project root folder as "python -m experiment.urban3d_training.py"
    assert(os.path.isfile("LICENSE"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_left', default = 'optimal_d',
                        help = f"First model to compare")

    parser.add_argument('--model_right', default='optimal_rgb',
                        help=f"Second model to compare")

    parser.add_argument('--dir_out', default = "logs/image_analysis", help = f"Output folder for images")

    parser.add_argument('--dataset', default = "dev", help = f"Dataset type (dev/test)")

    parser.add_argument('--image_count', default = 10, help = f"Number of images to process. -1 to process all images")

    args = parser.parse_args()

    # recreate output folder for images
    shutil.rmtree(args.dir_out, ignore_errors = True)
    os.makedirs(args.dir_out, exist_ok = True)

    config_left, model_left = load_model(args.model_left)
    config_right, model_right = load_model(args.model_right)

    create_images(model_left, config_left, model_right, config_right, args.dir_out, args.dataset, args.image_count)