from src.mrcnn_imports import *
from configs.config_factory import *
from src.evaluation import *
from src.point_cloud import *
import src.dataset as ds
import src.dataset_urban3d as dsu
import argparse
import shutil
from pathlib import Path
import numpy as np

MAX_DISPLAY_COUNT = 30

def save_image(path, name, img, convert = True):
    _, ax = plt.subplots(1, figsize=(16, 16))
    os.makedirs(path, exist_ok = True)
    if convert:
        img = img.astype(np.uint32).copy()
        img = img.copy().astype(np.uint8)
    height, width = img.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.imshow(img)
    plt.savefig(os.path.join(path, name))

def display_masked(image, gt_bbox, gt_mask, gt_class_id, scores, dataset, dest_folder, config, prefix):
    _, ax = plt.subplots(1, figsize=(16, 16))
    visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset.class_names, scores = scores, ax = ax)
    plt.savefig(os.path.join(dest_folder, f"{prefix}_{config.IMAGE_TYPE}".lower()))

def masks_to_ids(mask, classes):
    ret = np.ndarray(shape = mask.shape[:2], dtype = np.int)
    ret.fill(0)

    for i in range(0, mask.shape[2]):
        if classes[i] != 0:
            ret[mask[:,:, i]] = i + 1

    return ret

def create_ground_truth_image(model, dataset, config, image_id, folder):
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask = False)
    oimage = image.copy()

    if config.IMAGE_TYPE == "D":
        image = dataset.load_image_d2g(image_id)

    elif config.IMAGE_TYPE == "RGBD":
        image = dataset.load_image_rgb(image_id)

    filename = Path(dataset.image_info[image_id]["path"]).stem
    dest_folder = os.path.join(folder, filename)

    # display original image as is
    save_image(dest_folder, f"original_{config.IMAGE_TYPE}".lower(), image)

    # display original image with ground truth mask
    display_masked(image, gt_bbox, gt_mask, gt_class_id, None, dataset, dest_folder, config, "ground_truth")

    # display original image with predictions
    r = model.detect([oimage], verbose=0)[0]
    display_masked(image, r['rois'], r['masks'], r['class_ids'], r['scores'], dataset, dest_folder, config, "prediction")

    depth_img = dataset.load_image_depth(image_id) / 255.0
    depth_img.shape = depth_img.shape[:2]

    create_pointcloud_ext(dataset.load_image_rgb(image_id),
                          depth_img,
                          [masks_to_ids(gt_mask, gt_class_id), masks_to_ids(r['masks'], r['class_ids'])],
                          [(255, 0, 0), (0, 255, 0)],
                          os.path.join(dest_folder, f"3d_{config.IMAGE_TYPE}.ply".lower()))

def create_image(model, config, dir_out, ds_type, image_name):

    dataset = dsu.create_urban3d_model(ds_type, config.IMAGE_TYPE)
    image_id = dataset.get_image_id(image_name)

    assert(image_id != -1)
    create_ground_truth_image(model, dataset, config, image_id, dir_out)


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
    # this script must run from project root folder
    assert(os.path.isfile("LICENSE"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', default = 'optimal', help = f"Model config prefix (such as 'optimal')")
    parser.add_argument('--dir_out', default = "analysis", help = f"Output folder for images")
    parser.add_argument('--dataset', default = "dev", help = f"Dataset type (dev/test)")
    parser.add_argument('--image_name', help = f"Image name to compare")

    args = parser.parse_args()
    dir_out = os.path.join(args.dir_out, "image_compare")

    # recreate output folder for images
    os.makedirs(dir_out, exist_ok = True)

    for mod in ["d", "rgb", "rgbd"]:
        config, model = load_model(args.model_prefix + "_" + mod)
        create_image(model, config, dir_out, args.dataset, args.image_name)