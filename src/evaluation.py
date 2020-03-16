import numpy as np
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import log
import matplotlib.pyplot as plt
import skimage, skimage.color
from .image_tools import *

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def random_detect(dataset_val, inference_config, model, count: int = 5):
    for i in range(0, count):
        # Test on a random image
        image_id = np.random.choice(dataset_val.image_ids)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask = False)

        log("original_image", original_image)
        log("image_meta", image_meta)
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)

        image = dataset_val.load_image(image_id)
        mask, class_ids = dataset_val.load_mask(image_id)

        image = map_image_to_rgb(image)
        visualize.display_top_masks(image, mask, class_ids, dataset_val.class_names, limit = 1)

        results = model.detect([original_image], verbose=1)
        r = results[0]
        disp_oritinal_image = map_image_to_rgb(original_image)
        visualize.display_instances(disp_oritinal_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'], ax = get_ax())


def compute_mAP(dataset_val, inference_config, model):
    # image_ids = np.random.choice(dataset_val.image_ids, 20)
    image_ids = dataset_val.image_ids
    APs = []
    i = 0
    for image_id in image_ids:

        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)

        # Run object detection
        results = model.detect([image], verbose = 0)
        r = results[0]

        # Compute AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        i += 1

    return np.mean(APs)