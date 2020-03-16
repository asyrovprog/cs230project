import numpy as np
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import log
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import skimage, skimage.color
from .image_tools import *
from .metrics import *

def get_occlusions(image, size=64):
    occlusion_pixels = np.full((size, size, image.shape[2]), [0], image.dtype)
    occlusions = []
    for x in range(image.shape[0] // size):
        occlusions.append([])
        for y in range(image.shape[1] // size):            
            image_new = image.copy()
            image_new[x*size:(x+1)*size, y*size:(y+1)*size, :] = occlusion_pixels
            occlusions[x].append(image_new)
    return occlusions

def plot_occlusion_map(image, heatmap):
    # rescale heatmap to [0..1]
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap/heatmap.max() 
    heatmap = 1 - heatmap
    heatmap = ndimage.gaussian_filter(heatmap, sigma=(5, 5), order=0)

    # plot image, heatmap and overlap
    plt.subplot(1, 3, 1)
    plt.title("Image", fontsize=9)
    plt.axis('off')
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.title("Occlusion map", fontsize=9)
    plt.axis('off')
    plt.imshow(heatmap)
    plt.subplot(1, 3, 3)
    plt.title("Occlusion map overlap", fontsize=9)
    plt.axis('off')
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.6)
    plt.show()


def visualize_occlusions(dataset_val, inference_config, model):
    # Test on a random image
    #image_id = 833 
    image_id = np.random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                                image_id, use_mini_mask = False)
    print(f"Visualizing occlusions for image {image_id}")

    # generate occluded images of a specified size
    occ_size = 32
    occlusions = get_occlusions(original_image, size=occ_size)
    # fill in heatmap for every occlusion
    heatmap = np.zeros((original_image.shape[0], original_image.shape[1]))
    for x in range(len(occlusions)):
        for y in range(len(occlusions[x])):
            image = occlusions[x][y]
            r = model.detect([image], verbose=0)[0]
            _, _, overlaps = utils.compute_matches(
                gt_bbox, gt_class_id, gt_mask,
                r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.5)
            overlaps_sum = np.sum(overlaps)
            heatmap[x*occ_size:(x+1)*occ_size,y*occ_size:(y+1)*occ_size] = overlaps_sum
            print(f"  {x},{y}: occlusion {x*len(occlusions[x])+y}/{len(occlusions)*len(occlusions[x])}, overlap sum {overlaps_sum}")

    plot_occlusion_map(map_image_to_rgb(original_image)/256, heatmap)
