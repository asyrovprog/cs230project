import numpy as np
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import log
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import skimage, skimage.color
from .image_tools import *
from .metrics import *
from vis.losses import ActivationMaximization
from vis.optimizer import Optimizer
from vis.utils import utils

def plot_saliency_maps(image, layer_grads):
    image_count = len(layer_grads) + 1
    grads_scaling_factor = 2
    # plot original image
    plt.subplot(1, image_count, 1)
    plt.title("Image", fontsize=9)
    plt.axis('off')
    plt.imshow(map_image_to_rgb(image) / 256)
    # plot layers saliency maps
    i = 2
    for layer_name in layer_grads:
        # rescale maps to [0..1]
        grads_img = layer_grads[layer_name][0,:,:,:]
        grads_img = map_image_to_rgb(grads_img)
        grads_img = grads_img - grads_img.min()
        grads_img = grads_img/grads_img.max() 
        grads_img = grads_img * grads_scaling_factor # for brighter visualization
        plt.subplot(1, image_count, i)
        plt.title(layer_name, fontsize=9)
        plt.axis('off')
        plt.imshow(grads_img)
        i += 1
    plt.show()


def get_saliency_grads(model, layer_idx, filter_indices, seed_input):
    # define loss to maximize pixels that need to be changed the least to affect activations the most
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]
    input_tensor = model.input[0]
    # run optimization for input image
    opt = Optimizer(input_tensor, losses, wrt_tensor=None, norm_grads=False)
    grads = opt.minimize(seed_input=seed_input, max_iter=1, grad_modifier='absolute', verbose=False)[1]

    return grads

def visualize_saliency_maps(dataset_val, inference_config, model):
    # Test on a random image
    #image_id = 7 
    image_id = np.random.choice(dataset_val.image_ids)
    print(f"Visualizing saliency maps for image {image_id}")

    # load test image and get molded images for  Keras model input
    original_image, _, _, _, _ = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask = False)
    molded_images, _, _ = model.mold_inputs([original_image])

    # list layers to run saliency gradients on
    layers = {
        'Low-level convolutions': 'res2c_out',
        'Mid-level convolutions': 'res3d_out',
        'High-level convolutions': 'res5c_out',
        'RPN network output': 'fpn_p6',        
    }

    # get saliency map for layers
    layer_grads = {}
    for layer_name in layers:
        layer_idx = utils.find_layer_idx(model.keras_model, layers[layer_name])
        grads = get_saliency_grads(model.keras_model, layer_idx, None, molded_images)
        layer_grads[layer_name] = grads

    plot_saliency_maps(original_image, layer_grads)
    
