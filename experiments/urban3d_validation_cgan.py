"""
    Evaluate RGB Urban3d dataset. This program should be called from root folder of the
    project (cs230project) as follows:

        python -m experiments.urban3d_validation --config "easy_debug"

    List of configs located at configs/config_factory.py
"""

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

    # list of images (array of {id, source, path} dictionaries)
    image_infos = config.ds_val.image_info

    # arrays of results for metrics first entry is for entire image metrics, second entry is with
    # 32 pixel border cut off. we need this to undestand how cutting off parts of building affects
    # model performance
    a_accuracy, a_precision, a_recall, a_f1, a_iou_score = [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]

    # calculate metrics image by image
    print("=================================================================================================================")
    print(" Ring |  Accuracy  | Precision |   Recall  |     F1    |    IoU     | File")
    print("------|------------|-----------|-----------|-----------|------------|--------------------------------------------")
    for img in image_infos:
        img_id  = img["id"]
        img_src = os.path.basename(img["path"])

        # get batch (single image) of source image (rgbd satellite image) and ground truth (building footprint)
        # these images are scaled to range [-1, 1]
        aimg_y, aimg_x, _ = config.batch_from_ids([img_id], config.ds_val, use_image_generator = False)

        # get prediction for rgbd source, this image pixels are in range [-1, 1], where -1 is "not a building"
        # and 1 is "building", and no other values are possible
        aimg_g = cgan.predict(aimg_x)

        for i in range(len(a_accuracy)):
            # converting to images (from arrays of single images)
            res = compute_cgan_metrics(aimg_y[0].copy(), aimg_g[0].copy(), i)

            if not res is None:
                accuracy, precision, recall, f1, iou_score = res
                a_accuracy[i].append(accuracy)
                a_precision[i].append(precision)
                a_recall[i].append(recall)
                a_f1[i].append(f1)
                a_iou_score[i].append(iou_score)

                print(f" {i:4} | {accuracy:8.6f} | {precision:9.6f} | {recall:9.6f} | {f1:9.6f} | {iou_score:9.6f} | {img_src}")


    print("------|-----------|-----------|-----------|-----------|------------|--------------------------------------------")
    for i in range(len(a_accuracy)):
        mu_accuracy, mu_precision, mu_recall, mu_f1, mu_iou_score = np.mean(a_accuracy[i]), \
            np.mean(a_precision[i]), np.mean(a_recall[i]), np.mean(a_f1[i]), np.mean(a_iou_score[i])

        std_accuracy, std_precision, std_recall, std_f1, std_iou_score = np.std(a_accuracy[i]), \
            np.std(a_precision[i]), np.std(a_recall[i]), np.std(a_f1[i]), np.std(a_iou_score[i])

        print(f" Mean {i}:")
        print(f" {i:4} | {mu_accuracy:8.6f} | {mu_precision:9.6f} | {mu_recall:9.6f} | {mu_f1:9.6f} | {mu_iou_score:9.6f} |")
        print(f" Std {i}:")
        print(f" {i:4} | {std_accuracy:8.6f} | {std_precision:9.6f} | {std_recall:9.6f} | {std_f1:9.6f} | {std_iou_score:9.6f} |")

    print("================================================================================================================")