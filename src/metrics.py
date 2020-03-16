import datetime
import time
import numpy as np

from mrcnn import model as modellib, utils


def compute_matches(model, inference_config, image_id, dataset_val):
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]

    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_bbox, gt_class_id, gt_mask,
        r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.5)

    true_positives = np.count_nonzero(gt_match > -1)
    false_negatives = np.count_nonzero(gt_match == -1)
    false_positives = np.count_nonzero(pred_match == -1)

    total_count = np.count_nonzero(overlaps > 0.0)
    total_iou = np.sum(overlaps)

    return true_positives, false_negatives, false_positives, total_count, total_iou


def get_f1_ratios(true_positives, false_negatives, false_positives):
    if true_positives + false_positives == 0:
        print('WARNING: Div 0:  true_positives + false_positives == 0')
        return 0, 0, 0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_f1(model, inference_config, dataset_val, print_each=None, samples=None):
    if samples is None:
        image_ids = dataset_val.image_ids
    else:
        image_ids = np.random.choice(dataset_val.image_ids, samples)
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    total_count = 0
    total_iou = 0
    i = 0
    for image_id in image_ids:
        tp, fn, fp, tc, tiou = compute_matches(model, inference_config, image_id, dataset_val)
        true_positives += tp
        false_negatives += fn
        false_positives += fp
        total_count += tc
        total_iou += tiou
        i += 1
        if (print_each is not None) and (i % print_each == 0):
            precision, recall, f1 = get_f1_ratios(true_positives, false_negatives, false_positives)
            iou_ratio = total_iou / total_count
            print(f"{datetime.datetime.now()}: processed images - {i}/{len(image_ids)}")
            print(
                f"   true positives total: {true_positives}, precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}, IoU: {iou_ratio:.4f}")
    precision, recall, f1 = get_f1_ratios(true_positives, false_negatives, false_positives)
    iou_ratio = total_iou / total_count
    return precision, recall, f1, iou_ratio