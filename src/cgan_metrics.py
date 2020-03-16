import numpy as np

def compute_cgan_metrics(img_y, img_g, i = 0):
    """
    Computes accuracy, precision, recall, f1, iou_score for passed image, return None in case of div 0
        img_y: ground truth building footprint semantic map
        img_g: generated image
        i:     0 for entire image, 1 for inner (excluding border)
    Note:
        image format is (n,n,1) and each pixel is either -1 (for 'no' building at pixel) or 1 (for 'yes' building at pixel)
    """

    # image size (it is square), and ring step
    iz, rz = int(img_y.shape[0]), int(img_y.shape[0] / (4 * 2))

    # building inner square mask (ring) where we calculate metrics
    # example of such mask:
    #  1 1 1 1
    #  1 0 0 1
    #  1 0 0 1
    #  1 1 1 1
    ring = np.ones(img_y.shape, dtype=bool)
    ring[i * rz:iz - i * rz, i * rz:iz - i * rz, 0] = False

    # now, erasing all areas which are not in ring with 0
    img_y[ring] = 0
    img_g[ring] = 0

    # TP (true positive), TN, FP, FN
    TP = np.sum(np.logical_and((img_y == 1), (img_g == 1)))
    TN = np.sum(np.logical_and((img_y == -1), (img_g == -1)))
    FP = np.sum(np.logical_and((img_y == -1), (img_g == 1)))
    FN = np.sum(np.logical_and((img_y == 1), (img_g == -1)))
    # IoU (intersection over union)
    intersection = np.logical_and((img_y == 1), (img_g == 1))
    union = np.logical_or((img_y == 1), (img_g == 1))

    if TP + FP == 0 or TP + FN == 0:
        return None

    # reporting metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    if precision == 0 and recall == 0:
        return None

    f1 = 2.0 * (precision * recall) / (precision + recall)
    iou_score = np.sum(intersection) / np.sum(union)

    return accuracy, precision, recall, f1, iou_score

def compute_cgan_metrics_batch(aimg_y, aimg_g):
    """
    Computes average metrics on list of images
    """
    a_accuracy, a_precision, a_recall, a_f1, a_iou_score = [], [], [], [], []
    for i in range(len(aimg_y)):
        res = compute_cgan_metrics(aimg_y[i], aimg_g[i])

        if not res is None:
            accuracy, precision, recall, f1, iou_score = res
            a_accuracy.append(accuracy)
            a_precision.append(precision)
            a_recall.append(recall)
            a_f1.append(f1)
            a_iou_score.append(iou_score)

    return np.mean(a_accuracy), np.mean(a_precision), np.mean(a_recall), np.mean(a_f1), np.mean(a_iou_score)