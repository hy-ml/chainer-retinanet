import numpy as np
from chainer.backends import cuda


def convert_xywh_to_xyxy(bboxes):
    x1 = bboxes[:, :1]
    y1 = bboxes[:, 1:2]
    w = bboxes[:, 2:3]
    h = bboxes[:, 3:]

    x2 = x1 + w
    y2 = y1 + h

    bboxes_convert = np.hstack((x1, y1, x2, y2))
    return bboxes_convert


def area(bboxes):
    return (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])


def intersection(bboxes1, bboxes2):
    xp = cuda.get_array_module(bboxes1)
    [y_min1, x_min1, y_max1, x_max1] = xp.split(bboxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = xp.split(bboxes2, 4, axis=1)

    all_pairs_min_ymax = xp.minimum(y_max1, xp.transpose(y_max2))
    all_pairs_max_ymin = xp.maximum(y_min1, xp.transpose(y_min2))
    intersect_heights = xp.maximum(
        xp.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = xp.minimum(x_max1, xp.transpose(x_max2))
    all_pairs_max_xmin = xp.maximum(x_min1, xp.transpose(x_min2))
    intersect_widths = xp.maximum(
        xp.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def calc_iou(bboxes1, bboxes2):
    xp = cuda.get_array_module(bboxes1)
    intersect = intersection(bboxes1, bboxes2)
    area1 = area(bboxes1)
    area2 = area(bboxes2)
    union = xp.expand_dims(area1, axis=1) + xp.expand_dims(
        area2, axis=0) - intersect + 1e-8
    return intersect / union
