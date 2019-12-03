import numpy as np


def convert_xywh_to_xyxy(bboxes):
    x1 = bboxes[:, :1]
    y1 = bboxes[:, 1:2]
    w = bboxes[:, 2:3]
    h = bboxes[:, 3:]

    x2 = x1 + w
    y2 = y1 + h

    bboxes_convert = np.hstack((x1, y1, x2, y2))
    return bboxes_convert
