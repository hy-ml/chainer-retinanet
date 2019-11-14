import numpy as np
import cv2
from chainercv.datasets import coco_bbox_label_names, voc_bbox_label_names


class Visualizer(object):
    """Visualizer for object detection

    """

    def __init__(self, dataset_type, thickness=2):
        if dataset_type == 'COCO':
            self._label_names = coco_bbox_label_names
        elif dataset_type == 'VOC':
            self._label_names = voc_bbox_label_names
        else:
            raise ValueError(
                'Not support visualization for dataset `{}`'.format(
                    dataset_type))

        self._thickness = thickness

    def visualize(self, img, outputs):
        img = img.copy()

        if len(outputs) == 3:
            bboxes, labels, scores = outputs
        elif len(outputs) == 2:
            bboxes, labels = outputs
            scores = [[None for _ in range(bboxes[0].shape[0])]]
        bboxes = bboxes[0].astype(np.int)
        labels = labels[0]
        scores = scores[0]

        for bbox, label, score in zip(bboxes, labels, scores):
            self._add_bbox(img, bbox)
            self._add_text(img, bbox, label, score)
        return img

    def _add_bbox(self, img, bbox):
        pt1 = (bbox[1], bbox[0])
        pt2 = (bbox[3], bbox[2])
        cv2.rectangle(img, pt1, pt2, (50, 50, 250), self._thickness)

    def _add_text(self, img, bbox, label, score):
        if score is None:
            score = ''
        else:
            score = ': {.2f}'.format(score)
        cat = self._label_names[label] + score
        font = cv2.FONT_HERSHEY_SIMPLEX

        cat_size = cv2.getTextSize(cat, font, 0.5, 2)[0]
        cv2.rectangle(img, (bbox[1], bbox[0] - cat_size[1] - 2),
                      (bbox[1] + cat_size[0], bbox[0] - 2), (50, 50, 250), -1)
        cv2.putText(img, cat, (bbox[1], bbox[0] - 2),
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
