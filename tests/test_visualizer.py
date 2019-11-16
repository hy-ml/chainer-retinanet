import argparse
import numpy as np
import cv2
from chainercv.datasets import COCOBboxDataset

import _init_path  # NOQA
from utils.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coco_dir', type=str,
                        help='Path to the dirctory of COCO dataset.')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'])
    args = parser.parse_args()

    dataset = COCOBboxDataset(args.coco_dir, args.split)
    visualizer = Visualizer('COCO')

    for img, bbox, label in dataset:
        result = visualizer.visualize(img, ([bbox], [label]))

        cv2.imshow('output', result)
        key = cv2.waitKey(0) & 0xff
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
