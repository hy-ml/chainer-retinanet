import argparse
import numpy as np
import cv2
from chainercv.datasets import COCOBboxDataset
from chainercv.datasets import VOCBboxDataset

import _init_path  # NOQA
from utils.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Path to the dirctory of COCO dataset.')
    parser.add_argument('dataset_type', type=str,
                        choices=['COCO', 'VOC'])
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'])
    args = parser.parse_args()

    if args.dataset_type == 'COCO':
        dataset = COCOBboxDataset(args.data_dir, args.split)
    elif args.dataset_type == 'VOC':
        dataset = VOCBboxDataset(args.data_dir, split=args.split)
    else:
        raise ValueError()
    visualizer = Visualizer(args.dataset_type)

    for img, bbox, label in dataset:
        result = visualizer.visualize(img, ([bbox], [label]))

        cv2.imshow('output', result)
        key = cv2.waitKey(0) & 0xff
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
