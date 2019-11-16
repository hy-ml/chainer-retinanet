import os
import argparse
import numpy as np
import cv2

from configs import cfg
from setup_helpers import setup_model, setup_dataset
from utils.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path to the config file.')
    parser.add_argument('--model_path', type=str,
                        help='Path to the model path. If not specified, '
                             'last model is automatically loaded.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID. `-1` means CPU.')
    parser.add_argument('--use_preset', type=str, default='visualize',
                        choices='visualize', 'evaluate')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    model = setup_model(cfg)
    model.use_preset(args.use_preset)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    dataset = setup_dataset(cfg, 'val')
    visualizer = Visualizer('COCO')

    for data in dataset:
        img = data[0]
        output = [[v[0][:10]] for v in model.predict([img.copy()])]
        result = visualizer.visualize(img, output)

        cv2.imshow('result', result)
        key = cv2.waitKey(0) & 0xff
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
