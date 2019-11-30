import os
import argparse
import cv2
from chainer import serializers

from configs import cfg
from configs.path_catalog import gdrive_ids
from setup_helpers import setup_model, setup_dataset
from utils.visualizer import Visualizer
from utils.path import get_outdir
from utils.download_file_from_gdrive import download_file_from_gdrive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path to the config file.')
    parser.add_argument('--pretrained_model', type=str,
                        help='Path to the pretrained model.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID. `-1` means CPU.')
    parser.add_argument('--use_preset', type=str, default='visualize',
                        choices=['visualize', 'evaluate'])
    parser.add_argument('--split', type=str, default='eval',
                        choices=['train', 'eval'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    model = setup_model(cfg)

    # load pretrained model
    if args.pretrained_model == 'auto':
        save_dir = os.path.join('./out/download', cfg.dataset.train)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        pretrained_model = os.path.join(save_dir, cfg.model.type)
        if not os.path.isfile(pretrained_model):
            gdrive_id = gdrive_ids[cfg.dataset.train][cfg.model.type]
            download_file_from_gdrive(gdrive_id, pretrained_model)
    elif args.pretrained_model:
        pretrained_model = args.pretrained_model
    else:
        pretrained_model = os.path.join(get_outdir(
            args.config), 'model_iter_{}'.format(cfg.solver.n_iteration))
    serializers.load_npz(pretrained_model, model)

    model.use_preset(args.use_preset)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    dataset = setup_dataset(cfg, args.split)
    visualizer = Visualizer(cfg.dataset.eval)

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
