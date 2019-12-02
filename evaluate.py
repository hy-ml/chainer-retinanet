import argparse
from chainer import iterators
from chainercv.datasets import voc_bbox_label_names
from chainercv.evaluations import eval_detection_voc, eval_detection_coco
from chainercv.utils import apply_to_iterator, ProgressHook

from configs import cfg
from utils.load_pretrained_model import load_pretrained_model
from setup_helpers import setup_model, setup_dataset


def eval_voc(out_values, rest_values):
    pred_bboxes, pred_labels, pred_scores = out_values
    gt_bboxes, gt_labels, gt_difficults = rest_values

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    print()
    print('mAP: {:f}'.format(result['map']))
    for l, name in enumerate(voc_bbox_label_names):
        if result['ap'][l]:
            print('{:s}: {:f}'.format(name, result['ap'][l]))
        else:
            print('{:s}: -'.format(name))


def eval_coco(out_values, rest_values):
    pred_bboxes, pred_labels, pred_scores = out_values
    gt_bboxes, gt_labels, gt_area, gt_crowded = rest_values

    result = eval_detection_coco(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_area, gt_crowded)

    print()
    for area in ('all', 'large', 'medium', 'small'):
        print('mmAP ({}):'.format(area),
              result['map/iou=0.50:0.95/area={}/max_dets=100'.format(area)])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path to the config file.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID. Default is 0. -1 means CPU.')
    parser.add_argument('--batchsize', type=int, default=8,
                        help='Default is 8.')
    parser.add_argument('--pretrained_model', type=str,
                        help='Path to the pretrained model.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.freeze

    model = setup_model(cfg)
    load_pretrained_model(cfg, args.config, model, args.pretrained_model)

    dataset = setup_dataset(cfg, 'eval')
    iterator = iterators.MultithreadIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    model.use_preset('evaluate')
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    # delete unused iterators explicitly
    del in_values

    if cfg.dataset.eval == 'COCO':
        eval_coco(out_values, rest_values)
    elif cfg.dataset.eval == 'VOC':
        eval_voc(out_values, rest_values)
    else:
        raise ValueError()


if __name__ == '__main__':
    main()
