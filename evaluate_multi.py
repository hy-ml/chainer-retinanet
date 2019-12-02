import argparse
import chainer
from chainer import iterators
from chainercv.utils import apply_to_iterator, ProgressHook
import chainermn

from configs import cfg
from utils.load_pretrained_model import load_pretrained_model
from setup_helpers import setup_dataset, setup_model
from evaluate import eval_coco, eval_voc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path to the config file.')
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

    comm = chainermn.create_communicator('pure_nccl')
    device = comm.intra_rank

    model = setup_model(cfg)
    load_pretrained_model(cfg, args.config, model, args.pretrained_model)
    dataset = setup_dataset(cfg, 'eval')

    model.use_preset('evaluate')
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    if not comm.rank == 0:
        apply_to_iterator(model.predict, None, comm=comm)
        return

    iterator = iterators.MultithreadIterator(
        dataset, args.batchsize * comm.size, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)), comm=comm)
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
