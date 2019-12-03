import argparse
import multiprocessing

import chainer
from chainer import serializers
from chainer import training
from chainercv import transforms
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset

from configs import cfg
from utils.path import get_outdir, get_logdir
from extensions import LogTensorboard
from setup_helpers import setup_dataset
from setup_helpers import setup_model, setup_train_chain, freeze_params
from setup_helpers import setup_optimizer, add_hook_optimizer


def converter(batch, device=None):
    # do not send data to gpu (device is ignored)
    return tuple(list(v) for v in zip(*batch))


class Transform(object):

    def __call__(self, in_data):
        img, bbox, label = in_data
        # Flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        x_flip = params['x_flip']
        bbox = transforms.flip_bbox(
            bbox, img.shape[1:], x_flip=x_flip)
        return img, bbox, label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path to the config file.')
    parser.add_argument('--tensorboard', type=bool, default=True,
                        help='Whether use Tensorboard. Default is True.')
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    model = setup_model(cfg)
    train_chain = setup_train_chain(cfg, model)
    train_chain.to_gpu(0)

    train_dataset = TransformDataset(
        setup_dataset(cfg, 'train'), ('img', 'bbox', 'label'), Transform())
    train_iter = chainer.iterators.MultithreadIterator(
        train_dataset, cfg.n_sample_per_gpu)
    optimizer = setup_optimizer(cfg)
    optimizer.setup(train_chain)
    add_hook_optimizer(optimizer, cfg)
    freeze_params(cfg, train_chain.model)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter)
    trainer = training.Trainer(
        updater, (cfg.solver.n_iteration, 'iteration'),
        get_outdir(args.config))

    # extention
    log_interval = 10, 'iteration'
    trainer.extend(training.extensions.LogReport(trigger=log_interval))
    trainer.extend(training.extensions.observe_lr(), trigger=log_interval)
    trainer.extend(training.extensions.PrintReport(
        ['epoch', 'iteration', 'lr', 'main/loss',
            'main/loss/loc', 'main/loss/conf',
         ]),
        trigger=log_interval)
    trainer.extend(training.extensions.ProgressBar(update_interval=10))

    trainer.extend(training.extensions.snapshot(),
                   trigger=(10000, 'iteration'))
    trainer.extend(
        training.extensions.snapshot_object(
            model, 'model_iter_{.updater.iteration}'),
        trigger=(cfg.solver.n_iteration, 'iteration'))
    if args.tensorboard:
        trainer.extend(LogTensorboard(
            ['lr', 'main/loss', 'main/loss/loc', 'main/loss/conf'],
            trigger=(10, 'iteration'), log_dir=get_logdir(args.config)))

    if len(cfg.solver.lr_step):
        trainer.extend(training.extensions.MultistepShift(
            'lr', 0.1, cfg.solver.lr_step, cfg.solver.base_lr, optimizer))

    if args.resume:
        serializers.load_npz(args.resume, trainer, strict=False)

    trainer.run()


if __name__ == '__main__':
    main()
