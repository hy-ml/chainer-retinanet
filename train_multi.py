import argparse
import multiprocessing
import numpy as np
import cProfile
import io
import pstats

import chainer
from chainer import serializers
from chainer import training
import chainermn
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv import transforms

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
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark option.')
    parser.add_argument('--benchmark_n_iteration', type=int, default=500,
                        help='Iteration in benchmark option. Default is 500.')
    parser.add_argument('--n_print_profile', type=int, default=100,
                        help='Default is 100.')
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

    comm = chainermn.create_communicator('pure_nccl')
    assert comm.size == cfg.n_gpu
    device = comm.intra_rank

    if comm.rank == 0:
        print(cfg)

    model = setup_model(cfg)
    train_chain = setup_train_chain(cfg, model)
    chainer.cuda.get_device_from_id(device).use()
    train_chain.to_gpu()

    train_dataset = TransformDataset(
        setup_dataset(cfg, 'train'), ('img', 'bbox', 'label'), Transform())
    if args.benchmark:
        shuffle = False
    else:
        shuffle = True

    if comm.rank == 0:
        indices = np.arange(len(train_dataset))
    else:
        indices = None

    indices = chainermn.scatter_dataset(indices, comm, shuffle=shuffle)
    train_dataset = train_dataset.slice[indices]
    train_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, cfg.n_sample_per_gpu,
        n_processes=cfg.n_worker,
        shared_mem=100 * 1000 * 1000 * 4, shuffle=shuffle)
    optimizer = chainermn.create_multi_node_optimizer(
        setup_optimizer(cfg), comm)
    optimizer = optimizer.setup(train_chain)
    optimizer = add_hook_optimizer(optimizer, cfg)
    freeze_params(cfg, train_chain.model)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device, converter=converter)
    if args.benchmark:
        stop_trigger = (args.benchmark_n_iteration, 'iteration')
        outdir = 'benchmark_out'
    else:
        stop_trigger = (cfg.solver.n_iteration, 'iteration')
        outdir = get_outdir(args.config)
    trainer = training.Trainer(updater, stop_trigger, outdir)

    if args.benchmark:
        if comm.rank == 0:
            log_interval = 10, 'iteration'
            trainer.extend(training.extensions.LogReport(trigger=log_interval))
            trainer.extend(training.extensions.PrintReport(
                ['epoch', 'iteration', 'main/loss',
                 'main/loss/loc', 'main/loss/conf']),
                trigger=log_interval)
        pr = cProfile.Profile()
        pr.enable()
        trainer.run()
        pr.disable()
        s = io.StringIO()
        sort_by = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
        ps.print_stats()
        if comm.rank == 0:
            lines = s.getvalue().split('\n')
            for line in lines[:args.n_print_profile]:
                print(line)

        pr.dump_stats(
            '{0}/train_multi_rank_{1}.cprofile'.format(outdir, comm.rank))
        exit()

    # extention
    if comm.rank == 0:
        log_interval = 10, 'iteration'
        trainer.extend(training.extensions.LogReport(trigger=log_interval))
        trainer.extend(training.extensions.observe_lr(), trigger=log_interval)
        trainer.extend(training.extensions.PrintReport(
            ['epoch', 'iteration', 'lr', 'main/loss',
             'main/loss/loc', 'main/loss/conf']),
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
