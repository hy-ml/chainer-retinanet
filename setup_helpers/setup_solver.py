from chainer.optimizer_hooks import WeightDecay
from chainer.optimizers import SGD, MomentumSGD


def setup_optimizer(cfg):
    if cfg.solver.optimizer == 'SGD':
        optimizer = SGD(cfg.optimizer.base_lr)
    elif cfg.solver.optimizer == 'MomentumSGD':
        optimizer = MomentumSGD(cfg.solver.base_lr, cfg.solver.momentum)
    else:
        raise ValueError(
            'Not support `optimizer`: {}.'.format(cfg.solver.optimizer))
    return optimizer


def add_hock_optimizer(optimizer, cfg):
    hooks = cfg.solver.hooks

    def _get_hock(hook):
        if hook == 'WeightDecay':
            return WeightDecay(cfg.solver.weight_decay)
        else:
            raise ValueError('Not support `hock`: {}.'.format(hook))

    for hook in hooks:
        optimizer.add_hook(_get_hock(hook))
    return optimizer
