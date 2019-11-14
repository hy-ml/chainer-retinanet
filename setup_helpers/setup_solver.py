from chainer.optimizer_hooks import WeightDecay
from chainer.optimizers import SGD, MomentumSGD


def setup_optimizer(cfg):
    if cfg.solver.optimizer == 'SGD':
        optimzier = SGD(cfg.optimizer.base_lr)
    elif cfg.solver.optimizer == 'MomentumSGD':
        optimzier = MomentumSGD(cfg.solver.base_lr, cfg.solver.momentum)
    else:
        raise ValueError(
            'Not support `optimizer`: {}.'.format(cfg.solver.optimizer))
    return optimzier


def add_hock_optimizer(optimizer, cfg):
    hooks = cfg.solver.hooks

    def _get_hock(hook):
        if hook == 'WeightDecay':
            return WeightDecay(cfg.solver.weight_decay)

    for hook in hooks:
        optimizer.add_hock(_get_hock(hook))
    return optimizer
