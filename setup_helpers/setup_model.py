import re
from models import RetinaNetResNet50, RetinaNetResNet101
from models import RetinaNetTrainChain
from models.loss import SmoothL1, SoftmaxCrossEntropy, SoftmaxFocalLoss
from models.suppressor import NMS


def setup_model(cfg):
    n_fg_class = cfg.dataset.n_fg_class
    pretrained_model = cfg.model.pretrained_model
    min_size = cfg.model.min_size
    max_size = cfg.model.max_size
    suppressor = setup_suppressor(cfg)

    if cfg.model.type == 'RetinaNetResNet50':
        model = RetinaNetResNet50(
            n_fg_class, pretrained_model, min_size, max_size, suppressor)
    elif cfg.model.type == 'RetinaNetResNet101':
        model = RetinaNetResNet101(
            n_fg_class, pretrained_model, min_size, max_size, suppressor)
    else:
        raise ValueError('Not support model `{}`.'.format(cfg.model.type))
    return model


def setup_train_chain(cfg, model):
    # setup loc_loss
    if cfg.model.loc_loss == 'SmoothL1':
        loc_loss = SmoothL1()
    else:
        raise ValueError(
            'Not support `loc_loss`: {}.'.format(cfg.model.loc_loss))

    # setup conf_loss
    if cfg.model.conf_loss == 'SoftmaxFocalLoss':
        conf_loss = SoftmaxFocalLoss(cfg.model.focal_loss_gamma)
    elif cfg.model.conf_loss == 'SoftmaxCrossEntropy':
        conf_loss = SoftmaxCrossEntropy()
    else:
        raise ValueError(
            'Not support `conf_loss`: {}.'.format(cfg.model.conf_loss))

    train_chain = RetinaNetTrainChain(model, loc_loss, conf_loss,
                                      cfg.model.fg_thresh, cfg.model.bg_thresh)
    return train_chain


def freeze_params(cfg, train_chain):
    for path, link in train_chain.model.namedlinks():
        for regex in cfg.model.freeze_layers:
            if re.fullmatch(regex, path):
                link.disable_update()
                break
    return train_chain


def setup_suppressor(cfg):
    if cfg.model.suppressor == 'NMS':
        suppressor = NMS(cfg.model.nms_thresh)
    else:
        raise ValueError(
            'Not support `suppressor`: {}.'.format(cfg.model.suppressor))
    return suppressor
