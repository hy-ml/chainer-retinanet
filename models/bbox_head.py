import numpy as np
import chainer
import chainer.functions as F
from chainer import initializers
from chainer.links import Convolution2D


def create_fcn(out_chanels, init_options=None):
    default_init = {'initialW': initializers.Normal(0.01)}
    if init_options is None:
        init_options = [None for _ in out_chanels]
    fcn = []
    for oc, init_option in zip(out_chanels, init_options):
        if init_option is None:
            init = default_init
        else:
            init = init_option
        fcn.append(Convolution2D(None, oc, 3, 1, 1, **init))
    return chainer.Sequential(*fcn)


class BboxHead(chainer.Chain):
    _prior = 0.01
    _conf_last_init = {
        'initialW': initializers.Normal(0.01),
        'initial_bias': initializers.Constant(-np.log((1 - _prior) / _prior))
    }

    def __init__(self, n_fg_class, ratios, scales):
        super(BboxHead, self).__init__()
        with self.init_scope():
            self.loc = create_fcn(
                (256, 256, 256, 256, 4 * len(ratios) * len(scales)))
            self.conf = create_fcn(
                (256, 256, 256, 256,
                 (n_fg_class) * len(ratios) * len(scales)),
                [None, None, None, None, self._conf_last_init])
        self._n_fg_class = n_fg_class

    def forward(self, hs):
        b = hs[0].shape[0]
        locs = [F.reshape(F.transpose(
            self.loc(h), (0, 2, 3, 1)), (b, -1, 4)) for h in hs]
        confs = [F.reshape(F.transpose(
            self.conf(h), (0, 2, 3, 1)), (b, -1, self._n_fg_class))
            for h in hs]
        locs = F.concat(locs, axis=1)
        confs = F.concat(confs, axis=1)

        return locs, confs
