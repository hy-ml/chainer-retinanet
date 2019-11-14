import chainer
import chainer.functions as F
from chainer.links import Convolution2D


# TODO: add initialization setting
def create_fcn(out_chanels):
    fcn = []
    for oc in out_chanels:
        fcn.append(Convolution2D(None, oc, 3, 1, 1))
    return chainer.Sequential(*fcn)


class BboxHead(chainer.Chain):
    def __init__(self, n_fg_class, ratios, scales):
        super(BboxHead, self).__init__()
        with self.init_scope():
            self.loc = create_fcn(
                (256, 256, 256, 256, 4 * len(ratios) * len(scales)))
            self.conf = create_fcn(
                (256, 256, 256, 256,
                 (n_fg_class + 1) * len(ratios) * len(scales)))
        self._n_fg_class = n_fg_class

    def forward(self, hs):
        b = hs[0].shape[0]
        locs = [F.reshape(F.transpose(
            self.loc(h), (0, 2, 3, 1)), (b, -1, 4)) for h in hs]
        confs = [F.reshape(F.transpose(
            self.conf(h), (0, 2, 3, 1)), (b, -1, self._n_fg_class + 1))
            for h in hs]
        locs = F.concat(locs, axis=1)
        confs = F.concat(confs, axis=1)

        return locs, confs
