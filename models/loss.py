import chainer.functions as F
from chainer import cuda
from chainercv.links.model.fpn.misc import smooth_l1


class SmoothL1(object):
    def __init__(self, beta=1):
        self._beta = beta

    def __call__(self, loc, gt_loc, gt_label):
        xp = cuda.get_array_module(loc.array)
        n_sample = loc.shape[0]
        loss = F.sum(smooth_l1(
            loc[xp.where(gt_label > 0)[0]],
            gt_loc[gt_label > 0], self._beta)) / n_sample
        return loss


class SoftmaxCrossEntropy(object):
    def __call__(self, conf, gt_label):
        loss = F.softmax_cross_entropy(conf, gt_label)
        return loss
