import chainer.functions as F
from chainer import cuda
from chainercv.links.model.fpn.misc import smooth_l1


class SmoothL1(object):
    def __init__(self, beta=1):
        self._beta = beta

    def __call__(self, loc, gt_loc, gt_label):
        xp = cuda.get_array_module(loc.array)
        loc = loc[xp.where(gt_label > 0)[0]]
        gt_loc = gt_loc[xp.where(gt_label > 0)[0]]
        n_sample = loc.shape[0] + 1e-10
        loss = F.sum(smooth_l1(loc, gt_loc, self._beta)) / n_sample
        return loss


class SoftmaxCrossEntropy(object):
    def __call__(self, conf, gt_label):
        loss = F.softmax_cross_entropy(conf, gt_label)
        return loss


class SoftmaxFocalLoss(object):
    def __init__(self, gamma=2, eps=1e-7):
        self._gamma = gamma
        self._eps = eps

    def __call__(self, conf, gt_label):
        return self._focal_loss(conf, gt_label)

    def _focal_loss(self, x, t):
        xp = cuda.get_array_module(t)
        n_sample = t.shape[0]
        n_class = x.shape[-1]
        t = xp.eye(n_class)[t]
        logit = F.softmax(x)
        logit = F.clip(logit, self._eps, 1 - self._eps)
        y = -1 * t * F.log(logit)
        y = y * (1 - logit) ** self._gamma
        loss = F.sum(y) / n_sample
        return loss
