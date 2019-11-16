import numpy as np
import chainer.functions as F
from chainer import cuda
from chainercv.links.model.fpn.misc import smooth_l1


class SmoothL1(object):
    def __init__(self, beta=1):
        self._beta = beta
        self._eps = 1e-5

    def __call__(self, loc, gt_loc, gt_label):
        xp = cuda.get_array_module(loc.array)
        loc = loc[xp.where(gt_label > 0)[0]]
        gt_loc = gt_loc[xp.where(gt_label > 0)[0]]
        n_sample = loc.shape[0] + self._eps
        loss = F.sum(smooth_l1(loc, gt_loc, self._beta)) / n_sample
        return loss


class SoftmaxCrossEntropy(object):
    def __call__(self, conf, gt_label):
        loss = F.softmax_cross_entropy(conf, gt_label)
        return loss


class FocalLoss(object):
    def __init__(self, alpha=0.25, gamma=2.0):
        self._alpha = alpha
        self._gamma = gamma
        self._eps = 1e-5

    def __call__(self, conf, gt_label):
        xp = cuda.get_array_module(gt_label)
        gt_label = cuda.to_cpu(gt_label)
        # n_fg = max(xp.where(gt_label > 0)[0].shape[0], 1)
        n_fg = max(np.where(gt_label > 0)[0].shape[0], 1)
        n_class = conf.shape[-1]
        gt_label = np.eye(n_class + 1, dtype=np.int32)[gt_label][:, 1:]
        gt_label = xp.array(gt_label, dtype=xp.int32)

        logit = F.sigmoid(conf)
        logit = F.clip(logit, self._eps, 1 - self._eps)

        alpha_factor = xp.ones_like(gt_label, dtype=xp.float32) * self._alpha
        alpha_factor = F.where(gt_label == 1, alpha_factor, 1 - alpha_factor)
        focal_weight = F.where(gt_label == 1, 1 - logit, logit)
        focal_weight = alpha_factor * focal_weight ** self._gamma

        loss = focal_weight.reshape(-1) * \
            F.sigmoid_cross_entropy(
                conf.reshape(-1), gt_label.reshape(-1), reduce='no')
        loss = F.sum(loss) / n_fg
        return loss
