import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.resnet import ResNet50, ResNet101

from models.fpn import FPN
from models.retinanet import RetinaNet
from models.bbox_head import BboxHead


class RetinaNetResNet(RetinaNet):
    def __init__(self, n_fg_class=None, pretrained_model=None,
                 min_size=800, max_size=1333, suppressor=None):
        base = self._base(n_class=1, arch='he')
        base.pick = ('res3', 'res4', 'res5')
        base.pool1 = lambda x: F.max_pooling_2d(
            x, 3, stride=2, pad=1, cover_all=False)
        base.remove_unused()
        extractor = FPN(
            base, len(base.pick), (1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128))
        bbox_head = BboxHead(
            n_fg_class, self._anchor_ratios, self._anchor_scales)

        super(RetinaNetResNet, self).__init__(
            extractor=extractor,
            bbox_head=bbox_head,
            min_size=max_size,
            max_size=max_size,
            suppressor=suppressor,
        )

        if pretrained_model == 'imagenet':
            _copyparams(
                self.extractor.base,
                self._base(pretrained_model='imagenet', arch='he'))
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)


class RetinaNetResNet50(RetinaNetResNet):
    _base = ResNet50


class RetinaNetResNet101(RetinaNetResNet):
    _base = ResNet101


def _copyparams(dst, src):
    if isinstance(dst, chainer.Chain):
        for link in dst.children():
            _copyparams(link, src[link.name])
    elif isinstance(dst, chainer.ChainList):
        for i, link in enumerate(dst):
            _copyparams(link, src[i])
    else:
        dst.copyparams(src)
        if isinstance(dst, L.BatchNormalization):
            dst.avg_mean = src.avg_mean
            dst.avg_var = src.avg_var
