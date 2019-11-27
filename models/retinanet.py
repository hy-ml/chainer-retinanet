import numpy as np
from chainer.backends import cuda
import chainer.functions as F
from chainer.link import Chain
from chainercv import transforms


class RetinaNet(Chain):

    _anchor_size = 32
    _anchor_ratios = (0.5, 1, 2)
    _anchor_scales = (1, 2**(1 / 3), 2**(2 / 3))
    _std = (0.1, 0.2)
    _eps = 1e-5

    def __init__(self, extractor, bbox_head, min_size=800, max_size=1333,
                 suppressor=None):
        super(RetinaNet, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.bbox_head = bbox_head
        self._suppressor = suppressor
        self._scales = self.extractor.scales
        self._min_size = min_size
        self._max_size = max_size
        self._stride = 32

    def use_preset(self, preset):
        if preset == 'visualize':
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.score_thresh = 0.1
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _forward_extractor(self, x):
        with chainer.using_config('train', False):
            hs = self.extractor(x)
        return hs

    def _forward_heads(self, hs):
        locs, confs = self.bbox_head(hs)
        return locs, confs

    def forward(self, x):
        if isinstance(x, self.xp.ndarray):
            b = x.shape[0]
        elif isinstance(x, list):
            b = len(x)
        else:
            raise ValueError()
        x, scales, _ = self._prepare(x)
        hs = self._forward_extractor(x)
        locs, confs = self._forward_heads(hs)
        anchors = self.xp.vstack([self.anchors(h.shape[2:] for h in hs)[
                                 self.xp.newaxis] for _ in range(b)])
        return anchors, locs, confs, scales

    def predict(self, x):
        x, scales, sizes = self._prepare(x)
        hs = self._forward_extractor(x)
        anchors = self.xp.vstack([self.anchors(h.shape[2:] for h in hs)[
                                 self.xp.newaxis] for _ in range(len(sizes))])
        locs, confs = self._forward_heads(hs)
        scores = F.sigmoid(F.max(confs, axis=-1))
        labels = F.argmax(confs, -1)

        locs = locs.array
        labels = labels.array
        scores = scores.array

        anchors, locs, labels, scores = self._remove_bg(
            anchors, locs, labels, scores)
        bboxes = self._decode(anchors, locs, scales, sizes)
        bboxes, labels, scores = self._suppress(bboxes, labels, scores)

        bboxes = [cuda.to_cpu(b) for b in bboxes]
        labels = [cuda.to_cpu(l) for l in labels]
        scores = [cuda.to_cpu(s) for s in scores]

        return bboxes, labels, scores

    def anchors(self, sizes):
        anchors = []
        for l, (H, W) in enumerate(sizes):
            _anchors = []
            for s in self._anchor_scales:
                v, u, ar = np.meshgrid(
                    np.arange(W), np.arange(H), self._anchor_ratios)
                w = np.round(1 / np.sqrt(ar) / self._scales[l])
                h = np.round(w * ar)
                anchor = np.stack((u, v, h, w)).reshape((4, -1)).transpose()
                anchor[:, :2] = (anchor[:, :2] + 0.5) / self._scales[l]
                anchor[:, 2:] *= (self._anchor_size << l) * self._scales[l] * s

                # yxhw -> tlbr
                anchor[:, :2] -= anchor[:, 2:] / 2
                anchor[:, 2:] += anchor[:, :2]
                _anchors.append(self.xp.array(
                    anchor[:, np.newaxis, :], dtype=np.float32))
            # anchors.append(self.xp.vstack(_anchors))
            anchors.append(self.xp.concatenate(
                _anchors, axis=-2).reshape((-1, 4)))
        anchors = self.xp.vstack(anchors)
        # anchors = self.xp.concatenate(anchors, axis=-2)
        # anchors = anchors.reshape(-1, 4)
        return anchors

    def _prepare(self, imgs):
        """Preprocess images.
        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.
        Returns:
            Two arrays: preprocessed images and \
            scales that were caluclated in prepocessing.
        """
        scales = []
        resized_imgs = []
        sizes = [(img.shape[1], img.shape[2]) for img in imgs]
        for img in imgs:
            img, scale = self._scale_img(img)
            img -= self.extractor.mean
            scales.append(scale)
            resized_imgs.append(img)
        pad_size = np.array(
            [im.shape[1:] for im in resized_imgs]).max(axis=0)
        pad_size = (
            np.ceil(pad_size / self._stride) * self._stride).astype(int)
        x = np.zeros(
            (len(imgs), 3, pad_size[0], pad_size[1]), dtype=np.float32)
        for i, im in enumerate(resized_imgs):
            _, H, W = im.shape
            x[i, :, :H, :W] = im
        x = self.xp.array(x)

        return x, scales, sizes

    def _scale_img(self, img):
        """Process image."""
        _, H, W = img.shape
        scale = self._min_size / min(H, W)
        if scale * max(H, W) > self._max_size:
            scale = self._max_size / max(H, W)
        H, W = int(H * scale), int(W * scale)
        img = transforms.resize(img, (H, W))
        return img, scale

    def _remove_bg(self, anchors, locs, labels, scores):
        _anchors, _locs, _labels, _scores = [], [], [], []
        for anchor, loc, label, score in zip(anchors, locs, labels, scores):
            mask = score > self.score_thresh
            _anchors.append(anchor[mask])
            _locs.append(loc[mask])
            _labels.append(label[mask])
            _scores.append(score[mask])
        return _anchors, _locs, _labels, _scores

    # TODO: check implement properly
    def _decode(self, anchors, locs, scales, sizes):
        bboxes = []
        for i in range(len(sizes)):
            anchor, loc, scale, size = anchors[i], locs[i], scales[i], sizes[i]
            if loc.shape[0] == 0:  # guard no fg
                bbox = self.xp.empty((0, 4), dtype=self.xp.float32)
                bboxes.append(bbox)
                continue

            # bbox = self.xp.broadcast_to(anchor[:, None], loc.shape) / scales[i]
            bbox = anchor
            # tlbr -> yxhw
            bbox[:, 2:] -= bbox[:, :2]
            bbox[:, :2] += bbox[:, 2:] / 2
            # offset
            bbox[:, :2] += loc[:, :2] * bbox[:, 2:] * self._std[0]
            bbox[:, 2:] *= self.xp.exp(
                self.xp.minimum(loc[:, 2:] * self._std[1], self._eps))
            # yxhw -> tlbr
            bbox[:, :2] -= bbox[:, 2:] / 2
            bbox[:, 2:] += bbox[:, :2]
            # scale bbox
            bbox = bbox / scale
            # clip
            bbox[:, :2] = self.xp.maximum(bbox[:, :2], 0)
            bbox[:, 2:] = self.xp.minimum(
                bbox[:, 2:], self.xp.array(size))

            bboxes.append(bbox)

        return bboxes

    def _suppress(self, bboxes, labels, scores):
        _bboxes, _labels, _scores = [], [], []
        for bbox, label, score in zip(bboxes, labels, scores):
            if bbox.shape[0] == 0:
                _bboxes.append(bbox)
                _labels.append(label)
                _scores.append(score)
                continue

            unique_label = self.xp.unique(label)
            _bbox = self.xp.empty((0, 4), dtype=self.xp.float32)
            _label = self.xp.empty(0, dtype=self.xp.int32)
            _score = self.xp.empty(0, dtype=self.xp.float32)
            for l in unique_label:
                mask = self.xp.where(label == l)[0]
                selc = self._suppressor(bbox[mask], score[mask])
                _bbox = self.xp.vstack((_bbox, bbox[mask][selc]))
                _label = self.xp.hstack((_label, label[mask][selc]))
                _score = self.xp.hstack((_score, score[mask][selc]))

            _bboxes.append(_bbox)
            _labels.append(_label)
            _scores.append(_score)
        return _bboxes, _labels, _scores
