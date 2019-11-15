import numpy as np
import cv2
from PIL import Image
from chainer.link import Chain
from chainercv import transforms


class RetinaNet(Chain):

    _anchor_size = 32
    _anchor_ratios = (0.5, 1, 2)
    _anchor_scales = (1, 2**(1 / 3), 2**(2 / 3))

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
            self.nms_thresh = 0.5
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.5
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _forward_extractor(self, x):
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
        x, scales = self._prepare(x)
        hs = self._forward_extractor(x)
        locs, confs = self._forward_heads(hs)
        anchors = self.xp.vstack([self.anchors(h.shape[2:] for h in hs)[
                                 self.xp.newaxis] for _ in range(b)])
        return anchors, locs, confs, scales

    def predict(self, x):
        x, scales = self._prepare(x)
        hs = self._forward_extractor(x)
        anchors, locs, confs = self._forward_heads(hs)

        bboxes, labels, scores = self._decode(anchors, confs, locs, scales)
        mask = self._suppressor(bboxes, scores)
        return bboxes[mask], labels[mask], scores[mask]

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
                _anchors.append(self.xp.array(anchor, dtype=np.float32))
            anchors.append(self.xp.vstack(_anchors))
        anchors = self.xp.vstack(anchors)
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

        return x, scales

    def _scale_img(self, img):
        """Process image."""
        _, H, W = img.shape
        scale = self._min_size / min(H, W)
        if scale * max(H, W) > self._max_size:
            scale = self._max_size / max(H, W)
        H, W = int(H * scale), int(W * scale)
        img = transforms.resize(img, (H, W))
        return img, scale

    # TODO: implement
    def _decode(self, anchors, locs, confs, scales):
        raise NotImplementedError()

    # TODO: implement
    def _distribute(self, confs, locs):
        raise NotImplementedError()
