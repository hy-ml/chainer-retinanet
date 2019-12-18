from chainercv.links.model.fpn.misc import scale_img


class Sacle(object):
    def __init__(self, min_size, max_size):
        self._min_size = min_size
        self._max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        img, scale = scale_img(
            img, self._min_size, self._max_size)
        bbox = bbox * scale
        return img, bbox, label
