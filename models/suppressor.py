from chainercv.utils import non_maximum_suppression


class NMS(object):
    def __init__(self, thresh):
        self._thresh = thresh

    def __call__(self, bbox, score):
        selc = non_maximum_suppression(bbox, self._thresh, score)
        return selc


# TODO: implement
class SoftNMS(object):
    def __call__(self):
        raise NotImplementedError()
