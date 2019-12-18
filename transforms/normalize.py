class Normalize(object):
    def __init__(self, mean):
        self._mean = mean

    def __call__(self, in_data):
        img, bbox, label = in_data
        img = in_data[0]
        img -= self._mean
        return img, bbox, label
