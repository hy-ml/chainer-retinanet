import numpy as np


class ConvertToFloat16(object):
    def __call__(self, in_data):
        img, bbox, label = in_data
        img = img.astype(np.float16)
        bbox = bbox.astype(np.float16)
        label = label.astype(np.float16)
        return img, bbox, label
