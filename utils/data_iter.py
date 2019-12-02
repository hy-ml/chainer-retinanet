import os
from glob import glob
import numpy as np
import cv2


def _preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    return img


class WebCamIter(object):
    def __init__(self):
        self._cap = None

    def start_device(self):
        self._cap = cv2.VideoCapture(0)

    def stop_device(self):
        self._cap.release()
        del self._cap
        self._cap = None

    def __iter__(self):
        return self

    def __next__(self):
        _, img = self._cap.read()
        img = _preprocess(img)
        return img


class DirectoryIter(object):
    def __init__(self, indir):
        self._inpaths = glob(os.path.join(indir, '*'))
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        img = cv2.imread(self._inpaths[self._i])
        img = _preprocess(img)
        self._i += 1
        return img
