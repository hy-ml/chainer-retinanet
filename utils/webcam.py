
import cv2


class WebCam(object):
    def __init__(self):
        self._cap = None

    def start_device(self):
        self._cap = cv2.VideoCapture(0)

    def stop_device(self):
        self._cap.release()
        del self._cap
        self._cap = None

    def __next__(self):
        _, frame = self._cap.read()
        return frame
