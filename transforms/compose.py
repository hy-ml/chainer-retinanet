class Compose(object):
    def __init__(self, transforms=None):
        assert type(transforms) == list or transforms is None
        if transforms is None:
            self._transforms = []
        else:
            self._transforms = transforms

    def __call__(self, in_data):
        for t in self._transforms:
            in_data = t(in_data)
        return in_data

    def append(self, transform):
        self._transforms.append(transform)
