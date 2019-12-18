from chainercv import transforms


class Flip(object):
    def __call__(self, in_data):
        img, bbox, label = in_data
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        x_flip = params['x_flip']
        bbox = transforms.flip_bbox(
            bbox, img.shape[1:], x_flip=x_flip)
        return img, bbox, label
