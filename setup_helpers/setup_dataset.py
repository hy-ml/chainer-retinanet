from chainer.datasets import ConcatenatedDataset
# from datasets import COCOBboxDataset
from chainercv.datasets import COCOBboxDataset
# from chainercv.datasets import VOCBboxDataset
from datasets import VOCBboxDataset

from configs.path_catalog import coco_dir


def setup_dataset(cfg, split):
    if split == 'train':
        dataset_type = cfg.dataset.train
    elif split == 'eval':
        dataset_type = cfg.dataset.eval
    else:
        raise ValueError()

    if dataset_type == 'COCO':
        if split == 'train':
            dataset = COCOBboxDataset(split='train', year='2017')
        elif split == 'eval':
            dataset = COCOBboxDataset(
                split='val', year='2017', use_crowded=True,
                return_area=True, return_crowded=True)
        else:
            raise ValueError()
    elif dataset_type == 'VOC':
        if split == 'train':
            dataset = ConcatenatedDataset(
                VOCBboxDataset(year='2007', split='trainval'),
                VOCBboxDataset(year='2012', split='trainval')
            )
        elif split == 'eval':
            dataset = VOCBboxDataset(
                split='test', year='2007', use_difficult=True,
                return_difficult=True)
        else:
            raise ValueError()
    else:
        raise ValueError()

    return dataset
