from configs.path_catalog import coco_dir
from datasets import COCOBboxDataset
from chainercv.datasets import VOCBboxDataset


def setup_dataset(cfg, split):
    if split == 'train':
        dataset_type = cfg.dataset.train
    elif split == 'val':
        dataset_type = cfg.dataset.val
    else:
        raise ValueError

    if dataset_type == 'COCO':
        dataset = COCOBboxDataset(coco_dir, split)
    elif dataset_type == 'VOC':
        dataset = VOCBboxDataset(split=split)
    else:
        raise ValueError()

    return dataset
