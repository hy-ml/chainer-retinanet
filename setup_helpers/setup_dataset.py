from configs.path_catalog import coco_dir
# from chainercv.datasets import COCOBboxDataset
from datasets import COCOBboxDataset


def setup_dataset(cfg, split):
    if split == 'train':
        dataset_type = cfg.dataset.train
    elif split == 'val':
        dataset_type = cfg.dataset.val
    else:
        raise ValueError

    if dataset_type == 'COCO':
        dataset = COCOBboxDataset(coco_dir, split)
    else:
        raise ValueError()

    return dataset
