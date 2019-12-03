import filelock
import os
from chainercv import utils

from configs.path_catalog import coco_dir


img_urls = {
    '2014': {
        'train': 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
        'val': 'http://msvocds.blob.core.windows.net/coco2014/val2014.zip'
    },
    '2017': {
        'train': 'http://images.cocodataset.org/zips/train2017.zip',
        'val': 'http://images.cocodataset.org/zips/val2017.zip'
    }
}
instances_anno_urls = {
    '2014': {
        'train': 'http://msvocds.blob.core.windows.net/annotations-1-0-3/'
        'instances_train-val2014.zip',
        'val': 'http://msvocds.blob.core.windows.net/annotations-1-0-3/'
        'instances_train-val2014.zip',
        'valminusminival': 'https://dl.dropboxusercontent.com/s/'
        's3tw5zcg7395368/instances_valminusminival2014.json.zip',
        'minival': 'https://dl.dropboxusercontent.com/s/o43o90bna78omob/'
        'instances_minival2014.json.zip'
    },
    '2017': {
        'train': 'http://images.cocodataset.org/annotations/'
        'annotations_trainval2017.zip',
        'val': 'http://images.cocodataset.org/annotations/'
        'annotations_trainval2017.zip'
    }
}


panoptic_anno_url = 'http://images.cocodataset.org/annotations/' +\
    'panoptic_annotations_trainval2017.zip'


def get_coco(split, img_split, year, mode):

    if not os.path.isdir(coco_dir):
        os.makedirs(coco_dir)

    # To support ChainerMN, the target directory should be locked.
    lockfile_path = os.path.join(coco_dir, 'coco.lock')
    with filelock.FileLock(lockfile_path):
        annos_root = os.path.join(coco_dir, 'annotations')
        img_root = os.path.join(coco_dir, 'images')
        created_img_root = os.path.join(
            img_root, '{}{}'.format(img_split, year))
        img_url = img_urls[year][img_split]
        if mode == 'instances':
            anno_url = instances_anno_urls[year][split]
            anno_path = os.path.join(
                annos_root, 'instances_{}{}.json'.format(split, year))
        elif mode == 'panoptic':
            anno_url = panoptic_anno_url
            anno_path = os.path.join(
                annos_root, 'panoptic_{}{}.json'.format(split, year))

        if not os.path.exists(created_img_root):
            download_file_path = utils.cached_download(img_url)
            ext = os.path.splitext(img_url)[1]
            utils.extractall(download_file_path, img_root, ext)
        if not os.path.exists(anno_path):
            download_file_path = utils.cached_download(anno_url)
            ext = os.path.splitext(anno_url)[1]
            if split in ['train', 'val']:
                utils.extractall(download_file_path, coco_dir, ext)
            elif split in ['valminusminival', 'minival']:
                utils.extractall(download_file_path, annos_root, ext)

        if mode == 'panoptic':
            pixelmap_path = os.path.join(
                annos_root, 'panoptic_{}{}'.format(split, year))
            if not os.path.exists(pixelmap_path):
                utils.extractall(pixelmap_path + '.zip', annos_root, '.zip')

    if os.path.isfile(lockfile_path):
        os.remove(lockfile_path)  # remove lockfile
    return coco_dir
