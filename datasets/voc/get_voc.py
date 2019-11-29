import os
import filelock
import yaml
from chainercv import utils

from configs.path_catalog import voc_dir

flag_file = '.downloaded.yaml'
urls = {
    '2012': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'
    'VOCtrainval_11-May-2012.tar',
    '2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
    'VOCtrainval_06-Nov-2007.tar',
    '2007_test': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
    'VOCtest_06-Nov-2007.tar'
}


def get_voc(year, split):
    if year not in urls:
        raise ValueError
    key = year

    with open(flag_file, 'r') as fp:
        flag = yaml.load(fp)
        if flag['VOC'][year][split]:
            return voc_dir

    if split == 'test' and year == '2007':
        key = '2007_test'

    if not os.path.isdir(voc_dir):
        os.makedirs(voc_dir)

    # To support ChainerMN, the target directory should be locked.
    with filelock.FileLock(os.path.join(voc_dir, 'voc.lock')):
        base_path = os.path.join(voc_dir, 'VOCdevkit/VOC{}'.format(year))
        split_file = os.path.join(
            base_path, 'ImageSets/Main/{}.txt'.format(split))
        if os.path.exists(split_file):
            # skip downloading
            return base_path

        download_file_path = utils.cached_download(urls[key])
        ext = os.path.splitext(urls[key])[1]
        utils.extractall(download_file_path, voc_dir, ext)

    with open(flag_file, 'w') as fp:
        flag['VOC'][year][split] = True
        yaml.dump(flag, fp)
    return base_path
