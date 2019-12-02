import os
import filelock
from chainer import serializers

from configs.path_catalog import gdrive_ids
from utils.path import get_outdir
from utils.download_file_from_gdrive import download_file_from_gdrive


def load_pretrained_model(cfg, config_path, model, pretrained_model):
    if pretrained_model == 'auto':
        lock_file = './.pretrained_model.lock'
        save_dir = os.path.join('./out/download', cfg.dataset.train)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        pretrained_model = os.path.join(save_dir, cfg.model.type)
        with filelock.FileLock(lock_file):
            if not os.path.isfile(pretrained_model):
                gdrive_id = gdrive_ids[cfg.dataset.train][cfg.model.type]
                download_file_from_gdrive(gdrive_id, pretrained_model)
        if os.path.isfile(lock_file):
            os.remove(lock_file)
    elif pretrained_model:
        pretrained_model = pretrained_model
    else:
        pretrained_model = os.path.join(get_outdir(
            config_path), 'model_iter_{}'.format(cfg.solver.n_iteration))
    serializers.load_npz(pretrained_model, model)
