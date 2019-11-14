import os
from configs import path_catalog


def _get_config_name(config_path):
    return os.path.splitext(os.path.basename(config_path))[0]


def get_logdir(config_path):
    logdir = os.path.join(path_catalog.logdir, _get_config_name(config_path))
    return logdir


def get_outdir(config_path):
    outdir = os.path.join(path_catalog.outdir, _get_config_name(config_path))
    return outdir
