import os
from configs import path_catalog


def _get_config_name(config_path):
    config_path = os.path.splitext(config_path)[0]
    config_name = config_path.split('/')[-2:]
    config_name = os.path.join(*config_name)
    return config_name


def get_logdir(config_path):
    logdir = os.path.join(path_catalog.logdir, _get_config_name(config_path))
    return logdir


def get_outdir(config_path):
    outdir = os.path.join(path_catalog.outdir, _get_config_name(config_path))
    return outdir
