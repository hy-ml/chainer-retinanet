import os

import six

from chainer import reporter
from chainer.training import extension
from chainer.training import trigger as trigger_module
from tensorboardX import SummaryWriter


class LogTensorboard(extension.Extension):
    def __init__(self, y_keys, x_key='iteration', trigger=(1, 'epoch'),
                 log_dir=None):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self._log_dir = log_dir

        self._writer = SummaryWriter(log_dir)

        self._x_key = x_key
        if isinstance(y_keys, str):
            y_keys = (y_keys,)

        self._y_keys = y_keys
        self._trigger = trigger_module.get_trigger(trigger)
        self._init_summary()
        self._data = {k: [] for k in y_keys}

    def __call__(self, trainer):
        keys = self._y_keys
        observation = trainer.observation
        summary = self._summary

        if keys is None:
            summary.add(observation)
        else:
            summary.add({k: observation[k] for k in keys if k in observation})

        if self._trigger(trainer):
            stats = self._summary.compute_mean()
            stats_cpu = {}
            for name, value in six.iteritems(stats):
                stats_cpu[name] = float(value)  # copy to CPU

            updater = trainer.updater

            if self._x_key == 'epoch':
                for k, v in stats_cpu.items():
                    self._writer.add_scalar(k, v, updater.epoch)
            elif self._x_key == 'iteration':
                for k, v in stats_cpu.items():
                    self._writer.add_scalar(k, v, updater.iteration)

            self._init_summary()

    def _init_summary(self):
        self._summary = reporter.DictSummary()
