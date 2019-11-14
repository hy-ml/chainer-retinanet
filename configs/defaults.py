from yacs.config import CfgNode as CN


_C = CN()

# model
_C.model = CN()
_C.model.type = ''
_C.model.pretrained_model = 'imagenet'
_C.model.min_size = 800
_C.model.max_size = 1333
_C.model.suppressor = 'NMS'
_C.model.loc_loss = 'SmoothL1'
_C.model.conf_loss = 'FocalLoss'
_C.model.fg_thresh = 0.5
_C.model.bg_thresh = 0.4

# dataset
_C.dataset = CN()
_C.dataset.train = ''
_C.dataset.val = ''
_C.dataset.n_fg_class = 0

# solver
_C.solver.optimizer = 'MomuntumSGD'
_C.solver.base_lr = 0.0025  # 0.02 / 8
_C.solver.weight_decay = 0.0001
_C.solver.hooks = ['WeightDecay']
_C.solver.n_iteration = 90000
_C.solver.lr_steps = [60000, 80000]

# misc
_C.n_gpu = 2
_C.n_sample_per_gpu = 4
