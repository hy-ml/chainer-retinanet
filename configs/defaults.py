from yacs.config import CfgNode as CN


_C = CN()

# model
_C.model = CN()
_C.model.type = ''
_C.model.pretrained_model = 'imagenet'
_C.model.min_size = 800
_C.model.max_size = 1333

# loss
_C.model.loc_loss = 'SmoothL1'
_C.model.conf_loss = 'FocalLoss'
_C.model.focal_loss_alpha = 0.25
_C.model.focal_loss_gamma = 2.0
_C.model.fg_thresh = 0.5
_C.model.bg_thresh = 0.4

# suppression
_C.model.suppressor = 'NMS'
_C.model.nms_thresh = 0.5
_C.model.freeze_param = [
    '/.+/bn',
    '/extractor/base/conv1',
    '/extractor/base/res2'
]

# dataset
_C.dataset = CN()
_C.dataset.train = ''
_C.dataset.eval = ''
_C.dataset.n_fg_class = 0

# solver
_C.solver = CN()
_C.solver.optimizer = 'MomentumSGD'
_C.solver.base_lr = 0.00125  # 0.02 / 16
_C.solver.weight_decay = 0.0001
_C.solver.momentum = 0.9
_C.solver.hooks = ['WeightDecay', 'GradientClipping']
_C.solver.n_iteration = 90000
_C.solver.lr_step = [60000, 80000]
# when you use gradient clip, you must add GradientClipping to _C.solver.hooks
_C.solver.gradient_clipping_thresh = 5

# misc
_C.n_gpu = 8
_C.n_sample_per_gpu = 2
_C.n_worker = 4
_C.dtype = 'float32'  # only support float32 or mixed16
_C.cudnn_fast_batch_normalization = True
_C.workspace_size = 512  # MB
_C.autotune = True
