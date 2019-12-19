from transforms import Compose, Flip, Normalize, Sacle, ConvertToFloat16


def setup_transform(cfg, mean):
    transforms = Compose()
    transforms.append(Flip())
    transforms.append(Sacle(cfg.model.min_size, cfg.model.max_size))
    transforms.append(Normalize(mean))
    if cfg.dtype == 'mixed16':
        transforms.append(ConvertToFloat16())
    return transforms
