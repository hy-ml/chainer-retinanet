from transforms import Compose, Flip, Normalize, Sacle


def setup_transform(cfg, mean):
    transforms = Compose()
    transforms.append(Flip())
    transforms.append(Sacle(cfg.model.min_size, cfg.model.max_size))
    transforms.append(Normalize(mean))
    return transforms
