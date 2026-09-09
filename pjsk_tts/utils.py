import json
import os

import torch
from loguru import logger


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    target = model.module if hasattr(model, "module") else model
    # strict=False + 参数化模块内置的 load 钩子：可自动转换旧版 weight_norm
    # 的 weight_g/weight_v 键名；缺失/多余键记录到日志而非静默跳过。
    result = target.load_state_dict(saved_state_dict, strict=False)
    for k in result.missing_keys:
        logger.warning(f"checkpoint 缺少参数，保留模型当前值: {k}")
    for k in result.unexpected_keys:
        logger.info(f"忽略 checkpoint 中多余的参数: {k}")
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration


def get_hparams_from_file(config_path):
    with open(config_path) as f:
        config = json.loads(f.read())
    return HParams(**config)


class HParams:
    # 常用配置节声明，供静态类型检查识别动态属性
    model: "HParams"
    data: "HParams"
    train: "HParams"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
