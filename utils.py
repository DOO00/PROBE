import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(labels, probs):
    """Compute AUROC, AUPRC, RecK in percentage."""
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.detach().cpu().numpy()

        labels = np.asarray(labels).astype(np.int64)
        probs = np.asarray(probs).astype(np.float64)

        score["AUROC"] = float(roc_auc_score(labels, probs) * 100)
        score["AUPRC"] = float(average_precision_score(labels, probs) * 100)
        k = int(labels.sum())

    if k <= 0:
        score["RecK"] = 0.0
    else:
        score["RecK"] = float(labels[probs.argsort()[-k:]].sum() / labels.sum() * 100)
    return score


def get_training_config(dataset, config_path="semi_train.conf.yaml"):
    with open(config_path, "r", encoding="utf-8") as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    default_config = full_config["default"]
    dataset_config = full_config.get(dataset, {})
    dataset_config = dict(default_config, **dataset_config)
    return dataset_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(filename, log_level=1, name=None, mode="a"):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[log_level])

    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(str(file_path), mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

