import torch
import logging
import re
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def get_config(config_path:str) -> dict:
    """load config from jsonfile

    Args:
        config_path (str): _description_

    Returns:
        dict: config in a dict format
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def retrieval(eval_prediction, ignore_index=-100):
    """
    Computes metric for retrieval training (at the batch-level)
    
    Parameters
    ----------
    eval_prediction: EvalPrediction (dict-like)
        predictions: np.ndarray
            shape (dataset_size, N*M)
            This corresponds to the log-probability of the relevant passages per batch (N*M == batch size)
        label_ids: np.ndarray
            shape (dataset_size, )
            Label at the batch-level (each value should be included in [0, N-1] inclusive)
    ignore_index: int, optional
        Labels with this value are not taken into account when computing metrics.
        Defaults to -100
    """
    # convert to numpy
    eval_prediction['predictions'] = eval_prediction['predictions'].cpu().numpy()
    eval_prediction['label_ids'] = eval_prediction['label_ids'].cpu().numpy()
    print(f"eval_prediction.predictions.shape: {eval_prediction['predictions'].shape}")
    print(f"               .label_ids.shape: {eval_prediction['label_ids'].shape}")
    metrics = {}

    log_probs = eval_prediction['predictions']
    dataset_size, N_times_M = log_probs.shape
    # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)
    rankings = (-log_probs).argsort(axis=1)
    mrr, ignored_predictions = 0, 0
    for ranking, label in zip(rankings, eval_prediction['label_ids']):
        if label == ignore_index:
            ignored_predictions += 1
            continue
        # +1 to count from 1 instead of 0
        rank = (ranking == label).nonzero()[0].item() + 1
        mrr += 1/rank
    mrr /= (dataset_size-ignored_predictions)
    # print(f"dataset_size: {dataset_size}, ignored_predictions: {ignored_predictions}")
    metrics["MRR@N*M"] = mrr

    # argmax to get index of prediction (equivalent to `log_probs.argmax(axis=1)`)
    predictions = rankings[:, 0]
    # print(f"predictions[:100] {predictions.shape}:\n{predictions[:100]}")
    # print(f"eval_prediction.label_ids[:100] {eval_prediction.label_ids.shape}:\n{eval_prediction.label_ids[:100]}")
    # hits@1
    where = eval_prediction['label_ids'] != ignore_index
    metrics["hits@1"] = (predictions[where] == eval_prediction['label_ids'][where]).mean()

    return metrics

