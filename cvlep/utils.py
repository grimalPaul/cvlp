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

def get_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:  # torchrun launch
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif int(os.environ.get('SLURM_NPROCS', 1)) > 1:  # slurm launch
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])
    else:  # single gpu & process launch
        rank = 0
        local_rank = 0
        world_size = 0
    return rank, local_rank, world_size


    
