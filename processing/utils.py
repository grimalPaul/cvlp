# coding: utf-8

import hashlib
import argparse
import torch
from cvlep.utils import device
from transformers.tokenization_utils_base import BatchEncoding

def create_kwargs(arg: argparse.Namespace) -> dict:
    kwargs = dict()
    for var, value in arg._get_kwargs():
        kwargs[var] = value
    return kwargs

def md5(string: str) -> str:
    """Utility function. Uses hashlib to compute the md5 sum of a string.
    First encodes the string and utf-8.
    Lastly decodes the hash using hexdigest.
    """
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def json_integer_keys(dictionary):
    """
    Convert all keys of the dictionay to an integer
    (so make sure all of the keys can be casted as integers and remain unique before using this)
    """
    return {int(k): v for k, v in dictionary.items()}