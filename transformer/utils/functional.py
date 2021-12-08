import json
from datetime import datetime
from functools import wraps
from typing import Tuple

import torch
#
#
#  -------- get_device -----------
#
from torch import Tensor


def get_device() -> torch.device:
    """
    Returns best possible computing device.

    :return: string (cuda, cpu)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
#
#  -------- generate_square_subsequent_mask -----------
#
def generate_square_subsequent_mask(size: int) -> Tensor:
    mask = (torch.triu(torch.ones((size, size), device=get_device())) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


#
#
#  -------- create_mask -----------
#
def create_mask(src: Tensor, tgt: Tensor, pad_idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=get_device())

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


#
#
#  -------- create_mask -----------
#
def sequential_transforms(*transforms: callable) -> callable:
    # helper function to club together sequential operations
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


#
#
# -------- time_track -----------
#
def time_track(func: callable) -> callable:
    @wraps(func)
    def wrap(*args, **kw):
        t_begin = datetime.now()
        result = func(*args, **kw)
        t_end = datetime.now()

        print(
            f"[--- TIMETRACK || method: {func.__name__} -- time: {t_end - t_begin} ---]"
        )

        return result

    return wrap


#
#
#  -------- load_json -----------
#
def load_json(path: str) -> dict:
    """Load JSON configuration file."""
    with open(path) as data:
        return json.load(data)
