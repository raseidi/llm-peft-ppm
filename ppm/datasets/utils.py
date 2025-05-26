import os
import pandas as pd
import torch
from functools import wraps
from torch.nn.utils.rnn import pad_sequence

PATH_ROOT = "data/"


def clear_cache(log=None):
    if log:
        pass
    # remove every "cached_log.pkl" file
    for root, dirs, files in os.walk(PATH_ROOT):
        for file in files:
            if file == "cached_log.pkl":
                os.remove(os.path.join(root, file))


def read_log(path):
    """Read log from path, either .pkl, .csv, or .xes.

    The method ensures that the log is cached as a .pkl file."""
    path = os.path.join(path, "log.pkl")
    if os.path.exists(path):
        log = pd.read_pickle(path)
        return log
    elif os.path.exists(path.replace(".pkl", ".csv")):
        log = pd.read_csv(path.replace(".pkl", ".csv"))
    elif os.path.exists(path.replace(".pkl", ".xes")):
        # log = pm4py.read_xes(path.replace(".pkl", ".xes"))
        raise NotImplementedError("XES format not implemented.")
    else:
        raise ValueError("Log not found at path: {}".format(path))

    return log


# def cache(file_name="cached_log", format=".pkl", fn_name=None):
def cache(fn):
    """Cache the log dataframe

    Check if a preprocessed log dataframe exists in the cache directory.
    If it does, load it. If it doesn't, preprocess the log dataframe and save it to the cache directory.

    Args:
        fn (function): function that returns a log dataframe

    Returns:
        function: decorator wrapper function that caches the log dataframe
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        cache_path = os.path.join("data", f"{fn.__name__}", "cached_log.pkl")
        if os.path.exists(cache_path):
            log = pd.read_pickle(cache_path)
        else:
            log = fn(*args, **kwargs)
            log.to_pickle(cache_path)

        return log.copy()

    return wrapper


def ensure_dir(dir_name: str):
    """Creates folder if it does not exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def continuous(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    def _pad_batch(b):
        return torch.nn.utils.rnn.pad_sequence(b, batch_first=True, padding_value=0)

    # categorical/numerical features/targets
    # each variable is a list of tensors
    cf, nf, ct, nt = zip(*batch)

    # pad batch
    # if the dim=1 is empty, then it is None (no act labels)
    cf = _pad_batch(cf) if cf[0].shape[1] else torch.tensor([])
    nf = _pad_batch(nf) if nf[0].shape[1] else torch.tensor([])
    ct = _pad_batch(ct) if ct[0].shape[1] else torch.tensor([])
    nt = _pad_batch(nt) if nt[0].shape[1] else torch.tensor([])

    return cf, nf, ct, nt


def prefix(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    cf, nf, ct, nt = zip(*batch)

    # stack
    cf = torch.stack(cf)
    nf = torch.stack(nf)
    ct = torch.stack(ct)
    nt = torch.stack(nt)

    mask = ct != 0
    last_nonzero_indices = mask.int().cumsum(1).argmax(1)

    batch_indices = torch.arange(ct.size(0)).unsqueeze(1).expand(-1, ct.size(2))
    embed_indices = torch.arange(ct.size(2)).unsqueeze(0).expand(ct.size(0), -1)
    ct = ct[batch_indices, last_nonzero_indices, embed_indices]
    nt = nt[batch_indices, last_nonzero_indices, embed_indices]

    return cf, nf, ct, nt


def get_collate_fn(sequence_encoding: str = "continuous"):
    if sequence_encoding == "continuous":
        return continuous
    elif sequence_encoding == "prefix":  # for NEP only; not for suffix prediction
        return None
    else:
        raise ValueError(
            f"Unknown sequence encoding: {sequence_encoding}. "
            f"Expected 'continuous' or 'prefix'."
        )
