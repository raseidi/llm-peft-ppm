from typing import List, Tuple, Optional, Dict, Union
from collections import OrderedDict
from pathlib import Path

import torch
from pandas import DataFrame
from torch.utils.data import Dataset

from ppm.datasets.event_logs import EventLog


def _ensure_list(obj: Union[str, List[str], None]) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    return obj


class ContinuousTraces(Dataset):
    def __init__(
        self,
        log: EventLog,
        device: str = "cuda",
        refresh_cache: bool = True,
        torch_dtype=torch.float32,
    ):
        self.log = log
        self.refresh_cache = refresh_cache
        self.torch_dtype = torch_dtype
        self.dataset = self._load_or_build_dataset(log)
        self.to(device)

    def _load_or_build_dataset(self, log: EventLog):
        split = "train" if log.train_split else "test"
        path = Path(f"data/{log.name}/cached_train_test/{split}.pt")

        if path.exists() and not self.refresh_cache:
            return torch.load(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        dataset = self._build_dataset(log)
        torch.save(dataset, path)
        return dataset

    def _build_dataset(self, log: EventLog):
        df = log.dataframe.reset_index(drop=True)
        df["event_id"] = df.index
        self.traces = (
            df.groupby(log.case_id, sort=True)["event_id"].apply(list).values.tolist()
        )

        # convert cases.values to a list of tensors
        self.traces = [torch.tensor(t) for t in self.traces]

        # build matrices of features and targets
        self.cat_features = torch.tensor(
            log.dataframe[log.features.categorical].values, dtype=torch.long
        )
        self.num_features = torch.tensor(
            log.dataframe[log.features.numerical].values, dtype=self.torch_dtype
        )
        self.cat_targets = torch.tensor(
            log.dataframe[log.targets.categorical].values, dtype=torch.long
        )
        self.num_targets = torch.tensor(
            log.dataframe[log.targets.numerical].values, dtype=self.torch_dtype
        )

    def __len__(self):
        """returns the number of cases in the dataset.

        if you need the number of events, use `len(self.cat_features)` instead
        """
        return len(self.traces)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get a case from the dataset.

        This method returns a 4-d tuple of features and targets for a given case index.
        If, e.g., a numerical feature is not defined in the construction method, the method will return a 4-d tuple regardless.
        """
        trace = self.traces[idx]
        return (
            self.cat_features[trace],
            self.num_features[trace],
            self.cat_targets[trace],
            self.num_targets[trace],
        )

    def to(self, device: str):
        if device in ["cuda", "cpu"]:
            for prop, value in self.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(self, prop, value.to(device))
        else:
            raise ValueError("Invalid device. Use 'cuda' or 'cpu'.")


class PrefixNEP(Dataset):
    def __init__(
        self,
        log: EventLog,
        device: str = None,
        prefix_len: int = 5,
    ):
        self.prefix_len = prefix_len
        self._build_dataset(log)
        self.to(device)

    def __getitem__(self, idx: int):
        prefix = self.prefix_traces[idx]
        return (
            self.cat_features[prefix],
            self.num_features[prefix],
            self.cat_targets[prefix],
            self.num_targets[prefix],
        )

    def __len__(self) -> int:
        return len(self.prefix_traces)

    def _build_dataset(self, log: EventLog):
        # add a last row in log.dataframe to represent the padding
        # thus, when we retrieve the prefix id -1, we are getting the padding
        # log.dataframe = log.dataframe.append(
        #     {col: 0 for col in log.dataframe.columns}, ignore_index=True
        # )
        import pandas as pd

        df = log.dataframe.copy()
        df = pd.concat(
            [
                df,
                pd.DataFrame({col: 0 for col in df.columns}, index=[-1]),
            ]
        )

        self.prefix_traces = self._build_prefixes(df)

        # build matrices of features and targets
        self.cat_features = torch.tensor(
            log.dataframe[log.features.categorical].values, dtype=torch.long
        )
        self.num_features = torch.tensor(
            log.dataframe[log.features.numerical].values, dtype=torch.float
        )
        self.cat_targets = torch.tensor(
            log.dataframe[log.targets.categorical].values, dtype=torch.long
        )
        self.num_targets = torch.tensor(
            log.dataframe[log.targets.numerical].values, dtype=torch.float
        )

    def _build_prefixes(self, df: DataFrame):
        import functools, operator

        # df = df.reset_index(drop=True) #
        df["event_id"] = df.index.values

        def fn(x):
            result = []
            x = list(x)
            if len(x) < self.prefix_len:
                for i in range(1, len(x) + 1):
                    prefix = x[:i]
                    padded = prefix + [-1] * (self.prefix_len - len(prefix))
                    result.append(tuple(padded))
            else:
                for i in range(len(x) - self.prefix_len + 1):
                    prefix = x[i : i + self.prefix_len]
                    padded = prefix + [-1] * (self.prefix_len - len(prefix))
                    result.append(tuple(padded))
                for i in range(1, self.prefix_len):
                    prefix = x[:i]
                    padded = prefix + [-1] * (self.prefix_len - len(prefix))
                    result.append(tuple(padded))
            return result

        prefix_traces = (
            df.iloc[:-1]
            .groupby("case_id")["event_id"]  # -1 to exclude the padding
            .apply(fn)
            .tolist()
        )

        prefix_traces = functools.reduce(operator.iconcat, prefix_traces, [])
        # prefix_traces = [torch.tensor(t) for t in prefix_traces]
        prefix_traces = list(map(list, prefix_traces))

        return prefix_traces

    def to(self, device: str):
        if device in ["cuda", "cpu"]:
            for prop, value in self.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(self, prop, value.to(device))
        else:
            raise ValueError("Invalid device. Use 'cuda' or 'cpu'.")


class PrefixSuffix(Dataset):
    def __init__(
        self,
        log: EventLog,
        device: str = None,
        prefix_length: int = 5,
    ):
        self.log = log
        self.prefix_len = prefix_length
        self._build_dataset(log)
        self.to(device)

    def __getitem__(self, idx: int):
        prefix, suffix = self.prefix_ixs[idx], self.suffix_ixs[idx]
        return (
            self.cat_features[prefix],
            self.num_features[prefix],
            self.cat_targets[suffix],
            self.num_targets[suffix],
        )

    def __len__(self) -> int:
        return len(self.prefix_ixs)

    def _build_dataset(self, log: EventLog):
        import pandas as pd

        df = log.dataframe.reset_index(drop=True).copy()
        df["event_id"] = df.index.values
        df = pd.concat(
            [
                df,  # pad row for reference
                pd.DataFrame({col: 0 for col in df.columns}, index=[-1]),
            ]
        )
        self.prefix_ixs, self.suffix_ixs = self._build_ps(df)

        # build matrices of features and targets
        self.cat_features = torch.tensor(
            df[log.features.categorical].values, dtype=torch.long
        )
        self.num_features = torch.tensor(
            df[log.features.numerical].values, dtype=torch.float
        )
        self.cat_targets = torch.tensor(
            df[log.targets.categorical].values, dtype=torch.long
        )
        self.num_targets = torch.tensor(
            df[log.targets.numerical].values, dtype=torch.float
        )

    def _build_ps(self, df: DataFrame):
        """builds prefixes and suffixes indices"""

        def fn(x):
            """
            suffixes are unshifted by one since they are stored in different columns;
            such columns (e.g., next_activity) are shifted by one in the preprocessing step.
            thus, the dataframe {"a": [1,2,3], "na": [2,3,4]} generates
            the prefix=[1,2] with suffix=[3,4]; both accessed by prefix_ix=[0,1] and suffix_ix=[1,2]
            which will result in the same values from the preprocessed dataframe.
            the suffix ix starting from the last prefix ix is confusing, so i'm leaving this note
            until i find a better solution;
            using only the feature column would work if I had an extra preprocessing step to add the eos token;
            this would crash the other tasks though; plus, it wouldn't work for other targets than `next_activity`
            """
            x = list(x)
            list_prefixes = []
            list_suffixes = []
            if len(x) < self.prefix_len:
                for i in range(1, len(x) + 1):
                    # get prefix and suffix
                    prefix = x[:i]
                    suffix = x[i - 1 : self.prefix_len + i - 1]

                    # first events from trace need to be padded for prefixes
                    prefix = prefix + [-1] * (self.prefix_len - len(prefix))
                    suffix = suffix + [-1] * (self.prefix_len - len(suffix))

                    # append prefix and suffix
                    list_prefixes.append(prefix)
                    list_suffixes.append(suffix)
                return list_prefixes, list_suffixes

            # build `prefix_len` first prefixes
            for i in range(1, self.prefix_len):
                # get prefix and suffix
                prefix = x[:i]
                suffix = x[i - 1 : self.prefix_len + i - 1]

                # first events from trace need to be padded for prefixes
                prefix = prefix + [-1] * (self.prefix_len - len(prefix))
                suffix = suffix + [-1] * (self.prefix_len - len(suffix))

                # append prefix and suffix
                list_prefixes.append(prefix)
                list_suffixes.append(suffix)

            # build remaining prefixes
            for i in range(1, len(x) - self.prefix_len + 1):
                # get prefix and suffix
                prefix = x[i : i + self.prefix_len]
                suffix = x[
                    (i + self.prefix_len)
                    - 1 : (i + self.prefix_len + self.prefix_len)
                    - 1
                ]

                # last events from trace need to be padded for suffixes
                suffix = suffix + [-1] * (self.prefix_len - len(suffix))
                list_prefixes.append(prefix)
                list_suffixes.append(suffix)

            return list_prefixes, list_suffixes

        group = df.iloc[:-1].groupby("case_id")["event_id"]
        prefix_ixs, suffix_ixs = [], []
        for i, x in group:
            prefix, suffix = fn(x)
            prefix_ixs.extend(prefix)
            suffix_ixs.extend(suffix)

        return prefix_ixs, suffix_ixs

    def to(self, device: str):
        if device in ["cuda", "cpu"]:
            for prop, value in self.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(self, prop, value.to(device))
        else:
            raise ValueError("Invalid device. Use 'cuda' or 'cpu'.")
