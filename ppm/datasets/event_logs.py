import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple, Union

from pandas import DataFrame
import pandas as pd  # TODO handle warning
import pprint

# pd.set_option("future.no_silent_downcasting", True)


@dataclass
class EventAttributes:
    """Event features

    This class is used to store the features' names.

    Args:
        categorical (str, List[str]): a list of categorical features
        continuous (str, List[str]): a list of continuous features
    """

    numerical: Union[str, List[str]] = None
    categorical: Union[str, List[str]] = None

    def __post_init__(self):
        # enforce attributes to be lists
        for attr in self.__dict__.keys():
            if getattr(self, attr) is None:
                setattr(self, attr, [])
            elif isinstance(getattr(self, attr), str):
                setattr(self, attr, [getattr(self, attr)])
            elif isinstance(getattr(self, attr), list):
                pass
            else:
                raise ValueError(
                    f"Invalid type for {attr}. Must be a string, a list, or None."
                )

    @property
    def total_numerical(self):
        return len(self.numerical)

    @property
    def total_categorical(self):
        return len(self.categorical)

    @property
    def n_features(self):
        return self.total_numerical + self.total_categorical


@dataclass
class EventFeatures(EventAttributes):
    """Event features

    This class is used to store the features' names.

    Args:
        categorical (str, List[str]): a list of categorical features
        continuous (str, List[str]): a list of continuous features
    """

    ...


@dataclass
class EventTargets(EventAttributes):
    """Event targets

    This class is used to store the targets' names.

    Args:
        categorical (str, List[str]): a list of categorical targets
        continuous (str, List[str]): a list of continuous targets
    """

    ...


@dataclass
class EventLog:
    """Event log for deep learning models.

    Validate case identifier, features, and targets. Set vocabularies for categorical features. Define special tokens and the split type (train/test).

    Args:
        case_id (str): the case identifier
        events (List[EventFeatures]): a list of events
        targets (List[EventTargets]): a list of targets
    """

    dataframe: DataFrame = field(repr=False)
    case_id: str = field(repr=False)
    features: EventFeatures
    targets: EventTargets
    name: str = "event_log"
    train_split: bool = True
    vocabs: Tuple[dict, dict] = field(default=None, repr=False)
    stoi: dict = field(default=None, repr=False)
    itos: dict = field(default=None, repr=False)
    special_tokens: dict = field(
        default_factory=lambda: {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}, repr=False
    )

    def __post_init__(self):
        self._cases = None
        self._validate_columns()
        self._set_vocabs()
        self._encode_categorical_features()
        self._set_categorical_sizes()
        self._ensure_dtypes()
        self._set_targets()

    def __len__(self):
        return len(self.cases)

    def _ensure_dtypes(self):
        for f in set(self.features.numerical + self.targets.numerical):
            self.dataframe[f] = self.dataframe[f].astype("float32")

        for f in set(self.features.categorical + self.targets.categorical):
            self.dataframe[f] = self.dataframe[f].astype("int32")

    def _encode_categorical_features(self):
        for cat in set(self.features.categorical + self.targets.categorical):

            self.dataframe.loc[:, cat] = self.dataframe.loc[:, cat].map(
                self.stoi[cat.replace("next_", "")], na_action="ignore"
            )  # replace operation ensures that the next_activity is encoded using the same stoi as activity
            self.dataframe[cat] = self.dataframe[cat].infer_objects()
            self.dataframe[cat] = self.dataframe[cat].fillna(
                self.special_tokens["<UNK>"]
            )

    def _validate_columns(self):
        """Check if case_id, features, and targets are columns in the `self.dataframe`. Excludes all other columns."""

        columns = {self.case_id}

        def add_columns(nested_lists):
            for item in nested_lists:
                columns.update(set(item))

        add_columns(vars(self.features).values())
        add_columns(vars(self.targets).values())

        missing_cols = columns - set(self.dataframe.columns)
        assert not missing_cols, f"Columns {missing_cols} not found in the log."
        assert len(self) > 0, "No cases found in the log."

        self.dataframe = self.dataframe.loc[:, list(columns)]

    def get_vocabs(self) -> Tuple[dict, dict]:
        """This method is necessary for the test set to
        use the same vocab as the training set"""
        return (self.stoi, self.itos)

    def _set_vocabs(self):
        if self.vocabs is not None:
            self.stoi, self.itos = self.vocabs
            return

        if not self.train_split:
            warnings.warn(
                "Vocabularies will be inferred from the test set but it should be set from the training set."
            )

        self.stoi = OrderedDict()
        self.itos = OrderedDict()
        # Process each categorical feature
        for feature in self.features.categorical:
            unique_values = sorted(self.dataframe[feature].unique().tolist())

            self.stoi[feature] = self.special_tokens.copy()
            self.itos[feature] = {
                idx: token for token, idx in self.special_tokens.items()
            }

            for idx, value in enumerate(unique_values, start=len(self.special_tokens)):
                self.stoi[feature][value] = idx
                self.itos[feature][idx] = value

        self.vocabs = (self.stoi, self.itos)

    def _set_categorical_sizes(self):
        self.categorical_sizes = {k: len(v) for k, v in self.stoi.items()}

    def _set_targets(self):
        """Target should be a column in the dataframe to be shifted (next event prediction)."""
        new_cat_target_names = []
        new_num_target_names = []
        for target in self.targets.categorical + self.targets.numerical:
            if target not in self.dataframe.columns:
                raise ValueError(f"Target {target} not found in the log.")
            else:
                if target in self.targets.categorical:
                    new_cat_target_names.append(f"next_{target}")
                    special_token = self.special_tokens["<EOS>"]
                    self.dataframe[f"next_{target}"] = self.dataframe.groupby(
                        "case_id", observed=True, as_index=False
                    )[target].shift(-1, fill_value=special_token)
                else:
                    new_num_target_names.append(f"next_{target}")
                    special_token = 0.0
                    # it's not actually next, but I'll fix later (the models sees the `next_` prefix)
                    self.dataframe[f"next_{target}"] = self.dataframe[target]

        self.targets = EventTargets(
            numerical=new_num_target_names, categorical=new_cat_target_names
        )

    @property
    def cases(self) -> List[str]:
        if self._cases is None:
            self._cases = self.dataframe[self.case_id].unique().tolist()
        return self._cases
