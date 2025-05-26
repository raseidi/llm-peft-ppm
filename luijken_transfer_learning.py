import pprint
import torch
import argparse
import pandas as pd

from typing import Tuple

from torch.utils.data import DataLoader

from peft import LoraConfig, TaskType
from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets

from skpm.event_logs import (
    BPI12,
    BPI17,
    BPI20PrepaidTravelCosts,
    BPI20TravelPermitData,
    BPI20RequestForPayment,
)
from skpm.event_logs.split import unbiased
from skpm.feature_extraction.event import TimestampExtractor

from sklearn.preprocessing import StandardScaler

from ppm.datasets import ContinuousTraces
from ppm.datasets.utils import continuous
from ppm.engine.nep import train_engine
from ppm.models.config import FreezeConfig
from ppm.models import NextEventPredictor

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

EVENT_LOGS = {
    "BPI12": BPI12,
    "BPI17": BPI17,
    "BPI20PrepaidTravelCosts": BPI20PrepaidTravelCosts,
    "BPI20TravelPermitData": BPI20TravelPermitData,
    "BPI20RequestForPayment": BPI20RequestForPayment,
}

NUMERICAL_FEATURES = [
    "accumulated_time",
    "day_of_month",
    "day_of_week",
    "day_of_year",
    "hour_of_day",
    "min_of_hour",
    "month_of_year",
    "sec_of_min",
    "secs_within_day",
    "week_of_year",
]

PRETRAINED_CONFIGS = {
    "gpt2": {
        "name": "openai-community/gpt2",
        "embedding_size": 768,
        "hidden_size": 768,
        "pretrained": False,  # TRAIN FROM SCRATCH
        # "fine_tuning": args.fine_tuning,
        "fine_tuning_module_path": "h",
    },
    "pm-gpt2": {
        "name": "persisted_models/pm-gpt2",
        "embedding_size": 768,
        "hidden_size": 768,
        "pretrained": True,
        # "fine_tuning": args.fine_tuning,
        "fine_tuning_module_path": "h",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BPI12")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--persist_model", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default="multi-task-icpm")

    """ training config """
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--categorical_features", nargs="+", default=None)
    parser.add_argument("--categorical_targets", nargs="+", default=None)
    parser.add_argument("--continuous_features", nargs="+", default=None)
    parser.add_argument("--continuous_targets", nargs="+", default=None)

    """ in layer config """
    parser.add_argument(
        "--strategy", type=str, default="concat", choices=["sum", "concat"]
    )

    """ model config """
    parser.add_argument(
        "--backbone",
        type=str,
        default="rnn",
        choices=["gpt2", "llama32-1b", "qwen25-05b", "rnn"],
    )
    # if rnn
    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument(
        "--rnn_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"]
    )

    """ if fine-tuning """
    parser.add_argument(
        "--fine_tuning", type=str, default=None, choices=["lora", "freeze"]
    )
    # if lora
    parser.add_argument("--r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    # if freeze
    parser.add_argument(
        "--freeze_layers",
        nargs="+",
        type=int,
        default=None,
        help="List of layer indices to freeze. If None, all layers are frozen.",
    )

    return parser.parse_args()


def prepare_data(
    df: pd.DataFrame, unbiased_split_params: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.loc[:, ["case:concept:name", "concept:name", "time:timestamp"]]
    cases_to_drop = df.groupby("case:concept:name").size() > 2
    cases_to_drop = cases_to_drop[cases_to_drop].index
    df = df[df["case:concept:name"].isin(cases_to_drop)]

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    train, test = unbiased(df, **unbiased_split_params)

    time_unit = "d"
    ts = TimestampExtractor(
        case_features=["accumulated_time", "remaining_time"],
        event_features="all",
        time_unit=time_unit,
    )
    train[ts.get_feature_names_out()] = ts.fit_transform(train)
    test[ts.get_feature_names_out()] = ts.transform(test)

    train = train.drop(columns=["time:timestamp", "nano_timestamp"])
    test = test.drop(columns=["time:timestamp", "nano_timestamp"])

    train = train.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )
    test = test.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )

    sc = StandardScaler()
    columns = NUMERICAL_FEATURES + ["remaining_time"]
    # columns = ["accumulated_time", "remaining_time"]
    train.loc[:, columns] = sc.fit_transform(train[columns])
    test.loc[:, columns] = sc.transform(test[columns])

    return train, test


def prepare_pretraining_data():
    df = pd.read_csv("data/Helpdesk/finale.csv")

    df["time:timestamp"] = pd.to_datetime(df["Complete Timestamp"])
    df["case:concept:name"] = df["Case ID"]
    df["concept:name"] = df["Activity"]
    df = df.loc[:, ["case:concept:name", "concept:name", "time:timestamp"]]
    cases_to_drop = df.groupby("case:concept:name").size() > 2
    cases_to_drop = cases_to_drop[cases_to_drop].index
    df = df[df["case:concept:name"].isin(cases_to_drop)]

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])

    # temporal split
    train_cases = df[df["time:timestamp"] < df["time:timestamp"].quantile(0.8)][
        "case:concept:name"
    ].unique()
    test_cases = df[df["time:timestamp"] >= df["time:timestamp"].quantile(0.8)][
        "case:concept:name"
    ].unique()
    train_cases = set(train_cases) - set(test_cases)
    train = df[df["case:concept:name"].isin(train_cases)].reset_index(drop=True).copy()
    test = df[df["case:concept:name"].isin(test_cases)].reset_index(drop=True).copy()
    assert (
        train["case:concept:name"].nunique() + test["case:concept:name"].nunique()
        == df["case:concept:name"].nunique()
    )

    time_unit = "d"
    ts = TimestampExtractor(
        case_features=["accumulated_time", "remaining_time"],
        event_features="all",
        time_unit=time_unit,
    )
    train[ts.get_feature_names_out()] = ts.fit_transform(train)
    test[ts.get_feature_names_out()] = ts.transform(test)

    train = train.drop(columns=["time:timestamp", "nano_timestamp"])
    test = test.drop(columns=["time:timestamp", "nano_timestamp"])

    train = train.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )
    test = test.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )

    sc = StandardScaler()
    columns = NUMERICAL_FEATURES + ["remaining_time"]
    # columns = ["accumulated_time", "remaining_time"]
    train.loc[:, columns] = sc.fit_transform(train[columns])
    test.loc[:, columns] = sc.transform(test[columns])

    return train, test


def get_fine_tuning(fine_tuning, **kwargs):
    if fine_tuning == "lora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=kwargs["r"],
            lora_alpha=kwargs["lora_alpha"],
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "up_proj",
                "down_proj",
                "o_proj",
                "gate_proj",
            ],  
            use_rslora=True,
        )
    elif fine_tuning == "freeze":
        return FreezeConfig(
            ix_layers=kwargs["freeze_layers"],
            module_path=kwargs["fine_tuning_module_path"],
        )
    elif fine_tuning is None:
        return
    else:
        raise ValueError("Invalid fine-tuning strategy")


def get_model_config(train_log: EventLog, training_config: dict):
    pretrained_config = PRETRAINED_CONFIGS.get(training_config["backbone"], {})
    if pretrained_config:
        fine_tuning = get_fine_tuning(
            fine_tuning=training_config["fine_tuning"],
            r=training_config["r"],
            lora_alpha=training_config["lora_alpha"],
            freeze_layers=training_config["freeze_layers"],
            fine_tuning_module_path=pretrained_config["fine_tuning_module_path"],
        )
        pretrained_config["fine_tuning"] = fine_tuning
    if training_config["backbone"] != "rnn":
        backbone_hf_name = pretrained_config["name"]
    else:
        backbone_hf_name = "rnn"
    return {
        "embedding_size": training_config["embedding_size"],
        "categorical_cols": train_log.features.categorical,
        "categorical_sizes": train_log.categorical_sizes,
        "numerical_cols": train_log.features.numerical,
        "categorical_targets": train_log.targets.categorical,
        "numerical_targets": train_log.targets.numerical,
        "padding_idx": train_log.special_tokens["<PAD>"],
        "strategy": training_config["strategy"],
        "backbone_name": backbone_hf_name,
        "backbone_pretrained": False,
        "backbone_finetuning": pretrained_config.get("fine_tuning", None),
        "backbone_type": training_config.get("rnn_type", None),
        "backbone_hidden_size": training_config["hidden_size"],
        "backbone_n_layers": training_config.get("n_layers", None),
        "device": training_config["device"],
    }


def main(training_config: dict):
    if training_config["log"] == "helpdesk":
        train, test = prepare_pretraining_data()
    else:
        log = EVENT_LOGS[training_config["log"]]()
        train, test = prepare_data(log.dataframe, log.unbiased_split_params)

    event_features = EventFeatures(
        categorical=training_config["categorical_features"],
        numerical=training_config["continuous_features"],
    )
    event_targets = EventTargets(
        categorical=training_config["categorical_targets"],
        numerical=training_config["continuous_targets"],
    )

    train_log = EventLog(
        dataframe=train,
        case_id="case_id",
        features=event_features,
        targets=event_targets,
        train_split=True,
        name=training_config["log"],
    )

    test_log = EventLog(
        dataframe=test,
        case_id="case_id",
        features=event_features,
        targets=event_targets,
        train_split=False,
        name=training_config["log"],
        vocabs=train_log.get_vocabs(),
    )

    dataset_device = (
        training_config["device"]
        if training_config["backbone"]
        not in ["gpt2", "llama32-1b", "llama2-7b", "qwen25-05b"]
        else "cpu"
    )

    train_dataset = ContinuousTraces(
        log=train_log,
        refresh_cache=True,
        device=dataset_device,
    )

    test_dataset = ContinuousTraces(
        log=test_log,
        refresh_cache=True,
        device=dataset_device,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        collate_fn=continuous,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=continuous,
    )

    model_config = get_model_config(train_log, training_config)

    model = NextEventPredictor(**model_config).to(device=training_config["device"])

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    # print(
    #     f"trainable params: {trainable_params:,d} || "
    #     f"all params: {all_param:,d} || "
    #     f"trainable%: {100 * trainable_params / all_param:.4f}"
    # )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )

    training_config.update(
        {
            "total_params": all_param,
            "trainable_params": trainable_params,
        }
    )

    use_wandb = training_config.pop("wandb")
    persist_model = training_config.pop("persist_model")
    if use_wandb and WANDB_AVAILABLE:
        if (
            "freeze_layers" in training_config
            and training_config["freeze_layers"] is not None
        ):
            training_config["freeze_layers"] = ",".join(
                [str(i) for i in training_config["freeze_layers"]]
            )
        wandb.init(project=training_config.pop("project_name"), config=training_config)
        wandb.watch(model, log="all")

    train_engine(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        config=training_config,
        use_wandb=use_wandb,
        persist_model=persist_model,
    )
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    model.backbone.save_pretrained("models/pm-gpt2")


if __name__ == "__main__":
    args = parse_args()

    import os

    if not os.path.exists(PRETRAINED_CONFIGS["pm-gpt2"]["name"]):
        # pre-training
        pretraining_config = {
            # args to pop before logging
            "project_name": args.project_name,
            "wandb": False,
            "persist_model": False,
            # args to log
            "log": "helpdesk",
            "device": args.device,
            # architecture
            "backbone": "gpt2",
            "rnn_type": args.rnn_type,
            "embedding_size": PRETRAINED_CONFIGS["gpt2"]["embedding_size"],
            "hidden_size": PRETRAINED_CONFIGS["gpt2"]["hidden_size"],
            "n_layers": args.n_layers,
            # hyperparameters
            "lr": 0.00001,
            "batch_size": 32,
            "weight_decay": 0.01,
            "grad_clip": 1,
            "epochs": 400,
            # fine-tuning
            "fine_tuning": args.fine_tuning,
            "r": args.r,  # LoRA
            "lora_alpha": args.lora_alpha,  # LoRA
            "freeze_layers": args.freeze_layers,  # Freeze
            # features and tasks
            "categorical_features": ["activity"],
            "continuous_features": NUMERICAL_FEATURES,
            "categorical_targets": "activity",
            "continuous_targets": ["remaining_time"],
            "strategy": "concat",
        }

        pprint.pprint(pretraining_config)
        print("=" * 80)
        main(pretraining_config)
