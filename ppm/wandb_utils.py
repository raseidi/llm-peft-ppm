import os
import ast
import numpy as np
import pandas as pd


def safe_convert(x):
    """Safely convert string representation of lists to actual lists"""
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x  # return original value if conversion fails


def read_local_experiments(project="cosmo-ltl") -> pd.DataFrame:
    path = project + "_experiments.csv"
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except:
            return None
    else:
        return fetch_experiments(project)


def fetch_experiments(
    project="multi-task-benchmark", include_metrics=False
) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    runs = api.runs("raseidi/" + project)

    experiments = pd.DataFrame()
    for r in runs:
        if r.state != "finished":
            continue

        new = pd.DataFrame([r.config])
        new["id"] = r.id
        new["name"] = r.name

        if include_metrics:
            metrics = {
                m
                for m in r.summary.keys()
                if not m.startswith(("parameters/", "gradients/"))
            }

            # format timestamp from number to datetime
            pd.to_datetime(r.summary["_timestamp"], unit="s")

            metrics = {m: r.summary[m] for m in metrics if m not in ["_wandb"]}
            new = pd.concat((new, pd.DataFrame([metrics])), axis=1)

        experiments = pd.concat((experiments, new), ignore_index=True)

    experiments.reset_index(inplace=True, drop=True)
    experiments.to_csv(project + "_experiments.csv", index=False)
    return experiments.reset_index(drop=True)


# log_config = {'log': 'BPI12', 'backbone': 'rnn', 'rnn_type': 'lstm', 'embedding_size': 32, 'hidden_size': 128, 'n_layers': 2, 'lr': 0.0005, 'batch_size': 32, 'fine_tuning': 'lora', 'r': 2, 'lora_alpha': 32, 'cat_features': ['activity'], 'num_features': None, 'cat_targets': ['activity'], 'num_targets': ['execution_time'], 'strategy': 'sum'}


def is_duplicate(log_config, project="multi-task-benchmark") -> bool:
    list_columns = ["cat_features", "num_features", "cat_targets", "num_targets"]
    experiments = read_local_experiments(project=project)

    # Convert string representation of lists to actual lists
    experiments[list_columns] = experiments[list_columns].map(safe_convert)
    # Replace NaNs with None to match the `log_config` format
    experiments[list_columns] = experiments[list_columns].replace({np.nan: None})
    # Drop unique columns from wandb
    experiments = experiments.drop(
        columns=["id", "name", "total_params", "trainable_params"], errors="ignore"
    )
    experiments = experiments[
        (experiments["log"] == log_config["log"])
        & (experiments["backbone"] == log_config["backbone"])
    ]

    # Check if the columns from log_config and wandb match
    # assert set(experiments.columns) == set(log_config.keys())
    for _, run in experiments.iterrows():
        if run.to_dict() == log_config:
            return True

    return False
