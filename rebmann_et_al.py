"""Code adapted from 

Evaluating the Ability of LLMs to Solve Semantics-Aware Process Mining Tasks
https://github.com/fdschmidt93/trident-bpm

by Adrian Rebmann, Fabian David Schmidt, Goran Glavaš, and Han van der Aa
"""

from pathlib import Path
import string
import pickle
import torch.nn as nn        

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from ppm import EVENT_LOGS
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Semantic-Aware Next Activity Prediction")
    parser.add_argument(
        "--dataset",
        type=str,
        default="BPI20PrepaidTravelCosts",
        choices=list(EVENT_LOGS.keys()),
        help="Dataset to use for training and evaluation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model to use for next activity prediction.",
    )
    return parser.parse_args()

MISTRAL_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
MODELS=[
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct", 
    "mistralai-Mistral-7B-Instruct-v0.2"
]
UNK_TKN = 1

BPI12_dutch_to_english = {
    "W_Afhandelen leads": "W_Handling leads",
    "W_Beoordelen fraude": "W_Assessing fraud",
    "W_Completeren aanvraag": "W_Completing application",
    "W_Nabellen incomplete dossiers": "W_Following up on incomplete files (by phone)",
    "W_Nabellen offertes": "W_Following up on quotes (by phone)",
    "W_Valideren aanvraag": "W_Validating application",
    "W_Wijzigen contractgegevens": "W_Changing contract details",
}

def setify(x: str):
    set_: set[str] = eval(x)
    assert isinstance(set_, set), f"Conversion failed for {x}"
    return set_


def convert_next_label(line: dict):
    if not line["next"] == "[END]":
        return list(line["unique_activities"]).index(line["next"]) + 1
    else:
        return 0

def create_prefixes(group):
    """Create prefixes for each activity in a case trace"""
    activities = group['activity'].tolist()
    prefixes = []
    
    for i in range(len(activities)):
        prefix = activities[:i+1]  # Include current activity and all previous
        prefixes.append({
            'model_id': group['case_id'].iloc[i],
            'revision_id': group['case_id'].iloc[i],  
            'prefix': prefix,                                                   # list
            'next': activities[i+1] if i+1 < len(activities) else '[END]',
            'unique_activities': set(activities),                               # set
            'trace': group['activity'].tolist()                                 # list
        })
    
    return pd.DataFrame(prefixes)

def _load_prefixes(df):
    eval_prefix = df.groupby('case_id').apply(create_prefixes, include_groups=True).reset_index(drop=True)
    
    eval_prefix["prefix"] = eval_prefix["prefix"].apply(lambda x: tuple(x))
    # eval_prefix.unique_activities = eval_prefix.unique_activities.apply(setify)

    eval_prefix["prefix"] = eval_prefix["prefix"].apply(lambda x: list(x))
    mask = ~(eval_prefix.next == "[END]")
    eval_prefix["labels"] = eval_prefix.apply(convert_next_label, axis=1)
    columns = [
        "model_id",
        "revision_id",
        "trace",
        "prefix",
        "next",
        "unique_activities",
        "labels",
    ]
    eval_prefix = eval_prefix.loc[:, columns]
    eval_prefix = eval_prefix.loc[mask]
    # train, val, test = split_by_model(eval_prefix)
    
    return Dataset.from_pandas(eval_prefix)

def load_prefixes(dataset, split: str = "train"):
    train_snap_dataset = Path(f"data/{dataset}/train_snap_dataset")
    test_snap_dataset = Path(f"data/{dataset}/test_snap_dataset")
    
    if train_snap_dataset.exists():
        from datasets import load_from_disk
        train = load_from_disk(train_snap_dataset)
        test = load_from_disk(test_snap_dataset)
        return train, test
    
    from ppm.datasets.preprocess import prepare_data 
    log = EVENT_LOGS[dataset](cache_folder="data/")
    if dataset == "BPI12":
        log.dataframe['concept:name'] = log.dataframe['concept:name'].replace(BPI12_dutch_to_english)
        
    train, test = prepare_data(log.dataframe, log.unbiased_split_params)    # df = log.dataframe
    
    # cases_to_drop = df.groupby("case:concept:name").size() > 2
    # cases_to_drop = cases_to_drop[cases_to_drop].index
    # df = df[df["case:concept:name"].isin(cases_to_drop)]
    
    train, test = _load_prefixes(train), _load_prefixes(test)
    # df = df[['case:concept:name', 'concept:name', 'time:timestamp']]
    # df = df.rename(columns={
    #     'case:concept:name': 'case_id',
    #     'concept:name': 'activity',
    #     'time:timestamp': 'timestamp'
    # })
    # df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    # df = df.sort_values(by=['case_id', 'timestamp'])

    # df = df[df.case_id.isin(df.case_id.unique()[:10])] # debugging

    # convert each row to a list of prefixes that contain all the previous activities

    # Apply the function to each case
    # eval_prefix = df.groupby('case_id').apply(create_prefixes, include_groups=True).reset_index(drop=True)
    
    # eval_prefix["prefix"] = eval_prefix["prefix"].apply(lambda x: tuple(x))
    # # eval_prefix.unique_activities = eval_prefix.unique_activities.apply(setify)

    # eval_prefix["prefix"] = eval_prefix["prefix"].apply(lambda x: list(x))
    # mask = ~(eval_prefix.next == "[END]")
    # eval_prefix["labels"] = eval_prefix.apply(convert_next_label, axis=1)
    # columns = [
    #     "model_id",
    #     "revision_id",
    #     "trace",
    #     "prefix",
    #     "next",
    #     "unique_activities",
    #     "labels",
    # ]
    # eval_prefix = eval_prefix.loc[:, columns]
    # eval_prefix = eval_prefix.loc[mask]
    # # train, val, test = split_by_model(eval_prefix)
    
    # return Dataset.from_pandas(eval_prefix), n_labels
    train.save_to_disk(train_snap_dataset)
    test.save_to_disk(test_snap_dataset)
    return train, test

# train, test = load_prefixes("BPI20RequestForPayment")

def preprocess_next_activity(
    examples: dict, tokenizer: PreTrainedTokenizerFast, unique_activities_: dict[str, int]
) -> BatchEncoding:
    # unique_activities: list[set[str]]
    # prefix: list[tuple[str]]
    # labels: list[string]

    # List of activities:
    # A. Activity
    # B. Activity
    # C. Activity
    # Which one of the above activities should follow the below sequence of activities?
    # Sequence of activites: []
    # Answer: A
    # trace: list[tuple[str]]
    # label: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for prefix_ in examples["prefix"]:
        string_ = "List of activites:\n0. [END]\n"
        for activity, ix in unique_activities_.items():
            string_ += f"{str(ix).upper()}. {activity}\n"

        string_ += "Which one of the above activities should follow the below sequence of activities?\n"
        string_ += f"Sequence of activities: {[p.capitalize() for p in prefix_]}\n"
        string_ += "Answer: "
        inputs.append(string_)

    batch = tokenizer(inputs)
    
    # batch["labels"] = torch.LongTensor([i for i, label in enumerate(unique_activities_)]) 
    batch["labels"] = torch.LongTensor([unique_activities_.get(activity.capitalize(), UNK_TKN) for activity in examples["next"]])
    batch["next"] = examples["next"]
    return batch


def collate_next_activity_pred(
    examples: list[dict], tokenizer: PreTrainedTokenizerFast
):
    input_ids = [{"input_ids": line["input_ids"]} for line in examples]
    batch = tokenizer.pad(
        input_ids, padding=True, return_tensors="pt", return_attention_mask=True
    )
    batch["next"] = [line["next"] for line in examples]
    batch["labels"] = torch.LongTensor([line["labels"] for line in examples])
    return batch

class SNAP(torch.nn.Module):
        def __init__(self, model: str, label_tokens: list[str]):
            super().__init__()
            self.backbone = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, use_cache=False)
            self.lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules="all-linear",
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.backbone = get_peft_model(self.backbone, self.lora_config)
            
            
            self.backbone.lm_head = torch.nn.Identity()
            
            label_tokens = [str(l) for l in label_tokens]
            ids = [tkn(token, add_special_tokens=False)["input_ids"][0] for token in label_tokens]
            embeddings = self.backbone.get_output_embeddings()
            
            self.clf_head = nn.Parameter(embeddings.weight[ids].clone().T)
            self.clf_head.requires_grad = False
        
        def forward(self, batch, attention_mask):
            outputs = self.backbone(
                input_ids=batch, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            # Get hidden states from the last layer
            hidden_states = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)
            
            last_token_ids = (attention_mask.sum(1) - 1).long()
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_token_states_ND = hidden_states[batch_indices, last_token_ids, :]
            
            logits_NC = last_token_states_ND @ self.clf_head
            return logits_NC
        
if __name__ == "__main__":
    args = parse_args()
    
    
    train, test = load_prefixes(args.dataset)
    # train[0]['trace']
    # train[0]['prefix']
    # train[0]['next']
    # train['labels']
    # len(train)
    # type(train)

    tkn = AutoTokenizer.from_pretrained(args.model)
    tkn.pad_token_id = tkn.eos_token_id

    unique_activities_ = {"UNK": UNK_TKN}
    sorted_acts = sorted(list(set(train["next"])))
    unique_activities_.update({activity.capitalize(): i+2 for i, activity in enumerate(sorted_acts)})
    # batch = preprocess_next_activity(df, tkn) # batch is actually the dataset
    train_dataset = train.map(
        preprocess_next_activity, batched=True, fn_kwargs={"tokenizer": tkn, "unique_activities_": unique_activities_}, 
    )
    # train_dataset['input_ids']
    # train_dataset['labels']
    # train_dataset.data['labels']
    test_dataset = test.map(
        preprocess_next_activity, batched=True, fn_kwargs={"tokenizer": tkn, "unique_activities_": unique_activities_},
    )



    from torch.utils.data import DataLoader
    from functools import partial

    collate = partial(
        collate_next_activity_pred,
        tokenizer=tkn,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16, # like original paper
        collate_fn=collate,
        shuffle=True,
    )
    x = next(iter(train_loader))
    x['input_ids'].shape, x['attention_mask'].shape, x['labels'].shape
    print(tkn.decode(x['input_ids'][0], skip_special_tokens=True))

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=collate,
        shuffle=False,
    )

    label_tokens = list(unique_activities_.keys()) + ["[END]"]
    model = SNAP(
        model=args.model, 
        label_tokens=label_tokens
    )
    
    # from ppm.metrics.benchmark import count_parameters
    # trainable, total = count_parameters(model)
    # print(f"\"{args.dataset}\", {trainable}, {total}")
    # import sys
    # sys.exit(0)

    model = model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) 

    from torchmetrics.functional.classification import accuracy
    import wandb

    wandb.init(project="ml4pm-prompt", config={"dataset": args.dataset, "model": args.model})
    wandb.watch(model)

    train_epoch_loss = []
    test_epoch_loss = []
    
    n_labels = len(label_tokens)
    for epoch in range(3):
        # Training
        train_epoch_loss.append(0.0)
        train_batches = 0
        train_accuracy = 0.0

        model.train()
        for batch in train_loader:
            batch = batch.to("cuda")
            optimizer.zero_grad()
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="mean")
            
            acc = accuracy(
                logits, 
                batch["labels"], 
                num_classes=n_labels, 
                task="multiclass",
                average="micro"
            )
            loss.backward()
            optimizer.step()
            
            train_epoch_loss[-1] += loss.item()
            train_accuracy += acc.item()
            train_batches += 1
            # break
        
        avg_train_loss = train_epoch_loss[-1] / train_batches
        avg_train_acc = train_accuracy / train_batches
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}")
        
        # Testing
        test_epoch_loss.append(0.0)
        test_batches = 0
        test_accuracy = 0.0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="mean")
                    
                acc = accuracy(
                    logits, 
                    batch["labels"], 
                    num_classes=n_labels, 
                    task="multiclass",
                    average="micro"
                )
                test_epoch_loss[-1] += loss.item()
                test_accuracy += acc.item()
                test_batches += 1
                # break
        
        avg_test_loss = test_epoch_loss[-1] / test_batches
        avg_test_acc = test_accuracy / test_batches
        print(f"Epoch {epoch + 1} - Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.4f}")

        wandb.log({
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "test_loss": avg_test_loss,
            "test_accuracy": avg_test_acc,
        })

    # PERSIST THE MODEL
    # torch.save(model.state_dict(), "model.pth")