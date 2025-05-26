# Fine-Tuning LLMs for Multi-Task Predictive Process Monitoring

*Paper under review.*

## Overview

* This repo has code and scripts to fine-tune large language models (LLMs) for multi-task PPM.
* We use [uv](https://docs.astral.sh/uv/guides/install-python/) to manage our local environment.
* Tested only on Ubuntu 24.04 using python 3.12.

## Requirements

Install all dependencies with:

```bash
uv venv .venvv -python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Scripts and Structure

```
.
├── data/                           # Event logs (automatically downloaded)
├── scripts/                        # Experiment scripts and configs
│   ├── *.sh                        
│   ├── *.txt                       
│   └── *.slurm                     
├── notebooks/                      # Analysis notebooks
├── ppm/                            # Source code
├── luijken_transfer_learning.py    # Main training script
├── next_event_prediction.py        # Competitor training script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Data

We use five public event logs. They will be downloaded via [SkPM](https://skpm.readthedocs.io/en/latest/install/installation.html) under `data/<LOG>/`:

* [BPI20PTC](https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165) (Prepaid Travel Costs)
* [BPI20RfP](https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030) (Request for Payment)
* [BPI20TPD](https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550) (Permit Data)
* [BPI12](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)
* [BPI17](https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11)

## Usage

### Single experiments

**RNN baseline**

```bash
python next_event_prediction.py \
  --dataset BPI20PrepaidTravelCosts \
  --backbone rnn \
  --embedding_size 32 \
  --hidden_size 128 \
  --lr 0.0005 \
  --batch_size 64 \
  --epochs 25 \
  --categorical_features activity \
  --continuous_features all \
  --categorical_targets activity \
  --continuous_targets remaining_time
```

**LLM fine-tuning**

In order to use LLMs, you need a [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens). A few options of how to use it:

* Create an `.env` file in the root of this repository and write your token like `HF_TOKEN=<YOUR_TOKEN>`
* Export a local variable `export HF_TOKEN="<YOUR_TOKEN>"`
* Hard code it [here](https://github.com/raseidi/llm-peft-ppm/blob/61d92a1aba81fd710bf37851862e3386b41816c4/ppm/models/models.py#L13)

For local debugging purposes, try the tiny setup below with a small `r` value for `BPI20PrepaidTravelCosts` and `qwen25-05b`. If it doens't fit your GPU memory, keep decreasing the `batch_size` (=4 uses less than 2gb). 

```bash
python next_event_prediction.py \
  --dataset BPI20PrepaidTravelCosts \
  --backbone qwen25-05b \
  --embedding_size 896 \
  --hidden_size 896 \
  --lr 0.00005 \
  --batch_size 64 \
  --epochs 1 \
  --categorical_features activity \
  --continuous_features all \
  --categorical_targets activity \
  --continuous_targets remaining_time \
  --fine_tuning lora \
  --r 2 \
  --lora_alpha 4
```

Alternatively, use the argument `--wandb` to enable wandb.

### Hyperparameter search

We used Slurm on our HPC clusters. Check `scripts/*.sh`, `scripts/*.txt`, and `scripts/*.slurm` to see how to reproduce our jobs.

## Results

All metrics and analysis notebooks are in the `notebooks/` folder.

## Contact

For questions or feedback, reach me at [rafael.oyamada@kuleuven.be](mailto:rafael.oyamada@kuleuven.be) or open an issue here.
