#!/usr/bin/env bash
#SBATCH --nodes="1"
#SBATCH --ntasks="1"
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node="1"
#SBATCH --cluster="genius"
#SBATCH --mem="32G"
#SBATCH --time="8:00:00"
#SBATCH --partition="gpu_v100"

#SBATCH --account="intro_vsc37611"
#SBATCH --mail-type="BEGIN,END,FAIL"
#SBATCH --mail-user="rafael.oyamada@kuleuven.be"
#SBATCH --output="/vsc-hard-mounts/leuven-data/376/vsc37611/repos/ppm-llm/log_slurm/%a_%x.o%A"
#SBATCH --error="/vsc-hard-mounts/leuven-data/376/vsc37611/repos/ppm-llm/log_slurm/%a_%x.e%A"

# loading necessary modules and variables
module purge
module load cluster/genius/gpu_v100
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.4.0

source ~/.bashrc

cd $VSC_DATA/repos/ppm-llm/

source .venv/bin/activate
wandb login $WANDB_API_KEY

python main.py --project_name ppm-llm-confidence --r 256 --lr 5e-05 --dataset BPI12 --backbone llama32-1b --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 8 --lora_alpha 512 --fine_tuning lora --hidden_size 2048 --weight_decay 0.1 --embedding_size 2048 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --r 256 --lr 5e-05 --dataset BPI12 --backbone qwen25-05b --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 8 --lora_alpha 256 --fine_tuning lora --hidden_size 896 --weight_decay 0.1 --embedding_size 896 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --lr 0.0001 --dataset BPI12 --backbone rnn --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 256 --hidden_size 512 --weight_decay 0.01 --embedding_size 256 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --r 256 --lr 5e-05 --dataset BPI17 --backbone llama32-1b --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 8 --lora_alpha 256 --fine_tuning lora --hidden_size 2048 --weight_decay 0.1 --embedding_size 2048 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --r 256 --lr 5e-05 --dataset BPI17 --backbone qwen25-05b --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 8 --lora_alpha 512 --fine_tuning lora --hidden_size 896 --weight_decay 0.1 --embedding_size 896 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --lr 0.0001 --dataset BPI17 --backbone rnn --n_layers 1 --rnn_type lstm --strategy concat --grad_clip 5 --batch_size 256 --hidden_size 128 --weight_decay 0.01 --embedding_size 32 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --r 256 --lr 5e-05 --dataset BPI20PrepaidTravelCosts --backbone llama32-1b --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 8 --lora_alpha 256 --fine_tuning lora --hidden_size 2048 --weight_decay 0.1 --embedding_size 2048 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --r 256 --lr 5e-05 --dataset BPI20PrepaidTravelCosts --backbone qwen25-05b --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 8 --lora_alpha 512 --fine_tuning lora --hidden_size 896 --weight_decay 0.1 --embedding_size 896 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --lr 0.0001 --dataset BPI20PrepaidTravelCosts --backbone rnn --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 32 --hidden_size 128 --weight_decay 0.01 --embedding_size 32 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --lr 5e-05 --dataset BPI20RequestForPayment --backbone llama32-1b --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 8 --fine_tuning freeze --hidden_size 2048 --weight_decay 0.1 --embedding_size 2048 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --lr 5e-05 --dataset BPI20RequestForPayment --backbone qwen25-05b --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 8 --fine_tuning freeze --hidden_size 896 --weight_decay 0.1 --embedding_size 896 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --lr 0.0001 --dataset BPI20RequestForPayment --backbone rnn --n_layers 1 --rnn_type lstm --strategy concat --grad_clip 5 --batch_size 256 --hidden_size 512 --weight_decay 0.01 --embedding_size 32 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --r 256 --lr 5e-05 --dataset BPI20TravelPermitData --backbone llama32-1b --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 8 --lora_alpha 512 --fine_tuning lora --hidden_size 2048 --weight_decay 0.1 --embedding_size 2048 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --r 256 --lr 5e-05 --dataset BPI20TravelPermitData --backbone qwen25-05b --n_layers 1 --rnn_type lstm --strategy sum --grad_clip 5 --batch_size 8 --lora_alpha 512 --fine_tuning lora --hidden_size 896 --weight_decay 0.1 --embedding_size 896 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb
python main.py --project_name ppm-llm-confidence --lr 0.0001 --dataset BPI20TravelPermitData --backbone rnn --n_layers 1 --rnn_type lstm --strategy concat --grad_clip 5 --batch_size 256 --hidden_size 512 --weight_decay 0.01 --embedding_size 32 --continuous_targets remaining_time --categorical_targets activity --continuous_features all --categorical_features activity --persist --wandb