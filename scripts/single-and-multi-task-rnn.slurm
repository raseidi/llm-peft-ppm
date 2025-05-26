#!/usr/bin/env bash
#SBATCH --nodes="1"
#SBATCH --ntasks="1"
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node="1"
#SBATCH --cluster="genius"
#SBATCH --mem="16G"
#SBATCH --time="06:00:00"
#SBATCH --partition="gpu_p100"

#SBATCH --account="lp_lirisnlp"
#SBATCH --mail-type="BEGIN,END,FAIL"
#SBATCH --mail-user="rafael.oyamada@kuleuven.be"
#SBATCH --output="/vsc-hard-mounts/leuven-data/376/vsc37611/repos/ppm-llm/log_slurm/%a_%x.o%A"
#SBATCH --error="/vsc-hard-mounts/leuven-data/376/vsc37611/repos/ppm-llm/log_slurm/%a_%x.e%A"

# loading necessary modules and variables
module purge
module load cluster/genius/gpu_p100
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.4.0

source ~/.bashrc

cd $VSC_DATA/repos/ppm-llm/

source .venv/bin/activate
wandb login $WANDB_API_KEY

ARGUMENTS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/rnn_params.txt)

python next_event_prediction.py $ARGUMENTS