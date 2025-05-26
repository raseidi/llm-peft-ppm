#bin/bash

source .venv/bin/activate

PROJECT="multi-task-icpm"
LR=(0.00005)
BATCH_SIZE=8
EPOCHS=10
R=(256)
FINE_TUNING=("lora" "freeze")
declare -a FREEZE_LAYERS=(
    "-1 -2"
    "0 1"
    "0"
    "-1"
    null
)

declare -a DATASETS=(BPI12 BPI17 BPI20PrepaidTravelCosts BPI20TravelPermitData BPI20RequestForPayment)

# python fetch_wandb.py --project $PROJECT

for dataset in "${DATASETS[@]}"
do
    for lr in "${LR[@]}"
    do
        for fine_tuning in "${FINE_TUNING[@]}"
        do
            cmd="--dataset $dataset \
                --backbone pm-gpt2 \
                --embedding_size 768 \
                --hidden_size 768 \
                --categorical_features activity \
                --categorical_targets activity \
                --continuous_features all \
                --continuous_targets remaining_time \
                --strategy concat \
                --lr $lr \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --fine_tuning $fine_tuning \
                --project_name $PROJECT"

            if [[ $fine_tuning == "lora" ]]; then
                for r in "${R[@]}"
                do
                    cmd2="$cmd \
                    --r $r \
                    --lora_alpha $(( r*2 ))"
                    python main.py $cmd2 --wandb
                    echo $cmd2 --wandb
                done
            else
                for freeze_layers in "${FREEZE_LAYERS[@]}"
                do
                    if [ "$freeze_layers" == "null" ]; then
                        cmd2=$cmd
                    else
                        cmd2="$cmd --freeze_layers $freeze_layers"
                    fi
                    echo $cmd2 --wandb
                    python main.py $cmd2 --wandb
                done
            fi
        done
    done
done



