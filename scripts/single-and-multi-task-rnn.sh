#bin/bash

source .venv/bin/activate

LR=(0.0005 0.0001 0.00005)
BATCH_SIZES=(32 64 256)
EPOCHS=(25)
INPUT_SIZES=(32 128 256 512)
HIDDEN_SIZES=(128 256 512)
RNN_TYPE=("lstm")
CATEGORICAL_FEATURES=("activity")
NUMERICAL_FEATURES=("all")
CATEGORICAL_TARGETS=("activity" null)
NUMERICAL_TARGETS=("remaining_time" null)
PROJECT="multi-task-icpm"
WEIGHT_DECAY=0.01
STRATEGY=("sum")

declare -a DATASETS=(BPI12 BPI17 BPI20PrepaidTravelCosts BPI20TravelPermitData BPI20RequestForPayment)

# python fetch_wandb.py --project $PROJECT 

for strategy in "${STRATEGY[@]}"
do 
    for lr in "${LR[@]}"
    do
        for batch_size in "${BATCH_SIZES[@]}"
        do
            for epochs in "${EPOCHS[@]}"
            do
                for input_size in "${INPUT_SIZES[@]}"
                do
                    for hidden_size in "${HIDDEN_SIZES[@]}"
                    do
                        for rnn_type in "${RNN_TYPE[@]}"
                        do
                            for categorical_feature in "${CATEGORICAL_FEATURES[@]}"
                            do
                                for numerical_feature in "${NUMERICAL_FEATURES[@]}"
                                do
                                    for categorical_target in "${CATEGORICAL_TARGETS[@]}"
                                    do
                                        for numerical_target in "${NUMERICAL_TARGETS[@]}"
                                        do 
                                            for dataset in "${DATASETS[@]}"
                                            do
                                                # if both targets are null, skip
                                                if [ $categorical_target == "null" ] && [ $numerical_target == "null" ]; then
                                                    continue
                                                fi

                                                cmd="--dataset $dataset \
                                                    --backbone rnn \
                                                    --lr $lr \
                                                    --batch_size $batch_size \
                                                    --epochs $epochs \
                                                    --embedding_size $input_size \
                                                    --hidden_size $hidden_size \
                                                    --weight_decay $WEIGHT_DECAY \
                                                    --grad_clip 5.0 \
                                                    --strategy $strategy"

                                                if [ $categorical_feature != "null" ]; then
                                                    cmd="$cmd --categorical_features $categorical_feature"
                                                fi
                                                if [ $numerical_feature != "null" ]; then
                                                    cmd="$cmd --continuous_features $numerical_feature"
                                                fi
                                                if [ $categorical_target != "null" ]; then
                                                    cmd="$cmd --categorical_targets $categorical_target"
                                                fi
                                                if [ $numerical_target != "null" ]; then
                                                    cmd="$cmd --continuous_targets $numerical_target"
                                                fi
                                                echo $cmd --project_name $PROJECT --wandb
                                                # python main.py $cmd --project_name $PROJECT --wandb
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
