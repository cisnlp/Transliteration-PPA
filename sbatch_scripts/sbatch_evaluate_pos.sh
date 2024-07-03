#!/bin/bash

#SBATCH --job-name=evaluate_pos
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output /dss/dsshome1/06/di35ver/logs/%j.out
#SBATCH --error /dss/dsshome1/06/di35ver/logs/%j.err

MODEL="xlm-roberta-base"
MODEL_TYPE="xlmr"
NUM_EPOCHS=10
LR=2e-5
BATCH_SIZE=32
GRAD_ACC=2
MAX_LENGTH=256
max_checkpoint_num=52000
min_checkpoint_num=52000
train_langs="eng_Latn,zho_Hani,kor_Hang"

DATA_DIR="/mnt/data/pos/"
OUTPUT_DIR="/mnt/models_evaluation/pos/"
tokenized_dir="/mnt/models_evaluation/pos_tokenized"
init_checkpoint="/mnt/models_group2_mnt_TCM_pool_weight_1.0_tlm"

srun --container-image=/dss/dsshome1/06/di35ver/containers/my_torch_container.sqsh \
  --container-mounts=/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/orxhelili:/mnt \
  torchrun --standalone --nproc_per_node=1 /dss/dsshome1/06/di35ver/evaluation/tagging/evaluate_all_pos.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --data_dir $DATA_DIR \
    --labels $DATA_DIR/labels.txt \
    --output_dir $OUTPUT_DIR \
    --max_seq_len $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --per_gpu_eval_batch_size 16 \
    --save_steps 500 \
    --nr_of_seeds 5 \
    --learning_rate $LR \
    --do_train \
    --do_eval \
    --do_predict \
    --train_langs $train_langs \
    --eval_all_checkpoints \
    --eval_patience -1 \
    --overwrite_output_dir \
    --save_only_best_checkpoint \
    --max_checkpoint_num $max_checkpoint_num \
    --min_checkpoint_num $min_checkpoint_num \
    --tokenized_dir $tokenized_dir \
    --init_checkpoint $init_checkpoint
