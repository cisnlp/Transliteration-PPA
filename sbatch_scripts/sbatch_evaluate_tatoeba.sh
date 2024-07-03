#!/bin/bash

#SBATCH --job-name=evaluate_tatoeba
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output /dss/dsshome1/06/di35ver/logs/%j.out
#SBATCH --error /dss/dsshome1/06/di35ver/logs/%j.err

MODEL="xlm-roberta-base"
MODEL_TYPE="xlmr"

MAX_LENGTH=512
BATCH_SIZE=128
DIM=768
NLAYER=12
LAYER=7

DATA_DIR="/mnt/data/retrieval_tatoeba/"
OUTPUT_DIR="/mnt/models_evaluation/retrieval_tatoeba/"
tokenized_dir="/mnt/models_evaluation/retrieval_tatoeba_tokenized"
init_checkpoint="/mnt/models_group2_mnt_TCM_pool_weight_1.0_tlm"

srun --container-image=/dss/dsshome1/06/di35ver/containers/my_torch_container.sqsh \
  --container-mounts=/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/orxhelili:/mnt \
  torchrun --standalone --nproc_per_node=1 /dss/dsshome1/06/di35ver/evaluation/retrieval/evaluate_retrieval_all_tatoeba.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --embed_size $DIM \
    --batch_size $BATCH_SIZE \
    --max_seq_len $MAX_LENGTH \
    --num_layers $NLAYER \
    --dist cosine $LC \
    --specific_layer $LAYER \
    --tokenized_dir $tokenized_dir \
    --init_checkpoint $init_checkpoint
