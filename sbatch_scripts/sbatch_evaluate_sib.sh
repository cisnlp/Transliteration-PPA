#!/bin/bash

#SBATCH --job-name=evaluate_sib
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

max_checkpoint_num=52000
min_checkpoint_num=52000
source_language="eng_Latn,yue_Hani,kor_Hang,tha_Thai"

OUTPUT_DIR="/mnt/models_evaluation/sib200/"
DATA_DIR="/mnt/data/sib-200"
init_checkpoint="/mnt/models_group2_mnt_TCM_pool_weight_1.0_tlm"

srun --container-image=/dss/dsshome1/06/di35ver/containers/my_torch_container.sqsh \
  --container-mounts=/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/orxhelili:/mnt \
  torchrun --standalone --nproc_per_node=1 /dss/dsshome1/06/di35ver/evaluation/sib200/evaluate_sib.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --epochs 40 \
    --nr_of_seeds 5 \
    --min_checkpoint_num $min_checkpoint_num \
    --max_checkpoint_num $max_checkpoint_num \
    --init_checkpoint $init_checkpoint \
    --data_dir $DATA_DIR \
    --source_language $source_language
