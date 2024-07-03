#!/bin/bash

#SBATCH --job-name=transliteration_training
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output /dss/dsshome1/06/di35ver/logs/%j.out
#SBATCH --error /dss/dsshome1/06/di35ver/logs/%j.err

srun --container-image=/dss/dsshome1/06/di35ver/containers/my_torch_container.sqsh \
  --container-mounts=/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/orxhelili:/mnt \
  torchrun --standalone --nproc_per_node=4 /dss/dsshome1/06/di35ver/run_finetune.py \
  --model_name_or_path cis-lmu/glot500-base \
  --tokenizer_name cis-lmu/glot500-base \
  --output_dir /mnt/models_group2 \
  --cache_dir /mnt/models_cache \
  --transliteration_train_file /mnt/data/text_transliterations_group2.csv \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --fp16 True \
  --do_train \
  --num_train_epochs 8 \
  --save_steps 2000 \
  --ddp_timeout 259200 \
  --preprocessing_num_workers 16 \
  --transliteration_loss TCM \
  --contrast_layer 8 \
  --train_cls False \
  --logging_steps 100 \
  --learning_rate 2e-5 \
  --ddp_find_unused_parameters False \
  --remove_unused_columns False \
  --tcm_loss_weight 1.0 \
  --use_contrastive True \
  --use_lm True \
  --use_tlm True
