#!/bin/bash

augmentations=("na" "shuffle" "jit_scal" "perm_jit" "resample" "noise" "scale" "negate" "t_flip" "rotation" "hfc" "lfc" "p_shift" "ap_p" "ap_f" "perm_harnet" "t_warp_harnet")

combinations=()

for ((i = 0; i < ${#augmentations[@]}; i++)); do
  for ((j = i + 1; j < ${#augmentations[@]}; j++)); do
    combo="['${augmentations[i]}', '${augmentations[j]}']"
    combinations+=("$combo")
  done
done

for combo in "${combinations[@]}"; do
   sbatch --time=24:00:00 --gpus-per-node=1 --cpus-per-task=1 --mem-per-cpu=32G --wrap="uv run main.py task='train_ssl' feature_extractor.augmentations=\"\$combo\" feature_extractor/ml_config=matrix_comp"
done