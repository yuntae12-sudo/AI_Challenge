#!/usr/bin/env bash
set -e

cd ~/AI_Challenge/V-Max

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python vmax/scripts/evaluate/evaluate.py --waymo_dataset=true --path_dataset=/mnt/e/AI_Challenge/datasets/splits/secondary_val_1024/secondary_val_1024.tfrecord@1024 --sdc_actor=ai --path_model=sec1024_v17_1538560 --batch_size=1 --render=true --sdc_pov=true --eval_name=failure_analysis --scenario_indexes 758 1118 1945 2538 1785 37 2548 2543 2707 2580 1679 1597 2278 1565 1635
