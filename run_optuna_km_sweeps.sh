#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="${LOG_DIR:-$(pwd)/logs/optuna}"
mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
enz_log="$LOG_DIR/${timestamp}_eitlem_km_enz_0.09.log"
sub_log="$LOG_DIR/${timestamp}_eitlem_km_sub_0.45.log"

echo "Writing enzyme sweep log to: $enz_log"
python emulator_bench/launch_parallel_optuna.py \
  --gpus 0 1 \
  --trials_per_gpu 3 \
  --base_dir ~/github/EMULaToR/data/processed/baselines/EITLEM-Kinetics \
  --embeddings_dir ~/github/EMULaToR/data/processed/baselines/EITLEM-Kinetics/embeddings \
  --split_groups enzyme_sequence_splits \
  --threshold threshold_0.09 \
  --predictor_type km \
  --mol_type MACCSKeys \
  --device cuda:0 \
  --cache_device cuda:0 \
  --metric rmse \
  --eval_split val \
  --epochs 25 \
  --n_trials 30 \
  --num_workers 4 \
  --persistent_workers \
  --pin_memory \
  --batch_size 256 \
  --storage sqlite:////home/da24s023/github/EMULaToR/data/processed/baselines/EITLEM-Kinetics/optuna_studies/eitlem_km_enz_0.09.db \
  2>&1 | tee "$enz_log"

echo "Writing substrate sweep log to: $sub_log"
python emulator_bench/launch_parallel_optuna.py \
  --gpus 0 1 \
  --trials_per_gpu 3 \
  --base_dir ~/github/EMULaToR/data/processed/baselines/EITLEM-Kinetics \
  --embeddings_dir ~/github/EMULaToR/data/processed/baselines/EITLEM-Kinetics/embeddings \
  --split_groups substrate_splits \
  --threshold threshold_0.45 \
  --predictor_type km \
  --mol_type MACCSKeys \
  --device cuda:0 \
  --cache_device cuda:0 \
  --metric rmse \
  --eval_split val \
  --epochs 25 \
  --n_trials 30 \
  --num_workers 4 \
  --persistent_workers \
  --pin_memory \
  --batch_size 256 \
  --storage sqlite:////home/da24s023/github/EMULaToR/data/processed/baselines/EITLEM-Kinetics/optuna_studies/eitlem_km_sub_0.45.db \
  2>&1 | tee "$sub_log"

echo "Finished. Logs:"
echo "  $enz_log"
echo "  $sub_log"
