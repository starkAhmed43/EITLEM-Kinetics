# emulator_bench

This bench adds a split-driven retraining workflow to `EITLEM-Kinetics` without touching the original `Code/` training path.

It is designed for EMULaToR-style train/val/test split trees under:

- `/home/mallikarjun/github/EMULaToR/data/processed/baselines/EITLEM-Kinetics`

You can override that with `--base_dir`.

## What The Model Uses

Inputs per sample:

- Protein sequence from the split file column `sequence`
- Substrate SMILES from the split file column `smiles`
- Target value from `log10_value` by default

Bench input encoding:

- Protein input uses per-residue `ESM1v` embeddings from `esm1v_t33_650M_UR90S_1`
- Molecule input uses a cached fingerprint generated from SMILES
- Default molecular representation is `MACCSKeys`
- Optional molecular representations are `ECFP` and `RDKIT`

Model path:

- The bench reuses the original repo modules from [Code/KCM.py](/home/mallikarjun/github/EITLEM-Kinetics/Code/KCM.py), [Code/KMP.py](/home/mallikarjun/github/EITLEM-Kinetics/Code/KMP.py), and [Code/ensemble.py](/home/mallikarjun/github/EITLEM-Kinetics/Code/ensemble.py)
- The default bench setting is direct single-target retraining with the original EITLEM architecture defaults: `hidden_dim=512`, `protein_dim=1280`, `layer=10`, `dropout=0.5`, `att_layer=10`

## How Embeddings Are Cached

Protein cache:

- One file per normalized sequence
- Stored under `embeddings/proteins/<hash-prefix>/<hash>.npz`
- Cache payload stores the ESM1v per-token embedding and sequence length
- Default cache dtype is `float16` to reduce disk and I/O cost

Ligand cache:

- One file per SMILES and fingerprint configuration
- Stored under `embeddings/ligands/<mol_type>/<hash-prefix>/<hash>.npz`
- Cache payload stores the precomputed fingerprint vector

Cache behavior:

- Embeddings are only computed when the cache file does not already exist
- Repeated training runs, Optuna trials, and multi-GPU retrains all reuse the same cache
- Long protein sequences are handled by a sliding-window ESM extraction path instead of failing on sequence length

## Bench Scripts

Core scripts:

- `cache_embeddings.py`: scans the split tree and builds reusable protein and ligand caches
- `train_single_target_tvt.py`: trains on one explicit train/val/test split and writes checkpoints, metrics, and predictions
- `predict_single_target.py`: runs inference from a saved bench checkpoint

Benchmark and tuning scripts:

- `run_split_benchmarks.py`: sequential benchmark runner across discovered split jobs
- `tune_optuna.py`: tunes only retraining-safe optimization hyperparameters
- `launch_parallel_optuna.py`: launches multiple single-GPU Optuna workers against one shared study
- `launch_parallel_retrain_from_optuna.py`: retrains many split jobs in parallel across multiple GPUs from the best Optuna result

Parallel Optuna note:

- `launch_parallel_optuna.py` supports `--trials_per_gpu` for multiple concurrent workers per GPU
- Example: `--gpus 0 1 --trials_per_gpu 3` launches 6 Optuna workers at once against the same shared study
- `--n_trials` remains the total trial budget across all workers, not per worker

Parallel retrain note:

- `launch_parallel_retrain_from_optuna.py` also supports `--trials_per_gpu`
- Example: `--gpus 0 1 --trials_per_gpu 2` lets 4 discovered retrain jobs run at once
- The script first discovers jobs from the requested split groups and thresholds, then drains them across all GPU worker slots

## Enhancements In This Bench

Compared with the original repo training flow, this bench adds:

- Explicit train/val/test split loading from parquet or CSV
- One-time reusable embedding and fingerprint caching
- Automatic mixed precision:
  `bf16` on Ampere-or-newer CUDA devices, otherwise `fp16`
- TF32 enabled for CUDA matmul and cuDNN where available
- Faster repeated runs by removing repeated ESM extraction and repeated RDKit fingerprint generation
- Direct checkpointed TVT training with `final_results_train.csv`, `final_results_val.csv`, `final_results_test.csv`, and prediction CSVs
- Optuna tuning restricted to optimization hyperparameters so the core model definition stays fixed
- Multi-GPU parallel retraining from the best study result

## Notes

- This bench intentionally focuses on direct retraining from explicit split files. It does not try to reproduce the original repo's full iterative KCAT/KM/KKM transfer-learning orchestration.
- If your split files use different column names, pass `--sequence_col`, `--smiles_col`, and `--target_col`.
- The default base directory assumes the EMULaToR baseline folder is named `EITLEM-Kinetics`. If your folder name differs, pass `--base_dir` explicitly.
