import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.run_split_benchmarks import maybe_cache_embeddings
from emulator_bench.tune_optuna import metric_direction, prepare_optuna_storage

TUNE_SCRIPT = REPO_ROOT / "emulator_bench" / "tune_optuna.py"


def split_trials(total_trials, num_workers):
    base = total_trials // num_workers
    remainder = total_trials % num_workers
    return [base + (1 if idx < remainder else 0) for idx in range(num_workers)]


def worker_cmd(args, worker_trials, worker_index):
    cmd = [
        sys.executable,
        str(TUNE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--predictor_type",
        args.predictor_type,
        "--mol_type",
        args.mol_type,
        "--radius",
        str(args.radius),
        "--nbits",
        str(args.nbits),
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
        "--hidden_dim",
        str(args.hidden_dim),
        "--protein_dim",
        str(args.protein_dim),
        "--layer",
        str(args.layer),
        "--dropout",
        str(args.dropout),
        "--att_layer",
        str(args.att_layer),
        "--epochs",
        str(args.epochs),
        "--device",
        "cuda:0" if args.device.startswith("cuda") else args.device,
        "--cache_device",
        "cuda:0" if args.cache_device.startswith("cuda") else args.cache_device,
        "--max_residues",
        str(args.max_residues),
        "--max_seq_len",
        str(args.max_seq_len),
        "--max_batch",
        str(args.max_batch),
        "--long_seq_stride",
        str(args.long_seq_stride),
        "--protein_dtype",
        args.protein_dtype,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--protein_cache_items",
        str(args.protein_cache_items),
        "--metric",
        args.metric,
        "--eval_split",
        args.eval_split,
        "--n_trials",
        str(worker_trials),
        "--sampler_seed",
        str(args.sampler_seed + worker_index),
        "--study_name",
        args.study_name,
        "--storage",
        args.storage,
        "--skip_cache",
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.seeds:
        cmd.extend(["--seeds", *[str(seed) for seed in args.seeds]])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.persistent_workers:
        cmd.append("--persistent_workers")
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.preload_proteins:
        cmd.append("--preload_proteins")
    if args.lazy_ligands:
        cmd.append("--lazy_ligands")
    if args.torch_compile:
        cmd.append("--torch_compile")
    if args.overwrite_runs:
        cmd.append("--overwrite_runs")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch parallel single-GPU Optuna workers for the EITLEM bench.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--predictor_type", choices=["kcat", "km", "kkm"], default="kcat")
    parser.add_argument("--mol_type", choices=["MACCSKeys", "ECFP", "RDKIT"], default="MACCSKeys")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--nbits", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--protein_dim", type=int, default=1280)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--att_layer", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=512)
    parser.add_argument("--lazy_ligands", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--max_residues", type=int, default=12000)
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument("--max_batch", type=int, default=16)
    parser.add_argument("--long_seq_stride", type=int, default=800)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="eitlem_optuna")
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--reset_storage", action="store_true")
    parser.add_argument("--stagger_seconds", type=float, default=3.0)
    args = parser.parse_args()

    args.thresholds = args.thresholds or ([args.threshold] if args.threshold else None)
    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    optuna.create_study(
        direction=metric_direction(args.metric),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    worker_trial_counts = split_trials(args.n_trials, len(args.gpus))
    processes = []
    try:
        for worker_index, (gpu_id, worker_trials) in enumerate(zip(args.gpus, worker_trial_counts)):
            if worker_trials <= 0:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd = worker_cmd(args, worker_trials, worker_index)
            print("Launching Optuna worker %s on GPU %s for %s trials" % (worker_index, gpu_id, worker_trials), flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
            processes.append((gpu_id, worker_trials, proc))
            if worker_index < len(args.gpus) - 1 and args.stagger_seconds > 0:
                time.sleep(args.stagger_seconds)

        failed = False
        for gpu_id, worker_trials, proc in processes:
            return_code = proc.wait()
            if return_code != 0:
                failed = True
                print("Worker on GPU %s failed after %s trials with exit code %s" % (gpu_id, worker_trials, return_code), flush=True)
        if failed:
            raise RuntimeError("One or more parallel Optuna workers failed.")
    finally:
        for _gpu_id, _worker_trials, proc in processes:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
