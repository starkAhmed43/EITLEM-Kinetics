import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_EMBEDDINGS_DIR, DEFAULT_SPLIT_GROUPS, discover_split_jobs, normalize_threshold_args, summarize_seed_runs
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings

TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def load_best_hparams(args):
    if args.hparams_json:
        with open(args.hparams_json, "r") as handle:
            payload = json.load(handle)
        return payload.get("best_hparams", payload)

    if not args.storage:
        raise ValueError("Provide either --hparams_json or --storage.")
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    return dict(study.best_params)


def resolve_training_hparams(raw_hparams, args):
    def choose(key, fallback):
        override = getattr(args, key)
        if override is not None:
            return override
        return raw_hparams.get(key, fallback)

    return {
        "batch_size": int(choose("batch_size", 128)),
        "lr": float(choose("lr", 1e-3)),
        "weight_decay": float(choose("weight_decay", 1e-4)),
        "beta1": float(choose("beta1", 0.9)),
        "beta2": float(choose("beta2", 0.999)),
        "eps": float(choose("eps", 1e-8)),
        "scheduler": str(choose("scheduler", "cosine")),
        "lr_decay_factor": float(choose("lr_decay_factor", 0.5)),
        "lr_decay_patience": int(choose("lr_decay_patience", 5)),
        "min_lr": float(choose("min_lr", 1e-6)),
        "lr_warmup_epochs": int(choose("lr_warmup_epochs", 3)),
        "lr_warmup_start_factor": float(choose("lr_warmup_start_factor", 0.1)),
        "clip_grad": float(choose("clip_grad", 1.0)),
        "patience": int(choose("patience", 20)),
        "min_delta": float(choose("min_delta", 0.0)),
        "amsgrad": bool(args.amsgrad or raw_hparams.get("amsgrad", False)),
    }


def build_experiments(jobs, seeds, output_root):
    experiments = []
    for job in jobs:
        for seed in seeds:
            run_dir = output_root / job["split_group"] / job["split_name"] / ("seed_%s" % seed)
            experiments.append(
                {
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "train_path": job["train_path"],
                    "val_path": job["val_path"],
                    "test_path": job["test_path"],
                    "seed": int(seed),
                    "run_dir": run_dir,
                }
            )
    return experiments


def train_command(exp, args, hparams, device):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_path",
        exp["train_path"],
        "--val_path",
        exp["val_path"],
        "--test_path",
        exp["test_path"],
        "--embeddings_dir",
        args.embeddings_dir,
        "--out_dir",
        str(exp["run_dir"]),
        "--task_name",
        "%s_%s_seed%s" % (exp["split_group"], exp["split_name"], exp["seed"]),
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
        "--batch_size",
        str(hparams["batch_size"]),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(hparams["lr"]),
        "--weight_decay",
        str(hparams["weight_decay"]),
        "--beta1",
        str(hparams["beta1"]),
        "--beta2",
        str(hparams["beta2"]),
        "--eps",
        str(hparams["eps"]),
        "--scheduler",
        hparams["scheduler"],
        "--lr_decay_factor",
        str(hparams["lr_decay_factor"]),
        "--lr_decay_patience",
        str(hparams["lr_decay_patience"]),
        "--min_lr",
        str(hparams["min_lr"]),
        "--lr_warmup_epochs",
        str(hparams["lr_warmup_epochs"]),
        "--lr_warmup_start_factor",
        str(hparams["lr_warmup_start_factor"]),
        "--clip_grad",
        str(hparams["clip_grad"]),
        "--patience",
        str(hparams["patience"]),
        "--min_delta",
        str(hparams["min_delta"]),
        "--device",
        device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--protein_cache_items",
        str(args.protein_cache_items),
        "--seed",
        str(exp["seed"]),
    ]
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.persistent_workers:
        cmd.append("--persistent_workers")
    if args.preload_proteins:
        cmd.append("--preload_proteins")
    if args.lazy_ligands:
        cmd.append("--lazy_ligands")
    if args.torch_compile:
        cmd.append("--torch_compile")
    if hparams["amsgrad"]:
        cmd.append("--amsgrad")
    return cmd


def run_experiment(exp, args, hparams, gpu_id):
    exp["run_dir"].mkdir(parents=True, exist_ok=True)
    metric_path = exp["run_dir"] / "final_results_test.csv"
    if metric_path.exists() and not args.overwrite:
        return {
            "status": "skipped_exists",
            "gpu_id": str(gpu_id),
            "run_dir": str(exp["run_dir"]),
            "split_group": exp["split_group"],
            "split_name": exp["split_name"],
            "difficulty": exp["difficulty"],
            "seed": exp["seed"],
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0" if args.device.startswith("cuda") else args.device
    cmd = train_command(exp, args, hparams, device)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return {
        "status": "completed",
        "gpu_id": str(gpu_id),
        "run_dir": str(exp["run_dir"]),
        "split_group": exp["split_group"],
        "split_name": exp["split_name"],
        "difficulty": exp["difficulty"],
        "seed": exp["seed"],
    }


def run_parallel(experiments, args, hparams):
    work_queue = queue.Queue()
    for exp in experiments:
        work_queue.put(exp)

    results = []
    result_lock = threading.Lock()

    def worker(gpu_id):
        while True:
            try:
                exp = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_experiment(exp, args, hparams, gpu_id)
            except Exception as exc:
                result = {
                    "status": "failed",
                    "gpu_id": str(gpu_id),
                    "run_dir": str(exp["run_dir"]),
                    "split_group": exp["split_group"],
                    "split_name": exp["split_name"],
                    "difficulty": exp["difficulty"],
                    "seed": exp["seed"],
                    "error": str(exc),
                }
            with result_lock:
                results.append(result)
            work_queue.task_done()

    threads = []
    for gpu_id in args.gpus:
        thread = threading.Thread(target=worker, args=(str(gpu_id),), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return results


def main():
    parser = argparse.ArgumentParser(description="Retrain all requested EITLEM split jobs in parallel from the best Optuna hyperparameters.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda:0")
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--max_residues", type=int, default=12000)
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument("--max_batch", type=int, default=16)
    parser.add_argument("--long_seq_stride", type=int, default=800)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lr_decay_factor", type=float, default=None)
    parser.add_argument("--lr_decay_patience", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--lr_warmup_epochs", type=int, default=None)
    parser.add_argument("--lr_warmup_start_factor", type=float, default=None)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--study_name", type=str, default="eitlem_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=None)
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache_embeddings(args)
    raw_hparams = load_best_hparams(args)
    hparams = resolve_training_hparams(raw_hparams, args)

    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError("No split jobs found in %s" % args.base_dir)
    output_root = Path(args.output_root) if args.output_root else Path(args.base_dir) / "retrain_from_optuna"
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments(jobs, args.seeds, output_root)
    results = run_parallel(experiments, args, hparams)

    summary_rows = []
    for result in results:
        if result["status"] == "failed":
            summary_rows.append(result)
            continue
        run_dir = Path(result["run_dir"])
        row = dict(result)
        for prefix in ["train", "val", "test"]:
            metrics_path = run_dir / ("final_results_%s.csv" % prefix)
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
                for key, value in metrics.items():
                    row["%s_%s" % (prefix, key)] = value
        summary_rows.append(row)

    runs_df = pd.DataFrame(summary_rows)
    runs_df.to_csv(output_root / "retrain_summary_runs.csv", index=False)
    metric_cols = [col for col in runs_df.columns if col.startswith("test_")]
    summarize_seed_runs(summary_rows, ["split_group", "split_name", "difficulty"], metric_cols).to_csv(
        output_root / "retrain_summary_thresholds.csv",
        index=False,
    )
    summarize_seed_runs(summary_rows, ["split_group"], metric_cols).to_csv(
        output_root / "retrain_summary_by_split_group.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
