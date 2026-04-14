import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    normalize_threshold_args,
)
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def metric_direction(metric):
    return "minimize" if metric in {"rmse", "mse", "mae", "loss"} else "maximize"


def sqlite_path_from_storage(storage):
    if not storage or not storage.startswith("sqlite:///"):
        return None
    parsed = urlparse(storage)
    raw_path = unquote(parsed.path or "")
    return Path(raw_path) if raw_path else None


def sqlite_has_optuna_schema(db_path):
    with sqlite3.connect(str(db_path)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    return "version_info" in tables


def prepare_optuna_storage(args):
    db_path = sqlite_path_from_storage(args.storage)
    if db_path is None:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        return
    if args.reset_storage:
        db_path.unlink()
        return
    if not sqlite_has_optuna_schema(db_path):
        raise RuntimeError(
            "Optuna storage exists but does not contain a valid Optuna schema: "
            "%s. Use a new --storage path or rerun with --reset_storage." % db_path
        )


def suggest_hparams(trial, args):
    batch_size = int(args.batch_size) if args.batch_size is not None else trial.suggest_categorical("batch_size", [32, 64, 96, 128, 160])
    return {
        "batch_size": batch_size,
        "lr": trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "min_lr": trial.suggest_float("min_lr", 1e-7, 5e-5, log=True),
        "lr_warmup_epochs": trial.suggest_int("lr_warmup_epochs", 0, 8),
        "lr_warmup_start_factor": trial.suggest_float("lr_warmup_start_factor", 0.05, 0.5),
        "clip_grad": trial.suggest_categorical("clip_grad", [0.5, 1.0, 2.0, 5.0]),
        "patience": trial.suggest_categorical("patience", [10, 15, 20, 30]),
        "scheduler": "cosine",
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "amsgrad": False,
        "lr_decay_factor": 0.5,
        "lr_decay_patience": 5,
        "min_delta": 0.0,
    }


def run_trial_job(job, seed, hparams, args, trial_number):
    trial_root = (
        Path(job["root_dir"])
        / "eitlem_optuna_runs"
        / ("trial_%s" % trial_number)
        / job["split_group"]
        / job["split_name"]
        / ("seed_%s" % seed)
    )
    metric_file = trial_root / ("final_results_%s.csv" % args.eval_split)
    if not metric_file.exists() or args.overwrite_runs:
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--train_path",
            job["train_path"],
            "--val_path",
            job["val_path"],
            "--test_path",
            job["test_path"],
            "--embeddings_dir",
            args.embeddings_dir,
            "--out_dir",
            str(trial_root),
            "--task_name",
            "optuna_trial_%s_%s_%s_seed%s" % (trial_number, job["split_group"], job["split_name"], seed),
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
            "--val_every",
            str(args.val_every),
            "--monitor_metric",
            args.metric,
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--prefetch_factor",
            str(args.prefetch_factor),
            "--protein_cache_items",
            str(args.protein_cache_items),
            "--seed",
            str(seed),
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
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    metrics = pd.read_csv(metric_file).iloc[0].to_dict()
    if args.metric not in metrics:
        raise RuntimeError("Metric `%s` not found in %s" % (args.metric, metric_file))
    return float(metrics[args.metric])


def main():
    parser = argparse.ArgumentParser(description="Tune non-architectural EITLEM retraining hyperparameters with Optuna.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
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
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--val_every", type=int, default=2)
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
    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "pearson", "spearman", "r2_score", "mae", "mse", "loss"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="eitlem_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--reset_storage", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    if args.storage is None:
        args.storage = "sqlite:///%s" % (Path(args.base_dir) / "optuna_studies" / (args.study_name + ".db"))

    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError("No split jobs found in %s" % args.base_dir)

    study = optuna.create_study(
        direction=metric_direction(args.metric),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    def objective(trial):
        hparams = suggest_hparams(trial, args)
        scores = []
        for job in jobs:
            for seed in args.seeds:
                score = run_trial_job(job, seed, hparams, args, trial.number)
                scores.append(score)
        trial.set_user_attr("n_jobs", len(jobs))
        trial.set_user_attr("n_scores", len(scores))
        return float(sum(scores) / len(scores))

    study.optimize(objective, n_trials=args.n_trials)

    out_dir = Path(args.base_dir) / "optuna_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_csv = out_dir / ("%s_trials.csv" % args.study_name)
    trials_df.to_csv(trials_csv, index=False)
    best_json = out_dir / ("%s_best_hparams.json" % args.study_name)
    with open(best_json, "w") as handle:
        json.dump(
            {
                "study_name": args.study_name,
                "storage": args.storage,
                "direction": study.direction.name.lower(),
                "best_trial_number": int(study.best_trial.number),
                "best_value": float(study.best_value),
                "best_hparams": dict(study.best_params),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
