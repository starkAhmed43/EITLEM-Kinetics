import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_RESULTS_DIRNAME,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    normalize_threshold_args,
    split_sizes,
    summarize_seed_runs,
)

CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def maybe_cache_embeddings(args):
    if args.skip_cache:
        return
    cmd = [
        sys.executable,
        str(CACHE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--device",
        args.cache_device,
        "--mol_type",
        args.mol_type,
        "--radius",
        str(args.radius),
        "--nbits",
        str(args.nbits),
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
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.cache_overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def maybe_load_hparams(args):
    if not args.hparams_json:
        return args
    with open(args.hparams_json, "r") as handle:
        payload = json.load(handle)
    hparams = payload.get("best_hparams", payload)
    for key in [
        "batch_size",
        "lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
        "amsgrad",
        "scheduler",
        "multistep_milestones",
        "multistep_gamma",
        "lr_decay_factor",
        "lr_decay_patience",
        "min_lr",
        "lr_warmup_epochs",
        "lr_warmup_start_factor",
        "clip_grad",
        "patience",
        "min_delta",
    ]:
        if key in hparams:
            setattr(args, key, hparams[key])
    return args


def train_one(job, seed, args):
    result_root = Path(job["root_dir"]) / args.results_dirname / f"seed_{seed}"
    metric_path = result_root / "final_results_test.csv"
    if metric_path.exists() and not args.overwrite:
        return result_root
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
        str(result_root),
        "--task_name",
        f"{job['split_group']}_{job['split_name']}_seed{seed}",
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
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--beta1",
        str(args.beta1),
        "--beta2",
        str(args.beta2),
        "--eps",
        str(args.eps),
        "--scheduler",
        args.scheduler,
        "--multistep_milestones",
        *[str(value) for value in args.multistep_milestones],
        "--multistep_gamma",
        str(args.multistep_gamma),
        "--lr_decay_factor",
        str(args.lr_decay_factor),
        "--lr_decay_patience",
        str(args.lr_decay_patience),
        "--min_lr",
        str(args.min_lr),
        "--lr_warmup_epochs",
        str(args.lr_warmup_epochs),
        "--lr_warmup_start_factor",
        str(args.lr_warmup_start_factor),
        "--clip_grad",
        str(args.clip_grad),
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
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
        "--results_dirname",
        args.results_dirname,
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
    if args.amsgrad:
        cmd.append("--amsgrad")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return result_root


def main():
    parser = argparse.ArgumentParser(description="Run the EITLEM emulator bench across EMULaToR split groups.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hparams_json", type=str, default=None)

    parser.add_argument("--predictor_type", choices=["kcat", "km", "kkm"], default="kcat")
    parser.add_argument("--mol_type", choices=["MACCSKeys", "ECFP", "RDKIT"], default="MACCSKeys")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--nbits", type=int, default=1024)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--protein_dim", type=int, default=1280)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--att_layer", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "multistep", "cosine", "plateau"], default="cosine")
    parser.add_argument("--multistep_milestones", nargs="+", type=int, default=[50, 80])
    parser.add_argument("--multistep_gamma", type=float, default=0.9)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_warmup_epochs", type=int, default=3)
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=512)
    parser.add_argument("--lazy_ligands", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")

    parser.add_argument("--max_residues", type=int, default=12000)
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument("--max_batch", type=int, default=16)
    parser.add_argument("--long_seq_stride", type=int, default=800)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args = maybe_load_hparams(args)
    maybe_cache_embeddings(args)

    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    summary_rows = []
    for job in tqdm(jobs, desc="Benchmark jobs", unit="job"):
        for seed in args.seeds:
            out_dir = train_one(job, seed, args)
            test_metrics = pd.read_csv(out_dir / "final_results_test.csv").iloc[0].to_dict()
            val_metrics = pd.read_csv(out_dir / "final_results_val.csv").iloc[0].to_dict()
            row = {
                "split_group": job["split_group"],
                "split_name": job["split_name"],
                "difficulty": job["difficulty"],
                "seed": int(seed),
                "run_dir": str(out_dir),
            }
            row.update(split_sizes(Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"])))
            for prefix, metrics in (("val", val_metrics), ("test", test_metrics)):
                for key, value in metrics.items():
                    row["%s_%s" % (prefix, key)] = value
            summary_rows.append(row)

    summary_path = Path(args.base_dir) / "eitlem_summary_runs.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    metric_cols = [col for col in pd.DataFrame(summary_rows).columns if col.startswith("test_")]
    thresholds_df = summarize_seed_runs(summary_rows, ["split_group", "split_name", "difficulty"], metric_cols)
    thresholds_df.to_csv(Path(args.base_dir) / "eitlem_summary_thresholds.csv", index=False)
    by_group_df = summarize_seed_runs(summary_rows, ["split_group"], metric_cols)
    by_group_df.to_csv(Path(args.base_dir) / "eitlem_summary_by_split_group.csv", index=False)


if __name__ == "__main__":
    main()
