"""
Aggregate EITLEM-Kcat and EITLEM-Km baseline results across seeds.

Walks all retrain_from_optuna directories under both target roots, reads
final_results_{train,val,test}.csv for every seed, and outputs
mean + variance per metric per TVT split.

Directory layout handled:
  <target>/retrain_from_optuna/<split_group>/threshold_X/seed_N/   (threshold splits)
  <target>/retrain_from_optuna/<split_group>/<split_group>/seed_N/ (non-threshold splits)

Output: aggregate_eitlem_results.csv saved next to this script.
"""

from pathlib import Path

import pandas as pd

EMULATOR_ROOT = Path("~/github/EMULaToR/data/processed/baselines").expanduser()
TARGETS = ("EITLEM-Kcat", "EITLEM-Km")
SPLITS = ("train", "val", "test")
METRICS = ("mae", "mse", "rmse", "r2_score", "pearson", "spearman", "loss")


def parse_path(seed_dir: Path, target_root: Path) -> dict:
    """Extract split_group / threshold from a seed_* path under retrain_from_optuna."""
    # seed_dir: <target_root>/retrain_from_optuna/<split_group>/[threshold_X | <split_group>]/seed_N
    retrain_root = target_root / "retrain_from_optuna"
    parts = seed_dir.relative_to(retrain_root).parts
    # parts[0] = split_group, parts[1] = threshold_X or split_group, parts[2] = seed_N
    split_group = parts[0]
    middle = parts[1]
    threshold = middle if middle.startswith("threshold_") else None
    return dict(split_group=split_group, threshold=threshold, seed=seed_dir.name)


def load_seed_results(seed_dir: Path) -> dict[str, pd.Series] | None:
    """Return {split: metrics_series} for one seed dir, or None if incomplete."""
    results = {}
    for split in SPLITS:
        fpath = seed_dir / f"final_results_{split}.csv"
        if not fpath.exists():
            return None
        df = pd.read_csv(fpath)
        results[split] = df.iloc[0]
    return results


def main():
    rows = []

    for target in TARGETS:
        target_root = EMULATOR_ROOT / target
        retrain_root = target_root / "retrain_from_optuna"

        for seed_dir in sorted(retrain_root.rglob("seed_*")):
            if not seed_dir.is_dir():
                continue

            meta = parse_path(seed_dir, target_root)
            meta["target"] = target
            split_results = load_seed_results(seed_dir)
            if split_results is None:
                print(f"  [skip] incomplete: {seed_dir.relative_to(EMULATOR_ROOT)}")
                continue

            for split, series in split_results.items():
                row = {**meta, "tvt_split": split}
                for metric in METRICS:
                    if metric in series.index:
                        row[metric] = series[metric]
                rows.append(row)

    if not rows:
        print("No complete results found.")
        return

    df = pd.DataFrame(rows)

    group_keys = ["target", "split_group", "threshold", "tvt_split"]
    agg = (
        df.groupby(group_keys, dropna=False)[list(METRICS)]
        .agg(["mean", "var"])
    )
    # Flatten MultiIndex columns: (metric, stat) -> metric_mean / metric_var
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_seeds"] = df.groupby(group_keys, dropna=False).size()
    agg = agg.reset_index()

    for target in TARGETS:
        target_agg = agg[agg["target"] == target].drop(columns="target")
        out_path = EMULATOR_ROOT / target / "aggregate_eitlem_results.csv"
        target_agg.to_csv(out_path, index=False)
        print(f"Saved {len(target_agg)} rows to {out_path}")


if __name__ == "__main__":
    main()
