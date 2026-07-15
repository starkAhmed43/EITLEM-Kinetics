import argparse
import time
from pathlib import Path

import numpy as np
import torch
try:
    from src.utils.rich_progress import progress, write
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress, write

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    ensure_parent,
    ligand_cache_path,
    normalize_sequence,
    normalize_threshold_args,
    protein_cache_path,
    read_table,
    save_json,
)
from emulator_bench.feature_pipeline import (
    build_esm_batches,
    embed_long_sequence,
    ligand_cache_item,
    load_esm1v_model,
    protein_cache_item,
    resolve_amp_dtype,
    _esm_forward,
)


def _collect_unique_values(jobs, sequence_col: str, smiles_col: str):
    sequences = set()
    smiles_values = set()
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            frame = read_table(Path(job[split_key]))
            if sequence_col not in frame.columns or smiles_col not in frame.columns:
                raise ValueError(f"Expected columns `{sequence_col}` and `{smiles_col}` in {job[split_key]}")
            sequences.update(normalize_sequence(value) for value in frame[sequence_col].astype(str))
            smiles_values.update(str(value) for value in frame[smiles_col].astype(str))
    return sorted(sequences), sorted(smiles_values)


def _save_npz(path: Path, item: dict) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **item)
    tmp_path.replace(path)


def cache_proteins(args, sequences):
    pending = [seq for seq in sequences if args.overwrite or not protein_cache_path(args.embeddings_dir, seq).exists()]
    if not pending:
        print("Protein cache is already complete.")
        return {"proteins_total": len(sequences), "proteins_written": 0}

    device = torch.device(args.device)
    autocast_dtype, precision_mode = resolve_amp_dtype(device)
    print(f"Protein cache device: {device} | precision: {precision_mode}")
    model, batch_converter = load_esm1v_model(device)
    batches = build_esm_batches(
        pending,
        max_residues=args.max_residues,
        max_seq_len=args.max_seq_len,
        max_batch=args.max_batch,
    )

    written = 0
    iterator = progress(batches, desc="Caching protein embeddings", unit="batch")
    for batch in iterator:
        if len(batch) == 1 and len(batch[0]) > args.max_seq_len:
            sequence = batch[0]
            embedding = embed_long_sequence(
                model,
                batch_converter,
                sequence,
                device=device,
                autocast_dtype=autocast_dtype,
                max_window=args.max_seq_len,
                stride=args.long_seq_stride,
            )
            _save_npz(
                protein_cache_path(args.embeddings_dir, sequence),
                protein_cache_item(sequence, embedding, protein_dtype=args.protein_dtype),
            )
            written += 1
            iterator.set_postfix(written=written, remaining=len(pending) - written)
            continue

        embedded = _esm_forward(model, batch_converter, batch, device=device, autocast_dtype=autocast_dtype)
        for sequence in batch:
            _save_npz(
                protein_cache_path(args.embeddings_dir, sequence),
                protein_cache_item(sequence, embedded[sequence], protein_dtype=args.protein_dtype),
            )
            written += 1
        iterator.set_postfix(written=written, remaining=len(pending) - written)

    return {"proteins_total": len(sequences), "proteins_written": written}


def cache_ligands(args, smiles_values):
    pending = [
        smiles
        for smiles in smiles_values
        if args.overwrite
        or not ligand_cache_path(
            args.embeddings_dir,
            smiles,
            mol_type=args.mol_type,
            radius=args.radius,
            nbits=args.nbits,
        ).exists()
    ]
    if not pending:
        print("Ligand cache is already complete.")
        return {"ligands_total": len(smiles_values), "ligands_written": 0}

    written = 0
    iterator = progress(pending, desc="Caching molecular fingerprints", unit="smiles")
    for smiles in iterator:
        item = ligand_cache_item(smiles, mol_type=args.mol_type, radius=args.radius, nbits=args.nbits)
        _save_npz(
            ligand_cache_path(
                args.embeddings_dir,
                smiles,
                mol_type=args.mol_type,
                radius=args.radius,
                nbits=args.nbits,
            ),
            item,
        )
        written += 1
        iterator.set_postfix(written=written, remaining=len(pending) - written)
    return {"ligands_total": len(smiles_values), "ligands_written": written}


def main():
    parser = argparse.ArgumentParser(description="Cache reusable EITLEM-Kinetics protein embeddings and molecular fingerprints.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mol_type", choices=["MACCSKeys", "ECFP", "RDKIT"], default="MACCSKeys")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--nbits", type=int, default=1024)
    parser.add_argument("--max_residues", type=int, default=12000)
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument("--max_batch", type=int, default=64)
    parser.add_argument("--long_seq_stride", type=int, default=800)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.base_dir = Path(args.base_dir)
    args.embeddings_dir = Path(args.embeddings_dir)
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)

    jobs = discover_split_jobs(args.base_dir, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")

    started = time.time()
    sequences, smiles_values = _collect_unique_values(jobs, sequence_col=args.sequence_col, smiles_col=args.smiles_col)
    print(f"Discovered {len(jobs)} split jobs")
    print(f"Unique normalized sequences: {len(sequences)}")
    print(f"Unique smiles: {len(smiles_values)}")

    protein_stats = cache_proteins(args, sequences)
    ligand_stats = cache_ligands(args, smiles_values)

    manifest = {
        "cache_version": 1,
        "base_dir": str(args.base_dir),
        "embeddings_dir": str(args.embeddings_dir),
        "sequence_col": args.sequence_col,
        "smiles_col": args.smiles_col,
        "split_groups": list(args.split_groups),
        "thresholds": args.thresholds,
        "mol_type": args.mol_type,
        "radius": int(args.radius),
        "nbits": int(args.nbits),
        "protein_dtype": args.protein_dtype,
        "protein_model": "esm1v_t33_650M_UR90S_1",
        "protein_max_seq_len": int(args.max_seq_len),
        "protein_long_seq_stride": int(args.long_seq_stride),
        "protein_cache": protein_stats,
        "ligand_cache": ligand_stats,
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.embeddings_dir / "manifest.json", manifest)
    print(f"Saved cache manifest to {args.embeddings_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
