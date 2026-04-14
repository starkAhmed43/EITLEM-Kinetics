from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from emulator_bench.common import normalize_sequence
from emulator_bench.feature_pipeline import LigandEmbeddingStore, ProteinEmbeddingStore


class CachedEitlemDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        protein_store: ProteinEmbeddingStore,
        ligand_store: LigandEmbeddingStore,
        sequence_col: str = "sequence",
        smiles_col: str = "smiles",
        target_col: Optional[str] = "log10_value",
    ):
        self.frame = frame.reset_index(drop=True)
        self.protein_store = protein_store
        self.ligand_store = ligand_store
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.target_col = target_col

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        smiles = str(row[self.smiles_col])
        sequence = normalize_sequence(str(row[self.sequence_col]))
        protein = self.protein_store.get(sequence)
        ligand = self.ligand_store.get(smiles)

        item = Data(
            x=torch.from_numpy(ligand["fingerprint"]).reshape(1, -1),
            pro_emb=torch.from_numpy(protein["embedding"]),
            sequence_length=torch.tensor([int(protein["length"][0])], dtype=torch.long),
            mol_key=smiles,
            prot_key=sequence,
        )
        if self.target_col is not None and self.target_col in self.frame.columns:
            value = row[self.target_col]
            if pd.notna(value):
                item.value = torch.tensor(float(value), dtype=torch.float32)
        return item

class EitlemDataLoader(DataLoader):
    def __init__(self, dataset: CachedEitlemDataset, **kwargs):
        super().__init__(dataset, follow_batch=["pro_emb"], **kwargs)
