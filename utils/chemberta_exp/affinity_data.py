# affinity_data.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Union

_CHECKPOINT = "seyonec/ChemBERTa-zinc-base-v1"


class AffinityDataset(Dataset):
    """Tokenised SMILES + normalised affinity targets."""
    def __init__(
        self,
        smiles: List[str],
        targets,
        tokenizer,
        max_length: int = 256,
    ):
        self.smiles = smiles
        self.targets = targets
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        y = self.targets[idx]
        toks = self.tok(
            s,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = toks["input_ids"].squeeze(0)
        attn_mask = toks["attention_mask"].squeeze(0)
        return input_ids, attn_mask, torch.tensor(y, dtype=torch.float)


def get_dataloaders(
    csv_path: str,
    smiles_col: str = "smiles",
    target_col: str = "E_score2",
    batch_size: int = 64,
    test_size: float = 0.30,
    random_state: int = 42,
    max_length: int = 256,
) -> Tuple[DataLoader, DataLoader, MinMaxScaler]:
    """
    Read the CSV, scale targets to [0, 1], split, and return train/test loaders
    plus the fitted scaler (so callers can inverse‑transform later).

    Example
    -------
    >>> train_loader, test_loader, scaler = get_dataloaders("docked_leadlike.csv")
    """
    df = pd.read_csv(csv_path)
    smiles = df[smiles_col].astype(str).tolist()
    y = df[target_col].astype(float).values.reshape(-1, 1)

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y).flatten()

    sm_train, sm_test, y_train, y_test = train_test_split(
        smiles, y_scaled, test_size=test_size, random_state=random_state
    )

    tokenizer = AutoTokenizer.from_pretrained(_CHECKPOINT)
    train_ds = AffinityDataset(sm_train, y_train, tokenizer, max_length)
    test_ds  = AffinityDataset(sm_test,  y_test,  tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, scaler

