"""
src/data/dataset.py
────────────────────
PyTorch Dataset that reads a processed Parquet file.
Expected columns: all numerical features + 'icd10_labels' (list[str]).

Training data preparation (not run here) should:
  1. Load MIMIC-IV / eICU encounters
  2. Join diagnoses → one-hot encode ICD-10 codes → save to Parquet
  3. Tokenize clinical notes using PatientPreprocessor
  4. Serialise tensors to the Parquet as byte columns
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from src.data.preprocessing import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, HISTORY_FLAGS,
    build_numerical_tensor, POPULATION_MEDIANS, POPULATION_STDS,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class MedicalDataset(Dataset):
    """
    Reads a pre-processed Parquet file produced by the data pipeline.

    Each row contains:
      - All numerical / categorical / history columns
      - 'clinical_text'   : str — concatenated clinical notes
      - 'icd10_labels'    : JSON list[str] — ICD-10 codes present
      - 'label_vector'    : bytes — pre-serialised numpy array of shape (num_diseases,)

    Args:
        parquet_path : path to the processed .parquet file
        icd_vocab    : dict mapping ICD-10 code → integer index
        tokenizer    : HuggingFace tokenizer for clinical text
        max_seq_len  : max token length
        augment      : enable simple text augmentation during training
    """

    def __init__(
        self,
        parquet_path: str | Path,
        icd_vocab: dict[str, int],
        tokenizer: AutoTokenizer,
        max_seq_len: int = 512,
        augment: bool = False,
    ):
        self.df = pd.read_parquet(parquet_path)
        self.icd_vocab = icd_vocab
        self.num_diseases = len(icd_vocab)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + HISTORY_FLAGS
        log.info("Dataset loaded", rows=len(self.df), num_diseases=self.num_diseases)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # ── 1. Numerical features ─────────────────────────────────────────
        flat = {feat: row.get(feat, None) for feat in self.all_features}
        numerical = build_numerical_tensor(flat, POPULATION_MEDIANS, POPULATION_STDS)

        # ── 2. Clinical text tokenisation ─────────────────────────────────
        text: str = str(row.get("clinical_text", ""))
        if self.augment:
            text = self._augment_text(text)

        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── 3. Multi-hot disease label vector ─────────────────────────────
        labels = torch.zeros(self.num_diseases, dtype=torch.float32)
        raw_labels = row.get("icd10_labels", "[]")
        if isinstance(raw_labels, str):
            raw_labels = json.loads(raw_labels)
        for code in raw_labels:
            if code in self.icd_vocab:
                labels[self.icd_vocab[code]] = 1.0

        return {
            "numerical": numerical,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
            "patient_id": str(row.get("patient_id", idx)),
        }

    @staticmethod
    def _augment_text(text: str) -> str:
        """
        Simple clinical text augmentation:
          - Randomly drop ~10% of sentences
          - Random synonym swap for common abbreviations
        """
        import random
        abbrev_map = {
            "SOB": "shortness of breath",
            "CP": "chest pain",
            "HTN": "hypertension",
            "DM": "diabetes mellitus",
            "CAD": "coronary artery disease",
            "HF": "heart failure",
            "COPD": "chronic obstructive pulmonary disease",
        }
        for abbr, full in abbrev_map.items():
            if random.random() < 0.3:
                text = text.replace(abbr, full)

        sentences = text.split(". ")
        kept = [s for s in sentences if random.random() > 0.10]
        return ". ".join(kept) if kept else text


def build_class_weights(dataset: MedicalDataset) -> torch.Tensor:
    """
    Compute per-class positive weights for weighted BCE loss.
    weight[i] = (N - pos_i) / pos_i  (clamped to [1, 100])
    """
    label_sum = torch.zeros(dataset.num_diseases)
    for i in range(len(dataset)):
        label_sum += dataset[i]["labels"]
    n = len(dataset)
    weights = (n - label_sum) / (label_sum + 1e-6)
    return weights.clamp(1.0, 100.0)


def build_weighted_sampler(dataset: MedicalDataset) -> WeightedRandomSampler:
    """
    Over-sample rare-disease encounters so each batch has diverse pathologies.
    Sample weight = max positive class weight across diseases present in that sample.
    """
    class_weights = build_class_weights(dataset)
    sample_weights = []
    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        active_weights = class_weights[labels.bool()]
        w = float(active_weights.max()) if len(active_weights) > 0 else 1.0
        sample_weights.append(w)
    return WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)


def get_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    icd_vocab: dict[str, int],
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_seq_len: int = 512,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test dataloaders."""
    train_ds = MedicalDataset(train_path, icd_vocab, tokenizer, max_seq_len, augment=True)
    val_ds   = MedicalDataset(val_path,   icd_vocab, tokenizer, max_seq_len, augment=False)
    test_ds  = MedicalDataset(test_path,  icd_vocab, tokenizer, max_seq_len, augment=False)

    sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
