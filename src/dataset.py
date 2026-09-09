"""
PyTorch datasets and data loaders for all tasks.
Handles GTZAN graphs, MusicCaps text, and paired fusion data.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from torch_geometric.data import Data, Dataset as PyGDataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# ============================================================================
# Tag extraction from MusicCaps aspect_list
# ============================================================================

def extract_tags_from_aspects(aspect_list_str: str) -> list[str]:
    """Parse MusicCaps aspect_list string into individual tags."""
    if pd.isna(aspect_list_str) or not aspect_list_str:
        return []
    # aspect_list is formatted as '["tag1", "tag2", ...]'
    try:
        tags = json.loads(aspect_list_str.replace("'", '"'))
    except (json.JSONDecodeError, Exception):
        # Fallback: split by comma
        tags = [t.strip().strip('"').strip("'").strip("[").strip("]")
                for t in aspect_list_str.split(",")]
    return [t.strip().lower() for t in tags if t.strip()]


def build_tag_vocabulary(df: pd.DataFrame, min_count: int = 10, max_tags: int = 50) -> list[str]:
    """Build vocabulary of most common tags from MusicCaps aspect_list."""
    from collections import Counter
    tag_counter = Counter()

    for _, row in df.iterrows():
        tags = extract_tags_from_aspects(row.get("aspect_list", ""))
        tag_counter.update(tags)

    # Filter by min count and take top-N
    common_tags = [tag for tag, count in tag_counter.most_common(max_tags)
                   if count >= min_count]
    return common_tags


# ============================================================================
# GTZAN Graph Dataset (Task 2)
# ============================================================================

class GTZANGraphDataset(PyGDataset):
    """
    GTZAN dataset as pre-built PyG graphs for genre classification.
    """

    def __init__(
        self,
        graph_dir: str,
        genre_labels: list[str],
        split_ids: list[str] = None,
        transform=None,
    ):
        self.graph_dir = Path(graph_dir)
        self.genre_labels = genre_labels
        self.genre_to_idx = {g: i for i, g in enumerate(genre_labels)}

        # List all .pt files
        all_files = sorted(self.graph_dir.glob("*.pt"))
        if split_ids is not None:
            self.files = [f for f in all_files if f.stem in split_ids]
        else:
            self.files = all_files

        self._transform = transform
        super().__init__(str(graph_dir), transform=transform)

    def len(self) -> int:
        return len(self.files)

    def get(self, idx: int) -> Data:
        data = torch.load(self.files[idx], weights_only=False)
        # Extract genre from filename (e.g., "blues.00001")
        fname = self.files[idx].stem
        genre = fname.split(".")[0]
        data.y = torch.tensor(self.genre_to_idx.get(genre, 0), dtype=torch.long)
        data.track_id = fname
        return data


# ============================================================================
# CNN Mel-Spectrogram Dataset (Baseline)
# ============================================================================

class GTZANMelDataset(Dataset):
    """
    GTZAN mel-spectrogram dataset for CNN baseline.
    """

    def __init__(
        self,
        mel_dir: str,
        genre_labels: list[str],
        split_ids: list[str] = None,
        fixed_length: int = 1292,  # ~30s at sr=22050, hop=512
    ):
        self.mel_dir = Path(mel_dir)
        self.genre_labels = genre_labels
        self.genre_to_idx = {g: i for i, g in enumerate(genre_labels)}
        self.fixed_length = fixed_length

        all_files = sorted(self.mel_dir.glob("*.npy"))
        if split_ids is not None:
            self.files = [f for f in all_files if f.stem in split_ids]
        else:
            self.files = all_files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        mel = np.load(self.files[idx])  # (n_mels, T)

        # Pad or truncate to fixed length
        if mel.shape[1] < self.fixed_length:
            mel = np.pad(mel, ((0, 0), (0, self.fixed_length - mel.shape[1])), mode="constant")
        else:
            mel = mel[:, :self.fixed_length]

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # Add channel dimension
        mel_tensor = torch.tensor(mel, dtype=torch.float).unsqueeze(0)  # (1, n_mels, T)

        # Genre label
        fname = self.files[idx].stem
        genre = fname.split(".")[0]
        label = self.genre_to_idx.get(genre, 0)

        return mel_tensor, label


# ============================================================================
# MusicCaps Text Dataset (Task 1)
# ============================================================================

class MusicCapsTextDataset(Dataset):
    """
    MusicCaps captions + aspect tags for BERT multi-label classification.
    """

    def __init__(
        self,
        metadata_path: str,
        tag_vocabulary: list[str],
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        split_indices: list[int] = None,
    ):
        self.df = pd.read_csv(metadata_path)
        if split_indices is not None:
            self.df = self.df.iloc[split_indices].reset_index(drop=True)

        self.tag_vocabulary = tag_vocabulary
        self.tag_to_idx = {t: i for i, t in enumerate(tag_vocabulary)}
        self.num_tags = len(tag_vocabulary)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        caption = str(row.get("caption", ""))

        # Tokenize caption
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Multi-label tag vector
        tags = extract_tags_from_aspects(row.get("aspect_list", ""))
        tag_vector = torch.zeros(self.num_tags, dtype=torch.float)
        for tag in tags:
            if tag in self.tag_to_idx:
                tag_vector[self.tag_to_idx[tag]] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": tag_vector,
            "ytid": row.get("ytid", ""),
            "caption": caption,
        }


# ============================================================================
# Paired Fusion Dataset (Tasks 3 & 4)
# ============================================================================

class MusicCapsFusionDataset(Dataset):
    """
    Paired (graph, caption, tags) dataset for GNN-BERT fusion and contrastive learning.
    Only includes entries where both audio graph and caption exist.
    """

    def __init__(
        self,
        metadata_path: str,
        graph_dir: str,
        tag_vocabulary: list[str],
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        split_indices: list[int] = None,
    ):
        self.df = pd.read_csv(metadata_path)
        if split_indices is not None:
            self.df = self.df.iloc[split_indices].reset_index(drop=True)

        self.graph_dir = Path(graph_dir)
        self.tag_vocabulary = tag_vocabulary
        self.tag_to_idx = {t: i for i, t in enumerate(tag_vocabulary)}
        self.num_tags = len(tag_vocabulary)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Filter to entries with available graphs
        valid_indices = []
        for i, row in self.df.iterrows():
            graph_path = self.graph_dir / f"{row['ytid']}.pt"
            if graph_path.exists():
                valid_indices.append(i)

        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        print(f"MusicCapsFusionDataset: {len(self.df)} paired samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        ytid = row["ytid"]
        caption = str(row.get("caption", ""))

        # Load graph
        graph_path = self.graph_dir / f"{ytid}.pt"
        graph_data = torch.load(graph_path, weights_only=False)

        # Tokenize caption
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tag vector
        tags = extract_tags_from_aspects(row.get("aspect_list", ""))
        tag_vector = torch.zeros(self.num_tags, dtype=torch.float)
        for tag in tags:
            if tag in self.tag_to_idx:
                tag_vector[self.tag_to_idx[tag]] = 1.0

        return {
            "graph": graph_data,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": tag_vector,
            "ytid": ytid,
            "caption": caption,
        }


# ============================================================================
# Data splitting utilities
# ============================================================================

def create_gtzan_splits(
    genre_labels: list[str],
    tracks_per_genre: int = 100,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str = "data/splits",
) -> dict:
    """Create stratified train/val/test splits for GTZAN."""
    all_ids = []
    all_labels = []
    for genre in genre_labels:
        for i in range(tracks_per_genre):
            track_id = f"{genre}.{i:05d}"
            all_ids.append(track_id)
            all_labels.append(genre)

    # Stratified split
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        all_ids, all_labels, test_size=(1 - train_ratio),
        stratify=all_labels, random_state=seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5,
        stratify=temp_labels, random_state=seed
    )

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    for split_name, ids in splits.items():
        filepath = os.path.join(output_dir, f"gtzan_{split_name}.json")
        with open(filepath, "w") as f:
            json.dump(ids, f)

    print(f"GTZAN splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    return splits


def create_musiccaps_splits(
    metadata_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str = "data/splits",
) -> dict:
    """Create train/val/test splits for MusicCaps."""
    df = pd.read_csv(metadata_path)
    indices = list(range(len(df)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - train_ratio), random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed
    )

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    os.makedirs(output_dir, exist_ok=True)
    for split_name, idx in splits.items():
        filepath = os.path.join(output_dir, f"musiccaps_{split_name}.json")
        with open(filepath, "w") as f:
            json.dump(idx, f)

    print(f"MusicCaps splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return splits
