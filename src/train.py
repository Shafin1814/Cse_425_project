"""
Unified training script for all tasks.
Usage:
    python src/train.py --task 1   # BERT tag classifier
    python src/train.py --task 2   # GNN genre classifier
    python src/train.py --task 3   # GNN-BERT fusion
    python src/train.py --task 4   # Contrastive alignment
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_geometric.loader import DataLoader as PyGDataLoader
from bert_encoder import BERTTagClassifier, BERTEncoder
from gnn_model import GraphSAGEModel, GATModel, CNNMelBaseline
from fusion_model import GNNBERTFusionModel
from contrastive import ContrastiveDualEncoder, compute_retrieval_metrics
from dataset import (
    GTZANGraphDataset, GTZANMelDataset,
    MusicCapsTextDataset, MusicCapsFusionDataset,
    create_gtzan_splits, create_musiccaps_splits,
    build_tag_vocabulary,
)
from evaluate import (
    compute_tag_metrics, compute_genre_metrics,
    plot_confusion_matrix, plot_training_curves, plot_tsne,
    print_example_predictions, print_retrieval_examples,
)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Task 1: BERT Tag Classifier
# ============================================================================

def train_task1(config: dict, device: torch.device):
    """Train BERT multi-label tag classifier on MusicCaps captions."""
    print("\n" + "=" * 60)
    print("TASK 1: BERT Multi-Label Tag Classifier")
    print("=" * 60)

    metadata_path = config["paths"]["musiccaps_metadata"]
    cfg = config["training"]["task1"]

    # Build tag vocabulary
    import pandas as pd
    df = pd.read_csv(metadata_path)
    tag_vocab = build_tag_vocabulary(df, min_count=10, max_tags=50)
    print(f"Tag vocabulary: {len(tag_vocab)} tags")
    print(f"Tags: {tag_vocab[:20]}...")

    # Save tag vocab
    os.makedirs(config["paths"]["splits"], exist_ok=True)
    with open(os.path.join(config["paths"]["splits"], "tag_vocabulary.json"), "w") as f:
        json.dump(tag_vocab, f)

    # Create splits
    splits = create_musiccaps_splits(metadata_path, output_dir=config["paths"]["splits"])

    # Create datasets
    train_ds = MusicCapsTextDataset(metadata_path, tag_vocab,
                                     tokenizer_name=config["bert"]["model_name"],
                                     max_length=config["bert"]["max_length"],
                                     split_indices=splits["train"])
    val_ds = MusicCapsTextDataset(metadata_path, tag_vocab,
                                   tokenizer_name=config["bert"]["model_name"],
                                   max_length=config["bert"]["max_length"],
                                   split_indices=splits["val"])
    test_ds = MusicCapsTextDataset(metadata_path, tag_vocab,
                                    tokenizer_name=config["bert"]["model_name"],
                                    max_length=config["bert"]["max_length"],
                                    split_indices=splits["test"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    # Model
    model = BERTTagClassifier(
        num_tags=len(tag_vocab),
        model_name=config["bert"]["model_name"],
        freeze_layers=config["bert"]["freeze_layers"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # Training
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        # Train
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        train_metrics = compute_tag_metrics(all_labels, all_preds)

        # Validate
        val_loss, val_metrics, _, _ = evaluate_task1(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_metrics["macro_f1"])
        val_f1s.append(val_metrics["macro_f1"])

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} F1={train_metrics['macro_f1']:.4f} | "
              f"Val Loss={val_loss:.4f} F1={val_metrics['macro_f1']:.4f} AUC-PR={val_metrics['mean_auc_pr']:.4f}")

        # Early stopping
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config["paths"]["results"], "task1_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and test
    model.load_state_dict(torch.load(os.path.join(config["paths"]["results"], "task1_best.pt"), weights_only=True))
    test_loss, test_metrics, test_preds, test_labels = evaluate_task1(model, test_loader, criterion, device)

    print(f"\nTest Results: Macro-F1={test_metrics['macro_f1']:.4f} Micro-F1={test_metrics['micro_f1']:.4f} "
          f"AUC-PR={test_metrics['mean_auc_pr']:.4f}")

    # Plot training curves
    plots_dir = config["paths"]["plots"]
    os.makedirs(plots_dir, exist_ok=True)
    plot_training_curves(train_losses, val_losses, train_f1s, val_f1s,
                         os.path.join(plots_dir, "task1_training_curves.png"),
                         title="Task 1: BERT Tag Classifier")

    # Example predictions
    captions = [test_ds[i]["caption"] for i in range(min(5, len(test_ds)))]
    print_example_predictions(captions, test_labels[:5], test_preds[:5], tag_vocab, n=5)

    # Save metrics
    with open(os.path.join(config["paths"]["results"], "task1_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    return model, test_metrics


def evaluate_task1(model, loader, criterion, device):
    """Evaluate BERT tagger."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_tag_metrics(all_labels, all_preds)

    return avg_loss, metrics, all_preds, all_labels


# ============================================================================
# Task 2: GNN Genre Classifier
# ============================================================================

def train_task2(config: dict, device: torch.device):
    """Train GNN on GTZAN segment graphs for genre classification."""
    print("\n" + "=" * 60)
    print("TASK 2: GNN Genre Classifier (GTZAN)")
    print("=" * 60)

    cfg = config["training"]["task2"]
    graph_dir = config["paths"]["processed_gtzan_graphs"]
    genre_labels = config["genre_labels"]

    # Create splits
    splits = create_gtzan_splits(genre_labels, output_dir=config["paths"]["splits"])

    # Datasets
    train_ds = GTZANGraphDataset(graph_dir, genre_labels, split_ids=splits["train"])
    val_ds = GTZANGraphDataset(graph_dir, genre_labels, split_ids=splits["val"])
    test_ds = GTZANGraphDataset(graph_dir, genre_labels, split_ids=splits["test"])

    train_loader = PyGDataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = PyGDataLoader(test_ds, batch_size=cfg["batch_size"])

    # GNN model
    model = GraphSAGEModel(
        input_dim=config["graph"]["node_feature_dim"],
        hidden_dim=config["gnn"]["hidden_dim"],
        output_dim=config["num_genres"],
        num_layers=config["gnn"]["num_layers"],
        dropout=config["gnn"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]", leave=False):
            data = data.to(device)
            logits = model(data)
            loss = criterion(logits, data.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        train_metrics = compute_genre_metrics(all_labels, all_preds)

        # Validate
        val_loss, val_metrics, _, _ = evaluate_task2(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_metrics["macro_f1"])
        val_f1s.append(val_metrics["macro_f1"])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_metrics['accuracy']:.4f} | "
                  f"Val Loss={val_loss:.4f} Acc={val_metrics['accuracy']:.4f} F1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config["paths"]["results"], "task2_gnn_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test
    model.load_state_dict(torch.load(os.path.join(config["paths"]["results"], "task2_gnn_best.pt"), weights_only=True))
    test_loss, test_metrics, test_preds, test_labels = evaluate_task2(model, test_loader, criterion, device)

    print(f"\nGNN Test: Accuracy={test_metrics['accuracy']:.4f} Macro-F1={test_metrics['macro_f1']:.4f}")

    # Confusion matrix
    plots_dir = config["paths"]["plots"]
    os.makedirs(plots_dir, exist_ok=True)
    plot_confusion_matrix(test_labels, test_preds, genre_labels,
                          os.path.join(plots_dir, "task2_gnn_confusion.png"),
                          title="Task 2: GNN Genre Classification")

    plot_training_curves(train_losses, val_losses, train_f1s, val_f1s,
                         os.path.join(plots_dir, "task2_gnn_training_curves.png"),
                         title="Task 2: GNN Genre Classifier")

    # Save metrics
    with open(os.path.join(config["paths"]["results"], "task2_gnn_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2, default=str)

    return model, test_metrics


def evaluate_task2(model, loader, criterion, device):
    """Evaluate GNN genre classifier."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            loss = criterion(logits, data.y)

            total_loss += loss.item()
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_genre_metrics(all_labels, all_preds)

    return avg_loss, metrics, all_preds, all_labels


# ============================================================================
# Task 2 CNN Baseline
# ============================================================================

def train_cnn_baseline(config: dict, device: torch.device):
    """Train CNN baseline on mel-spectrograms."""
    print("\n" + "=" * 60)
    print("BASELINE: CNN Mel-Spectrogram Genre Classifier")
    print("=" * 60)

    cfg = config["training"]["task2"]
    mel_dir = config["paths"]["mel_specs"]
    genre_labels = config["genre_labels"]

    # Load splits
    splits_dir = config["paths"]["splits"]
    with open(os.path.join(splits_dir, "gtzan_train.json")) as f:
        train_ids = json.load(f)
    with open(os.path.join(splits_dir, "gtzan_val.json")) as f:
        val_ids = json.load(f)
    with open(os.path.join(splits_dir, "gtzan_test.json")) as f:
        test_ids = json.load(f)

    train_ds = GTZANMelDataset(mel_dir, genre_labels, split_ids=train_ids)
    val_ds = GTZANMelDataset(mel_dir, genre_labels, split_ids=val_ids)
    test_ds = GTZANMelDataset(mel_dir, genre_labels, split_ids=test_ids)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    model = CNNMelBaseline(num_classes=config["num_genres"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0
        train_preds, train_labels_list = [], []
        for mel, label in train_loader:
            mel, label = mel.to(device), label.to(device)
            logits = model(mel)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_preds.append(logits.argmax(-1).cpu().numpy())
            train_labels_list.append(label.cpu().numpy())

        train_loss = epoch_loss / max(len(train_loader), 1)
        train_preds = np.concatenate(train_preds)
        train_labels_arr = np.concatenate(train_labels_list)
        train_acc = accuracy_score(train_labels_arr, train_preds)

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for mel, label in val_loader:
                mel, label = mel.to(device), label.to(device)
                logits = model(mel)
                loss = criterion(logits, label)
                val_loss += loss.item()
                val_preds.append(logits.argmax(-1).cpu().numpy())
                val_labels.append(label.cpu().numpy())

        val_loss = val_loss / max(len(val_loader), 1)
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_acc = accuracy_score(val_labels, val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config["paths"]["results"], "task2_cnn_best.pt"))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

    # Test
    model.load_state_dict(torch.load(os.path.join(config["paths"]["results"], "task2_cnn_best.pt"), weights_only=True))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for mel, label in test_loader:
            mel = mel.to(device)
            logits = model(mel)
            test_preds.append(logits.argmax(-1).cpu().numpy())
            test_labels.append(label.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)
    test_metrics = compute_genre_metrics(test_labels, test_preds, genre_labels)

    print(f"\nCNN Baseline Test: Accuracy={test_metrics['accuracy']:.4f} Macro-F1={test_metrics['macro_f1']:.4f}")

    plot_confusion_matrix(test_labels, test_preds, genre_labels,
                          os.path.join(config["paths"]["plots"], "task2_cnn_confusion.png"),
                          title="Baseline: CNN Mel-Spec Genre Classification")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         os.path.join(config["paths"]["plots"], "task2_cnn_training_curves.png"),
                         title="Baseline: CNN Mel-Spec Classifier")

    with open(os.path.join(config["paths"]["results"], "task2_cnn_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2, default=str)

    return model, test_metrics


# ============================================================================
# Task 3: GNN-BERT Fusion
# ============================================================================

def train_task3(config: dict, device: torch.device):
    """Train GNN-BERT fusion model."""
    print("\n" + "=" * 60)
    print("TASK 3: GNN-BERT Fusion for Multi-Context Understanding")
    print("=" * 60)

    cfg = config["training"]["task3"]
    metadata_path = config["paths"]["musiccaps_metadata"]
    graph_dir = config["paths"]["processed_musiccaps_graphs"]

    # Load tag vocabulary
    with open(os.path.join(config["paths"]["splits"], "tag_vocabulary.json")) as f:
        tag_vocab = json.load(f)

    # Load splits
    with open(os.path.join(config["paths"]["splits"], "musiccaps_train.json")) as f:
        train_idx = json.load(f)
    with open(os.path.join(config["paths"]["splits"], "musiccaps_val.json")) as f:
        val_idx = json.load(f)
    with open(os.path.join(config["paths"]["splits"], "musiccaps_test.json")) as f:
        test_idx = json.load(f)

    # Datasets
    train_ds = MusicCapsFusionDataset(metadata_path, graph_dir, tag_vocab,
                                       tokenizer_name=config["bert"]["model_name"],
                                       split_indices=train_idx)
    val_ds = MusicCapsFusionDataset(metadata_path, graph_dir, tag_vocab,
                                     tokenizer_name=config["bert"]["model_name"],
                                     split_indices=val_idx)
    test_ds = MusicCapsFusionDataset(metadata_path, graph_dir, tag_vocab,
                                      tokenizer_name=config["bert"]["model_name"],
                                      split_indices=test_idx)

    def collate_fusion(batch):
        from torch_geometric.data import Batch
        graphs = Batch.from_data_list([b["graph"] for b in batch])
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        return {"graph": graphs, "input_ids": input_ids,
                "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fusion)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], collate_fn=collate_fusion)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], collate_fn=collate_fusion)

    # Build model components
    gnn_encoder = GraphSAGEModel(
        input_dim=config["graph"]["node_feature_dim"],
        hidden_dim=config["gnn"]["hidden_dim"],
        output_dim=config["num_genres"],  # will use encode_graph, not classifier
        num_layers=config["gnn"]["num_layers"],
        dropout=config["gnn"]["dropout"],
    )
    bert_encoder = BERTEncoder(
        model_name=config["bert"]["model_name"],
        freeze_layers=config["bert"]["freeze_layers"],
    )

    model = GNNBERTFusionModel(
        gnn_encoder=gnn_encoder,
        bert_encoder=bert_encoder,
        num_tags=len(tag_vocab),
        fusion_method=config["fusion"]["method"],
        graph_dim=config["gnn"]["hidden_dim"],
        text_dim=config["bert"]["hidden_dim"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]", leave=False):
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(graph, input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        train_loss = epoch_loss / max(len(train_loader), 1)
        all_preds = np.concatenate(all_preds) if all_preds else np.array([])
        all_labels = np.concatenate(all_labels) if all_labels else np.array([])
        train_metrics = compute_tag_metrics(all_labels, all_preds) if len(all_labels) > 0 else {"macro_f1": 0}

        # Validate
        val_loss, val_metrics = evaluate_fusion(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_metrics["macro_f1"])
        val_f1s.append(val_metrics["macro_f1"])

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} F1={train_metrics['macro_f1']:.4f} | "
              f"Val Loss={val_loss:.4f} F1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config["paths"]["results"], "task3_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test
    model.load_state_dict(torch.load(os.path.join(config["paths"]["results"], "task3_best.pt"), weights_only=True))
    test_loss, test_metrics = evaluate_fusion(model, test_loader, criterion, device)
    print(f"\nFusion Test: Macro-F1={test_metrics['macro_f1']:.4f} AUC-PR={test_metrics['mean_auc_pr']:.4f}")

    # Plot
    plots_dir = config["paths"]["plots"]
    plot_training_curves(train_losses, val_losses, train_f1s, val_f1s,
                         os.path.join(plots_dir, "task3_training_curves.png"),
                         title="Task 3: GNN-BERT Fusion")

    # t-SNE visualization
    collect_and_plot_tsne(model, test_loader, device,
                          os.path.join(plots_dir, "task3_tsne.png"), tag_vocab)

    with open(os.path.join(config["paths"]["results"], "task3_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    return model, test_metrics


def evaluate_fusion(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(graph, input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    metrics = compute_tag_metrics(all_labels, all_preds) if len(all_labels) > 0 else {"macro_f1": 0, "mean_auc_pr": 0}
    return avg_loss, metrics


def collect_and_plot_tsne(model, loader, device, save_path, tag_vocab):
    """Collect fused embeddings and plot t-SNE."""
    model.eval()
    embeddings, labels_list = [], []

    with torch.no_grad():
        for batch in loader:
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lab = batch["labels"]

            z = model.get_fused_embedding(graph, input_ids, attention_mask)
            embeddings.append(z.cpu().numpy())
            # Use the dominant tag as label
            dominant = lab.argmax(dim=-1).numpy()
            labels_list.append(dominant)

    if embeddings:
        embeddings = np.concatenate(embeddings)
        labels_arr = np.concatenate(labels_list)
        if len(embeddings) >= 5:
            top_tags = tag_vocab[:10] if len(tag_vocab) >= 10 else tag_vocab
            plot_tsne(embeddings, labels_arr, top_tags, save_path,
                      title="Task 3: t-SNE of Fused Embeddings")


# ============================================================================
# Task 4: Contrastive Alignment
# ============================================================================

def train_task4(config: dict, device: torch.device):
    """Train contrastive dual-encoder."""
    print("\n" + "=" * 60)
    print("TASK 4: Contrastive Cross-Modal Alignment (MusicCaps)")
    print("=" * 60)

    cfg = config["training"]["task4"]
    metadata_path = config["paths"]["musiccaps_metadata"]
    graph_dir = config["paths"]["processed_musiccaps_graphs"]

    with open(os.path.join(config["paths"]["splits"], "tag_vocabulary.json")) as f:
        tag_vocab = json.load(f)
    with open(os.path.join(config["paths"]["splits"], "musiccaps_train.json")) as f:
        train_idx = json.load(f)
    with open(os.path.join(config["paths"]["splits"], "musiccaps_val.json")) as f:
        val_idx = json.load(f)
    with open(os.path.join(config["paths"]["splits"], "musiccaps_test.json")) as f:
        test_idx = json.load(f)

    train_ds = MusicCapsFusionDataset(metadata_path, graph_dir, tag_vocab,
                                       tokenizer_name=config["bert"]["model_name"],
                                       split_indices=train_idx)
    val_ds = MusicCapsFusionDataset(metadata_path, graph_dir, tag_vocab,
                                     tokenizer_name=config["bert"]["model_name"],
                                     split_indices=val_idx)
    test_ds = MusicCapsFusionDataset(metadata_path, graph_dir, tag_vocab,
                                      tokenizer_name=config["bert"]["model_name"],
                                      split_indices=test_idx)

    def collate_fusion(batch):
        from torch_geometric.data import Batch
        graphs = Batch.from_data_list([b["graph"] for b in batch])
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        return {"graph": graphs, "input_ids": input_ids,
                "attention_mask": attention_mask, "labels": labels}

    drop_train = (len(train_ds) >= cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collate_fusion, drop_last=drop_train)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            collate_fn=collate_fusion, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             collate_fn=collate_fusion, drop_last=False)

    gnn_encoder = GraphSAGEModel(
        input_dim=config["graph"]["node_feature_dim"],
        hidden_dim=config["gnn"]["hidden_dim"],
        output_dim=config["num_genres"],
        num_layers=config["gnn"]["num_layers"],
        dropout=config["gnn"]["dropout"],
    )
    bert_encoder = BERTEncoder(
        model_name=config["bert"]["model_name"],
        freeze_layers=config["bert"]["freeze_layers"],
    )

    model = ContrastiveDualEncoder(
        gnn_encoder=gnn_encoder,
        bert_encoder=bert_encoder,
        graph_dim=config["gnn"]["hidden_dim"],
        text_dim=config["bert"]["hidden_dim"],
        projection_dim=config["contrastive"]["projection_dim"],
        temperature=config["contrastive"]["temperature"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", leave=False):
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            result = model(graph, input_ids, attention_mask)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / max(len(train_loader), 1)

        # Validate
        val_loss = evaluate_contrastive(model, val_loader, device)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config["paths"]["results"], "task4_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Evaluate retrieval
    model.load_state_dict(torch.load(os.path.join(config["paths"]["results"], "task4_best.pt"), weights_only=True))
    retrieval_metrics = evaluate_retrieval(model, test_loader, device)

    print(f"\nRetrieval Metrics:")
    for k, v in retrieval_metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(config["paths"]["results"], "task4_metrics.json"), "w") as f:
        json.dump(retrieval_metrics, f, indent=2)

    return model, retrieval_metrics


def evaluate_contrastive(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            result = model(graph, input_ids, attention_mask)
            total_loss += result["loss"].item()
    return total_loss / max(len(loader), 1)


def evaluate_retrieval(model, loader, device):
    model.eval()
    all_g_emb, all_t_emb = [], []

    with torch.no_grad():
        for batch in loader:
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            g_emb = model.encode_graph(graph)
            t_emb = model.encode_text(input_ids, attention_mask)

            all_g_emb.append(g_emb.cpu())
            all_t_emb.append(t_emb.cpu())

    all_g_emb = torch.cat(all_g_emb)
    all_t_emb = torch.cat(all_t_emb)

    similarity = torch.matmul(all_g_emb, all_t_emb.t())
    metrics = compute_retrieval_metrics(similarity, ks=[1, 5, 10])

    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GNN-BERT Music Context Models")
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4], required=True,
                        help="Task number to train")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--cnn-baseline", action="store_true",
                        help="Train CNN baseline (Task 2 only)")
    parser.add_argument("--cnn-only", action="store_true",
                        help="Train ONLY CNN baseline (Task 2 only)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config["paths"]["results"], exist_ok=True)
    os.makedirs(config["paths"]["plots"], exist_ok=True)

    if args.task == 1:
        train_task1(config, device)
    elif args.task == 2:
        if args.cnn_only:
            train_cnn_baseline(config, device)
        else:
            train_task2(config, device)
            if args.cnn_baseline:
                train_cnn_baseline(config, device)
    elif args.task == 3:
        train_task3(config, device)
    elif args.task == 4:
        train_task4(config, device)


if __name__ == "__main__":
    main()
