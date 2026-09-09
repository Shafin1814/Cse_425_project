"""
Evaluation metrics for all tasks.
Macro-F1, Micro-F1, AUC-PR, confusion matrix, t-SNE, retrieval R@K.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
    classification_report, accuracy_score,
)
from sklearn.manifold import TSNE


def compute_tag_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute multi-label tag classification metrics.

    Args:
        y_true: (N, K) binary ground truth
        y_pred: (N, K) predicted probabilities
        threshold: decision threshold

    Returns:
        dict with macro/micro F1, precision, recall, AUC-PR
    """
    y_pred_binary = (y_pred >= threshold).astype(int)

    metrics = {
        "macro_f1": f1_score(y_true, y_pred_binary, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred_binary, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred_binary, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred_binary, average="macro", zero_division=0),
    }

    # AUC-PR per tag (only for tags with positive samples)
    auc_scores = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() > 0:
            try:
                auc = average_precision_score(y_true[:, k], y_pred[:, k])
                auc_scores.append(auc)
            except ValueError:
                pass
    metrics["mean_auc_pr"] = np.mean(auc_scores) if auc_scores else 0.0

    return metrics


def compute_genre_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genre_labels: list[str] = None,
) -> dict:
    """
    Compute genre classification metrics.

    Args:
        y_true: (N,) true genre indices
        y_pred: (N,) predicted genre indices
        genre_labels: list of genre names
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }

    if genre_labels:
        report = classification_report(
            y_true, y_pred, target_names=genre_labels,
            output_dict=True, zero_division=0
        )
        metrics["per_class"] = report

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    save_path: str,
    title: str = "Confusion Matrix",
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_f1s: list[float],
    val_f1s: list[float],
    save_path: str,
    title: str = "Training Curves",
):
    """Plot training loss and F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_f1s, label="Train F1", linewidth=2)
    ax2.plot(epochs, val_f1s, label="Val F1", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title(f"{title} — Macro-F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    save_path: str,
    title: str = "t-SNE Visualization",
    perplexity: int = 30,
):
    """Plot t-SNE of embeddings colored by labels."""
    effective_perplexity = min(perplexity, max((len(embeddings) - 1) // 3, 2))
    tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label < len(label_names) else str(label)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=name, alpha=0.6, s=20)

    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


def print_example_predictions(
    captions: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tag_names: list[str],
    n: int = 5,
    threshold: float = 0.5,
):
    """Print n example predictions with true vs predicted tags."""
    y_pred_binary = (y_pred >= threshold).astype(int)

    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)

    for i in range(min(n, len(captions))):
        true_tags = [tag_names[j] for j in range(len(tag_names)) if y_true[i, j] == 1]
        pred_tags = [tag_names[j] for j in range(len(tag_names)) if y_pred_binary[i, j] == 1]

        print(f"\n--- Example {i + 1} ---")
        print(f"Caption: {captions[i][:200]}")
        print(f"True tags:      {true_tags}")
        print(f"Predicted tags:  {pred_tags}")


def print_retrieval_examples(
    query_captions: list[str],
    retrieved_captions: list[list[str]],
    scores: np.ndarray,
    n: int = 10,
):
    """Print retrieval examples: query caption → top retrieved."""
    print("\n" + "=" * 80)
    print("RETRIEVAL EXAMPLES (Caption → Audio → Top-3 Matched Captions)")
    print("=" * 80)

    for i in range(min(n, len(query_captions))):
        print(f"\n--- Query {i + 1} ---")
        print(f"Query: {query_captions[i][:200]}")
        for j, (cap, score) in enumerate(zip(retrieved_captions[i][:3], scores[i][:3])):
            print(f"  Match {j + 1} (sim={score:.4f}): {cap[:200]}")
