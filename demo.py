"""
Interactive Multimodal Music Context Demo CLI.
Run all 4 tasks from the command line with instant results.

Usage:
    python demo.py
    python demo.py --prompt "fast energetic guitar solo with heavy drums and rock bass"
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Safe Windows stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from bert_encoder import BERTTagClassifier, get_tokenizer


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="GNN-BERT Music Context Demo")
    parser.add_argument(
        "--prompt",
        type=str,
        default="An energetic rock track with heavy electric guitar distortion, punchy drums, and intense bass guitar.",
        help="Custom text prompt to predict musical tags for",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" GNN-BERT FOR UNDERSTANDING CONTEXT FROM MUSIC -- INTERACTIVE DEMO")
    print("=" * 70)

    config = load_config()
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Python: {sys.version.split()[0]}")

    # Load tag vocabulary
    vocab_path = os.path.join(config["paths"]["splits"], "tag_vocabulary.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        tag_vocab = json.load(f)

    # -------------------------------------------------------------------------
    # Task 1: BERT Multi-Label Tag Predictor
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TASK 1: BERT Musical Context Tag Predictor (MusicCaps)")
    print("-" * 70)

    task1_weights = os.path.join(config["paths"]["results"], "task1_best.pt")
    if os.path.exists(task1_weights):
        print("[*] Loading Task 1 model weights...")
        model = BERTTagClassifier(
            model_name=config["bert"]["model_name"],
            num_tags=len(tag_vocab),
            freeze_layers=config["bert"]["freeze_layers"],
        ).to(device)
        model.load_state_dict(torch.load(task1_weights, map_location=device, weights_only=True))
        model.eval()

        tokenizer = get_tokenizer(config["bert"]["model_name"])

        def predict(prompt_text, top_k=8, threshold=0.30):
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                max_length=128,
                padding="max_length",
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.sigmoid(logits).cpu().squeeze().numpy()
            sorted_idx = np.argsort(probs)[::-1][:top_k]
            return [(tag_vocab[i], float(probs[i])) for i in sorted_idx if probs[i] >= threshold]

        test_prompts = [
            args.prompt,
            "A slow tempo classical piano piece with emotional, sad, and gentle acoustic melodies.",
            "A groovy hip hop track with deep 808 bass, punchy kick, fast tempo rapping, and synthetic snares.",
        ]

        for p in test_prompts:
            print(f"\n[Prompt]: \"{p}\"")
            tags = predict(p)
            print("  Predicted Aspect Tags:")
            if tags:
                for tag, conf in tags:
                    print(f"    - {tag:22s} ({conf * 100:.1f}%)")
            else:
                print("    - (No tags above threshold)")
    else:
        print("Task 1 model weights not found at results/task1_best.pt")

    # -------------------------------------------------------------------------
    # Task 2: GNN vs CNN Genre Classification Comparison
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TASK 2: GNN vs CNN Baseline on GTZAN Genre Classification")
    print("-" * 70)

    gnn_metrics_file = os.path.join(config["paths"]["results"], "task2_gnn_metrics.json")
    cnn_metrics_file = os.path.join(config["paths"]["results"], "task2_cnn_metrics.json")

    rows = []
    if os.path.exists(gnn_metrics_file):
        with open(gnn_metrics_file, "r", encoding="utf-8") as f:
            m_gnn = json.load(f)
        rows.append({
            "Model Architecture": "GNN (GraphSAGE on Segment Graphs)",
            "Accuracy": f"{m_gnn.get('accuracy', 0)*100:.2f}%",
            "Macro-F1": f"{m_gnn.get('macro_f1', 0):.4f}",
            "Micro-F1": f"{m_gnn.get('micro_f1', 0):.4f}",
        })

    if os.path.exists(cnn_metrics_file):
        with open(cnn_metrics_file, "r", encoding="utf-8") as f:
            m_cnn = json.load(f)
        rows.append({
            "Model Architecture": "CNN Baseline (2D Mel-Spectrograms)",
            "Accuracy": f"{m_cnn.get('accuracy', 0)*100:.2f}%",
            "Macro-F1": f"{m_cnn.get('macro_f1', 0):.4f}",
            "Micro-F1": f"{m_cnn.get('micro_f1', 0):.4f}",
        })

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    else:
        print("Task 2 metrics not found in results/")

    # -------------------------------------------------------------------------
    # Task 3 & 4: Contrastive Retrieval
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TASK 4: Contrastive Cross-Modal Alignment & Retrieval (InfoNCE)")
    print("-" * 70)

    t4_metrics_file = os.path.join(config["paths"]["results"], "task4_metrics.json")
    if os.path.exists(t4_metrics_file):
        with open(t4_metrics_file, "r", encoding="utf-8") as f:
            t4_m = json.load(f)
        print("Retrieval Recall@K Metrics:")
        for k, v in t4_m.items():
            print(f"   {k:15s}: {v:.4f}")

    qual_file = os.path.join(config["paths"]["retrieval_examples"], "retrieval_qualitative.json")
    if os.path.exists(qual_file):
        with open(qual_file, "r", encoding="utf-8") as f:
            examples = json.load(f)
        print(f"\nSample Cross-Modal Retrieval (Text -> Audio Matches):")
        for i, ex in enumerate(examples[:2]):
            print(f"\n   Query {i+1}: \"{ex['query_caption'][:90]}...\"")
            print(f"   Ground Truth Clip: {ex['ground_truth_ytid']}")
            print("   Top Matches:")
            for m in ex['top_matches']:
                print(f"     -> Audio: {m['matched_ytid']} (similarity={m['similarity']:.4f})")

    print("\n" + "=" * 70)
    print(" [OK] All tasks demonstrated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
