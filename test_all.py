"""
Comprehensive Full-Project Test Suite.
Verifies all datasets, models, weights, outputs, metrics, and inference pipelines.

Usage:
    python test_all.py
"""

import os
import sys
import json
import yaml
import time
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


def run_tests():
    print("=" * 75)
    print("   GNN-BERT MUSIC CONTEXT UNDERSTANDING -- FULL PROJECT TEST SUITE")
    print("=" * 75)

    passed_tests = 0
    total_tests = 0

    def check(name, condition, info=""):
        nonlocal passed_tests, total_tests
        total_tests += 1
        status = "[PASS]" if condition else "[FAIL]"
        if condition:
            passed_tests += 1
            print(f"  {status} {name}" + (f" ({info})" if info else ""))
        else:
            print(f"  {status} {name} -- {info}")
        return condition

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # TEST 1: Environment & GPU
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 1: Environment & Hardware Acceleration]")
    check("Python Version >= 3.10", sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}")
    check("CUDA Availability", torch.cuda.is_available(), f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # -------------------------------------------------------------------------
    # TEST 2: Data Pipeline & Splits
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 2: Data Directories & Preprocessed Datasets]")
    splits_dir = Path(config["paths"]["splits"])
    check("Split Files Exist", all((splits_dir / f).exists() for f in [
        "gtzan_train.json", "gtzan_val.json", "gtzan_test.json",
        "musiccaps_train.json", "musiccaps_val.json", "musiccaps_test.json",
        "tag_vocabulary.json"
    ]), "All 7 split files present")

    gtzan_graphs = list(Path(config["paths"]["processed_gtzan_graphs"]).glob("*.pt"))
    check("GTZAN PyG Segment Graphs", len(gtzan_graphs) >= 990, f"{len(gtzan_graphs)} / 1000 tracks converted")

    gtzan_mels = list(Path(config["paths"]["mel_specs"]).glob("*.npy"))
    check("GTZAN Mel-Spectrograms", len(gtzan_mels) >= 990, f"{len(gtzan_mels)} / 1000 arrays extracted")

    musiccaps_graphs = list(Path(config["paths"]["processed_musiccaps_graphs"]).glob("*.pt"))
    check("MusicCaps Segment Graphs", len(musiccaps_graphs) >= 50, f"{len(musiccaps_graphs)} paired clips converted")

    # -------------------------------------------------------------------------
    # TEST 3: Graph Construction Verification
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 3: Graph Structure & Feature Verification]")
    sample_graph = torch.load(gtzan_graphs[0], weights_only=False)
    check("Node Features Dimension", sample_graph.x.shape[1] == 160, f"Expected 160, got {sample_graph.x.shape[1]}")
    check("Graph Edge Indices Valid", sample_graph.edge_index.dim() == 2 and sample_graph.edge_index.shape[0] == 2, f"{sample_graph.edge_index.shape[1]} edges")

    # -------------------------------------------------------------------------
    # TEST 4: Model Checkpoints & Metrics
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 4: Trained Model Checkpoints & Metric Files]")
    results_dir = Path(config["paths"]["results"])

    models_info = [
        ("Task 1 (BERT Tag Classifier)", "task1_best.pt", "task1_metrics.json"),
        ("Task 2 (GraphSAGE GNN)", "task2_gnn_best.pt", "task2_gnn_metrics.json"),
        ("Task 2 (CNN Baseline)", "task2_cnn_best.pt", "task2_cnn_metrics.json"),
        ("Task 3 (GNN-BERT Fusion)", "task3_best.pt", "task3_metrics.json"),
        ("Task 4 (Contrastive Dual-Encoder)", "task4_best.pt", "task4_metrics.json"),
    ]

    for label, pt_name, json_name in models_info:
        pt_path = results_dir / pt_name
        json_path = results_dir / json_name
        pt_ok = pt_path.exists() and pt_path.stat().st_size > 1000
        json_ok = json_path.exists() and json_path.stat().st_size > 10
        check(f"{label} Weights", pt_ok, f"{pt_path.stat().st_size / (1024*1024):.1f} MB" if pt_ok else "Missing")
        check(f"{label} Metrics JSON", json_ok, json_path.name if json_ok else "Missing")

    # -------------------------------------------------------------------------
    # TEST 5: Plot Artifacts
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 5: Evaluation Plots & Visualizations]")
    plots_dir = Path(config["paths"]["plots"])
    expected_plots = [
        "task1_training_curves.png",
        "task2_gnn_training_curves.png",
        "task2_gnn_confusion.png",
        "task2_cnn_training_curves.png",
        "task2_cnn_confusion.png",
        "task3_training_curves.png",
        "task3_tsne.png",
    ]
    for p in expected_plots:
        check(f"Plot: {p}", (plots_dir / p).exists(), f"{(plots_dir / p).stat().st_size / 1024:.1f} KB" if (plots_dir / p).exists() else "Missing")

    # -------------------------------------------------------------------------
    # TEST 6: Live Inference Forward Pass
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 6: Live Model Inference Verification]")
    from bert_encoder import BERTTagClassifier, get_tokenizer
    from gnn_model import GraphSAGEModel
    from torch_geometric.data import Batch

    with open(splits_dir / "tag_vocabulary.json", "r", encoding="utf-8") as f:
        tag_vocab = json.load(f)

    # Task 1 live test
    t1_model = BERTTagClassifier(
        model_name=config["bert"]["model_name"],
        num_tags=len(tag_vocab),
    ).to(device)
    t1_model.load_state_dict(torch.load(results_dir / "task1_best.pt", map_location=device, weights_only=True))
    t1_model.eval()

    tokenizer = get_tokenizer(config["bert"]["model_name"])
    inp = tokenizer("rock music with electric guitar and energetic drums", return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out1 = t1_model(inp["input_ids"], inp["attention_mask"])
    check("Task 1 BERT Forward Pass", out1.shape == (1, len(tag_vocab)), f"Predicted logits shape: {out1.shape}")

    # Task 2 GNN live test
    t2_gnn = GraphSAGEModel(
        input_dim=config["graph"]["node_feature_dim"],
        hidden_dim=config["gnn"]["hidden_dim"],
        output_dim=config["num_genres"],
    ).to(device)
    t2_gnn.load_state_dict(torch.load(results_dir / "task2_gnn_best.pt", map_location=device, weights_only=True))
    t2_gnn.eval()

    batched_graph = Batch.from_data_list([sample_graph]).to(device)
    with torch.no_grad():
        out2 = t2_gnn(batched_graph)
    check("Task 2 GNN Forward Pass", out2.shape == (1, config["num_genres"]), f"Predicted genre logits: {out2.shape}")

    # -------------------------------------------------------------------------
    # TEST 7: Jupyter Notebooks Pre-Rendered
    # -------------------------------------------------------------------------
    print("\n[TEST GROUP 7: Jupyter Notebooks Integrity]")
    eda_nb_path = Path("notebooks/eda.ipynb")
    demo_nb_path = Path("notebooks/demo_context.ipynb")
    check("EDA Notebook (eda.ipynb)", eda_nb_path.exists() and eda_nb_path.stat().st_size > 5000, f"{eda_nb_path.stat().st_size / 1024:.1f} KB")
    check("Demo Notebook (demo_context.ipynb)", demo_nb_path.exists() and demo_nb_path.stat().st_size > 5000, f"{demo_nb_path.stat().st_size / 1024:.1f} KB")

    # -------------------------------------------------------------------------
    # FINAL SCORE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    score_pct = (passed_tests / total_tests) * 100
    print(f"   TEST SUMMARY: {passed_tests} / {total_tests} Tests Passed ({score_pct:.1f}%)")
    print("=" * 75)

    if passed_tests == total_tests:
        print("\n [SUCCESS] Full project passed all validation tests! Ready for submission & grading.")
    else:
        print(f"\n [WARNING] {total_tests - passed_tests} test(s) failed. Please review above logs.")


if __name__ == "__main__":
    run_tests()
