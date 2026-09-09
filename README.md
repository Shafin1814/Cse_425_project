# GNN-Based BERT for Understanding Context from Music

**Course:** Neural Networks (CSE425 / EEE474 / CSE715)

A hybrid BERT + Graph Neural Network (GNN) system that understands musical context by combining:
- **BERT** — contextual language representations from captions and tags (MusicCaps)
- **GNN** — message passing on music structure graphs built from audio segments

## Project Structure

```
├── config.yaml              # All hyperparameters and paths
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/
│   │   ├── gtzan/            # 1000 GTZAN tracks (10 genres × 100)
│   │   └── musiccaps_audio/  # Downloaded MusicCaps WAV clips
│   ├── processed/            # PyG graphs, mel-specs, BERT caches
│   └── splits/               # train/val/test JSON splits
├── notebooks/
│   ├── eda.ipynb
│   └── demo_context.ipynb
├── src/
│   ├── audio_features.py     # Mel, chroma, MFCC extraction & segmentation
│   ├── graph_builder.py      # Segment + chord-transition graph construction
│   ├── bert_encoder.py       # DistilBERT wrapper + tag classifier (Task 1)
│   ├── gnn_model.py          # GraphSAGE, GAT, CNN baseline (Task 2)
│   ├── fusion_model.py       # Cross-attention GNN-BERT fusion (Task 3)
│   ├── contrastive.py        # InfoNCE dual-encoder (Task 4)
│   ├── dataset.py            # All PyTorch datasets & data loaders
│   ├── preprocess.py         # Audio → graphs preprocessing pipeline
│   ├── train.py              # Unified training script for all tasks
│   └── evaluate.py           # Metrics: F1, AUC-PR, R@K, t-SNE, plots
├── results/
│   ├── metrics.json
│   ├── plots/
│   └── retrieval_examples/
└── report/
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download MusicCaps metadata
```bash
python src/download_musiccaps.py --metadata-only
```

### 2. Download MusicCaps audio (optional, for Tasks 2-4)
```bash
python src/download_musiccaps.py --limit 500
```

### 3. Preprocess audio → graphs
```bash
python src/preprocess.py --dataset all
```

### 4. Train models
```bash
# Task 1: BERT Tag Classifier
python src/train.py --task 1

# Task 2: GNN Genre Classifier (+ CNN baseline)
python src/train.py --task 2 --cnn-baseline

# Task 3: GNN-BERT Fusion
python src/train.py --task 3

# Task 4: Contrastive Alignment
python src/train.py --task 4
```

## Testing & Verification

### Automated Full Project Test Suite (29/29 checks)
Run the automated test suite to verify environment, datasets, trained checkpoints, forward passes, plots, and notebooks:
```bash
python test_all.py
```

### Interactive Live Demo (All 4 Tasks in ~3s)
Run live inference demonstrations across Task 1 (tag prediction), Task 2 (GNN vs. CNN table), Task 3 (fusion metrics), and Task 4 (cross-modal retrieval):
```bash
python demo.py
```

### Interactive Jupyter Notebooks
Open either pre-rendered notebook in VS Code or JupyterLab:
- `notebooks/demo_context.ipynb` — Visual interactive walkthrough of all 4 tasks with live prediction widgets.
- `notebooks/eda.ipynb` — Full exploratory data analysis with waveforms, spectrograms, NetworkX segment graphs, and class distributions.


## Tasks

| Task | Model | Dataset | Goal |
|------|-------|---------|------|
| 1 (Easy) | DistilBERT classifier | MusicCaps captions | Multi-label tag prediction |
| 2 (Medium) | GraphSAGE on segment graphs | GTZAN audio | Genre classification |
| 3 (Hard) | Cross-attention GNN-BERT | MusicCaps + audio graphs | Multi-context tag prediction |
| 4 (Advanced) | Contrastive dual-encoder | MusicCaps pairs | Cross-modal retrieval |

## Environment

- Python 3.12
- PyTorch 2.6 + CUDA 12.4
- PyTorch Geometric 2.8
- Transformers 5.15
- Librosa 1.0

## Results

Results are saved to `results/` including:
- Training curves (loss, F1 vs. epoch)
- Confusion matrices
- t-SNE visualizations
- Retrieval examples
- Metrics JSON files
