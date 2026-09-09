"""
Preprocessing script: builds graphs and mel-spectrograms from raw audio.
Usage:
    python src/preprocess.py --dataset gtzan
    python src/preprocess.py --dataset musiccaps
    python src/preprocess.py --dataset all
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_features import extract_track_features, extract_full_mel_spectrogram
from graph_builder import build_segment_graph


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_gtzan(config: dict):
    """Build segment graphs and mel-spectrograms for GTZAN dataset."""
    print("\n" + "=" * 60)
    print("Preprocessing GTZAN Dataset")
    print("=" * 60)

    audio_dir = Path(config["paths"]["gtzan_audio"])
    graph_dir = Path(config["paths"]["processed_gtzan_graphs"])
    mel_dir = Path(config["paths"]["mel_specs"])
    graph_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)

    sr = config["audio"]["sample_rate"]
    segment_duration = config["audio"]["segment_duration"]
    similarity_threshold = config["graph"]["similarity_threshold"]

    genre_labels = config["genre_labels"]
    total = 0
    errors = 0

    for genre in genre_labels:
        genre_dir = audio_dir / genre
        if not genre_dir.exists():
            print(f"WARNING: Genre dir not found: {genre_dir}")
            continue

        audio_files = sorted(genre_dir.glob("*.wav")) + sorted(genre_dir.glob("*.au"))

        for audio_path in tqdm(audio_files, desc=f"Processing {genre}"):
            track_id = audio_path.stem  # e.g., "blues.00001"
            graph_path = graph_dir / f"{track_id}.pt"
            mel_path = mel_dir / f"{track_id}.npy"

            # Skip if already processed
            if graph_path.exists() and mel_path.exists():
                total += 1
                continue

            try:
                # Extract segment features and build graph
                node_features, _ = extract_track_features(
                    str(audio_path), sr=sr,
                    segment_duration=segment_duration,
                    n_mels=config["audio"]["n_mels"],
                    n_chroma=config["audio"]["n_chroma"],
                    n_mfcc=config["audio"]["n_mfcc"],
                    hop_length=config["audio"]["hop_length"],
                )

                graph = build_segment_graph(
                    node_features,
                    similarity_threshold=similarity_threshold,
                )

                torch.save(graph, graph_path)

                # Extract mel-spectrogram for CNN baseline
                mel = extract_full_mel_spectrogram(
                    str(audio_path), sr=sr,
                    n_mels=config["audio"]["n_mels"],
                    hop_length=config["audio"]["hop_length"],
                )
                np.save(mel_path, mel)

                total += 1

            except Exception as e:
                errors += 1
                print(f"  ERROR processing {audio_path.name}: {e}")

    print(f"\nGTZAN preprocessing complete: {total} tracks, {errors} errors")
    print(f"Graphs saved to: {graph_dir}")
    print(f"Mel-specs saved to: {mel_dir}")


def preprocess_musiccaps(config: dict):
    """Build segment graphs for MusicCaps audio clips."""
    print("\n" + "=" * 60)
    print("Preprocessing MusicCaps Dataset")
    print("=" * 60)

    import pandas as pd

    audio_dir = Path(config["paths"]["musiccaps_audio"])
    graph_dir = Path(config["paths"]["processed_musiccaps_graphs"])
    graph_dir.mkdir(parents=True, exist_ok=True)

    sr = config["audio"]["sample_rate"]
    segment_duration = config["audio"]["segment_duration"]
    similarity_threshold = config["graph"]["similarity_threshold"]

    # List available audio files
    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")

    total = 0
    errors = 0

    for audio_path in tqdm(audio_files, desc="Processing MusicCaps"):
        ytid = audio_path.stem
        graph_path = graph_dir / f"{ytid}.pt"

        if graph_path.exists():
            total += 1
            continue

        try:
            node_features, _ = extract_track_features(
                str(audio_path), sr=sr,
                segment_duration=segment_duration,
                n_mels=config["audio"]["n_mels"],
                n_chroma=config["audio"]["n_chroma"],
                n_mfcc=config["audio"]["n_mfcc"],
                hop_length=config["audio"]["hop_length"],
                duration=10.0,  # MusicCaps clips are 10s
            )

            graph = build_segment_graph(
                node_features,
                similarity_threshold=similarity_threshold,
            )

            torch.save(graph, graph_path)
            total += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ERROR processing {ytid}: {e}")

    print(f"\nMusicCaps preprocessing complete: {total} graphs, {errors} errors")


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio datasets")
    parser.add_argument("--dataset", choices=["gtzan", "musiccaps", "all"], default="all")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dataset in ("gtzan", "all"):
        preprocess_gtzan(config)

    if args.dataset in ("musiccaps", "all"):
        preprocess_musiccaps(config)


if __name__ == "__main__":
    main()
