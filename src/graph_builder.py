"""
Graph construction for music structure understanding.
Builds segment graphs and chord-transition graphs from audio features.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


def build_segment_graph(
    node_features: np.ndarray,
    similarity_threshold: float = 0.85,
    include_temporal: bool = True,
) -> Data:
    """
    Build a segment graph from node feature matrix.

    Args:
        node_features: (num_segments, feature_dim) feature matrix
        similarity_threshold: cosine similarity threshold for non-temporal edges
        include_temporal: whether to include temporal adjacency edges

    Returns:
        PyG Data object with node features and edge index
    """
    num_nodes = node_features.shape[0]

    if num_nodes == 0:
        return Data(
            x=torch.zeros(1, node_features.shape[1] if node_features.ndim > 1 else 160, dtype=torch.float),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
        )

    edge_list = []

    # 1. Temporal adjacency edges (bidirectional)
    if include_temporal and num_nodes > 1:
        for i in range(num_nodes - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])

    # 2. Similarity-based edges
    if num_nodes > 1:
        sim_matrix = cosine_similarity(node_features)
        for i in range(num_nodes):
            for j in range(i + 2, num_nodes):  # skip adjacent (already added)
                if sim_matrix[i, j] > similarity_threshold:
                    edge_list.append([i, j])
                    edge_list.append([j, i])

    # Self-loops for message passing
    for i in range(num_nodes):
        edge_list.append([i, i])

    if len(edge_list) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    x = torch.tensor(node_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def build_chord_transition_graph(
    chroma_sequence: np.ndarray,
    n_chroma: int = 12,
) -> Data:
    """
    Build a chord-transition graph from chroma features.

    Nodes = unique chord templates (major/minor triads)
    Edges = observed transitions weighted by count

    Args:
        chroma_sequence: (12, T) chroma feature matrix
        n_chroma: number of chroma bins

    Returns:
        PyG Data object
    """
    # Define 24 basic chord templates (12 major + 12 minor)
    chord_templates = _get_chord_templates()
    num_chords = len(chord_templates)

    # Classify each frame to nearest chord
    chord_sequence = []
    for t in range(chroma_sequence.shape[1]):
        frame = chroma_sequence[:, t]
        if np.sum(frame) < 0.01:
            continue
        frame_norm = frame / (np.linalg.norm(frame) + 1e-8)
        similarities = [np.dot(frame_norm, template) for template in chord_templates]
        chord_id = np.argmax(similarities)
        chord_sequence.append(chord_id)

    if len(chord_sequence) < 2:
        return Data(
            x=torch.tensor(chord_templates, dtype=torch.float),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            num_nodes=num_chords,
        )

    # Build transition matrix
    transition_counts = np.zeros((num_chords, num_chords))
    for i in range(len(chord_sequence) - 1):
        src, dst = chord_sequence[i], chord_sequence[i + 1]
        transition_counts[src, dst] += 1

    # Create edges from transitions
    edge_list = []
    edge_weights = []
    for i in range(num_chords):
        for j in range(num_chords):
            if transition_counts[i, j] > 0:
                edge_list.append([i, j])
                edge_weights.append(transition_counts[i, j])

    # Add self-loops for isolated nodes
    active_nodes = set()
    for e in edge_list:
        active_nodes.add(e[0])
        active_nodes.add(e[1])
    for i in range(num_chords):
        if i not in active_nodes:
            edge_list.append([i, i])
            edge_weights.append(0.0)

    if len(edge_list) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    x = torch.tensor(chord_templates, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_chords,
    )


def _get_chord_templates() -> list[np.ndarray]:
    """Generate 24 chord templates: 12 major + 12 minor triads."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    templates = []

    for root in range(12):
        # Major triad: root, root+4, root+7
        major = np.zeros(12)
        major[root] = 1.0
        major[(root + 4) % 12] = 1.0
        major[(root + 7) % 12] = 1.0
        major /= np.linalg.norm(major)
        templates.append(major)

    for root in range(12):
        # Minor triad: root, root+3, root+7
        minor = np.zeros(12)
        minor[root] = 1.0
        minor[(root + 3) % 12] = 1.0
        minor[(root + 7) % 12] = 1.0
        minor /= np.linalg.norm(minor)
        templates.append(minor)

    return templates


def build_graph_from_audio(
    filepath: str,
    graph_type: str = "segment",
    sr: int = 22050,
    segment_duration: float = 5.0,
    similarity_threshold: float = 0.85,
    duration: Optional[float] = None,
) -> Data:
    """
    High-level function: build a graph from an audio file.

    Args:
        filepath: path to audio file
        graph_type: 'segment' or 'chord'
        sr: sample rate
        segment_duration: segment duration for segment graphs
        similarity_threshold: for segment graph edge construction
        duration: max audio duration to load

    Returns:
        PyG Data object
    """
    from audio_features import extract_track_features, load_audio, extract_chroma

    if graph_type == "segment":
        node_features, _ = extract_track_features(
            filepath, sr=sr, segment_duration=segment_duration, duration=duration
        )
        graph = build_segment_graph(node_features, similarity_threshold=similarity_threshold)
    elif graph_type == "chord":
        y, sr = load_audio(filepath, sr=sr, duration=duration)
        chroma = extract_chroma(y, sr=sr)
        graph = build_chord_transition_graph(chroma)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    return graph
