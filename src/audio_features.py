"""
Audio feature extraction for music context understanding.
Extracts log-mel spectrograms, chroma, MFCCs, and handles segmentation.
"""

import numpy as np
import librosa
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)


def load_audio(filepath: str, sr: int = 22050, duration: Optional[float] = None) -> tuple[np.ndarray, int]:
    """Load audio file with resampling."""
    y, sr_out = librosa.load(filepath, sr=sr, duration=duration)
    return y, sr_out


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def extract_chroma(
    y: np.ndarray,
    sr: int = 22050,
    n_chroma: int = 12,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract chroma features using CQT."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_chroma, hop_length=hop_length)
    return chroma


def extract_mfcc(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract MFCCs."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc


def segment_audio(
    y: np.ndarray,
    sr: int = 22050,
    segment_duration: float = 5.0,
) -> list[np.ndarray]:
    """Split audio into fixed-length segments."""
    segment_samples = int(segment_duration * sr)
    total_samples = len(y)
    segments = []

    for start in range(0, total_samples, segment_samples):
        end = start + segment_samples
        if end <= total_samples:
            segments.append(y[start:end])
        else:
            # Pad the last segment if it's too short (at least half length)
            remainder = y[start:]
            if len(remainder) >= segment_samples // 2:
                padded = np.pad(remainder, (0, segment_samples - len(remainder)), mode="constant")
                segments.append(padded)

    return segments


def extract_segment_features(
    segment: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_chroma: int = 12,
    n_mfcc: int = 20,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extract a fixed-size feature vector for a single segment.
    Returns: concatenation of [mel_mean(128), chroma_mean(12), mfcc_mean(20)] = 160-dim.
    """
    # Mel spectrogram mean
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(log_mel, axis=1)  # (n_mels,)

    # Chroma mean
    chroma = librosa.feature.chroma_cqt(y=segment, sr=sr, n_chroma=n_chroma, hop_length=hop_length)
    chroma_mean = np.mean(chroma, axis=1)  # (n_chroma,)

    # MFCC mean
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_mean = np.mean(mfcc, axis=1)  # (n_mfcc,)

    # Concatenate
    feature_vector = np.concatenate([mel_mean, chroma_mean, mfcc_mean])
    return feature_vector  # (160,)


def extract_track_features(
    filepath: str,
    sr: int = 22050,
    segment_duration: float = 5.0,
    n_mels: int = 128,
    n_chroma: int = 12,
    n_mfcc: int = 20,
    hop_length: int = 512,
    duration: Optional[float] = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Extract features for all segments in a track.

    Returns:
        node_features: (num_segments, 160) feature matrix
        segments: list of raw audio segments
    """
    y, sr = load_audio(filepath, sr=sr, duration=duration)
    segments = segment_audio(y, sr=sr, segment_duration=segment_duration)

    if len(segments) == 0:
        # Fallback: treat entire audio as one segment
        segments = [y]

    node_features = []
    for seg in segments:
        feat = extract_segment_features(
            seg, sr=sr, n_mels=n_mels, n_chroma=n_chroma,
            n_mfcc=n_mfcc, hop_length=hop_length
        )
        node_features.append(feat)

    node_features = np.stack(node_features, axis=0)
    return node_features, segments


def extract_full_mel_spectrogram(
    filepath: str,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    duration: Optional[float] = None,
) -> np.ndarray:
    """Extract full mel spectrogram for CNN baseline. Returns (n_mels, T)."""
    y, sr = load_audio(filepath, sr=sr, duration=duration)
    mel = extract_mel_spectrogram(y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    return mel
