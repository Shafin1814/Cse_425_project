"""
Download MusicCaps metadata and audio clips.
Saves metadata CSV and downloads audio via yt-dlp.
"""

import os
import subprocess
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def download_musiccaps_metadata(output_path: str) -> pd.DataFrame:
    """Download MusicCaps metadata from HuggingFace and save as CSV."""
    print("Loading MusicCaps metadata from HuggingFace...")
    ds = load_dataset("google/MusicCaps", split="train")
    df = ds.to_pandas()
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} entries to {output_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def download_clip(
    video_identifier: str,
    output_filename: str,
    start_time: float,
    end_time: float,
    num_attempts: int = 3,
    url_base: str = "https://www.youtube.com/watch?v=",
) -> tuple[bool, str]:
    """Download a single audio clip from YouTube using yt-dlp."""
    command = (
        f'yt-dlp --quiet --force-keyframes-at-cuts --no-warnings '
        f'-x --audio-format wav -f bestaudio '
        f'-o "{output_filename}" '
        f'--download-sections "*{start_time}-{end_time}" '
        f'{url_base}{video_identifier}'
    )

    for attempt in range(num_attempts):
        try:
            subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
            if os.path.exists(output_filename):
                return True, "Downloaded"
        except subprocess.CalledProcessError as err:
            if attempt == num_attempts - 1:
                return False, str(err.output)
    return False, "Failed"


def download_musiccaps_audio(
    metadata_path: str,
    output_dir: str,
    limit: int = None,
) -> dict:
    """Download MusicCaps audio clips."""
    df = pd.read_csv(metadata_path)
    if limit is not None:
        df = df.head(limit)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    stats = {"success": 0, "failed": 0, "skipped": 0}
    failed_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading clips"):
        ytid = row["ytid"]
        outfile = str(output_dir / f"{ytid}.wav")

        if os.path.exists(outfile):
            stats["skipped"] += 1
            continue

        success, msg = download_clip(
            ytid, outfile, row["start_s"], row["end_s"]
        )
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            failed_ids.append(ytid)

    print(f"\nDownload complete: {stats}")
    print(f"Failed IDs ({len(failed_ids)}): {failed_ids[:20]}...")

    # Save failed IDs for reference
    if failed_ids:
        failed_path = output_dir / "failed_downloads.txt"
        with open(failed_path, "w") as f:
            f.write("\n".join(failed_ids))

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MusicCaps dataset")
    parser.add_argument("--metadata-output", default="data/raw/musiccaps_metadata.csv")
    parser.add_argument("--audio-output", default="data/raw/musiccaps_audio")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max clips to download (None for all)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only download metadata, skip audio")
    args = parser.parse_args()

    # Step 1: Download metadata
    if not os.path.exists(args.metadata_output):
        download_musiccaps_metadata(args.metadata_output)
    else:
        print(f"Metadata already exists at {args.metadata_output}")

    # Step 2: Download audio
    if not args.metadata_only:
        download_musiccaps_audio(
            args.metadata_output,
            args.audio_output,
            limit=args.limit,
        )
