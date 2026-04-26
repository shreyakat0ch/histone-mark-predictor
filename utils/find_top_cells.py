import os
import pickle
import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import config


def find_and_copy_top_cells(n_cells=70):
    processed_dir = config.PROCESSED_DIR
    data_dir = os.environ.get(
        "HISTONE_FULL_DATA_DIR",
        os.path.join(config.RELEASE_DIR, "data"),
    )
    subset_dir = os.path.join(config.RELEASE_DIR, "data_subset")

    index_path = os.path.join(processed_dir, "cell_peak_indexes.pkl")
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found. Please run the data pipeline first.")
        return

    with open(index_path, "rb") as f:
        indexes = pickle.load(f)

    target_marks = config.TARGET_MARKS
    print(f"Index contains {len(indexes)} cell lines.")
    print(f"Looking for marks: {target_marks}")
    print(f"Source data directory: {data_dir}")

    cell_completeness = []
    for cell, payload in indexes.items():
        peak_index = payload["index"]
        count = sum(
            1 for m_idx in range(len(target_marks))
            if peak_index.get(m_idx) is not None
        )
        cell_completeness.append((cell, count))

    cell_completeness.sort(key=lambda x: x[1], reverse=True)

    print(f"\nCompleteness breakdown across all {len(indexes)} cell lines:")
    for n in range(len(target_marks), -1, -1):
        n_with_marks = sum(1 for _, count in cell_completeness if count == n)
        if n_with_marks:
            print(f"  {n}/{len(target_marks)} marks: {n_with_marks} cell lines")

    top_cells = cell_completeness[:n_cells]
    if not top_cells:
        print("No cells found.")
        return

    print(f"\nTop {len(top_cells)}: completeness {top_cells[0][1]} to {top_cells[-1][1]} marks")

    os.makedirs(subset_dir, exist_ok=True)

    copied = 0
    for cell, count in top_cells:
        src = os.path.join(data_dir, cell)
        dst = os.path.join(subset_dir, cell)
        if os.path.exists(src):
            print(f"  Copying {cell} ({count}/{len(target_marks)} marks)...")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied += 1
        else:
            print(f"  Warning: {src} not found.")

    print(f"\nDone! {copied} cell lines copied to {subset_dir}")


if __name__ == "__main__":
    find_and_copy_top_cells()
