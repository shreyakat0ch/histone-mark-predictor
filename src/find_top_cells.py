import pickle
import os
import shutil
import config

def find_and_copy_top_cells():
    processed_dir = config.PROCESSED_DIR
    data_dir      = config.DATA_DIR
    subset_dir    = os.path.join(config.RELEASE_DIR, "data_subset")

    index_path = os.path.join(processed_dir, 'cell_peak_indexes.pkl')
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found. Please run the data pipeline first.")
        return

    with open(index_path, 'rb') as f:
        indexes = pickle.load(f)

    target_marks = config.TARGET_MARKS  # ['H3K27ac', 'H3K4me3', 'H3K27me3', 'H3K9me3']
    print(f"Index contains {len(indexes)} cell lines.")
    print(f"Looking for marks: {target_marks}")

    # Debug: show what the index actually looks like for first cell
    first_cell = list(indexes.keys())[0]
    payload = indexes[first_cell]
    peak_index = payload['index']  # {0: chrom_data, 1: chrom_data, ...} or None
    print(f"\nSample cell '{first_cell}':")
    for m_idx, mark in enumerate(target_marks):
        has_data = peak_index.get(m_idx) is not None
        n_peaks = 0
        if has_data:
            for chrom_data in peak_index[m_idx].values():
                n_peaks += len(chrom_data[0])  # chrom_data = (centers, signals)
        print(f"  [{m_idx}] {mark}: {'✓' if has_data else '✗'}  ({n_peaks} peaks)")

    # Count marks per cell using integer index keys
    cell_completeness = []
    for cell, payload in indexes.items():
        peak_index = payload['index']
        count = sum(
            1 for m_idx in range(len(target_marks))
            if peak_index.get(m_idx) is not None
        )
        cell_completeness.append((cell, count))

    # Sort by completeness
    cell_completeness.sort(key=lambda x: x[1], reverse=True)

    # Print breakdown
    print(f"\nCompleteness breakdown across all {len(indexes)} cell lines:")
    for n in range(len(target_marks), -1, -1):
        cells_n = [(c, cnt) for c, cnt in cell_completeness if cnt == n]
        if cells_n:
            print(f"  {n}/{len(target_marks)} marks: {len(cells_n)} cell lines")

    # Take top 70
    top_70 = cell_completeness[:70]
    print(f"\nTop 70: completeness {top_70[0][1]} to {top_70[-1][1]} marks")

    os.makedirs(subset_dir, exist_ok=True)

    copied = 0
    for cell, count in top_70:
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
    print(f"\nNext steps:")
    print(f"  1. rm {processed_dir}/windows_*.pkl")
    print(f"  2. python data_pipeline.py")
    print(f"  3. python train.py")

if __name__ == "__main__":
    find_and_copy_top_cells()
