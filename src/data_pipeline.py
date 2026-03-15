import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import config
from cell_peak_index import CellPeakIndex

def discover_cell_lines():
    print("Discovering cell lines...")
    cells = []
    for cell in sorted(os.listdir(config.DATA_DIR)):
        cell_dir = os.path.join(config.DATA_DIR, cell)
        if not os.path.isdir(cell_dir):
            continue
            
        has_bed = any(
            os.path.exists(os.path.join(cell_dir, "chip_histone", f"{m}.05.bed"))
            for m in config.TARGET_MARKS
        )
        
        rna_dir = os.path.join(cell_dir, "RNA")
        has_rna = os.path.isdir(rna_dir) and any(f.endswith(".tsv") for f in os.listdir(rna_dir))
        
        if has_bed and has_rna:
            cells.append(cell)
            
    print(f"Found {len(cells)} valid cell lines with both RNA and targeted histone marks.")
    return cells

def process_rna(cells):
    print("Processing RNA data...")
    cell_series_list = []
    valid_cells = []
    
    for cell in tqdm(cells, desc="Reading RNA files"):
        rna_dir = os.path.join(config.DATA_DIR, cell, "RNA")
        rna_files = [f for f in os.listdir(rna_dir) if f.endswith(".tsv")]
        if not rna_files:
            continue
            
        try:
            df = pd.read_csv(os.path.join(rna_dir, rna_files[0]), sep="\t", index_col=0)
            avg_tpm = df.mean(axis=1) # average replicates
            cell_series_list.append(avg_tpm)
            valid_cells.append(cell)
        except Exception as e:
            print(f"Error processing {cell} RNA: {e}")
            
    # Combine into master matrix
    master = pd.concat(cell_series_list, axis=1)
    master.columns = valid_cells
    master = master.fillna(0)
    
    # Log transform
    log_tpm = np.log2(master.values.astype(np.float32) + 1.0)
    log_tpm_df = pd.DataFrame(log_tpm, index=master.index, columns=master.columns)
    
    # Find top most variable genes
    variances = log_tpm_df.var(axis=1).sort_values(ascending=False)
    top_genes = variances.head(config.RNA_NUM_TOP_GENES).index.tolist()
    
    print(f"Selected top {len(top_genes)} variable genes.")
    
    # Build per-cell RNA dictionary
    rna_map = {}
    for cell in valid_cells:
        cell_log = log_tpm_df[cell]
        rna_map[cell] = cell_log.reindex(top_genes).fillna(0).values.astype(np.float32)
        
    with open(os.path.join(config.PROCESSED_DIR, "rna_map.pkl"), "wb") as f:
        pickle.dump({
            "top_genes": top_genes,
            "rna_map": rna_map,
            "valid_cells": valid_cells
        }, f)
        
    # Save a JSON cell list for splits
    with open(os.path.join(config.PROCESSED_DIR, "cell_order.json"), "w") as f:
        json.dump(valid_cells, f)
        
    return valid_cells, rna_map

def build_peak_indexes(cells):
    print("Building peak indexes for all cells...")
    cell_index_payloads = {}
    
    for cell in tqdm(cells, desc="Indexing peaks"):
        cell_dir = os.path.join(config.DATA_DIR, cell)
        idx = CellPeakIndex.from_bed_dir(cell_dir, config.TARGET_MARKS)
        cell_index_payloads[cell] = idx.to_serialized()
        
    with open(os.path.join(config.PROCESSED_DIR, "cell_peak_indexes.pkl"), "wb") as f:
        pickle.dump(cell_index_payloads, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Peak indexing complete.")

if __name__ == "__main__":
    cells = discover_cell_lines()
    valid_cells, rna_map = process_rna(cells)
    build_peak_indexes(valid_cells)
    print("Data pipeline finished successfully.")
