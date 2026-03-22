import os
import json
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pyfaidx import Fasta
import config
from genome_cache import GenomeCache
from cell_peak_index import CellPeakIndex
from tqdm import tqdm

def rc_one_hot(oh):
    """Reverse-complement a one-hot encoded DNA sequence."""
    return np.flip(oh, axis=0)[:, [3, 2, 1, 0]].copy()

class ChromaDataset(Dataset):
    def __init__(self, cell_lines, split="train", seed=42):
        self.cell_lines = cell_lines
        self.split = split
        self.training = (split == "train")
        
        # Load RNA map
        with open(os.path.join(config.PROCESSED_DIR, "rna_map.pkl"), "rb") as f:
            rna_data = pickle.load(f)
            self.rna_map = rna_data["rna_map"]
            
        dummy_rna_dim = next(iter(self.rna_map.values())).shape[0]
        self._dummy_rna = np.zeros(dummy_rna_dim, dtype=np.float32)
            
        # Load peak indexes
        print(f"[{split}] Loading peak indexes...")
        with open(os.path.join(config.PROCESSED_DIR, "cell_peak_indexes.pkl"), "rb") as f:
            all_indexes = pickle.load(f)
            
        self._cell_index = {}
        for cell in self.cell_lines:
            if cell in all_indexes:
                self._cell_index[cell] = CellPeakIndex.from_serialized(all_indexes[cell], config.TARGET_MARKS)
                
        # Setup Genome Cache
        self.genome = GenomeCache(config.FASTA_PATH, config.GENOME_CACHE_PATH)
        
        # Normalization stats
        self.target_mean = np.zeros(config.NUM_MARKS, dtype=np.float32)
        self.target_std = np.ones(config.NUM_MARKS, dtype=np.float32)
        
        # Check for cached windows
        cache_file = os.path.join(config.PROCESSED_DIR, f"windows_{split}.pkl")
        if os.path.exists(cache_file):
            print(f"[{split}] Loading cached windows from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.windows = pickle.load(f)
            print(f"[{split}] Total windows: {len(self.windows)} (cached)")
        else:
            self.build_dataset(seed)
            print(f"[{split}] Saving sampled windows to {cache_file}...")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.windows, f)
        
    def build_dataset(self, seed):
        rng = np.random.RandomState(seed)
        
        # Load chroms
        fasta = Fasta(config.FASTA_PATH)
        fasta_chroms = set(fasta.keys())
        valid_chroms = [c for c in fasta_chroms if c.startswith("chr") and "_" not in c]
        
        # 1. Collect peaks
        print(f"[{self.split}] Sampling peaks...")
        peak_windows = []
        peak_set = set()
        
        for cell in tqdm(self.cell_lines, desc=f"Peaks ({self.split})"):
            idx = self._cell_index[cell]
            for m_idx in range(config.NUM_MARKS):
                locs = idx.peak_locs.get(m_idx, [])
                if len(locs) > config.MAX_PEAKS_PER_MARK_PER_CELL:
                    locs = [locs[i] for i in rng.choice(len(locs), config.MAX_PEAKS_PER_MARK_PER_CELL, replace=False)]
                    
                for chrom, center in locs:
                    if chrom not in fasta_chroms:
                        continue
                    chrom_len = len(fasta[chrom])
                    start = center - config.HALF_WINDOW
                    if start < 0 or start + config.WINDOW_SIZE > chrom_len:
                        continue
                    
                    key = (cell, chrom, center)
                    if key not in peak_set:
                        peak_set.add(key)
                        peak_windows.append(key)
                        
        # 2. Sample backgrounds
        if config.PEAK_FRAC > 0:
            n_bg = int(len(peak_windows) * (1 - config.PEAK_FRAC) / max(config.PEAK_FRAC, 1e-6))
        else:
            n_bg = 0
            
        print(f"[{self.split}] Sampling {n_bg} background windows...")
        bg_windows = []
        bg_set = set()
        attempts = 0
        bg_margin = config.HALF_WINDOW
        
        while len(bg_windows) < n_bg and attempts < max(n_bg * 50, 10000):
            attempts += 1
            chrom = valid_chroms[rng.randint(len(valid_chroms))]
            chrom_len = len(fasta[chrom])
            if chrom_len <= config.WINDOW_SIZE: continue
            
            center = int(rng.randint(config.HALF_WINDOW, chrom_len - config.HALF_WINDOW))
            cell = self.cell_lines[rng.randint(len(self.cell_lines))]
            
            key = (cell, chrom, center)
            if key in bg_set: continue
            
            if not self._cell_index[cell].is_background_window(chrom, center, bg_margin):
                continue
                
            bg_set.add(key)
            bg_windows.append(key)
            
        self.windows = peak_windows + bg_windows
        rng.shuffle(self.windows)
        print(f"[{self.split}] Total windows: {len(self.windows)}")

    def compute_target_stats(self):
        """Called ONLY on the train split to calculate norm stats."""
        print("Computing training target statistics (Z-score)...")
        sum_x = np.zeros(config.NUM_MARKS, dtype=np.float64)
        sum_x2 = np.zeros(config.NUM_MARKS, dtype=np.float64)
        counts = np.zeros(config.NUM_MARKS, dtype=np.int64)

        for cell, chrom, center in tqdm(self.windows, desc="Target stats"):
            start = center - config.HALF_WINDOW
            raw_target = self._cell_index[cell].lookup_signal(chrom, start, start + config.WINDOW_SIZE)
            valid = raw_target != -1.0
            if not np.any(valid):
                continue
            vals = raw_target[valid].astype(np.float64)
            sum_x[valid] += vals
            sum_x2[valid] += vals * vals
            counts[valid] += 1

        valid_counts = counts > 0
        self.target_mean[valid_counts] = (sum_x[valid_counts] / counts[valid_counts]).astype(np.float32)
        variance = np.zeros(config.NUM_MARKS, dtype=np.float64)
        variance[valid_counts] = (sum_x2[valid_counts] / counts[valid_counts]) - np.square(self.target_mean[valid_counts].astype(np.float64))
        self.target_std[valid_counts] = np.sqrt(np.maximum(variance[valid_counts], 1e-12)).astype(np.float32)
        
        # Save so val split can load them
        with open(os.path.join(config.PROCESSED_DIR, "norm_stats.json"), 'w') as f:
            json.dump({
                "mean": self.target_mean.tolist(),
                "std": self.target_std.tolist()
            }, f)
            
        print(f"Mean: {self.target_mean}")
        print(f"Std: {self.target_std}")

    def load_target_stats(self):
        """Called on val/test splits to load train norm stats."""
        stats_path = os.path.join(config.PROCESSED_DIR, "norm_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                d = json.load(f)
                self.target_mean = np.array(d["mean"], dtype=np.float32)
                self.target_std = np.array(d["std"], dtype=np.float32)

    def normalize_targets(self, target):
        target = target.astype(np.float32, copy=True)
        valid = target != -1.0
        if np.any(valid):
            target[valid] = (target[valid] - self.target_mean[valid]) / self.target_std[valid]
        return target
        
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        cell, chrom, center = self.windows[idx]
        
        if self.training and config.AUGMENT_OFFSET_MAX_BP > 0:
            center += random.randint(-config.AUGMENT_OFFSET_MAX_BP, config.AUGMENT_OFFSET_MAX_BP)

        start = center - config.HALF_WINDOW
            
        # 1. DNA (Binary Fetch)
        dna_indices = self.genome.get_seq(chrom, start, start + config.WINDOW_SIZE)
        
        # Fast one-hot conversion
        dna_onehot = np.zeros((config.WINDOW_SIZE, 4), dtype=np.float32)
        for i in range(4):
            dna_onehot[dna_indices == i, i] = 1.0
        
        # 2. RNA
        rna = self.rna_map.get(cell, self._dummy_rna)
        
        # 3. Targets
        raw_target = self._cell_index[cell].lookup_signal(chrom, start, start + config.WINDOW_SIZE)
        target = self.normalize_targets(raw_target)
        
        # Extract validity mask
        mask = (raw_target != -1.0).astype(np.float32)
        
        # 4. Augmentation
        if self.training and config.AUGMENT_RC and random.random() < 0.5:
            dna_onehot = rc_one_hot(dna_onehot)
            
        # Return format expected by existing logic
        # target, mask: [NUM_MARKS]
        # dna_onehot is [L, 4] but previous model expected [4, L]
        return (
            torch.from_numpy(dna_onehot.T).float(),
            torch.from_numpy(rna).float(),
            torch.from_numpy(target).float(),
            torch.from_numpy(mask).float()
        )
