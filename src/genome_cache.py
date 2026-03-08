import numpy as np
import pyfaidx
import os
import tqdm
import json

class GenomeCache:
    """
    Converts a FASTA file to a binary numpy file for extremely fast random access.
    Maps DNA to uint8: A=0, C=1, G=2, T=3, N=4
    """
    MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    
    def __init__(self, fasta_path, cache_path):
        self.fasta_path = fasta_path
        self.cache_path = cache_path
        self.meta_path = cache_path + ".meta.json"
        
        if not os.path.exists(self.cache_path) or not os.path.exists(self.meta_path):
            self._build_cache()
            
        # Load metadata
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        self.chrom_offsets = meta['offsets']
        self.chrom_lens = meta['lens']
        self.total_len = meta['total_len']
        
        # Memory map the binary file
        self.data = np.memmap(self.cache_path, dtype='uint8', mode='r', shape=(self.total_len,))

    def _build_cache(self):
        print(f"Building binary genome cache from {self.fasta_path}...")
        fasta = pyfaidx.Fasta(self.fasta_path)
        
        chroms = sorted([c for c in fasta.keys() if len(c) < 6]) # Standard chroms only
        offsets = {}
        lens = {}
        current_offset = 0
        
        # First pass: calculate total length
        for chrom in chroms:
            l = len(fasta[chrom])
            offsets[chrom] = current_offset
            lens[chrom] = l
            current_offset += l
            
        total_len = current_offset
        
        # Second pass: write to binary file
        bin_data = np.memmap(self.cache_path, dtype='uint8', mode='w+', shape=(total_len,))
        
        for chrom in tqdm.tqdm(chroms, desc="Caching chromosomes"):
            seq_str = str(fasta[chrom][:]).upper()
            seq_bytes = np.frombuffer(seq_str.encode('ascii'), dtype='uint8')
            
            # Efficiently map characters to 0-4
            arr = np.full(len(seq_bytes), 4, dtype='uint8') # Default to N (4)
            arr[seq_bytes == 65] = 0 # A
            arr[seq_bytes == 67] = 1 # C
            arr[seq_bytes == 71] = 2 # G
            arr[seq_bytes == 84] = 3 # T
            
            start = offsets[chrom]
            bin_data[start : start + len(arr)] = arr
            bin_data.flush()
            
        with open(self.meta_path, 'w') as f:
            json.dump({'offsets': offsets, 'lens': lens, 'total_len': total_len}, f)
        print(f"Cache built at {self.cache_path}")

    def get_seq(self, chrom, start, end):
        """Fetches a sequence as uint8 array. O(1) complexity."""
        if chrom not in self.chrom_offsets:
            return np.full(end - start, 4, dtype='uint8')
            
        abs_start = self.chrom_offsets[chrom] + start
        abs_end = self.chrom_offsets[chrom] + end
        
        # Handle boundaries
        if start < 0 or end > self.chrom_lens[chrom]:
            seq = np.full(end - start, 4, dtype='uint8')
            valid_start = max(0, start)
            valid_end = min(self.chrom_lens[chrom], end)
            
            if valid_start < valid_end:
                out_start = valid_start - start
                out_end = out_start + (valid_end - valid_start)
                seq[out_start:out_end] = self.data[self.chrom_offsets[chrom] + valid_start : self.chrom_offsets[chrom] + valid_end]
            return seq
            
        return self.data[abs_start:abs_end]

if __name__ == "__main__":
    import config
    cache = GenomeCache(config.FASTA_PATH, os.path.join(config.PROCESSED_DIR, "genome_hg38.bin"))
    print("Test fetch:", cache.get_seq("chr1", 1000000, 1000010))
