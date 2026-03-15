"""
Shared CellPeakIndex class used by both data_pipeline.py (write path) and
dataset.py (read path) to avoid duplicating the peak-lookup logic.
"""

import numpy as np
from bisect import bisect_left, bisect_right


class CellPeakIndex:
    """
    For a given cell line, indexes ALL peak centers and signals per mark per chrom.
    Allows fast lookup: "What is the max signal for each mark in this window?"
    """

    def __init__(self, target_marks):
        self.target_marks = target_marks
        self._index = {}
        self.peak_locs = {}
        self.all_peak_centers = {}

    # ------------------------------------------------------------------
    # Construction from raw BED files (used by data_pipeline.py)
    # ------------------------------------------------------------------
    @classmethod
    def from_bed_dir(cls, cell_dir, target_marks):
        """Build the index by reading narrowPeak BED files from *cell_dir*."""
        import os
        import pandas as pd

        obj = cls(target_marks)
        all_centers = {}

        for m_idx, mark in enumerate(target_marks):
            bed_path = os.path.join(cell_dir, "chip_histone", f"{mark}.05.bed")
            if not os.path.exists(bed_path):
                obj._index[m_idx] = None
                obj.peak_locs[m_idx] = []
                continue

            try:
                df = pd.read_csv(
                    bed_path,
                    sep="\t",
                    header=None,
                    usecols=[0, 1, 2, 6],
                    names=["chrom", "start", "end", "signal"],
                )
            except Exception as e:
                print(f"Error reading {bed_path}: {e}")
                obj._index[m_idx] = None
                obj.peak_locs[m_idx] = []
                continue

            df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0)
            df["center"] = (df["start"] + df["end"]) // 2

            chrom_index = {}
            locs = []
            for chrom, grp in df.groupby("chrom"):
                grp_sorted = grp.sort_values("center")
                centers = grp_sorted["center"].values.astype(np.int64)
                signals = grp_sorted["signal"].values.astype(np.float32)
                chrom_index[chrom] = (centers, signals)
                locs.extend([(chrom, int(c)) for c in centers])
                all_centers.setdefault(chrom, []).extend(int(c) for c in centers)

            obj._index[m_idx] = chrom_index
            obj.peak_locs[m_idx] = locs

        obj.all_peak_centers = {
            chrom: np.array(sorted(centers), dtype=np.int64)
            for chrom, centers in all_centers.items()
        }
        return obj

    # ------------------------------------------------------------------
    # Serialisation helpers (pickle-friendly dicts)
    # ------------------------------------------------------------------
    def to_serialized(self):
        return {
            "index": self._index,
            "peak_locs": self.peak_locs,
            "all_peak_centers": self.all_peak_centers,
        }

    @classmethod
    def from_serialized(cls, payload, target_marks):
        obj = cls(target_marks)
        obj._index = payload["index"]
        obj.peak_locs = payload["peak_locs"]
        obj.all_peak_centers = payload["all_peak_centers"]
        return obj

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------
    def has_mark(self, m_idx):
        return self._index.get(m_idx) is not None

    def lookup_signal(self, chrom, win_start, win_end):
        result = np.full(len(self.target_marks), -1.0, dtype=np.float32)

        for m_idx in range(len(self.target_marks)):
            mark_data = self._index.get(m_idx)
            if mark_data is None:
                continue

            result[m_idx] = 0.0

            if chrom not in mark_data:
                continue

            centers, signals = mark_data[chrom]
            lo = bisect_left(centers, win_start)
            hi = bisect_right(centers, win_end)

            if lo < hi:
                result[m_idx] = float(np.log2(signals[lo:hi].max() + 1.0))

        return result

    def is_background_window(self, chrom, center, margin):
        centers = self.all_peak_centers.get(chrom)
        if centers is None or len(centers) == 0:
            return True

        left = bisect_left(centers, center - margin)
        right = bisect_right(centers, center + margin)
        return left == right
