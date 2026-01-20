import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Dict, Tuple


class Pattern:
    def __init__(self, pattern_type: str, indices: Dict, prices: Dict, bars: int):
        self.type = pattern_type
        self.indices = indices
        self.prices = prices
        self.bars = bars


class PatternDetector:
    def __init__(self, min_height_ratio: float = 0.01, min_bars: int = 10, max_bars: int = 50):
        self.min_height_ratio = min_height_ratio
        self.min_bars = min_bars
        self.max_bars = max_bars

    def find_local_peaks(self, data: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Find local peaks in data.
        A peak at index i means data[i] >= data[i-window:i] and data[i] >= data[i+1:i+window+1]
        """
        n = len(data)
        peaks = np.zeros(n, dtype=int)

        for i in range(window, n - window):
            if (data[i] >= data[max(0, i - window):i].max() and
                    data[i] >= data[i + 1:min(n, i + window + 1)].max()):
                peaks[i] = 1

        return peaks

    def find_local_valleys(self, data: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Find local valleys in data.
        A valley at index i means data[i] <= data[i-window:i] and data[i] <= data[i+1:i+window+1]
        """
        n = len(data)
        valleys = np.zeros(n, dtype=int)

        for i in range(window, n - window):
            if (data[i] <= data[max(0, i - window):i].min() and
                    data[i] <= data[i + 1:min(n, i + window + 1)].min()):
                valleys[i] = 1

        return valleys

    def detect_double_top(self, high: np.ndarray, low: np.ndarray) -> List[Pattern]:
        """
        Detect double top pattern.
        Double top: two peaks at similar heights with a valley between them.
        """
        n = len(high)
        peaks = self.find_local_peaks(high, window=5)
        peak_indices = np.where(peaks == 1)[0]

        patterns = []

        for i in range(len(peak_indices) - 1):
            peak1_idx = peak_indices[i]
            peak2_idx = peak_indices[i + 1]

            time_diff = peak2_idx - peak1_idx

            if not (self.min_bars <= time_diff <= self.max_bars):
                continue

            peak1_price = high[peak1_idx]
            peak2_price = high[peak2_idx]
            height_diff = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)

            if height_diff > self.min_height_ratio:
                continue

            valley_idx = np.argmin(low[peak1_idx + 1:peak2_idx]) + peak1_idx + 1
            valley_price = low[valley_idx]

            pattern = Pattern(
                pattern_type="double_top",
                indices={"peak1": peak1_idx, "peak2": peak2_idx, "valley": valley_idx},
                prices={"peak1": peak1_price, "peak2": peak2_price, "valley": valley_price},
                bars=time_diff
            )

            patterns.append(pattern)

        return patterns

    def detect_double_bottom(self, high: np.ndarray, low: np.ndarray) -> List[Pattern]:
        """
        Detect double bottom pattern.
        Double bottom: two valleys at similar heights with a peak between them.
        """
        n = len(low)
        valleys = self.find_local_valleys(low, window=5)
        valley_indices = np.where(valleys == 1)[0]

        patterns = []

        for i in range(len(valley_indices) - 1):
            valley1_idx = valley_indices[i]
            valley2_idx = valley_indices[i + 1]

            time_diff = valley2_idx - valley1_idx

            if not (self.min_bars <= time_diff <= self.max_bars):
                continue

            valley1_price = low[valley1_idx]
            valley2_price = low[valley2_idx]
            height_diff = abs(valley1_price - valley2_price) / max(valley1_price, valley2_price)

            if height_diff > self.min_height_ratio:
                continue

            peak_idx = np.argmax(high[valley1_idx + 1:valley2_idx]) + valley1_idx + 1
            peak_price = high[peak_idx]

            pattern = Pattern(
                pattern_type="double_bottom",
                indices={"valley1": valley1_idx, "valley2": valley2_idx, "peak": peak_idx},
                prices={"valley1": valley1_price, "valley2": valley2_price, "peak": peak_price},
                bars=time_diff
            )

            patterns.append(pattern)

        return patterns

    def extract_features(self, pattern: Pattern, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> Dict:
        """
        Extract ML features from a pattern.
        """
        if pattern.type == "double_top":
            peak1 = pattern.prices["peak1"]
            peak2 = pattern.prices["peak2"]
            valley = pattern.prices["valley"]

            avg_peak = (peak1 + peak2) / 2
            drop_from_peak = (avg_peak - valley) / avg_peak
            height_diff = abs(peak1 - peak2) / max(peak1, peak2)

            features = {
                "pattern_type": 0,  # 0 for double_top
                "formation_bars": pattern.bars,
                "peak_height": avg_peak,
                "height_symmetry": 1 - height_diff,
                "drop_pct": drop_from_peak,
                "valley_to_peak_ratio": valley / avg_peak,
                "neckline_level": valley,
            }

        else:  # double_bottom
            valley1 = pattern.prices["valley1"]
            valley2 = pattern.prices["valley2"]
            peak = pattern.prices["peak"]

            avg_valley = (valley1 + valley2) / 2
            rise_from_valley = (peak - avg_valley) / avg_valley
            height_diff = abs(valley1 - valley2) / max(valley1, valley2)

            features = {
                "pattern_type": 1,  # 1 for double_bottom
                "formation_bars": pattern.bars,
                "valley_level": avg_valley,
                "height_symmetry": 1 - height_diff,
                "rise_pct": rise_from_valley,
                "peak_to_valley_ratio": peak / avg_valley,
                "support_level": avg_valley,
            }

        return features
