import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Dict, Tuple


class Pattern:
    def __init__(self, pattern_type: str, indices: Dict, prices: Dict, bars: int, quality_score: float = 0.0):
        self.type = pattern_type
        self.indices = indices
        self.prices = prices
        self.bars = bars
        self.quality_score = quality_score


class PatternDetector:
    def __init__(self, min_height_ratio: float = 0.005, min_bars: int = 15, max_bars: int = 40,
                 min_volume_ratio: float = 0.8, min_quality_score: float = 60.0):
        """
        Improved pattern detector with stricter quality criteria.
        
        Args:
        - min_height_ratio: Maximum allowed height difference between peaks (0.005 = 0.5%)
        - min_bars: Minimum bars for pattern formation
        - max_bars: Maximum bars for pattern formation
        - min_volume_ratio: Minimum volume ratio for quality validation
        - min_quality_score: Minimum quality score (0-100) to include pattern
        """
        self.min_height_ratio = min_height_ratio
        self.min_bars = min_bars
        self.max_bars = max_bars
        self.min_volume_ratio = min_volume_ratio
        self.min_quality_score = min_quality_score

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

    def calculate_quality_score(self, pattern_type: str, high: np.ndarray, low: np.ndarray,
                                volume: np.ndarray, pattern: 'Pattern') -> float:
        """
        Calculate pattern quality score (0-100).
        Higher score = higher confidence in the pattern.
        """
        score = 0.0

        if pattern_type == "double_top":
            peak1_idx = pattern.indices['peak1']
            peak2_idx = pattern.indices['peak2']
            valley_idx = pattern.indices['valley']

            peak1 = pattern.prices['peak1']
            peak2 = pattern.prices['peak2']
            valley = pattern.prices['valley']

            # 1. Height symmetry (0-25 points)
            # Perfect symmetry at 0%, loose at 0.5%, fail at > 0.5%
            height_diff = abs(peak1 - peak2) / max(peak1, peak2)
            if height_diff < 0.002:
                score += 25
            elif height_diff < 0.005:
                score += 20
            elif height_diff < 0.01:
                score += 10

            # 2. Valley support strength (0-25 points)
            # Valley should be clearly below midline
            midline = (peak1 + peak2) / 2
            valley_depth = (midline - valley) / midline
            if valley_depth > 0.05:  # > 5% below midline
                score += 25
            elif valley_depth > 0.03:
                score += 20
            elif valley_depth > 0.01:
                score += 10

            # 3. Formation time (0-15 points)
            # Ideal: 20-35 bars
            formation_bars = pattern.bars
            if 20 <= formation_bars <= 35:
                score += 15
            elif 15 <= formation_bars <= 40:
                score += 10
            elif 10 <= formation_bars <= 50:
                score += 5

            # 4. Volume pattern (0-20 points)
            # Peak1 volume should be higher than peak2 (weakening signal)
            if volume is not None and len(volume) > peak2_idx:
                try:
                    vol_peak1 = volume[peak1_idx]
                    vol_peak2 = volume[peak2_idx]
                    avg_vol = np.mean(volume[max(0, peak1_idx - 20):peak1_idx])

                    if vol_peak2 < vol_peak1 * 0.8:  # Clear volume weakness
                        score += 20
                    elif vol_peak2 < vol_peak1:
                        score += 15
                    elif vol_peak1 > avg_vol:
                        score += 10
                except:
                    score += 5

            # 5. Right shoulder weakness (0-15 points)
            # Right peak should be lower or equal to left peak (weakening)
            if peak2 <= peak1 * 0.995:  # Slightly lower is better
                score += 15
            elif peak2 <= peak1:
                score += 10

        elif pattern_type == "double_bottom":
            valley1_idx = pattern.indices['valley1']
            valley2_idx = pattern.indices['valley2']
            peak_idx = pattern.indices['peak']

            valley1 = pattern.prices['valley1']
            valley2 = pattern.prices['valley2']
            peak = pattern.prices['peak']

            # 1. Depth symmetry (0-25 points)
            depth_diff = abs(valley1 - valley2) / max(valley1, valley2)
            if depth_diff < 0.002:
                score += 25
            elif depth_diff < 0.005:
                score += 20
            elif depth_diff < 0.01:
                score += 10

            # 2. Peak resistance strength (0-25 points)
            midline = (valley1 + valley2) / 2
            peak_height = (peak - midline) / midline
            if peak_height > 0.05:  # > 5% above midline
                score += 25
            elif peak_height > 0.03:
                score += 20
            elif peak_height > 0.01:
                score += 10

            # 3. Formation time (0-15 points)
            formation_bars = pattern.bars
            if 20 <= formation_bars <= 35:
                score += 15
            elif 15 <= formation_bars <= 40:
                score += 10
            elif 10 <= formation_bars <= 50:
                score += 5

            # 4. Volume pattern (0-20 points)
            if volume is not None and len(volume) > valley2_idx:
                try:
                    vol_valley1 = volume[valley1_idx]
                    vol_valley2 = volume[valley2_idx]
                    avg_vol = np.mean(volume[max(0, valley1_idx - 20):valley1_idx])

                    if vol_valley2 < vol_valley1 * 0.8:  # Clear volume weakness
                        score += 20
                    elif vol_valley2 < vol_valley1:
                        score += 15
                    elif vol_valley1 > avg_vol:
                        score += 10
                except:
                    score += 5

            # 5. Right valley strength (0-15 points)
            if valley2 >= valley1 * 0.995:  # Slightly higher is better
                score += 15
            elif valley2 >= valley1:
                score += 10

        return score

    def detect_double_top(self, high: np.ndarray, low: np.ndarray, volume: np.ndarray = None) -> List[Pattern]:
        """
        Detect double top pattern with quality filtering.
        Double top: two peaks at similar heights with a valley between them.
        """
        n = len(high)
        peaks = self.find_local_peaks(high, window=5)
        peak_indices = np.where(peaks == 1)[0]

        patterns = []
        rejected = 0

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
                rejected += 1
                continue

            valley_idx = np.argmin(low[peak1_idx + 1:peak2_idx]) + peak1_idx + 1
            valley_price = low[valley_idx]

            pattern = Pattern(
                pattern_type="double_top",
                indices={"peak1": peak1_idx, "peak2": peak2_idx, "valley": valley_idx},
                prices={"peak1": peak1_price, "peak2": peak2_price, "valley": valley_price},
                bars=time_diff
            )

            quality_score = self.calculate_quality_score("double_top", high, low, volume, pattern)
            pattern.quality_score = quality_score

            if quality_score >= self.min_quality_score:
                patterns.append(pattern)
            else:
                rejected += 1

        logger.debug(f"Double tops: {len(patterns)} kept, {rejected} rejected for low quality")
        return patterns

    def detect_double_bottom(self, high: np.ndarray, low: np.ndarray, volume: np.ndarray = None) -> List[Pattern]:
        """
        Detect double bottom pattern with quality filtering.
        Double bottom: two valleys at similar depths with a peak between them.
        """
        n = len(low)
        valleys = self.find_local_valleys(low, window=5)
        valley_indices = np.where(valleys == 1)[0]

        patterns = []
        rejected = 0

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
                rejected += 1
                continue

            peak_idx = np.argmax(high[valley1_idx + 1:valley2_idx]) + valley1_idx + 1
            peak_price = high[peak_idx]

            pattern = Pattern(
                pattern_type="double_bottom",
                indices={"valley1": valley1_idx, "valley2": valley2_idx, "peak": peak_idx},
                prices={"valley1": valley1_price, "valley2": valley2_price, "peak": peak_price},
                bars=time_diff
            )

            quality_score = self.calculate_quality_score("double_bottom", high, low, volume, pattern)
            pattern.quality_score = quality_score

            if quality_score >= self.min_quality_score:
                patterns.append(pattern)
            else:
                rejected += 1

        logger.debug(f"Double bottoms: {len(patterns)} kept, {rejected} rejected for low quality")
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
                "quality_score": pattern.quality_score,
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
                "quality_score": pattern.quality_score,
            }

        return features
