from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import find_peaks

from xrf.constants import EPSILON


@dataclass
class Peak:

    start_index: int
    end_index: int
    peak_indices: List[int]

    def __contains__(self, i):
        return self.start_index <= i < self.end_index

    def overlapping(self, other: "Peak") -> bool:
        if any(peak_index in self for peak_index in other.peak_indices):
            return True
        if any(peak_index in other for peak_index in self.peak_indices):
            return True
        return False

    def merge(self, other: "Peak") -> "Peak":
        start_index = min(self.start_index, other.start_index)
        end_index = max(self.end_index, other.end_index)
        peak_indices = self.peak_indices + other.peak_indices
        peak_indices.sort()
        return Peak(
            start_index=start_index, end_index=end_index, peak_indices=peak_indices
        )


class Spectrum:
    def __init__(self, x: np.ndarray, y: np.ndarray, n: int):
        self.x = x
        self.y = y
        self.peaks = self._build_peaks(y, n)

    def trim_to_peak(self, peak: "Peak"):
        return (
            self.x[peak.start_index : peak.end_index],
            self.y[peak.start_index : peak.end_index],
        )

    @classmethod
    def _build_peaks(cls, y: np.ndarray, n: int):
        peaks = []
        for peak_index in cls._find_peaks(y, n):
            peaks.append(cls._build_peak(y, peak_index))
        return cls._merge_peaks(peaks)

    @classmethod
    def _find_peaks(cls, y: np.ndarray, n: int) -> List[int]:
        return list(find_peaks(y, width=n, height=EPSILON)[0])

    @classmethod
    def _build_peak(cls, y: np.ndarray, peak_index: int):
        peak_value = y[peak_index]
        end_index = peak_index + 1
        while end_index < y.shape[0] and y[end_index] > peak_value / 2:
            end_index += 1
        start_index = peak_index - 1
        while start_index >= 0 and y[start_index] > peak_value / 2:
            start_index -= 1
        return Peak(
            start_index=start_index, end_index=end_index, peak_indices=[peak_index]
        )

    @classmethod
    def _merge_peaks(cls, peaks: List[Peak]) -> List[Peak]:
        if len(peaks) == 0:
            return []
        new_peaks = []
        new_peak = peaks[0]
        for peak in peaks[1:]:
            if new_peak.overlapping(peak):
                new_peak = new_peak.merge(peak)
            else:
                new_peaks.append(new_peak)
                new_peak = peak
        if new_peak is not None:
            new_peaks.append(new_peak)
        return new_peaks
