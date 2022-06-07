import itertools
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from xrf.constants import CHECK_POINTS, MAX_PEAKS_STD, PEAK_HEIGHT_THRESHOLD
from xrf.gaussian_util import fit_gaussian, gaussian


@dataclass
class Peak:

    start_index: int
    end_index: int
    peak_indices: List[int]

    def __contains__(self, i):
        return self.start_index <= i < self.end_index

    @property
    def right_arm(self):
        return self.end_index - self.peak_indices[-1]

    @property
    def left_arm(self):
        return self.peak_indices[0] - self.start_index

    def overlapping(self, other: "Peak") -> bool:
        if any(peak_index in self for peak_index in other.peak_indices):
            return True
        if any(peak_index in other for peak_index in self.peak_indices):
            return True
        return False


class Spectrum:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n: int,
        peaks_indices: Optional[List[int]] = None,
    ):
        self.x = x
        self.y = y
        self.peaks = self._build_peaks(y, n, peaks_indices)

    @property
    def peaks_indices(self):
        return list(
            itertools.chain.from_iterable(peak.peak_indices for peak in self.peaks)
        )

    def trim_to_peak(self, peak: "Peak"):
        return (
            self.x[peak.start_index : peak.end_index],
            self.y[peak.start_index : peak.end_index],
        )

    def fit_gaussians(self):
        results = []
        for peak in self.peaks:
            x, y = self.trim_to_peak(peak)
            deg_of_freedom = x.shape[0] - 3
            res, cov = fit_gaussian(
                x,
                y,
                [peak_index - peak.start_index for peak_index in peak.peak_indices],
            )
            error = np.sqrt(np.diag(cov))
            y_pred = gaussian(x, *res)
            chi2 = np.sum((y - y_pred) ** 2)
            chi2_red = chi2 / deg_of_freedom
            for i in range(len(res) // 3):
                a, mean, std = res[3 * i], res[3 * i + 1], res[3 * i + 2]
                a_err, mean_err, std_err = (
                    error[3 * i],
                    error[3 * i + 1],
                    error[3 * i + 2],
                )
                a_perr, mean_perr, std_perr = (
                    (a_err / a) * 100,
                    (mean_err / mean) * 100,
                    (std_err / std) * 100,
                )
                results.append(
                    dict(
                        a=a,
                        a_err=a_err,
                        a_perr=a_perr,
                        mean=mean,
                        mean_err=mean_err,
                        mean_perr=mean_perr,
                        std=std,
                        std_err=std_err,
                        std_perr=std_perr,
                        deg_of_freedom=deg_of_freedom,
                        chi2=chi2,
                        chi2_red=chi2_red,
                    )
                )
        return pd.DataFrame(results)

    @classmethod
    def _build_peaks(cls, y: np.ndarray, n: int, peaks_indices: Optional[List[int]]):
        peaks = []
        if peaks_indices is None or len(peaks_indices) == 0:
            peaks_indices = cls._find_peaks(y, n)
        for peak_index in peaks_indices:
            peak = cls._build_peak(y, peak_index)
            if cls._valid_peak(peak):
                peaks.append(peak)
        return cls._merge_peaks_list(y, peaks)

    @classmethod
    def _find_peaks(cls, y: np.ndarray, n: int) -> List[int]:
        min_height = PEAK_HEIGHT_THRESHOLD * y.max()
        return list(find_peaks(y, width=n, distance=CHECK_POINTS, height=min_height)[0])

    @classmethod
    def _build_peak(cls, y: np.ndarray, peak_index: int):
        peak_value = y[peak_index]
        end_index = peak_index + 1
        while (
            end_index < y.shape[0] - CHECK_POINTS
            and y[end_index : end_index + CHECK_POINTS].max() > peak_value / 2
        ):
            end_index += 1
        start_index = peak_index - 1
        while (
            start_index >= CHECK_POINTS - 1
            and y[start_index - CHECK_POINTS + 1 : start_index + 1].max()
            > peak_value / 2
        ):
            start_index -= 1
        return Peak(
            start_index=start_index, end_index=end_index, peak_indices=[peak_index]
        )

    @classmethod
    def _valid_peak(cls, peak: Peak):
        if peak.left_arm < CHECK_POINTS:
            return False
        if peak.right_arm < CHECK_POINTS:
            return False
        return True

    @classmethod
    def _merge_peaks_list(cls, y: np.ndarray, peaks: List[Peak]) -> List[Peak]:
        if len(peaks) == 0:
            return []
        new_peaks = []
        new_peak = peaks[0]
        for peak in peaks[1:]:
            if new_peak.overlapping(peak):
                new_peak = cls._merge_peaks(y, new_peak, peak)
            else:
                new_peaks.append(new_peak)
                new_peak = peak
        if new_peak is not None:
            new_peaks.append(new_peak)
        return new_peaks

    @classmethod
    def _merge_peaks(cls, y: np.ndarray, peak1: Peak, peak2: Peak) -> Peak:
        start_index = min(peak1.start_index, peak2.start_index)
        end_index = max(peak1.end_index, peak2.end_index)
        peak_indices = peak1.peak_indices + peak2.peak_indices
        peak_indices.sort()
        return Peak(
            start_index=start_index,
            end_index=end_index,
            peak_indices=cls._remove_outliers(peak_indices, y[peak_indices]),
        )

    @classmethod
    def _remove_outliers(cls, indices, values):
        values_max, values_std = values.max(), values.std()
        values_min = values_max - MAX_PEAKS_STD * values_std
        return [
            indices[i]
            for i, value in enumerate(values)
            if values_min <= value <= values_max
        ]
