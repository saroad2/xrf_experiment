from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from xrf.constants import CHECK_POINTS
from xrf.gaussian_util import fit_gaussian, gaussian


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
    def _build_peaks(cls, y: np.ndarray, n: int):
        peaks = []
        for peak_index in cls._find_peaks(y, n):
            peaks.append(cls._build_peak(y, peak_index))
        return cls._merge_peaks(peaks)

    @classmethod
    def _find_peaks(cls, y: np.ndarray, n: int) -> List[int]:
        return list(find_peaks(y, width=n, distance=CHECK_POINTS)[0])

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
