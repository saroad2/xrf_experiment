from pathlib import Path
from typing import Sequence

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xrf.constants import CHANNEL, COUNTS_PER_SECOND, DEFAULT_NEIGHBOURHOOD, ENCODING
from xrf.gaussian_util import (
    extract_local_maxima_indices,
    fit_gaussian,
    gaussian,
    trim_gaussian,
)


@click.group("xrf")
def xrf_group():
    pass


@xrf_group.command("parse-mca")
@click.argument("input_path", type=click.Path(dir_okay=False, path_type=Path))
def parse_mca_cli(input_path: Path):
    with open(input_path, encoding=ENCODING) as input_file:
        lines = [line.replace("\n", "").strip() for line in input_file.readlines()]
    start_data_index = lines.index("<<DATA>>")
    end_data_index = lines.index("<<END>>")
    data = np.array(
        [float(count) for count in lines[start_data_index + 1 : end_data_index]],
    )
    parameters_lines = lines[:start_data_index] + lines[end_data_index + 1 :]
    live_time_line = next(
        filter(lambda line: line.startswith("LIVE_TIME"), parameters_lines),
    )
    live_time = float(live_time_line.split("-")[1].strip())
    data /= live_time
    data_frame = pd.DataFrame(
        enumerate(data, start=1),
        columns=[
            CHANNEL,
            COUNTS_PER_SECOND,
        ],
    )
    name = input_path.stem.replace(" ", "_")
    data_frame.to_csv(input_path.with_name(f"{name}.csv"), index=False)


@xrf_group.command("maximum-candidates")
@click.argument("input_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("-n", "--neighbourhood", type=int, default=DEFAULT_NEIGHBOURHOOD)
def maximum_candidates_cli(input_path: Path, neighbourhood: int):
    df = pd.read_csv(input_path)
    max_indices = extract_local_maxima_indices(
        df[COUNTS_PER_SECOND].to_numpy(), n=neighbourhood
    )
    name = input_path.stem.replace(" ", "_")
    maxima_df = df.iloc[max_indices].copy()
    maxima_df.sort_values(by=[COUNTS_PER_SECOND], ascending=False, inplace=True)
    maxima_df.to_csv(input_path.with_name(f"{name}_max_candidates.csv"), index=False)


@xrf_group.command("fit-gaussian")
@click.argument("input_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--start-x", required=True, type=int, multiple=True)
@click.option("--show-plot", is_flag=True, default=False)
def fit_gaussian_cli(input_path: Path, start_x: Sequence[int], show_plot):
    df = pd.read_csv(input_path)
    start_x = list(start_x)
    x, y = df[CHANNEL].to_numpy(), df[COUNTS_PER_SECOND].to_numpy()
    x, y = trim_gaussian(x, y, start_x)
    deg_of_freedom = x.shape[0] - 3
    res, cov = fit_gaussian(x, y, start_x)
    error = np.sqrt(np.diag(cov))
    y_pred = gaussian(x, *res)
    chi2 = np.sum((y - y_pred) ** 2)
    chi2_red = chi2 / deg_of_freedom
    print("Result:")
    print(res)
    print("Error:")
    print(error)
    print("Covariance:")
    print(cov)
    print(f"{chi2=:.2e}, {deg_of_freedom=} {chi2_red=:.2e}")
    if show_plot:
        plt.plot(x, y, label="Raw data")
        plt.plot(x, y_pred, label="Gaussian fit")
        plt.legend()
        plt.show()


@xrf_group.command("plot-data")
@click.argument("input_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--show-local-maxima", is_flag=True, default=False)
@click.option("--min-x", type=int)
@click.option("--max-x", type=int)
@click.option("-n", "--neighbourhood", type=int, default=DEFAULT_NEIGHBOURHOOD)
def plot_data_cli(
    input_path: Path,
    min_x: int,
    max_x: int,
    neighbourhood: int,
    show_local_maxima: bool,
):
    df = pd.read_csv(input_path)
    x, y = df[CHANNEL].to_numpy(), df[COUNTS_PER_SECOND].to_numpy()
    if min_x is not None:
        x, y = x[x >= min_x], y[x >= min_x]
    if max_x is not None:
        x, y = x[x <= max_x], y[x <= max_x]
    plt.plot(x, y)
    title = "Channel to Count"
    if show_local_maxima:
        max_indices = extract_local_maxima_indices(y, n=neighbourhood)
        title += f" ({len(max_indices)} local maxima)"
        plt.scatter(x[max_indices], y[max_indices], s=5, c="red")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    xrf_group()
