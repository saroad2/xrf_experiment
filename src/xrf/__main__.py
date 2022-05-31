from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xrf.constants import CHANNEL, COUNTS_PER_SECOND, ENCODING


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


@xrf_group.command("plot-data")
@click.argument("input_path", type=click.Path(dir_okay=False, path_type=Path))
def plot_data_cli(input_path: Path):
    df = pd.read_csv(input_path)
    plt.plot(df[CHANNEL], df[COUNTS_PER_SECOND])
    plt.show()


if __name__ == "__main__":
    xrf_group()
