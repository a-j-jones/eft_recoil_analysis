import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from scipy.signal import find_peaks


# ---------------- Set the style: ----------------
SCATTER_COLOUR = "#31c974"
GRID_COLOR = "#333333"
BORDER_COLOUR = "#9a8866"
BACKGROUND_COLOUR = "black"
TEXT_COLOUR = "white"
FONT_PATH = 'styling/bender.regular.ttf'

FONT = fm.FontProperties(fname=FONT_PATH)
fm.fontManager.addfont(FONT_PATH)

sns.set_style("darkgrid", {
    # Axes:
    "axes.facecolor": BACKGROUND_COLOUR,
    "axes.labelcolor": BORDER_COLOUR,
    "axes.edgecolor": BORDER_COLOUR,

    # Figure:
    "figure.facecolor": BACKGROUND_COLOUR,

    # Grid:
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,

    # Ticks:
    "xtick.color": BORDER_COLOUR,
    "ytick.color": BORDER_COLOUR,

    "font.family": FONT.get_name(),

})


def get_recoil_pattern(file: str) -> pd.DataFrame:
    """
    Get the recoil pattern from the json file, and return a DataFrame with the data.

    @param file:   JSON file to read.
    @return df:    DataFrame with the recoil pattern.
    """

    df = pd.read_json(file).set_index("frame")

    df["filename"] = Path(file).stem
    df[["reticle_x", "reticle_y"]] = pd.DataFrame(df.reticle.tolist(), index=df.index)
    df[["camera_x", "camera_y"]] = pd.DataFrame(df["optical_flow"].tolist(), index=df.index)
    for col in ["camera_x", "camera_y"]:
        df[col] = df[col].cumsum()

    df.drop(columns=["optical_flow", "reticle"], inplace=True)

    for col in df.columns:
        if col == "filename":
            continue

        df["base"] = df[col].iloc[0]
        df[col] = df[col] - df.base
        df.drop(columns=["base"], inplace=True)

    df["combined_x"] = df.reticle_x + df.camera_x
    df["combined_y"] = df.reticle_y + df.camera_y
    df["combined_y_move"] = df.combined_y - df.combined_y.shift(1)
    df["combined_y_color"] = df.combined_y_move.apply(lambda x: "red" if x > 0 else "blue")

    return df


def create_plots(files: List[str | Path]) -> plt.Figure:
    """
    Create the plots for the recoil pattern and movement.

    @param files: JSON files to read.
    @return:        Image of the plots.
    """

    width = len(files) * 3
    fig, axs = plt.subplots(figsize=(width, 8))

    df = pd.concat([get_recoil_pattern(file) for file in files])

    offset = 0
    for filename, df_plot in df.groupby("filename"):
        # Test camera differences
        x_difference = df_plot.camera_x.iloc[0] - df_plot.camera_x.iloc[-1]
        y_difference = df_plot.camera_y.iloc[0] - df_plot.camera_y.iloc[-1]
        euc_distance = np.sqrt(x_difference ** 2 + y_difference ** 2)

        # Add label for camera movement:
        axs.text(
            x=offset,
            y=df_plot.combined_y.max() + 30,
            s=filename.replace("_", "\n"),
            color=TEXT_COLOUR,
            horizontalalignment="center"
        )
        if euc_distance > 25:
            axs.text(
                x=offset,
                y=df_plot.combined_y.max() + 15,
                s=f"CAMERA MISALIGNED {euc_distance:.0f}px",
                color="red",
                horizontalalignment="center"
            )

        # Update offset for the next scatter
        additional_offset = df_plot.combined_x.max() + 100
        df_plot["combined_x"] = df_plot["combined_x"] + offset
        offset += additional_offset

        axs.plot(
            df_plot.combined_x,
            df_plot.combined_y,
            color=SCATTER_COLOUR,
            linewidth=1,
            alpha=0.5,
            zorder=1
        )

        # Find the individual shots and scatter plot them:
        diff_arr = (df_plot.combined_y - df_plot.combined_y.shift(1))
        peaks, _ = find_peaks(diff_arr, height=(diff_arr.max()*0.25), distance=3)
        shots = df_plot.iloc[peaks - 1]

        # Spray pattern plotted
        axs.scatter(
            shots.combined_x,
            shots.combined_y,
            color=SCATTER_COLOUR,
            linewidth=1,
            alpha=1,
            zorder=1,
            marker="x"
        )

    axs.set_title(f"Combined reticle and camera recoil")
    axs.axis('equal')
    axs.set_ylim(df.combined_y.min() - 25, df.combined_y.max() + 75)

    # Set title color
    axs.title.set_color(BORDER_COLOUR)

    return fig


def fig_to_img(fig: plt.Figure) -> np.ndarray:
    """
    Convert a matplotlib figure to a numpy array with RGBA channels and return it.

    @param fig: matplotlib figure to convert.
    @return:
    """

    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))

    return im
