import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
    df = df[["filename", "combined_x", "combined_y", "combined_y_color", "camera_x", "camera_y"]]

    return df


def create_plots(files: List[str | Path]) -> np.ndarray:
    """
    Create the plots for the recoil pattern and movement.

    @param files: JSON files to read.
    @return:        Image of the plots.
    """

    plots = 1
    width = len(files) * 3
    fig, axs = plt.subplots(figsize=(width, 8))

    df = pd.concat([get_recoil_pattern(file) for file in files])

    offset = 0
    labels = []
    for file in df.filename.unique():
        # Test camera differences:
        df_test = df[df.filename == file]
        x_difference = df_test.camera_x.iloc[0] - df_test.camera_x.iloc[-1]
        y_difference = df_test.camera_y.iloc[0] - df_test.camera_y.iloc[-1]
        euc_distance = np.sqrt(x_difference ** 2 + y_difference ** 2)

        # Add label for camera movement:
        if euc_distance > 25:
            label = {
                "x": offset,
                "y": df[df.filename == file].combined_y.max() + 15,
                "s": f"CAMERA MISALIGNED {euc_distance:.0f}px",
                "color": "red",
                "horizontalalignment": "center"
            }
            labels.append(label)

        # Create labels for plot:
        label = {
            "x": offset,
            "y": df[df.filename == file].combined_y.max() + 30,
            "s": file.replace("_", "\n"),
            "horizontalalignment": "center"
        }
        labels.append(label)

        # Create offset for the next scatter:
        additional_offset = df[df.filename == file].combined_x.max() + 100
        df.loc[df.filename == file, "combined_x"] = df[df.filename == file].combined_x + offset
        offset += additional_offset

    # Plot labels:
    for label in labels:
        axs.text(**label)

    # Spray pattern plotted:
    axs.scatter(df.combined_x, df.combined_y, c=df.combined_y_color)
    axs.set_title(f"Combined reticle and camera recoil")
    axs.axis('equal')
    axs.set_ylim(df.combined_y.min() - 25, df.combined_y.max() + 75)

    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))

    return im
