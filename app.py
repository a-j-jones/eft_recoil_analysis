import io
from pathlib import Path
from typing import List

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageFile

from tracker import Tracker

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_recoil_pattern(file: str) -> pd.DataFrame:
    """
    Get the recoil pattern from the json file, and return a DataFrame with the data.

    @param file:   JSON file to read.
    @return df:    DataFrame with the recoil pattern.
    """

    df = pd.read_json(file).set_index("frame")

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


def create_plots(file: str) -> np.ndarray:
    """
    Create the plots for the recoil pattern and movement.

    @param file:    JSON file to read.
    @return:        Image of the plots.
    """

    plots = 1
    width = 6
    fig, axs = plt.subplots(2, plots, figsize=(width, 10), sharey='row')

    df = get_recoil_pattern(file)

    # Spray pattern plotted:
    axs[0].scatter(df.combined_x, df.combined_y, c=df.combined_y_color)
    axs[0].set_title(f"Combined reticle and camera recoil")
    axs[0].axis('equal')

    # Movement plotted:
    axs[1].plot(df.index, df.combined_y_move)
    axs[1].set_title(f"Combined reticle and camera recoil")

    fig.tight_layout()

    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))

    return im


def create_tracker(file: str, files: list) -> List[np.ndarray]:
    """
    Create the tracker for the recoil pattern.

    @param file:  Video file to track.
    @param files: List of additional files to track.
    @return:
    """
    if files is None:
        files = []

    images = []
    for filepath in [file, *[x.name for x in files]]:
        if filepath is None:
            continue

        print(filepath)
        filename = Path(filepath).stem
        output = Path("results", filename).with_suffix(".json")
        t = Tracker(filepath)
        t.track()
        t.save(output)
        img = create_plots(output)
        images.append(img)

    return images


interface = gr.Interface(
    fn=create_tracker,
    inputs=[gr.Video(), gr.File(file_count="multiple")],
    outputs=[gr.Gallery()]
)

interface.launch(debug=True)
