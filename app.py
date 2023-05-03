import io
from pathlib import Path
from typing import List

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageFile

from utils.tracker import Tracker

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    df = df[["filename", "combined_x", "combined_y", "combined_y_color"]]

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
        # Create labels for plot:
        label = {
            "x": offset,
            "y": df[df.filename == file].combined_y.max() + 25,
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


def create_tracker(checkboxes: List[str], files: list) -> np.ndarray:
    """
    Create the tracker for the recoil pattern.

    @param checkboxes: List of checkbox names which are checked.
    @param files: List of additional files to track.
    @return:
    """

    # Get the options from the checkboxes:
    high_precision = True if "High precision tracking" in checkboxes else False
    debug_level = 1 if "Debug mode" in checkboxes else 0

    if files is None:
        files = []

    images = []
    results = []
    for file in files:
        filepath = file.name
        if filepath is None:
            continue

        print(filepath)
        video_file = Path(filepath).stem[:-8]
        results_file = Path("results", video_file).with_suffix(".json")
        t = Tracker(
            filepath,
            high_precision=high_precision,
            debug_level=debug_level
        )
        t.track()
        t.save(results_file)
        results.append(results_file)

    image = create_plots(results)

    return image


if __name__ == "__main__":
    interface = gr.Interface(
        fn=create_tracker,
        inputs=[
            gr.components.CheckboxGroup(choices=["High precision tracking", "Debug mode"], label="Options"),
            gr.components.File(file_count="multiple", label="Video files to track")
        ],
        outputs=gr.components.Image(type="numpy", label="Recoil patterns")
    )

    interface.launch(debug=True)
