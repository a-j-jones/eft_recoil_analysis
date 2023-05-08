import multiprocessing
from pathlib import Path
from typing import List

import gradio as gr
import numpy as np
from PIL import ImageFile

from utils.plotting import create_plots, fig_to_img
from utils.tracker import Tracker

ImageFile.LOAD_TRUNCATED_IMAGES = True


def track_file(filepath: str, high_precision: bool, debug_level: int) -> Path:
    # Get the file paths:
    results_file = Path("results", Path(filepath).stem[:-8]).with_suffix(".json")

    t = Tracker(
        filepath,
        high_precision=high_precision,
        debug_level=debug_level
    )
    t.track()
    t.save(results_file)

    return results_file


def create_tracker(num_processes: float, checkboxes: List[str], files: list) -> np.ndarray:
    """
    Create the tracker for the recoil pattern.

    @param num_processes: Number of processes for tracking.
    @param checkboxes: List of checkbox names which are checked.
    @param files: List of additional files to track.
    @return:
    """

    # Get the options from the checkboxes:
    high_precision = True if "High precision tracking" in checkboxes else False
    debug_level = 1 if "Debug mode" in checkboxes else 0

    if files is None:
        files = []

    num_processes = min(int(num_processes), len(files))
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(track_file, [(file.name, high_precision, debug_level) for file in files])

    fig = create_plots(results)
    image = fig_to_img(fig)

    return image


if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    interface = gr.Interface(
        fn=create_tracker,
        inputs=[
            gr.components.Slider(minimum=1, maximum=cpu_count, value=cpu_count, step=1, label="Number of processes"),
            gr.components.CheckboxGroup(choices=["High precision tracking", "Debug mode"], label="Options"),
            gr.components.File(file_count="multiple", label="Video files to track")
        ],
        outputs=gr.components.Image(type="numpy", label="Recoil patterns"),
        title="Recoil pattern tracker",
        description="This tool was designed to track the recoil pattern of a weapon in Escape from Tarkov.",
        allow_flagging="never",
    )

    interface.launch(debug=True)
