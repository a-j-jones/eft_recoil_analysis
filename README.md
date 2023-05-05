# EFT Recoil Analysis
This project uses opencv to track objects within a recorded video in order to analyse the recoil pattern of a weapon in a video game, where this is otherwise not possible by the use of in-game tools.

The tool is specifically designed for use within Escape From Tarkov, however in theory it should work for other games.

<p align="center">
   <img src="https://user-images.githubusercontent.com/64594018/188312056-c76147cb-56ca-43fd-b427-5ddaaf663361.gif" />
</p>

## Installation:
```cmd
git clone https://github.com/a-j-jones/eft_recoil_analysis.git
cd eft_recoil_analysis
pip install -r requirements.txt
```

## Usage:
### Gradio interface
The tool can be run using the gradio web interface by running the following command in your terminal:
```cmd
python app.py
```
From there you can open the url and use the interface to upload your video files and run the tool. Currently there are the following options:
 - **Number of processes:** The tool will execute in parallel (one process per video file) to speed up the processing time. This option allows you to set the number of processes to use. The default is the number of cores on your machine.
 - **High precision tracking:** This option will use a more precise camera tracking method, however it will take longer to process. The output will indicate if there is a camera misalignment on any of the tracked recoil patterns.
 - **Debug mode:** This option will show the tracking in real time. This is useful for testing the tracking on a video file before running the tool on a large number of files. (Note: This was designed to be used with a single video file and may cause problems if multiple files are uploaded in debug mode)

<p align="center">
  <img src="https://user-images.githubusercontent.com/64594018/236353323-f33a9c2f-48df-4ae4-93d5-b39683ae0a44.png" />
</p>

Once tracking has finished for all of the uploaded files, you will be able to see output such as the following:

<p align="center">
  <img src="https://user-images.githubusercontent.com/64594018/236354280-898823ea-f0e6-4731-b7e8-1a32db1dded2.png" />
</p>

## Usage notes:
 - In the `templates` folder you should have a template image of the sight reticle that is used in your video files (a small square around the edge of the reticle).
 - If the tool is unable to detect the sight in the first frame of the video, you will be prompted to draw a box around the reticle which will be used for the rest of the video.
 - JSON files containing the recoil data will be saved in the `results` folder with the same name as the video file.

## Video file guidance:
 - Begin and end the recording (or cut the video file afterwards) with the weapon ADS'd right before and right after shooting.
 - A relatively short video clip will mean that you don't waste time tracking outside of the spray pattern (though the tool shouldn't take too long to run per clip).
 - Raid/Game settings:
   - Choose a location with enough texture above the sight for better camera tracking.
   - Use a sharp red dot, I haven't tested many, but the holographics & dovetail OKP-7 seem to work well.
   - Offline Factory works well as there are some longer distances which means the sparks from bullet impacts have a lesser effect on the tracking, though with some tweaking you may find that the shooting range can provide similar results.

## Python example:

* Create a file `main.py` with:

```Python
from utils.tracker import Tracker

# Add your in / out filenames
video_filename = "recoil.mp4"
output_filename = "recoil.json"

# Track video file:
t = Tracker(video_filename, debug_level=1)  # Debug level == 1 shows the tracking in real time.
t.track()
t.save(output_filename)
```

This will create a json file which can be analysed using Pandas with the following format.

```JSON
[
    {
        "filename": "videos\\recoil.mp4",
        "frame": 1,
        "reticle": [
            1723.0,
            736.0
        ],
        "optical_flow": [
            0.0,
            0.1
        ]
    },
    ...
]
```