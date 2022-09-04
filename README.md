# EFT Recoil Analysis
This project uses opencv to track objects within a recorded video in order to analyse the recoil pattern of a weapon in a video game, where this is otherwise not possible by the use of in-game tools.

The tool is specifically designed for use within Escape From Tarkov, however in theory it should work for other games.

<p align="center">
  <img src="https://user-images.githubusercontent.com/64594018/188312056-c76147cb-56ca-43fd-b427-5ddaaf663361.gif" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/64594018/182401951-8336208e-e8bc-4acb-b305-cb6de584eecb.PNG" />
</p>

For best results in EFT:
 - Begin and end the recording (or cut the video file afterwards) with the weapon ADS'd right before and right after shooting.
 - A relatively short video clip will mean that you don't waste time tracking outside of the spray pattern (though the tool shouldn't take too long to run per clip).
 - Choose something with high contrast that doesn't end up obscured by the gun during the spray/recoil as a camera tracking point (experiment here for best results).
 - Raid/Game settings:
    - Use a sharp red dot, I haven't tested many, but the holographics & dovetail OKP-7 seem to work well.
    - Offline Factory works well as there are some longer distances which means the sparks from bullet impacts have a lesser effect on the tracking, though with some tweaking you may find that the shooting range can provide similar results.

## Example

* Create a file `main.py` with:

```Python
from tracker import Tracker

# Add your in / out filenames
video_filename = "recoil.mp4"
output_filename = "recoil.json"

# Track video file:
t = Tracker(video_filename, debug_level=1) # Debug level == 1 shows the tracking in real time.
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

 - "reticle" is a value that describes the pixel position of the red dot / reticle. (absolute value)
 - "optical_flow" is the difference between the camera position of the current and last frame (relative value)
