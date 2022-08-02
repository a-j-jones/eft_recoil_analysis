# EFT Recoil Analysis
This project uses opencv to track objects within a recorded video in order to analyse the recoil pattern of a weapon in a video game, where this is otherwise not possible by the use of in-game tools.

The tool is specifically designed for use within Escape From Tarkov, however in theory it should work for other games.

![recoil_cropped](https://user-images.githubusercontent.com/64594018/182376402-5c842ed5-0ca1-4258-acfa-5d9ea6efeac9.gif)

For best results in EFT:
 - Begin and end the recording (or cut the video file afterwards) with the weapon ADS'd right before and right after shooting.
 - A relatively short video clip will mean that you don't waste time tracking outside of the spray pattern (though the tool should take too long to run per clip).
 - Choose something with high contrast that doesn't end up obscured by the gun during the spray/recoil as a camera tracking point (experiment here for best results).
 - Raid/Game settings:
    - Use a sharp red dot, I haven't tested many, but the holographics & dovetail OKP-7 seem to work well.
    - Offline Factory
    - Night time (important for video contrast, day-time introduces )
    - Stand at pumping station, look towards Gate 3 extract / Delivery from the past objective.
      this should mean that your bullets hit farther away, reducing the impact that sparks will have on the video tracking.


## Example

* Create a file `main.py` with:

```Python
from tracker import Tracker

# Add your in / out filenames
video_filename = "recoil.mp4"
output_filename = "recoil.json"

# Track video file:
t = Tracker(video_filename, debug=True)
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
        "camera recoil": [
            1289.0,
            784.0
        ]
    },
    ...
]
```
