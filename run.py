from tracker import Tracker
from glob import glob

# folder = "videos"
# files = list(glob(f"{folder}/*.mp4"))
# trackers = [Tracker(file, debug_level=0) for file in files]
#
# for tracker in trackers:
# 	tracker.track()
# 	tracker.save()

t = Tracker("videos/__recoil.mp4", debug_level=1)
t.track()
t.save("recoil data/__recoil.json")


