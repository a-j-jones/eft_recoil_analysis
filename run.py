from tracker import Tracker
from glob import glob
import cv2

# folder = "videos"
# files = list(glob(f"{folder}/*.mp4"))
# trackers = [Tracker(file, debug_level=0) for file in files]
#
# for tracker in trackers:
# 	try:
# 		tracker.track()
# 		tracker.save()
# 	except cv2.error:
# 		continue

t = Tracker("videos/__recoil.mp4", debug_level=0)
t.track()
t.save("recoil data/__recoil.json")


