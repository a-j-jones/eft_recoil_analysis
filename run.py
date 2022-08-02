from tracker import Tracker
from glob import glob

folder = "videos"
trackers = [Tracker(file) for file in glob(f"{folder}/*.mp4")]

for tracker in trackers:
	tracker.track()
	tracker.save()


