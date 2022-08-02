from tracker import Tracker

# ADD FILENAME:
filename = ""
output_filename = ""

# Track video file:
t = Tracker(filename, debug=True)
t.track()
t.save(output_filename)
