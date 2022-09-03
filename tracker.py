# Code:
from selector import Selector

# Public packages:
import cv2
import json
import numpy as np
from tqdm import tqdm
from scipy import stats

# Typing:
from typing import Tuple, Optional
from numpy import ndarray


def search_margin(middle: Tuple[int, int], margin_px: int = 100) -> Tuple[Tuple[int, int], Tuple[int, int]]:
	"""
	Create a rectangle around a given point for use with the template matching.
	@param middle:      x and y coordinates
	@param margin_px:   integer value for the size of the margin
	@return:            rectangle (top-left and bottom-right coordinates).
	"""
	return (int(middle[0] - margin_px), int(middle[1] - margin_px)), (
		int(middle[0] + margin_px), int(middle[1] + margin_px))


def get_middle(point1: Tuple[int, int], point2: Tuple[int, int]) -> Tuple[int, int]:
	"""
	Get a single coordinate for the middle of a rectangle.
	@param point1:      Tuple of x and y coordinates for Top-Left of a rectangle
	@param point2:      Tuple of x and y coordinates for Bottom-Right of a rectangle
	@return:            Tuple of a single point in the middle of the given rectangle.
	"""
	return ((point2[0] - point1[0]) / 2) + point1[0], ((point2[1] - point1[1]) / 2) + point1[1]


def matchSymbol(img: ndarray, template: ndarray,
                search_area: Tuple[Tuple[int, int], Tuple[int, int]] = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
	"""
	Match a template image within a larger image using cv2.matchTemplate, returns the Top-Left
	and Bottom-Right coordinates of the template image at the matched location.
	Requires a grayscale image.

	@param img:         Larger greyscale cv2 image for which to find the template within.
	@param template:    Smaller greyscale cv2 image to search for within the template.
	@param search_area: Optional rectangle parameter to specify a smaller area to search within
						the larger img.

	@return:            Tuple of coordinates for the rectangle representing the position of the
						template image.
	"""
	w, h = template.shape[::-1]

	if search_area:
		(x1, y1), (x2, y2) = search_area
		img = img[y1:y2, x1:x2]

	# Apply template Matching
	res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

	top_left = (max_loc[0] + x1, max_loc[1] + y1)
	bottom_right = (top_left[0] + w, top_left[1] + h)

	return top_left, bottom_right


class TrackedObject:
	def __init__(self, selector: Selector, name: str):
		"""
		Initialises TrackedObject which contains the information required for tracking a specific object.
		@param selector:    Selector object which defines the initial img and location of the object.
		@param name:        Name of the tracked object.
		"""
		self.__search_area = None
		self.name = name
		self.middle = None
		self.loc, self.img = selector.get_rectangle()
		self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

	@property
	def search_area(self):
		return self.__search_area

	@search_area.getter
	def search_area(self):
		"""
		Getter method for search_area which returns either the margin around the middle of the object
		if available, otherwise it calculates the middle first.
		@return:    A Tuple of coordinates of a rectangle to search for the object.
		"""
		if self.middle:
			return search_margin(self.middle, 100)
		else:
			return search_margin(get_middle(*self.loc), 100)


class Tracker:
	def __init__(self, video_file: str, debug_level: int = 0):
		"""
		Initialises Tracker class which will track objects in a 2D space using cv2. The class records coordinates
		data for the position of tracked objects which can be used to calculate the recoil of a weapon in EFT.

		Additional data recorded includes the frame number and video file path.

		@param video_file:  File path for the video file to be analysed.
		@param debug_level: Integer value to define level of debugging.
		"""
		# ----------------- Initialise frame data -----------------
		self.data = []
		self.frame_id = 0
		self.frame_grey = self.frame = self.prev_frame = self.orig_frame = None
		self.video_file = video_file

		# -------------------- Tracking points --------------------
		self.selector = Selector(video_file)
		self.reticle = TrackedObject(self.selector, name="Reticle")
		self.camera = TrackedObject(self.selector, name="Camera")
		self.selector.close()
		self.original_points = None

		# ------------------------ Capture ------------------------
		self.cap = cv2.VideoCapture(video_file)
		self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

		# ----------------------- Debugging -----------------------
		self.debug_level = debug_level

	def track_camera(self, window_pct=((0, 0), (0.40, 1.00))):
		# Lucas kanade params
		lk_params = dict(winSize=(30, 30),
		                 maxLevel=4,
		                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		x1 = int(self.frame_grey.shape[1] * window_pct[0][1])
		x2 = int(self.frame_grey.shape[1] * window_pct[1][1])
		y1 = int(self.frame_grey.shape[0] * window_pct[0][0])
		y2 = int(self.frame_grey.shape[0] * window_pct[1][0])
		curr_frame = self.frame_grey.copy()[x1:x2, y1:y2]
		prev_frame = self.prev_frame.copy()[x1:x2, y1:y2]

		old_points = cv2.goodFeaturesToTrack(prev_frame,
		                                     maxCorners=200,
		                                     qualityLevel=0.01,
		                                     minDistance=2,
		                                     useHarrisDetector=True,
		                                     k=0.03)

		new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame,
		                                                     curr_frame,
		                                                     old_points,
		                                                     None,
		                                                     **lk_params)

		idx = np.where(status == 1)
		new_points = new_points[idx]
		old_points = old_points[idx]

		# ----------------------------- START DEBUGGING -----------------------------
		if self.debug_level == 1:
			for new, old in zip(new_points, old_points):
				cv2.line(self.frame,
				         (int(new[0]), int(new[1])),
				         (int(old[0]), int(old[1])),
				         (0, 255, 0), thickness=2, lineType=8)

		elif self.debug_level == 2:
			check_frame = self.frame.copy()
			for pt in new_points:
				cv2.circle(check_frame, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), 3)

			cv2.imshow("Checking", check_frame)
			cv2.waitKey(60)
		# ------------------------------ END DEBUGGING ------------------------------

		movement = new_points - old_points
		movement_round = np.round_(movement, decimals=2)

		# Median:
		x_average = np.mean(movement[:, 0])
		y_average = np.mean(movement[:, 1])

		# Mode:
		x_average = stats.mode(movement_round[:, 0]).mode[0]
		y_average = stats.mode(movement_round[:, 1]).mode[0]

		return str(x_average), str(y_average)

	def track_object(self, obj: TrackedObject):
		"""
		Executes the tracking functionality of the TrackedObject object, and draws tracking rectangle/text
		on the frame for debugging purposes.
		@param obj:     TrackedObject object (e.g. Reticle / Camera)
		"""
		top_left, bottom_right = matchSymbol(img=self.frame_grey,
		                                     template=obj.img,
		                                     search_area=obj.search_area
		                                     )

		# Draw rectangle over tracked object and the search area:
		cv2.rectangle(self.frame, top_left, bottom_right, (255, 255, 255), 1)
		cv2.rectangle(self.frame, obj.search_area[0], obj.search_area[1], (255, 255, 255), 1)

		# Add text to the search areas for information:
		cv2.putText(self.frame, obj.name, (top_left[0], top_left[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
		            (255, 255, 255),
		            1)
		cv2.putText(self.frame, f"{obj.name} search area", (obj.search_area[0][0], obj.search_area[0][1] - 5),
		            cv2.FONT_HERSHEY_SIMPLEX,
		            0.4,
		            (255, 255, 255),
		            1)
		obj.middle = get_middle(top_left, bottom_right)

	def track(self):
		"""
		Loops through every frame in the video file and tracks each object which has been selected by the user.
		"""
		if self.debug_level:
			cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
			cv2.moveWindow("Frame", 40, 30)

		with tqdm(total=self.length) as pbar:
			while True:
				# Get new frame from the video:
				ret, self.frame = self.cap.read()
				if not ret:
					break

				# Convert frame to Grayscale
				self.frame_grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
				self.frame_id += 1

				# Track camera:
				if self.frame_id > 1:
					camera_shake = self.track_camera()
				else:
					self.orig_frame = self.frame_grey.copy()
					camera_shake = (0, 0)
				self.prev_frame = self.frame_grey.copy()

				# Track objects:
				self.track_object(self.reticle)
				self.track_object(self.camera)


				self.data.append({
					"filename": self.video_file,
					"frame": self.frame_id,
					"reticle": self.reticle.middle,
					"camera recoil": self.camera.middle,
					"camera shake": camera_shake
				})

				# Show frame:
				if self.debug_level > 0:
					if self.frame_id == 1:
						scale = 0.8
						width = int(self.frame.shape[1]*scale)
						height = int(self.frame.shape[0]*scale)
						cv2.resizeWindow("Frame", width, height)
					cv2.imshow("Frame", self.frame)
					cv2.waitKey(15)

				pbar.update()

			pbar.close()
			self.cap.release()
			cv2.destroyAllWindows()

	def save(self, filename: Optional[str] = None):
		"""
		Saves the positional data to a given JSON file.
		@param filename:    JSON file path (e.g. position_data.json)
		"""

		if not filename:
			file_stripped = self.video_file.split("\\")[-1].split(".")[0]
			filename = f"{file_stripped}.json"

		if not self.data:
			print(f"No data has been recorded yet.")
		else:
			with open(filename, "w") as f:
				json.dump(self.data, f, indent=4)
				print(f"Data saved to {filename}")
