# Code:
from selector import Selector

# Public packages:
import cv2
import json
from tqdm import tqdm

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
	def __init__(self, video_file: str, debug: bool = False):
		"""
		Initialises Tracker class which will track objects in a 2D space using cv2. The class records coordinates
		data for the position of tracked objects which can be used to calculate the recoil of a weapon in EFT.

		Additional data recorded includes the frame number and video file path.

		@param video_file:  File path for the video file to be analysed.
		@param debug:       Boolean value to enable debug-mode (shows the tracking on screen in real-time).
		"""
		# ----------------- Initialise frame data -----------------
		self.data = []
		self.frame_id = 0
		self.frame = None
		self.video_file = video_file

		# -------------------- Tracking points --------------------
		self.selector = Selector(video_file)
		self.reticle = TrackedObject(self.selector, name="Reticle")
		self.camera = TrackedObject(self.selector, name="Camera")
		self.selector.close()

		# ------------------------ Capture ------------------------
		self.cap = cv2.VideoCapture(video_file)
		self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

		# ----------------------- Debugging -----------------------
		self.debug = debug

	def track_object(self, obj: TrackedObject):
		"""
		Executes the tracking functionality of the TrackedObject object, and draws tracking rectangle/text
		on the frame for debugging purposes.
		@param obj:     TrackedObject object (e.g. Reticle / Camera)
		"""
		top_left, bottom_right = matchSymbol(img=self.frame,
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
		if self.debug:
			cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
			cv2.moveWindow("Frame", 40, 30)

		with tqdm(total=self.length) as pbar:
			while True:
				# Get new frame from the video:
				ret, self.frame = self.cap.read()

				try:
					# Convert frame to Grayscale
					self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
					self.frame_id += 1
				except cv2.error:
					break

				# Track objects:
				self.track_object(self.reticle)
				self.track_object(self.camera)
				self.data.append({
					"filename": self.video_file,
					"frame": self.frame_id,
					"reticle": self.reticle.middle,
					"camera recoil": self.camera.middle
				})

				# Show frame:
				if self.debug:
					if self.frame_id == 1:
						print(self.frame.shape)
						cv2.resizeWindow("Frame", self.frame.shape[1], self.frame.shape[0])
					cv2.imshow("Frame", self.frame)
					cv2.waitKey(16)

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
