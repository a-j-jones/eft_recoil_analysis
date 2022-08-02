# Public packages:
import cv2

# Typing:
from numpy import ndarray


class Selector:
	def __init__(self, video_file: str):
		"""
		Initialises the Selector class which enables the user to select an object to track within the frame.
		@param video_file:  Video file to open.
		"""
		self.drawing = True
		self.point1 = ()
		self.point2 = ()

		# Create window:
		self.cap = cv2.VideoCapture(video_file)
		cv2.namedWindow("Frame")
		cv2.setMouseCallback("Frame", self.mouse_drawing)

		# Initial Frame:
		_, self.base_frame = self.cap.read()

	def close(self):
		"""
		Releases the connection to the video file and closes all cv2 windows.
		"""
		self.cap.release()
		cv2.destroyAllWindows()

	def mouse_drawing(self, event: int, x: int, y: int, flags, params):
		"""
		Uses the cv2 event type to determine which action to take (i.e. select point1 or point2 / stop drawing).
		@param event:   cv2 Callback event type.
		@param x:       current x position of mouse.
		@param y:       current y position of mouse.
		"""
		if event == cv2.EVENT_LBUTTONDOWN:
			self.point1 = (x, y)

		if event == cv2.EVENT_LBUTTONUP:
			self.point2 = (x, y)
			self.drawing = False

		elif event == cv2.EVENT_MOUSEMOVE:
			if self.drawing is True:
				self.point2 = (x, y)

	def get_rectangle(self) -> tuple[tuple[tuple, tuple], ndarray]:
		"""
		Loops until a rectangle has been drawn around an object to be tracked.
		@return:    Tuple of coordinates of a rectangle and corresponding cv2 image as a tracking template.
		"""
		self.drawing = True
		self.point1 = ()
		self.point2 = ()
		while True:
			frame = self.base_frame.copy()
			if self.point1 and self.point2:
				cv2.rectangle(frame, self.point1, self.point2, (0, 255, 0))

			cv2.imshow("Frame", frame)

			if not self.drawing:
				cv2.rectangle(self.base_frame, self.point1, self.point2, (0, 0, 255))
				break

			key = cv2.waitKey(1)
			if key == 27:
				break

		return (self.point1, self.point2), frame[self.point1[1]:self.point2[1], self.point1[0]:self.point2[0]]