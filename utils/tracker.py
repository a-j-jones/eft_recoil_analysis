# Public packages:
import glob
import json
from typing import Tuple, Optional

import cv2
import numpy as np
from imutils import resize
from imutils.video import FileVideoStream
from numpy import ndarray
from tqdm import tqdm

# Private packages:
from utils.selector import Selector


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
    point1_x, point1_y = point1
    point2_x, point2_y = point2

    return int(round((point2_x - point1_x) / 2) + point1_x), int(round((point2_y - point1_y) / 2) + point1_y)


def matchSymbol(img: ndarray, template: ndarray,
                search_area: Tuple[Tuple[int, int], Tuple[int, int]] = None) -> Tuple[
    Tuple[int, int], Tuple[int, int], int]:
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
    else:
        x1 = y1 = 0

    # Apply template Matching
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Extract coordinates of max value:
    top_left = (max_loc[0] + x1, max_loc[1] + y1)
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left, bottom_right, max_val


class TrackedObject:
    def __init__(self,
                 name: str,
                 selector: Selector = None,
                 loc: Tuple[Tuple[int, int], Tuple[int, int]] = None,
                 img: ndarray = None) -> None:
        """
        Initialises TrackedObject which contains the information required for tracking a specific object.
        @param name:        Name of the tracked object.
        @param selector:    Selector object which defines the initial img and location of the object.
        @param loc:         Location of the object to be tracked.
        @param img:         Image to be tracked.
        """
        self.__search_area = None
        self.name = name
        self.middle = None

        if selector:
            self.loc, self.img = selector.get_rectangle()
        elif loc is None or img is None:
            raise ValueError("If Selector object is not provided then Location and Image template must be instead.")
        else:
            self.loc = loc
            self.img = img

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    @property
    def search_area(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return self.__search_area

    @search_area.getter
    def search_area(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Getter method for search_area which returns either the margin around the middle of the object
        if available, otherwise it calculates the middle first.
        @return:    A Tuple of coordinates of a rectangle to search for the object.
        """
        if self.middle:
            return search_margin(self.middle, 75)
        else:
            return search_margin(get_middle(*self.loc), 75)


class Tracker:
    def __init__(self, video_file: str, debug_level: int = 0, find_new_points: int = 0) -> None:
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

        # Get number of frames and initial frame:
        video = cv2.VideoCapture(video_file)
        self.length = video.get(cv2.CAP_PROP_FRAME_COUNT)
        _, self.initial_frame = video.read()
        video.release()

        # -------------------- Tracking points --------------------
        self.reticle = self.get_reticle()
        self.original_points = None
        self.new_points = self.old_points = None
        self.find_new_points = find_new_points

        # ------------------------ Capture ------------------------
        self.cap = FileVideoStream(video_file, queue_size=256).start()

        # ----------------------- Debugging -----------------------
        self.debug_level = debug_level

    def get_reticle(self, window_pct=((0.25, 0.40), (0.75, 0.60))) -> TrackedObject:
        """
        Loops through the saved reticle templates to see if we can identify a reticle in the screen, if one
        cannot be found then the code reverts to the manual selection tool to find the appropriate reticle on
        the screen.

        @param window_pct:  Window size to limit the area of the screen that is searched by cv2.MatchTemplate for
                            performance.
        """

        # Gets the grey frame and the window dimensions:
        grey_frame = cv2.cvtColor(self.initial_frame, cv2.COLOR_BGR2GRAY)
        x1 = int(grey_frame.shape[1] * window_pct[0][1])
        x2 = int(grey_frame.shape[1] * window_pct[1][1])
        y1 = int(grey_frame.shape[0] * window_pct[0][0])
        y2 = int(grey_frame.shape[0] * window_pct[1][0])

        # Loop through each file in the 'templates' folder:
        for file in glob.glob("templates/*.jpg"):
            template = cv2.imread(file)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            match_data = {scale: None for scale in np.linspace(0.8, 1.2, 25)}
            for scale in match_data.keys():
                resized_template = resize(template, width=int(template.shape[1] * scale))

                top_left, bottom_right, match = matchSymbol(img=grey_frame,
                                                            template=resized_template,
                                                            search_area=((x1, y1), (x2, y2)))
                match_data[scale] = {
                    "match": match,
                    "loc": (top_left, bottom_right)
                }

            match_value = np.array([m["match"] for k, m in match_data.items()])
            match_scales = list(match_data.keys())

            best_match = match_scales[np.argmax(match_value)]
            data = match_data[best_match]
            print(f"The best template match for the reticle template was {data['match']:.2f}")
            if data["match"] > 0.65:
                img = self.initial_frame[data["loc"][0][1]:data["loc"][1][1], data["loc"][0][0]:data["loc"][1][0]]
                return TrackedObject(name="Reticle", loc=(data["loc"][0], data["loc"][1]), img=img)

        # If no match found, revert to manual selection and return object.
        selector = Selector(self.video_file, best_match=(data["loc"][0], data["loc"][1]))
        t = TrackedObject(name="Reticle", selector=selector)
        selector.close()

        return t

    def track_camera(self,
                     window_pct: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 0.45), (0.36, 0.55))) -> None:
        """
        Tracks the camera position using cv2.goodFeaturesToTrack and cv2.calcOpticalFlowPyrLK, this contributes
        to the overall recoil of the weapon.

        @param window_pct: 	Tuple of screen size relative points (percentage of the screen width/height)
                            e.g. (y1, x1), (y2, x2)
        @return:
        """
        lk_params = dict(
            winSize=(150, 150),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        maxCorners = 50
        feature_params = dict(
            maxCorners=maxCorners,
            qualityLevel=0.1,
            minDistance=10,
            blockSize=3
        )

        x1 = int(self.frame_grey.shape[1] * window_pct[0][1])
        x2 = int(self.frame_grey.shape[1] * window_pct[1][1])
        y1 = int(self.frame_grey.shape[0] * window_pct[0][0])
        y2 = int(self.frame_grey.shape[0] * window_pct[1][0])

        curr_frame = self.frame_grey[y1:y2, x1:x2]
        prev_frame = self.prev_frame[y1:y2, x1:x2]

        if self.new_points is None or len(self.new_points) < (maxCorners * 0.75) or self.find_new_points:
            self.old_points = cv2.goodFeaturesToTrack(prev_frame,
                                                      **feature_params)
        else:
            self.old_points = self.new_points.reshape(len(self.new_points), 1, 2)

        self.new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame,
                                                                  curr_frame,
                                                                  self.old_points,
                                                                  None,
                                                                  **lk_params)

        idx = np.where(status == 1)
        self.new_points = self.new_points[idx]
        self.old_points = self.old_points[idx]

        if self.debug_level == 1:
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            for new, old in zip(self.new_points, self.old_points):
                cv2.line(self.frame,
                         (int(new[0] + x1), int(new[1] + y1)),
                         (int(old[0] + x1), int(old[1] + y1)),
                         (0, 255, 0), thickness=4, lineType=8)

        movement = self.new_points - self.old_points

        # Median:
        x_average = np.median(movement[:, 0])
        y_average = np.median(movement[:, 1])

        return float(x_average), float(y_average)

    def track_object(self, obj: TrackedObject) -> None:
        """
        Executes the tracking functionality of the TrackedObject object, and draws tracking rectangle/text
        on the frame for debugging purposes.
        @param obj:     TrackedObject object (e.g. Reticle / Camera)
        """
        top_left, bottom_right, _ = matchSymbol(img=self.frame_grey,
                                                template=obj.img,
                                                search_area=obj.search_area
                                                )

        if self.debug_level > 0:
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

    def track(self) -> None:
        """
        Loops through every frame in the video file and tracks each object which has been selected by the user.
        """
        # if self.debug_level:
        # 	cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        # 	cv2.moveWindow("Frame", 40, 30)

        with tqdm(total=self.length) as pbar:
            while self.cap.more():
                # Get new frame from the video:
                self.frame = self.cap.read()
                if self.frame is None:
                    break

                if self.debug_level > 0:
                    cv2.putText(self.frame, "Queue Size: {}".format(self.cap.Q.qsize()),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Convert frame to Grayscale
                self.frame_grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                # After converting the frame to grayscale, apply histogram equalization
                self.frame_grey = cv2.equalizeHist(self.frame_grey)

                self.frame_id += 1

                # Track camera:
                if self.frame_id > 1:
                    camera_shake = self.track_camera()
                else:
                    self.orig_frame = self.frame_grey
                    camera_shake = (0, 0)
                self.prev_frame = self.frame_grey

                # Track objects:
                self.track_object(self.reticle)

                self.data.append({
                    "filename": self.video_file,
                    "frame": self.frame_id,
                    "reticle": self.reticle.middle,
                    "optical_flow": camera_shake
                })

                # Show frame:
                if self.debug_level > 0:
                    self.draw_recoil()
                    cv2.imshow("Frame", self.frame)
                    cv2.waitKey(15)

                pbar.update()

            pbar.close()
            self.cap.stop()
            cv2.destroyAllWindows()

    def draw_recoil(self, rel_start_x=0.4, rel_start_y=0.5) -> None:
        """
        Draws the recoil pattern on the frame in real time as a debugging tool:

        @param rel_start_x: Relative start x position of the recoil (percentage of screen)
        @param rel_start_y: Relative start y position of the recoil (percentage of screen)
        """
        start_x = rel_start_x * self.frame.shape[1]
        start_y = rel_start_y * self.frame.shape[0]

        arrs = {
            "reticle_x": np.array([x["reticle"][0] for x in self.data]),
            "reticle_y": np.array([x["reticle"][1] for x in self.data]),
            "camera_x": np.array([x["optical_flow"][0] for x in self.data]),
            "camera_y": np.array([x["optical_flow"][1] for x in self.data])
        }

        # Shifts:
        shift_by = 1
        shift_nan = np.empty(shift_by)
        shift_nan[:] = np.nan

        for axis in ["x", "y"]:
            # Reticle processing:
            base_arr = arrs[f"reticle_{axis}"]
            shift_arr = np.append(shift_nan, base_arr[:-shift_by])
            abs_move_arr = base_arr - shift_arr
            arrs[f"reticle_{axis}"] = np.append(np.zeros(1), np.cumsum(abs_move_arr[1:]))

            # Camera processing:
            base_arr = arrs[f"camera_{axis}"]
            arrs[f"camera_{axis}"] = np.append(np.zeros(1), np.cumsum(base_arr[1:]))

            # Combined:
            arrs[f"combined_{axis}"] = arrs[f"reticle_{axis}"] + arrs[f"camera_{axis}"]

        x_arr = arrs[f"combined_x"]
        y_arr = arrs[f"combined_y"] * -1

        for i in range(len(y_arr)):
            if i == 0:
                continue

            cv2.line(self.frame,
                     (int(x_arr[i] + start_x), int(y_arr[i] + start_y)),
                     (int(x_arr[i - 1] + start_x), int(y_arr[i - 1] + start_y)),
                     (0, 255, 0), thickness=4, lineType=8)

            cv2.putText(self.frame, "Recoil", (int(start_x - 20), int(start_y + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1)

    def save(self, filename: Optional[str] = None) -> None:
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
