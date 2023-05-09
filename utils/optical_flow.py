from typing import Tuple

import cv2
import numpy as np


def lucas_kanade(tracker: "Tracker",
                 prev_frame: np.ndarray,
                 curr_frame: np.ndarray,
                 x1: float,
                 y1: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the movement of the frame using the Lucas-Kanade method.

    @param tracker: Tracker object
    @param prev_frame: Previous video frame
    @param curr_frame: Current video frame
    @param x1: Top-left x coordinate of the portion of frame to track
    @param y1: Top-left y coordinate of the portion of frame to track
    @return:
    """

    # Parameters for lucas kanade optical flow:
    LK_PARAMS = dict(
        winSize=(150, 150),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    MAX_CORNERS = 50
    FEATURE_PARAMS = dict(
        maxCorners=MAX_CORNERS,
        qualityLevel=0.1,
        minDistance=10,
        blockSize=3
    )

    if tracker.new_points is None or len(tracker.new_points) < (MAX_CORNERS * 0.75) or tracker.find_new_points:
        tracker.old_points = cv2.goodFeaturesToTrack(prev_frame,
                                                     **FEATURE_PARAMS)
    else:
        tracker.old_points = tracker.new_points.reshape(len(tracker.new_points), 1, 2)

    tracker.new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame,
                                                                 curr_frame,
                                                                 tracker.old_points,
                                                                 None,
                                                                 **LK_PARAMS)

    idx = np.where(status == 1)
    tracker.new_points = tracker.new_points[idx]
    tracker.old_points = tracker.old_points[idx]
    movement = tracker.new_points - tracker.old_points
    movement = np.median(movement, axis=0)

    # Create an array of lines:
    lines = []
    for new, old in zip(tracker.new_points, tracker.old_points):
        lines.append([int(new[0] + x1), int(new[1] + y1), int(old[0] + x1), int(old[1] + y1)])
    lines = np.array(lines)

    return movement, lines


def farneback(prev_frame: np.ndarray,
              curr_frame: np.ndarray,
              x1: float,
              y1: float,
              frame: int = 0,
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the movement of the frame using the Farneback method.

    @param prev_frame: Previous video frame
    @param curr_frame: Current video frame
    @param x1: Top-left x coordinate of the portion of frame to track
    @param y1: Top-left y coordinate of the portion of frame to track
    @param frame: Frame number
    @return:
    """
    # Parameters for farneback optical flow:
    FB_PARAMS = dict(
        pyr_scale=0.1,
        levels=3,
        winsize=100,
        iterations=4,
        poly_n=9,
        poly_sigma=1.2,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # if frame > 50:
    #     FB_PARAMS["winsize"] = int(FB_PARAMS["winsize"] * 0.75)
    #     pass

    # Calculate the movement:
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, **FB_PARAMS)
    flow_magnitude = np.linalg.norm(flow, axis=2)
    threshold = 0.20  # Adjust this value based on your requirements
    mask = flow_magnitude > threshold

    # Only filter if >25% of the pixels are moving:
    if (mask.sum() / flow[:, :, 0].size) > 0.25:
        filtered_flow = flow[mask]
    else:
        filtered_flow = flow

    # After filtering the flow, calculate movement base on the mode bin:
    movement = [0, 0]
    for idx in range(2):
        x_dir = np.histogram(filtered_flow[:, idx], bins=100)
        movement[idx] = x_dir[1][np.argmax(x_dir[0])]

    lines = []
    for y in range(0, flow.shape[0], 5):
        for x in range(0, flow.shape[1], 5):
            fx, fy = flow[y, x]
            lines.append([x1 + x, y1 + y, x1 + x + int(fx), y1 + y + int(fy)])
    lines = np.array(lines)

    return movement, lines
