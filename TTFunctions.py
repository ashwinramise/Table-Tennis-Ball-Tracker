# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import pandas as pd
import tkinter as tk


ap = argparse.ArgumentParser()
args = vars(ap.parse_args())
# Define the lower and upper boundaries of the "White" ball in the HSV color space,
# WhiteLower = (0, 5, 210)  # WTT
# WhiteLower = (0, 3, 240) # 1080 60fps
WhiteLower = (0, 12, 230) # 720p 240fps
# WhiteLower = (0, 7, 224) # 1080p 120fps
WhiteUpper = (255, 255, 255)  # White

lower_table = np.array([97, 53, 134])
upper_table = np.array([255, 255, 255])


def apply_ball_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "white", then perform a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, WhiteLower, WhiteUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
    # close_mask = cv2.morphologyEx(open_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    return mask


def apply_table_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_Table = cv2.inRange(hsv, lower_table, upper_table)
    mask_Table = cv2.erode(mask_Table, None, iterations=2)
    mask_Table = cv2.dilate(mask_Table, None, iterations=2)
    return mask_Table


def contour_detection(mask):
    # find contours in the mask and initialize the current (x, y) center of the ball
    contours1 = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours1)
    return contours


def contour_detection_table(mask):
    # find contours in the mask and initialize the current (x, y) center of the ball
    contours1 = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours1


def encircling(contours, frame):
    c = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # only proceed if the radius meets a minimum size
    if radius > 2:
        # draw the circle and centroid on the frame, then update the list of tracked points
        cv2.circle(frame, (int(x), int(y)), int(radius),
                   (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        return center


def line_draw(points, args, frame):
    for i in range(1, len(points)):
        # if either of the tracked points are None, ignore them
        if points[i - 1] is None or points[i] is None:
            continue
        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, points[i - 1], points[i], (0, 153, 0), thickness)


def record_writer(records, filename):
    all_recs = [ele for ele in records if type(ele) is tuple]
    df = pd.DataFrame()
    df['x'] = [ele[0] for ele in all_recs]
    df['y'] = [ele[1] for ele in all_recs]
    df.to_csv(f"{filename}")
    return df['x'].tolist(), df['y'].tolist()


def stream(streamer):
    # Provide options for live stream or recording
    if streamer == 1:
        vs = VideoStream(src=0).start()
    elif streamer == 2:
        vs = VideoStream(input("Enter the full path for the video file: ")).start()
    else:
        vs = cv2.VideoCapture(args["video"])
    return vs


def get_bounces(table, x, y):
    Ball_in_table = 0
    for i in range(0, len(table[0])):
        k = table[0][i][0]
        for j in range(len(x)):
            if k[0] == x[j] and k[1] == y[j]:
                Ball_in_table = Ball_in_table + 1
    return round(Ball_in_table**0.5 * 10)


def display_results(bounces, total_pos):
    window = tk.Tk()
    window.title('Bounces and Points')
    label1 = tk.Label(text=f"The total bounces are {bounces}")
    label2 = tk.Label(text=f"The total points detected are {total_pos}")
    label1.pack()
    label2.pack()
    window.mainloop()


def extrapolate(history):
    import warnings
    x_val = np.array([p.center[0] for p in history['x'].tolist()])
    y_val = np.array([p.center[1] for p in history['y'].tolist()])
    warnings.filterwarnings('error')
    pol = np.polyfit(x_val, y_val, 2, full=False)
    # We grab last 2 balls let them be extrapolates or not
    b1, b2 = history[-1], history[-2]
    x1, x2 = b1.center[0], b2.center[0]
    # Predicted value is at x = x1 + (x1-x2) = 2*x1 - x2
    extrapol_x = 2 * x1 - x2
    # Evaluate parabola at x = 2*x1 - x2
    extrapol_y = np.polyval(pol, extrapol_x)
    return extrapol_x, extrapol_y


def bounce_detector(history):
    center_x, center_y = extrapolate(history)
    bounces = get_bounces(history, center_x, center_y)
    return bounces

