# import the necessary packages
from collections import deque
import argparse
import cv2
import time
import TTFunctions as bd
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=10,
                help="max buffer size")
args = vars(ap.parse_args())
# Initialize the list of tracked points
points = deque(maxlen=args["buffer"])
# Ask user to define what option to go with
streamer = int(input("Live Stream(1) or Recording(2): "))
# Manage life feed or video stream
vs = bd.stream(streamer)
time.sleep(2.0)
records = []
table = []
while True:
    # grab the current frame
    frame = vs.read()
    # if we are using a video and it did not grab a frame, then it has reached the end of the video
    if frame is None:
        break
    else:
        frame = imutils.resize(frame, width=1500)
        # frame = frame[300:660, 0:1300] # 720p 240fps
        frame = frame[300:670, 200:1350] #1080 120fps
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame
    ball_mask = bd.apply_ball_mask(frame)
    table_mask = bd.apply_table_mask(frame)
    while True:
        table.append(cv2.findNonZero(table_mask))
        break
    ball_contours = bd.contour_detection(ball_mask)
    table_contour = bd.contour_detection_table(table_mask)
    center = None
    # only proceed if at least one contour was found
    if len(ball_contours) > 0:
        # encircle the ball in the image with center marked
        center = bd.encircling(ball_contours, frame)
    points.appendleft(center)
    point_list = list(points)
    records.append(point_list[0])
    bd.line_draw(points, args, frame)
    cv2.imshow("Frame", frame)  # Show the Frame
    key = cv2.waitKey(1) & 0xFF
    # if the 'ESC' key is pressed, stop the loop
    if key == 27:
        break
    elif frame is None:
        break
X, Y = bd.record_writer(records, filename='points.csv')
bounces = bd.get_bounces(table, X, Y)  # get the bounces
total = len(X)  # get total points
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
    vs.release()
# Display the results
bd.display_results(bounces, total)
# close all windows
cv2.destroyAllWindows()
