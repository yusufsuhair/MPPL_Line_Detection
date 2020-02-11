import cv2
import numpy as np

upper_left = (100, 600)
bottom_right = (800, 2000)
video = cv2.VideoCapture("video.mp4")

while True:
    ret, orig_frame_ori = video.read()
    orig_frame = orig_frame_ori[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    if not ret:
        video = cv2.VideoCapture("video.mp4")
        continue

    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_white = np.array([0, 0, 168])
    up_white = np.array([172, 111, 255])
    mask = cv2.inRange(hsv, low_white, up_white)
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()