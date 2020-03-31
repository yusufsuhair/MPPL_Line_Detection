import cv2
import numpy as np

video = cv2.VideoCapture("video.mp4")


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture("video.mp4")
        continue
    height = orig_frame.shape[0]
    widht = orig_frame.shape[1]
    region_of_interest_vertices = [

        (0, height),
        (0.25 * widht, 0.5 * height),
        (0.75 * widht, 0.5 * height),
        (widht, height)
    ]

    cropped_image = region_of_interest(orig_frame,
                                       np.array([region_of_interest_vertices], np.int32), )

    frame = cv2.GaussianBlur(cropped_image, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([0, 0, 168])
    up_yellow = np.array([172, 111, 255])

    mask = cv2.inRange(hsv, low_yellow, up_yellow)
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
