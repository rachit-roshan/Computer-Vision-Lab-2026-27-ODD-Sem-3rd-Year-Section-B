# BTCS301P-AS07 - EXPERIMENT 7
# Motion Estimation using Optical Flow Algorithms in Video Sequences

# Name: Rachit Roshan
# Roll No.: 48

import cv2
import numpy as np

VIDEO_SOURCE = "sample.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot open sample.mp4. Make sure sample.mp4 is in the same folder."
    )

feature_params = dict(maxCorners=100, qualityLevel=0.3,
                      minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS |
                           cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame.")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Lucas-Kanade sparse optical flow
    if p0 is None or len(p0) == 0:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        old_gray, gray, p0, None, **lk_params
    )

    sparse = frame.copy()

    if p1 is not None:
        good_new = p1[status == 1]
        good_old = p0[status == 1]

        for new, old in zip(good_new, good_old):
            xn, yn = new.ravel()
            xo, yo = old.ravel()
            cv2.arrowedLine(sparse, (int(xo), int(yo)),
                            (int(xn), int(yn)), (0, 255, 0), 2,
                            tipLength=0.3)
            cv2.circle(sparse, (int(xn), int(yn)), 3, (0, 0, 255), -1)

        p0 = good_new.reshape(-1, 1, 2)

    # Farneback dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros_like(frame)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    dense = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Lucas-Kanade Sparse Optical Flow", sparse)
    cv2.imshow("Farneback Dense Optical Flow", dense)

    old_gray = gray.copy()

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
