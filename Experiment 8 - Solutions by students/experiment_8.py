# ============================================================
# BTCS301P - EXPERIMENT 8
# Application of Optical Flow for Real-Time Object Tracking
# and Motion Analysis

# Name: Rachit Roshan
# Roll No.: 48
# ============================================================

import cv2
import numpy as np


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

# 0 = webcam
# "sample.mp4" = saved video
VIDEO_SOURCE = 0


# ------------------------------------------------------------
# OPEN VIDEO
# ------------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("ERROR: Could not open webcam/video.")
    print("Try VIDEO_SOURCE = 1 if you have another camera.")
    exit()


# ------------------------------------------------------------
# SHI-TOMASI FEATURE PARAMETERS
# ------------------------------------------------------------

feature_params = {
    "maxCorners": 100,
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7
}


# ------------------------------------------------------------
# LUCAS-KANADE PARAMETERS
# ------------------------------------------------------------

lk_params = {
    "winSize": (15, 15),
    "maxLevel": 2,
    "criteria": (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
}


# ------------------------------------------------------------
# READ FIRST FRAME
# ------------------------------------------------------------

ret, first_frame = cap.read()

if not ret:
    print("ERROR: Could not read the first video frame.")
    cap.release()
    exit()


# Convert first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)


# ------------------------------------------------------------
# SHI-TOMASI CORNER DETECTION
# ------------------------------------------------------------

# IMPORTANT:
# Do NOT pass None as a positional mask argument.
# Do NOT pass an empty (0,0) mask.
#
# This form is compatible with OpenCV 5.

p0 = cv2.goodFeaturesToTrack(
    prev_gray,
    maxCorners=feature_params["maxCorners"],
    qualityLevel=feature_params["qualityLevel"],
    minDistance=feature_params["minDistance"],
    blockSize=feature_params["blockSize"]
)


if p0 is None:
    print("No feature points detected.")
    print("Try better lighting or a video with more visible texture.")
    cap.release()
    exit()


print("Experiment 8 started successfully.")
print("Shi-Tomasi feature points detected:", len(p0))
print("Press Q to quit.")


# ------------------------------------------------------------
# TRAJECTORY MASK
# ------------------------------------------------------------

trajectory_mask = np.zeros_like(first_frame)


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended or frame could not be read.")
        break


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # --------------------------------------------------------
    # LUCAS-KANADE SPARSE OPTICAL FLOW
    # --------------------------------------------------------

    sparse_frame = frame.copy()

    if p0 is not None and len(p0) > 0:

        p1, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            p0,
            None,
            **lk_params
        )


        if p1 is not None:

            good_new = p1[status == 1]
            good_old = p0[status == 1]

            magnitudes = []


            # ------------------------------------------------
            # DRAW TRAJECTORIES AND VECTORS
            # ------------------------------------------------

            for new, old in zip(good_new, good_old):

                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()


                # Displacement
                dx = x_new - x_old
                dy = y_new - y_old


                # Magnitude
                magnitude = float(
                    np.sqrt(dx * dx + dy * dy)
                )

                magnitudes.append(magnitude)


                # Direction
                direction = np.degrees(
                    np.arctan2(dy, dx)
                )


                # Draw trajectory
                cv2.line(
                    trajectory_mask,
                    (int(x_old), int(y_old)),
                    (int(x_new), int(y_new)),
                    (0, 255, 0),
                    2
                )


                # Draw motion arrow
                cv2.arrowedLine(
                    sparse_frame,
                    (int(x_old), int(y_old)),
                    (int(x_new), int(y_new)),
                    (255, 0, 0),
                    2,
                    tipLength=0.25
                )


                # Draw feature point
                cv2.circle(
                    sparse_frame,
                    (int(x_new), int(y_new)),
                    4,
                    (0, 0, 255),
                    -1
                )


            # Add trajectories
            sparse_frame = cv2.add(
                sparse_frame,
                trajectory_mask
            )


            # ------------------------------------------------
            # AVERAGE SPEED
            # ------------------------------------------------

            if len(magnitudes) > 0:

                average_speed = float(
                    np.mean(magnitudes)
                )

            else:

                average_speed = 0.0


            cv2.putText(
                sparse_frame,
                f"Lucas-Kanade Avg Speed: "
                f"{average_speed:.2f} px/frame",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            # Update feature points
            p0 = good_new.reshape(-1, 1, 2)


            # If too few points remain, detect new ones
            if len(p0) < 10:

                p0 = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=feature_params["maxCorners"],
                    qualityLevel=feature_params["qualityLevel"],
                    minDistance=feature_params["minDistance"],
                    blockSize=feature_params["blockSize"]
                )


    # --------------------------------------------------------
    # FARNEBACK DENSE OPTICAL FLOW
    # --------------------------------------------------------

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )


    # Calculate magnitude and direction
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )


    # --------------------------------------------------------
    # HSV VISUALIZATION
    # --------------------------------------------------------

    hsv = np.zeros_like(frame)

    # Direction
    hsv[..., 0] = (
        angle * 180 / np.pi / 2
    )

    # Saturation
    hsv[..., 1] = 255

    # Magnitude
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )


    dense_frame = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


    dense_average = float(
        np.mean(magnitude)
    )


    cv2.putText(
        dense_frame,
        f"Farneback Avg Motion: "
        f"{dense_average:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    cv2.imshow(
        "Experiment 8 - Lucas-Kanade",
        sparse_frame
    )

    cv2.imshow(
        "Experiment 8 - Farneback",
        dense_frame
    )


    # Update previous frame
    prev_gray = gray.copy()


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Experiment 8 completed.")