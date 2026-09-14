"""
Experiment No. 5
Feature Extraction and Image Analysis using SIFT and HOG Descriptors

Name: Rachit Roshan
Roll No.: 48

"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, transform
from skimage.feature import hog


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def save_sample_images():
    """Save a real-world sample image and a transformed similar image."""
    rgb = data.astronaut()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Resize to a manageable, consistent size.
    gray = cv2.resize(gray, (640, 512), interpolation=cv2.INTER_AREA)

    # Create a similar image for matching: small rotation + brightness change.
    matrix = cv2.getRotationMatrix2D((320, 256), 8, 1.03)
    transformed = cv2.warpAffine(
        gray, matrix, (640, 512), borderMode=cv2.BORDER_REFLECT
    )
    transformed = cv2.convertScaleAbs(transformed, alpha=1.02, beta=5)

    original_path = DATA_DIR / "real_world_image.png"
    similar_path = DATA_DIR / "similar_image.png"
    cv2.imwrite(str(original_path), gray)
    cv2.imwrite(str(similar_path), transformed)

    return original_path, similar_path


def load_and_preprocess(path):
    """Load image and convert to grayscale if necessary."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # Mild contrast normalization.
    image = cv2.equalizeHist(image)
    return image


def run_sift(image, name):
    """Detect SIFT keypoints and compute descriptors."""
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "SIFT is not available in this OpenCV installation. "
            "Install a recent opencv-python package."
        )

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    visualization = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_sift_keypoints.png"), visualization)

    return keypoints, descriptors


def run_hog(image, name):
    """Extract HOG features and save a visualization."""
    image_float = image.astype(np.float32) / 255.0

    features, hog_image = hog(
        image_float,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
    )

    # Convert HOG visualization to uint8 for saving.
    hog_uint8 = np.uint8(255 * (hog_image / (hog_image.max() + 1e-9)))
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_hog_visualization.png"), hog_uint8)

    return features, hog_image


def save_preprocessing_plot(image, transformed):
    """Visualize the two images used in the experiment."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Preprocessed Real-World Image")
    axes[0].axis("off")

    axes[1].imshow(transformed, cmap="gray")
    axes[1].set_title("Similar Transformed Image")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_input_images.png", dpi=160)
    plt.close(fig)


def save_sift_hog_plot(image, keypoints, hog_image):
    """Create a side-by-side SIFT/HOG visualization."""
    sift_vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(cv2.cvtColor(sift_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"SIFT Keypoints ({len(keypoints)})")
    axes[0].axis("off")

    axes[1].imshow(hog_image, cmap="gray")
    axes[1].set_title("HOG Visualization")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_sift_and_hog.png", dpi=160)
    plt.close(fig)


def match_sift(des1, des2, keypoints1, keypoints2, image1, image2):
    """Perform basic SIFT matching using Lowe's ratio test."""
    if des1 is None or des2 is None:
        return [], None, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good.append(first)

    good = sorted(good, key=lambda m: m.distance)

    match_image = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        good[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(OUTPUT_DIR / "03_sift_image_matching.png"), match_image)

    match_ratio = len(good) / max(1, len(knn_matches))
    return good, match_image, match_ratio


def create_report(
    keypoints1,
    descriptors1,
    hog_features1,
    keypoints2,
    descriptors2,
    hog_features2,
    good_matches,
    match_ratio,
):
    """Write reproducible numerical observations to a text report."""
    report = f"""
EXPERIMENT NO. 5 — SIFT AND HOG FEATURE EXTRACTION
==================================================

Image source:
skimage.data.astronaut() — used as the real-world sample image.
A rotated/scaled/brightness-adjusted copy was generated for basic matching.

SIFT RESULTS
------------
Original image keypoints: {len(keypoints1)}
Original SIFT descriptor shape: {None if descriptors1 is None else descriptors1.shape}
Similar image keypoints: {len(keypoints2)}
Similar SIFT descriptor shape: {None if descriptors2 is None else descriptors2.shape}

HOG RESULTS
-----------
Original HOG feature length: {len(hog_features1)}
Similar image HOG feature length: {len(hog_features2)}
Parameters:
- Orientations: 9
- Pixels per cell: 8 x 8
- Cells per block: 2 x 2
- Block normalization: L2-Hys

BASIC IMAGE MATCHING
--------------------
Good SIFT matches after Lowe's ratio test: {len(good_matches)}
Good-match ratio: {match_ratio:.4f}

OBSERVATION
-----------
SIFT produces localized keypoints and descriptors that are useful for matching
distinctive image structures under moderate scale, rotation, and illumination
changes. HOG summarizes local gradient orientations and is particularly useful
for describing shape and edge patterns. HOG is not inherently rotation invariant
and depends on the chosen cell/block parameters.

The exact SIFT keypoint count and matching count can vary slightly with OpenCV
versions and runtime environments.
"""
    (OUTPUT_DIR / "experiment_report.txt").write_text(
        report.strip() + "\n", encoding="utf-8"
    )


def main():
    print("Experiment No. 5 — SIFT and HOG")
    print("=" * 45)

    original_path, similar_path = save_sample_images()
    image = load_and_preprocess(original_path)
    similar = load_and_preprocess(similar_path)

    print(f"Original image: {original_path}")
    print(f"Similar image:  {similar_path}")

    save_preprocessing_plot(image, similar)

    kp1, des1 = run_sift(image, "original")
    kp2, des2 = run_sift(similar, "similar")

    hog_features1, hog_image1 = run_hog(image, "original")
    hog_features2, hog_image2 = run_hog(similar, "similar")

    save_sift_hog_plot(image, kp1, hog_image1)

    good_matches, _, match_ratio = match_sift(
        des1, des2, kp1, kp2, image, similar
    )

    create_report(
        kp1, des1, hog_features1,
        kp2, des2, hog_features2,
        good_matches, match_ratio,
    )

    print("\nSIFT keypoints (original):", len(kp1))
    print("SIFT keypoints (similar): ", len(kp2))
    print("HOG feature length:", len(hog_features1))
    print("Good SIFT matches:", len(good_matches))
    print("\nCompleted. Check the 'outputs' folder for visualizations and report.")


if __name__ == "__main__":
    main()
