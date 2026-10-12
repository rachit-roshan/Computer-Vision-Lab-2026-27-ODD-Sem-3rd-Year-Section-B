from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
DEFAULT_IMAGE = ASSET_DIR / "sample_image.png"
OUTPUT_DIR = ROOT / "outputs"


def create_sample_image(path: Path) -> None:
    """Create a colorful sample PNG so the program works without a supplied image."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    height, width = 320, 480
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # OpenCV uses BGR channel order.
            image[y, x] = (
                int(200 - 140 * x / (width - 1)),
                int(55 + 160 * y / (height - 1)),
                int(35 + 180 * x / (width - 1)),
            )
    cv2.rectangle(image, (65, 65), (205, 255), (40, 185, 245), -1)
    cv2.circle(image, (325, 160), 68, (110, 210, 40), -1)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not create sample image at: {path}")


def load_image(path: Path) -> np.ndarray:
    """Read a color image and give a useful error if it cannot be opened."""
    if not path.exists() and path.resolve() == DEFAULT_IMAGE.resolve():
        print("Sample image was missing. Creating a new sample image...")
        create_sample_image(path)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}\n"
            "If using the default sample, make sure the assets folder is writable. "
            "For your own image, pass its full path using --image."
        )
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fundamental OpenCV image processing lab")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load image and report its properties.
    image = load_image(args.image)
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    print("\n--- Original Image Properties ---")
    print(f"Image: {args.image.resolve()}")
    print(f"Width x Height: {width} x {height} pixels")
    print(f"Resolution: {width * height:,} pixels")
    print(f"Channels: {channels}")
    print(f"Data type: {image.dtype}")
    print(f"Array shape: {image.shape}")

    # Save the input in JPEG and PNG formats.
    save_image(OUTPUT_DIR / "original_copy.jpg", image)
    save_image(OUTPUT_DIR / "original_copy.png", image)

    # Convert BGR to grayscale, HSV, and LAB.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    save_image(OUTPUT_DIR / "grayscale.png", gray)
    save_image(OUTPUT_DIR / "hsv.png", hsv)
    save_image(OUTPUT_DIR / "lab.png", lab)

    # Resize, rotate, and flip horizontally/vertically.
    resized = cv2.resize(image, (max(1, width // 2), max(1, height // 2)))
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), 45, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height))
    horizontal_flip = cv2.flip(image, 1)
    vertical_flip = cv2.flip(image, 0)
    save_image(OUTPUT_DIR / "resized.png", resized)
    save_image(OUTPUT_DIR / "rotated_45_degrees.png", rotated)
    save_image(OUTPUT_DIR / "horizontal_flip.png", horizontal_flip)
    save_image(OUTPUT_DIR / "vertical_flip.png", vertical_flip)

    # Complement / negative image.
    negative = cv2.bitwise_not(image)
    save_image(OUTPUT_DIR / "negative.png", negative)

    # Crop a central Region of Interest (ROI).
    x1, x2 = width // 4, max(width // 4 + 1, (3 * width) // 4)
    y1, y2 = height // 4, max(height // 4 + 1, (3 * height) // 4)
    roi = image[y1:min(y2, height), x1:min(x2, width)]
    if roi.size == 0:
        raise ValueError("Image is too small to crop a region of interest.")
    save_image(OUTPUT_DIR / "roi.png", roi)
    print(f"ROI shape: {roi.shape}")

    # Compare original and processed images.
    items = [
        ("Original", image), ("Grayscale", gray), ("HSV", hsv), ("LAB", lab),
        ("Resized", resized), ("Rotated 45 degrees", rotated),
        ("Horizontal flip", horizontal_flip), ("Vertical flip", vertical_flip),
        ("Negative", negative), ("ROI", roi)
    ]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, (title, processed) in zip(axes.flat, items):
        if processed.ndim == 2:
            ax.imshow(processed, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Fundamental Image Processing Operations")
    fig.tight_layout()
    comparison = OUTPUT_DIR / "comparison.png"
    fig.savefig(comparison, dpi=150, bbox_inches="tight")
    print(f"Processed images saved in: {OUTPUT_DIR.resolve()}")
    print(f"Comparison figure saved as: {comparison.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
