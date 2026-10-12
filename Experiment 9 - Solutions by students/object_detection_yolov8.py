
import argparse
import os
import time

import cv2
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="YOLOv8 object detection using a pre-trained COCO model."
    )
    parser.add_argument(
        "--source",
        default="https://ultralytics.com/images/image.jpg",
        help="Path/URL to an image, video, webcam index, or folder of images."
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model. yolov8n.pt is lightweight and suitable for lab work."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold."
    )
    parser.add_argument(
        "--output",
        default="runs/detect/experiment9",
        help="Directory where detection results are saved."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 65)
    print("EXPERIMENT 9: OBJECT DETECTION USING YOLOv8")
    print("=" * 65)

    # Step 1 & 2: Load the pre-trained deep learning model.
    print("\n[1] Loading YOLOv8 model...")
    model = YOLO(args.model)
    print("[OK] Model loaded.")

    # Step 3 & 4: Access image and perform preprocessing internally.
    print(f"\n[2] Input source: {args.source}")
    print("[3] Running object detection...")

    start_time = time.perf_counter()

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=640,
        save=True,
        project="runs/detect",
        name="experiment9",
        exist_ok=True,
        verbose=False
    )

    elapsed = time.perf_counter() - start_time

    # Step 5-7: Display detected objects, bounding boxes,
    # class labels, confidence scores and basic statistics.
    total_detections = 0

    for image_number, result in enumerate(results, start=1):
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        total_detections += count

        print(f"\nImage {image_number}:")
        print(f"  Detected objects: {count}")

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes, start=1):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_name = result.names[class_id]

                print(
                    f"  {i}. {class_name:15s} "
                    f"confidence={confidence:.2f} "
                    f"box=({x1}, {y1}, {x2}, {y2})"
                )

    # Approximate inference speed for the complete prediction call.
    print("\n" + "-" * 65)
    print("RESULT SUMMARY")
    print("-" * 65)
    print(f"Total detected objects : {total_detections}")
    print(f"Prediction time        : {elapsed:.3f} seconds")
    if elapsed > 0:
        print(f"Approx. processing rate: {len(results) / elapsed:.2f} image(s)/sec")
    print(f"Confidence threshold   : {args.conf}")
    print(f"Output directory       : {os.path.abspath(args.output)}")
    print("-" * 65)

    # Optional display for a normal local image source.
    if os.path.isfile(args.source):
        image = cv2.imread(args.source)
        if image is not None:
            annotated = results[0].plot()
            cv2.imshow("YOLOv8 Object Detection - Experiment 9", annotated)
            print("\nPress any key on the image window to close it.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
