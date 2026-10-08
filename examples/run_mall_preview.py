import csv
import glob
import os
import sys

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from detector import PeopleDetector


def main() -> None:
    frame_paths = sorted(glob.glob(r"data/frames/frames/*.jpg"))[:300]
    if not frame_paths:
        raise FileNotFoundError("No frames found in data/frames/frames")

    os.makedirs("output", exist_ok=True)
    output_video = "output/mall_detection_preview.mp4"
    output_csv = "output/mall_detection_counts.csv"

    detector = PeopleDetector(model_path="yolov8n.pt", confidence_threshold=0.15, imgsz=1280)
    writer = None
    rows = [("frame", "people_count")]

    for idx, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        detections, count = detector.detect_people(frame)
        annotated = detector.draw_detections(frame, detections, count)

        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(
                output_video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                15,
                (w, h),
            )

        writer.write(annotated)
        rows.append((os.path.basename(frame_path), count))
        print(f"{idx}/{len(frame_paths)} {os.path.basename(frame_path)} count={count}")

    if writer is not None:
        writer.release()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"Saved preview video: {output_video}")
    print(f"Saved counts csv: {output_csv}")


if __name__ == "__main__":
    main()
