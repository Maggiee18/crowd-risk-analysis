import glob
import os
import sys
from typing import List, Tuple

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from detector import PeopleDetector


def put_alert_text(img, text: str, color: Tuple[int, int, int], y: int = 40) -> None:
    cv2.rectangle(img, (10, y - 30), (630, y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main() -> None:
    frame_paths: List[str] = sorted(glob.glob(r"data/frames/frames/*.jpg"))
    if not frame_paths:
        raise FileNotFoundError("No frames found in data/frames/frames")

    os.makedirs("output/alerts", exist_ok=True)
    detector = PeopleDetector(model_path="yolov8n.pt", confidence_threshold=0.5)

    best_over_30 = None  # (count, path, annotated)
    best_over_80 = None  # (count, path, annotated)
    global_best = None

    for idx, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        detections, count = detector.detect_people(frame)
        annotated = detector.draw_detections(frame, detections, count)

        if global_best is None or count > global_best[0]:
            global_best = (count, frame_path, annotated.copy())

        if count > 30 and (best_over_30 is None or count > best_over_30[0]):
            img = annotated.copy()
            put_alert_text(img, f"CRITICAL: Count {count} exceeds 30", (0, 0, 255))
            best_over_30 = (count, frame_path, img)

        if count >= 24 and (best_over_80 is None or count > best_over_80[0]):
            img = annotated.copy()
            put_alert_text(
                img,
                f"ALERT: {count}/30 ({(count/30)*100:.1f}%) capacity reached",
                (0, 165, 255),
            )
            best_over_80 = (count, frame_path, img)

        if idx % 200 == 0:
            print(f"Processed {idx}/{len(frame_paths)}")

    # Save outputs
    if best_over_30 is not None:
        out_30 = "output/alerts/frame_count_over_30.jpg"
        cv2.imwrite(out_30, best_over_30[2])
        print(f"Saved >30 frame: {out_30} from {os.path.basename(best_over_30[1])}, count={best_over_30[0]}")
    else:
        print("No frame detected with count > 30")

    if best_over_80 is not None:
        out_80 = "output/alerts/frame_over_80_percent.jpg"
        cv2.imwrite(out_80, best_over_80[2])
        print(f"Saved >=80% frame: {out_80} from {os.path.basename(best_over_80[1])}, count={best_over_80[0]}")
    else:
        print("No frame detected with count >= 24 (80% of 30)")

    if global_best is not None:
        out_top = "output/alerts/frame_max_detected.jpg"
        top = global_best[2].copy()
        put_alert_text(top, f"MAX DETECTED COUNT: {global_best[0]}", (255, 255, 0))
        cv2.imwrite(out_top, top)
        print(f"Saved max-count frame: {out_top} from {os.path.basename(global_best[1])}, count={global_best[0]}")


if __name__ == "__main__":
    main()
