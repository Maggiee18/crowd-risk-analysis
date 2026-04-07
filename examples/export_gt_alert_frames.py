import os

import cv2
import numpy as np
import scipy.io as sio


def annotate(img, text, color, y):
    cv2.rectangle(img, (10, y - 30), (630, y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    gt = sio.loadmat("data/mall_gt.mat")
    counts = np.array(gt["count"]).reshape(-1)
    frames = sorted(
        [
            os.path.join("data/frames/frames", f)
            for f in os.listdir("data/frames/frames")
            if f.lower().endswith(".jpg")
        ]
    )

    if len(frames) != len(counts):
        raise RuntimeError(f"Frame/GT mismatch: {len(frames)} frames vs {len(counts)} gt counts")

    os.makedirs("output/alerts", exist_ok=True)

    idx_over_30 = int(np.where(counts > 30)[0][0])
    idx_over_80 = int(np.where(counts >= 24)[0][0])

    # > 30 frame
    img30 = cv2.imread(frames[idx_over_30])
    annotate(img30, f"GT COUNT: {int(counts[idx_over_30])} (> 30)", (0, 0, 255), 40)
    annotate(img30, "CRITICAL: Capacity exceeded", (0, 0, 255), 78)
    out30 = "output/alerts/gt_count_over_30.jpg"
    cv2.imwrite(out30, img30)

    # >= 80% of 30 frame
    img80 = cv2.imread(frames[idx_over_80])
    util = (float(counts[idx_over_80]) / 30.0) * 100.0
    annotate(img80, f"GT COUNT: {int(counts[idx_over_80])} (>= 24)", (0, 165, 255), 40)
    annotate(img80, f"ALERT: {util:.1f}% of capacity (30)", (0, 165, 255), 78)
    out80 = "output/alerts/gt_over_80_percent.jpg"
    cv2.imwrite(out80, img80)

    print(f"Saved: {out30} from frame index {idx_over_30 + 1}")
    print(f"Saved: {out80} from frame index {idx_over_80 + 1}")


if __name__ == "__main__":
    main()
