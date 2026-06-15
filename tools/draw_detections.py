#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2

def intersection_over_smaller(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller > 0 else 0



def read_mot_detections(det_txt_path, frame_id):
    """
    Reads MOT-format detections:

    frame, id, x, y, w, h, score, -1, -1, -1
    """
    detections = []

    with open(det_txt_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 7:
                continue

            row_frame = int(float(parts[0]))

            if row_frame != frame_id:
                continue

            detections.append(
                {
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "w": float(parts[4]),
                    "h": float(parts[5]),
                    "score": float(parts[6]),
                }
            )

    return detections


def draw_detections(
    image,
    detections,
    min_score=0.0,
    min_height=0.0,
    min_area=0.0,
):
    """
    Draw detections on an image.

    Green: passes filters
    Red: filtered out
    """
    image = image.copy()

    boxes = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    for box in boxes:
        duplicate = False

        for kept_box in kept:
            overlap = intersection_over_smaller(
                (box["x"], box["y"], box["w"], box["h"]),
                (kept_box["x"], kept_box["y"], kept_box["w"], kept_box["h"]),
            )

            if overlap > 0.75:
                duplicate = True
                break

        if not duplicate:
            kept.append(box)

    kept_count = 0

    for det in detections:
        x = det["x"]
        y = det["y"]
        w = det["w"]
        h = det["h"]
        score = det["score"]

        area = w * h

        keep = (
            score >= min_score
            and h >= min_height
            and area >= min_area
            and det in kept
        )

        if keep:
            kept_count += 1
            color = (0, 255, 0)
            thickness = 2
        else:
            color = (0, 0, 255)
            thickness = 1

        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        label = f"{score:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    print(
        f"Frame: {len(detections)} detections, "
        f"{kept_count} pass filters"
    )

    return image


def find_frame_image(image_dir, frame_id):
    """
    Searches for common MOT image naming conventions.
    """
    candidates = [
        image_dir / f"{frame_id:06d}.jpg",
        image_dir / f"{frame_id:06d}.png",
        image_dir / f"{frame_id:08d}.jpg",
        image_dir / f"{frame_id:08d}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not locate image for frame {frame_id}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing frame images",
    )

    parser.add_argument(
        "--detections",
        required=True,
        help="Path to MOT-format det.txt",
    )

    parser.add_argument(
        "--frame",
        type=int,
        required=True,
        help="Frame number to display",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--min-height",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--save",
        help="Optional output image path",
    )

    args = parser.parse_args()

    image_dir = Path(args.images)
    det_path = Path(args.detections)

    image_path = find_frame_image(
        image_dir,
        args.frame,
    )

    print(f"Image: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Failed to load {image_path}"
        )

    detections = read_mot_detections(
        det_path,
        args.frame,
    )

    image = draw_detections(
        image,
        detections,
        min_score=args.min_score,
        min_height=args.min_height,
        min_area=args.min_area,
    )

    if args.save:
        cv2.imwrite(args.save, image)
        print(f"Saved to {args.save}")

    cv2.imshow(
        f"Frame {args.frame}",
        image,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
