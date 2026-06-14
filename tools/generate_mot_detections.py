"""
generate_mot_detections.py

Preprocessor to create new detections based on the specified detector model.
"""

from pathlib import Path
import sys
import os
import errno
import argparse
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import detectors

import os
import cv2
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\..*",
)

def read_seqinfo(sequence_dir):
    """
    Read a MOT-format seqinfo.ini file
    """
    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if not os.path.exists(info_filename):
        return None

    info = {}
    with open(info_filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("[") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()

    return info

def write_mot_row(f, frame_idx, track_id, x, y, w, h, confidence):
    """
    Write a MOT-format detection item
    """
    f.write(
        f"{frame_idx},{track_id},"
        f"{x:.3f},{y:.3f},{w:.3f},{h:.3f},"
        f"{confidence:.6f},-1,-1,-1\n"
    )

def generate_mot_detections(model_name, video_dir, output_dir, yolov5_dir):
    """Generate MOT-format detections from the images in the video datasets.

    Parameters
    ----------
    modelname
        Object with detect(image) method, or callable returning detections.
        Each detection should have x, y, w, h, confidence attributes.
    video_dir : str
        Path to the videos directory.
    output_dir : str
        Path to output detections directory.
    """
    if not os.path.isdir(video_dir):
        raise ValueError(f"Video directory does not exist: {video_dir}")

    try:
        output_dir = os.path.join(output_dir, model_name)
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    detector = detectors.create_detector(
        model_name,
        yolov5_dir=yolov5_dir
    )

    sequences = sorted(
        s for s in os.listdir(video_dir)
        if not s.startswith(".") and os.path.isdir(os.path.join(video_dir, s))
    )

    for sequence in sequences:
        print(f"Processing {sequence}")

        sequence_dir = os.path.join(video_dir, sequence)
        image_dir = os.path.join(sequence_dir, "img1")

        info_dict = read_seqinfo(sequence_dir)
        if not info_dict:
            print(f"WARNING: could not find/read seqinfo.ini for {sequence}")
            continue

        if not os.path.isdir(image_dir):
            print(f"WARNING: could not find image directory: {image_dir}")
            continue

        seq_length = int(info_dict["seqLength"])

        sequence_output_dir = os.path.join(output_dir, sequence, "det")
        os.makedirs(sequence_output_dir, exist_ok=True)

        output_file = os.path.join(sequence_output_dir, "det.txt")

        with open(output_file, "w") as f:
            for frame_idx in range(1, seq_length + 1):
                image_filename = os.path.join(image_dir, f"{frame_idx:06d}.jpg")
                bgr_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)

                if bgr_image is None:
                    print(f"WARNING: could not read image: {image_filename}")
                    continue

                if hasattr(detector, "detect"):
                    detections = detector.detect(bgr_image)
                else:
                    detections = detector(bgr_image)

                for det in detections:
                    write_mot_row(
                        f,
                        frame_idx,
                        -1,
                        det.x,
                        det.y,
                        det.w,
                        det.h,
                        det.confidence,
                    )

        print(f"  wrote {output_file}")

def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Detections generator")
    parser.add_argument(
        "--model", help="Name of the detector model to use.",
        required=True)
    parser.add_argument(
        "--video_dir", help="Input directory.", default="videos",
        required=True)
    parser.add_argument(
        "--output_dir", help="Output directory.", default="detections",
        required=True)
    parser.add_argument(
        "--yolov5_dir", help="Location of ylov5 repo.", default="external/yolov5",
        required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    generate_mot_detections(args.model, args.video_dir, args.output_dir, args.yolov5_dir)

if __name__ == "__main__":
    main()
