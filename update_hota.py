from pathlib import Path
import csv
import yaml
import argparse


def safe_divide(numerator, denominator):
    """Return numerator / denominator, or 0.0 if denominator is zero."""
    return numerator / denominator if denominator else 0.0


def compute_detector_metrics(tp, fp, fn):
    """Compute precision, recall, and F1 from detector TP/FP/FN counts."""
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return precision, recall, f1


def compute_combined_det_stats(metadata):
    """
    Compute combined detector statistics, averaging over all frames in all sequences.
    """
    sequences = metadata.get("sequences", {})

    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_thresholds = set()
    found_stats = False

    for seq_name, seq_data in sequences.items():
        det_stats = seq_data.get("det_stats")
        if not det_stats:
            continue

        required_keys = {"tp", "fp", "fn"}
        missing_keys = required_keys - set(det_stats.keys())
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(
                f"Sequence {seq_name} det_stats is missing required key(s): {missing}"
            )

        total_tp += int(det_stats["tp"])
        total_fp += int(det_stats["fp"])
        total_fn += int(det_stats["fn"])
        found_stats = True

        if "iou_threshold" in det_stats:
            iou_thresholds.add(float(det_stats["iou_threshold"]))

    if not found_stats:
        return None

    if len(iou_thresholds) > 1:
        thresholds = ", ".join(str(x) for x in sorted(iou_thresholds))
        raise ValueError(
            f"Cannot compute combined det_stats with mixed IoU thresholds: {thresholds}"
        )

    precision, recall, f1 = compute_detector_metrics(total_tp, total_fp, total_fn)

    combined_det_stats = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if iou_thresholds:
        combined_det_stats = {
            "iou_threshold": next(iter(iou_thresholds)),
            **combined_det_stats,
        }

    return combined_det_stats


def update_hota(metadata_dir):
    """
    Update sequence HOTA values in metadata from MOT evaluation, and create
    a combined det_stats based on total TP/FP/FN counts
    """
    metadata_path = Path(metadata_dir) / "metadata.yaml"
    hota_csv_path = metadata_path.parent.with_name("pedestrian_detailed.csv")

    with metadata_path.open("r") as f:
        metadata = yaml.safe_load(f) or {}

    metadata.setdefault("sequences", {})
    metadata.setdefault("combined", {})

    with hota_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seq_name = row["seq"]
            hota_value = float(row["HOTA___AUC"]) * 100.0  # 0..1 => 0..100%

            if seq_name == "COMBINED":
                metadata["combined"]["hota"] = hota_value
            else:
                metadata["sequences"].setdefault(seq_name, {})
                metadata["sequences"][seq_name]["hota"] = hota_value

    combined_det_stats = compute_combined_det_stats(metadata)
    if combined_det_stats is not None:
        metadata["combined"]["det_stats"] = combined_det_stats

    with metadata_path.open("w") as f:
        yaml.safe_dump(
            metadata,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Update sequence HOTA and detector stats."
        )
    )
    parser.add_argument(
        "--metadata_dir", help="Path to metadata directory",
        type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    update_hota(args.metadata_dir)
