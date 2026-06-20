from pathlib import Path
import csv
import yaml
import argparse


def update_hota(metadata_dir):
    """
    Update sequence HOTA values in metadata from MOT evaluation.
    """
    metadata_path = Path(metadata_dir) / "metadata.yaml"
    hota_csv_path = metadata_path.parent.with_name("pedestrian_detailed.csv")

    with metadata_path.open("r") as f:
        metadata = yaml.safe_load(f) or {}

    metadata.setdefault("sequences", {})

    with hota_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seq_name = row["seq"]
            hota_value = float(row["HOTA___AUC"])
            hota_value *= 100.0                                     # 0..1 => 0..100%

            if seq_name == "COMBINED":
                metadata.setdefault("combined", {})
                metadata["combined"]["hota"] = hota_value
            else:
                metadata["sequences"].setdefault(seq_name, {})
                metadata["sequences"][seq_name]["hota"] = hota_value

    with metadata_path.open("w") as f:
        yaml.safe_dump(
            metadata,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    return metadata


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Update sequence HOTA values in metadata from MOT evaluation."
    )
    parser.add_argument(
        "--metadata_dir", help="Path to metadata directory",
        type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    update_hota(args.metadata_dir)
