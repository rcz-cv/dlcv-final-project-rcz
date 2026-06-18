# DLCV Week 6 - Final Project

**`README-dlcv-rcz.md`**

## Installation & Preparation

### Python Requirements

```
python3.12 -m venv .venv

# Change PS1 prompt on line ~70 to:
# PS1="(.venv) ${PS1:-}"
vi .venv/bin/activate

source .venv/bin/activate

# Confirm version:
% python --version
Python 3.12.12

pip install --upgrade pip 

### x86/CUDA
pip install numpy opencv-python scipy tensorflow tf-slim tf-keras torch

### AppleSilicon
pip install tensorflow-macos tensorflow-metal numpy opencv-python scipy tf-slim tf-keras torch

### Ultralytics
pip install ultralytics

```

### Install resources

As of 20260612 the resource files are available on Google Drive [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)

Move the unzipped network folder into resources. The detections will be created by us below.

```
resources/
    detections/
        mars-small128/
    networks/
        mars-small128/
            mars-small*
```

### Smoke Test

The following example runs the tracker against one of the MOT16 sequences:

```
python run_tracker.py \
    --sequence_dir=./videos/MOT16-09/ \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
```

### Create Detections

The following example generates person re-identification features from standard MOT challenge
detections. We apply it to our new videos:

```
python Closet/tools/generate_detections.py \
    --model resources/networks/mars-small128/mars-small128.pb \
    --detection_dir detections/yolov5mu.pt-mars/
    --mot_dir videos \
    --output_dir resources/detections/mars-small128/DLCV
```

#### After creating detections, run the tracker against them:

```
for VIDEO in KITTI-17 MOT16-09 MOT16-11 PETS09-S2L1 TUD-Campus TUD-Stadtmitte; do
    python deep_sort_app.py \
        --sequence_dir="videos/${VIDEO}" \
        --detection_file="resources/detections/mars-small128/DLCV/${VIDEO}.npy" \
        --output_file="eval/trackers/DLCV/DLCV-train/deep_sort_baseline/data/${VIDEO}.txt"
done
```

## Evaluating the tracker

bash track_eval.sh

## show results

(export VIDEO="TUD-Stadtmitte";
    python show_results.py \
        --sequence_dir="videos/${VIDEO}" \
        --detection_file="resources/detections/mars-small128/DLCV/${VIDEO}.npy" \
        --result_file="eval/trackers/DLCV/DLCV-train/deep_sort_baseline/data/${VIDEO}.txt"
)

### Run online tracker

python run_tracker.py \
    --sequence_dir=./videos/MOT16-09 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True

### Run mot challenge

python run_motchallenge.py \
    --min_confidence=0.3 \
    --nn_budget=100

### Create offline REID from online detections

```
python Closet/tools/generate_detections.py \
    --model resources/networks/mars-small128/mars-small128.pb \
    --detection_dir detections/yolov5mu.pt-mars \
    --mot_dir videos \
    --output_dir resources/detections/mars-small128/yolov5mu.pt-mars-off
```

#### After creating detections, run the tracker against them:

```
for VIDEO in KITTI-17 MOT16-09 MOT16-11 PETS09-S2L1 TUD-Campus TUD-Stadtmitte; do
    python Closet/deep_sort_app.py \
        --sequence_dir="videos/${VIDEO}" \
        --detection_file="resources/detections/mars-small128/yolov5mu.pt-mars-off/${VIDEO}.npy" \
        --output_file="eval/trackers/DLCV/DLCV-train/yolov5mu.pt-mars-off/data/${VIDEO}.txt"
done
```

### Created a complete execution/compare script

e.g.

```
bash run_eval.sh --tracker yolov5mu-mars-mc35 --detector yolov5mu --min_confidence 0.35
```
