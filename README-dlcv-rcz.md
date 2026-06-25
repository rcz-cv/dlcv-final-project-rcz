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

### run_identity_metrics.py
pip install scikit-learn

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
    --gt_eval \
    --sequence_dir=./videos/MOT16-09 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/temporary/data \
    --detector=mot16 \
    --reid=mars \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display
```

### Videos to Google Drive

```
# store to archive
tar --no-xattrs -czf dlcv-final-project-videos.tar.gz videos/
shasum -a 256 dlcv-final-project-videos.tar.gz
# 56b276c25370eec2eb198b6273d16939fffac04fa5c9fd7ff8117215171b035e  dlcv-final-project-videos.tar.gz

# https://drive.google.com/file/d/1ujjjDlQZ6eEfdfWqJx-L_pgbJkSqRkU8/view?usp=sharing

# restore from archive
tar xzf dlcv-final-project-videos.tar.gz
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
    --display

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

### Run Tracker with mot16-mars baseline

```
python run_tracker.py \
    --sequence_dir=./videos/MOT16-09 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/mot16-mars/data \
    --detector=mot16 \
    --reid=mars \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display

bash scripts/run_eval.sh --tracker mot16-mars --detector mot16 --reid mars --min_confidence=0.3

```

### Run Tracker with osnet

```
python run_tracker.py \
    --sequence_dir=./videos/MOT16-09 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/yolo26m-osnet_x1_0/data \
    --detector=yolo26m \
    --reid=osnet_x1_0 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display

bash run_eval.sh --tracker yolo26m-osnet_x1_0 --detector yolov5mu --reid osnet_x1_0 --min_confidence 0.35

```

### Created a complete execution/compare script

e.g.

```
bash run_eval.sh --tracker yolo26m-osnet_x1_0 --detector yolo26m --reid osnet_x1_0
```

Just update HOTA:
```
python update_hota.py --metadata_dir ./eval/trackers/DLCV/DLCV-train/yolo26m-osnet_x1_0/data
```
 
### Run Identity with osnet

```
python run_identity.py \
    --sequence_dir=./videos/MOT16-09 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/BR-yolo26s-osnet_x1_0/data \
    --detector=yolo26s \
    --reid=osnet_x1_0 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display

bash run_eval.sh --tracker BR-yolo26s-osnet_x1_0 --detector yolo26s --reid osnet_x1_0

```
python run_identity.py \
    --sequence_dir=./videos/MOT16-09 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/BR-yolo26s-osnet_x0_75/data \
    --detector=yolo26s \
    --reid=osnet_x0_75 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display

bash run_eval.sh --tracker BR-yolo26s-osnet_x0_75 --detector yolo26s --reid osnet_x0_75

### Evaluate Identity

```
python run_identity_metrics.py \
  --sequence_dir videos/MOT16-09 \
  --output_dir eval/metrics/identity \
  --reid osnet_x0_75 \
  --knn_k 1 \
  --identity_max_distance 0.22

bash scripts/run_eval_id.sh \
    --tracker osnet_x0_75-k4-imax14 \
    --reid osnet_x0_75 \
    --knn_k 4 \
    --identity_max_distance 0.14

```

### Run Tracker with gt_eval

```
python run_tracker.py \
    --gt_eval \
    --sequence_dir=./videos/MOT16-11 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/yolo26m-osnet_x1_0-eval/data \
    --detector=yolo26m \
    --reid=osnet_x1_0 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display

bash run_eval.sh \
    --tracker yolo26m-osnet_x1_0-eval \
    --detector yolo26s \
    --reid osnet_x1_0 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --gt_eval
```

### Run baseline Tracker with gt_eval

```
python run_tracker.py \
    --gt_eval \
    --sequence_dir=./videos/MOT16-11 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/mot16-mars-gteval/data \
    --detector=mot16 \
    --reid=mars \
    --min_confidence=0.3 \
    --display

bash run_eval.sh \
    --tracker mot16-mars-gteval \
    --detector mot16 \
    --reid mars \
    --min_confidence=0.3 \
    --gt_eval
```


### debugging

python run_tracker.py \
    --sequence_dir=./videos/PETS09-S2L1 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/mot16-mars-pets/data \
    --detector=mot16 \
    --reid=mars \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display

### Report experiments

Section 3.1 Person Detector Evaluation

```
bash scripts/run_eval.sh \
    --tracker mot16-mars-rep \
    --detector mot16 \
    --reid mars \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --gt_eval

bash scripts/run_eval.sh \
    --tracker yolov5mu-osnet_x0_5-rep \
    --detector yolov5mu \
    --reid osnet_x0_5 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --gt_eval

bash scripts/run_eval.sh \
    --tracker yolov5lu-osnet_x0_75-rep \
    --detector yolov5lu \
    --reid osnet_x0_75 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --gt_eval

bash scripts/run_eval.sh \
    --tracker yolo26s-osnet_x1_0-rep \
    --detector yolo26s \
    --reid osnet_x1_0 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --gt_eval

bash scripts/run_eval.sh \
    --tracker yolo26m-osnet_x1_0-rep \
    --detector yolo26m \
    --reid osnet_x1_0 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --gt_eval

bash scripts/run_eval.sh \
    --tracker yolo26l-osnet_ain_x1_0-rep \
    --detector yolo26l \
    --reid osnet_ain_x1_0 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --gt_eval

```
