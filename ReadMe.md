# DLCV Week 6 - Final Project

**Improvements to the DeepSORT implementation**  
**By Richard Zulch**  
**25-June-2026**  

**`ReadMe.md`**  

## Installation & Preparation

**NOTE:**

* For Google Colab, please clone the repo and open `final-project.ipynb` there.
* For use on the local computer, please follow the instructions below.

**First:**

1. Clone the repo into a local `project-folder`
2. **cd** `project-folder`


### 1. Python Requirements

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
pip install numpy opencv-python scipy tensorflow tf-slim tf-keras torch ultralytics scikit-learn

### AppleSilicon
pip install tensorflow-macos tensorflow-metal numpy opencv-python scipy tf-slim tf-keras torch ultralytics scikit-learn

```


### 2. Install mars model

DeepSORT's baseline **mars** model is available on Google Drive [here](https://drive.google.com/file/d/1mye0DtoestFP9GqJp3ylSBVAXzFr0Mr_/view?usp=sharing)

Steps to install project `resources` in the project:

1. Visit the link above with a browser
2. Download the file, which will be named `mars-small128.tar.gz`
4. Put the `mars-small128.tar.gz` file into your project folder
4. It will be used by the setup_resources.sh script in step 4 below.

The resulting folder structure will look like this:

To verify the mars archive:

```
shasum -a 256 mars-small128.tar.gz
# 55077730c5c1f806e45048ce8c77eba4f9e950566f8d46ff4a7ce00cb0c70563  mars-small128.tar.gz
```


### 3. Install videos

The selected MOT videos are available on Google Drive [here](https://drive.google.com/file/d/1ujjjDlQZ6eEfdfWqJx-L_pgbJkSqRkU8/view?usp=sharing)

Steps to install `videos` in the project:

1. Visit the link above with a browser
2. Download the file, which will be named `dlcv-final-project-videos.tar.gz`
4. Put the `dlcv-final-project-videos.tar.gz` file into your project folder
5. Execute `tar xzf dlcv-final-project-videos.tar.gz` to create and populate the `videos` folder

To verify the videos archive:

```
shasum -a 256 dlcv-final-project-videos.tar.gz
# 56b276c25370eec2eb198b6273d16939fffac04fa5c9fd7ff8117215171b035e  dlcv-final-project-videos.tar.gz
```


### 4. Prepare the project

Execute the following bash scripts to create the necessary folder structure within the project that refers to the downloaded files and videos.

```
bash scripts/setup_eval.sh
bash scripts/setup_videos.sh
bash scripts/setup_resources.sh
bash scripts/setup_osnet.sh
```

If you wish to save disk space now, discard the downloaded `resources-*.zip` and `dlcv-final-project-videos.tar.gz` files. 


### 5. Smoke Test

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

### 6. Execution examples

#### Experiment-1: Run Tracker with osnet with display

```
python run_tracker.py \
    --sequence_dir=./videos/MOT16-09 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/experiment-1/data \
    --detector=yolo26m \
    --reid=osnet_x1_0 \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display
```

#### Experiment-2: Run Tracker with osnet on all videos, HOTA output

```
bash scripts/run_eval.sh \
	--tracker experiment-2 \
	--detector yolov5mu \
	--reid osnet_x1_0 \
	--min_confidence 0.35
```

#### Experiment-3: Run tracker with segmentation model on KITTI-17

```
python run_tracker.py \
    --sequence_dir=./videos/KITTI-17 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/experiment-3/data \
    --detector yolo26s-seg \
    --reid mars \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --max_age=30 \
    --max_cosine_distance=0.19 \
    --display
```

#### Experiment-4: Evaluate yolo26s and osnet_x0_75 ReID on groundtruth for video MOT16-11

```
python run_tracker.py \
    --sequence_dir=./videos/MOT16-11 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/experiment-4/data \
    --detector yolo26s \
    --reid osnet_x0_75 \
    --min_confidence=0.70 \
    --nn_budget=100 \
    --max_age=30 \
    --max_cosine_distance=0.19 \
    --gt_eval
```

#### Experiment-5: Run Identity with mars and display

```
python run_identity.py \
    --sequence_dir=./videos/MOT16-09 \
    --output_dir=./eval/trackers/DLCV/DLCV-train/experiment-5/data \
    --detector=yolo26s \
    --reid=mars \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display
```

#### Experiment-6: Run Identity with osnet_x1_0 on all videos

```
bash scripts/run_eval.sh \
	--tracker experiment-6 \
	--detector yolov5mu \
	--reid osnet_x1_0 \
	--min_confidence 0.35
```

#### Experiment-7: Run Identity metrics on a video

```
python run_identity_metrics.py \
	--sequence_dir videos/MOT16-11 \
	--output_dir eval/metrics/identity \
	--reid osnet_x0_75 \
	--knn_k 9 \
	--identity_max_distance 0.27
```
