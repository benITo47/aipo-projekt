# One-command flows. Run `make help` for the list.
# All commands shell out to the four top-level scripts:
#   dataset.py · train.py · eval.py · demo.py
.DEFAULT_GOAL := help
SHELL := /bin/bash
PYTHON ?= python3
VENV ?= .venv
PY := source $(VENV)/bin/activate && python

# Demo source — override on the command line: `make demo SOURCE=path/to/clip.mp4`
SOURCE ?= 0

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | awk -F':.*?## ' '{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------- install ----------

install-mac:  ## Mac dev install (CPU torch via pip, inference only)
	$(PYTHON) -m venv $(VENV)
	source $(VENV)/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"

install-cuda:  ## RTX box install (CUDA 12.4 torch + training extras)
	$(PYTHON) -m venv $(VENV)
	source $(VENV)/bin/activate && pip install --upgrade pip && \
	  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
	  pip install -e ".[gpu,dev]"

# ---------- data ----------

data-soccernet:  ## Download SoccerNet (needs SOCCERNET_PASSWORD)
	$(PY) dataset.py soccernet --out data/raw/soccernet

data-roboflow:  ## Download Roboflow players dataset (needs ROBOFLOW_API_KEY)
	$(PY) dataset.py roboflow --out data/raw/roboflow

data-pitch:  ## Download Roboflow pitch-keypoints dataset (needs ROBOFLOW_API_KEY)
	$(PY) dataset.py pitch --out data/raw/pitch

data-merge:  ## Merge whatever player datasets exist into configs/combined.yaml
	$(PY) dataset.py merge

data:  ## Roboflow players + pitch + merge (no NDA needed)
	$(PY) dataset.py all

data-full: data-soccernet data-roboflow data-pitch data-merge  ## SoccerNet + Roboflow + pitch

# ---------- training (RTX box) ----------

train:  ## Fine-tune detector (YOLO26) → models/checkpoints/best.pt
	$(PY) train.py detector

train-pitch:  ## Fine-tune pitch model (YOLO26n-pose) → models/checkpoints/pitch.pt
	$(PY) train.py pitch

train-all:  ## Train both models end-to-end
	$(PY) train.py all

train-export:  ## Export detector .pt → ONNX for fast Mac inference
	$(PY) train.py export --weights models/checkpoints/best.pt

# ---------- eval ----------

eval-detector:  ## mAP / PR on detector val
	$(PY) eval.py detector --weights models/checkpoints/best.pt

eval-pitch:  ## Box + keypoint mAP for pitch model
	$(PY) eval.py pitch --weights models/checkpoints/pitch.pt

eval-homography:  ## Per-frame homography stability on a video (SOURCE=clip.mp4)
	$(PY) eval.py homography --source $(SOURCE)

eval-all:  ## Detector + pitch eval (skips homography — needs a video)
	$(PY) eval.py all

# ---------- inference / demo ----------

demo:  ## Run full pipeline. `make demo SOURCE=clip.mp4` (or SOURCE=0 for webcam)
	$(PY) demo.py --source $(SOURCE)

demo-pretrained:  ## Same, but with stock yolo26n.pt + COCO classes — no fine-tuning needed
	$(PY) demo.py --source $(SOURCE) --weights /dev/null --no-pitch

demo-record:  ## Headless render to file: `make demo-record SOURCE=clip.mp4 OUT=out.mp4`
	$(PY) demo.py --source $(SOURCE) --output $(OUT) --no-show

# ---------- dev ----------

smoke:
	$(PY) scripts/smoke_test.py
lint:
	$(PY) -m ruff check src dataset.py train.py eval.py demo.py
fmt:
	$(PY) -m ruff format src dataset.py train.py eval.py demo.py
test:
	$(PY) -m pytest -q

.PHONY: help install-mac install-cuda \
        data data-full data-soccernet data-roboflow data-pitch data-merge \
        train train-pitch train-all train-export \
        eval-detector eval-pitch eval-homography eval-all \
        demo demo-pretrained demo-record \
        smoke lint fmt test
