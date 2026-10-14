# AutoAgent0

SparseDriveV2 planning + BridgeDrive detection_ft + Claude Opus 5.

## Setup

```bash
git lfs install
git clone https://github.com/jiagengliu02/AutoAgent0.git
cd AutoAgent0
git lfs pull
```

CARLA 0.9.15, Fail2Drive (`start_carla.sh`, `fail2drive_split/`), CUDA 11.8, FFmpeg, and Vulkan tools.

| Environment | Dependencies |
| --- | --- |
| Evaluator | Python 3.10, Fail2Drive, OmegaConf, PyYAML, NumPy, SciPy, Shapely, OpenCV, Pillow, imageio, imageio-ffmpeg |
| SparseDriveV2 | Python 3.8, PyTorch 2.0.1+cu118, [requirements](third_party/SparseDriveV2/requirement.txt) |
| BridgeDrive | Python 3.10, PyTorch 2.5.0, torchvision 0.20.0, [requirements](third_party/LEAD/requirements.txt), scikit-learn 1.7.0, joblib 1.5.1 |

```bash
export FAIL2DRIVE_ROOT=/path/to/fail2drive
export CARLA_ROOT=/path/to/carla
export EVALUATOR_PYTHON_BIN=/path/to/fail2drive-env/bin/python
export SPARSEDRIVEV2_PYTHON_BIN=/path/to/sparsedrivev2-env/bin/python
export BRIDGEDRIVE_PYTHON_BIN=/path/to/lead-env/bin/python
export CUDA_HOME=/path/to/cuda-11.8

(
  cd third_party/SparseDriveV2/projects/mmdet3d_plugin/ops
  FORCE_CUDA=1 "$SPARSEDRIVEV2_PYTHON_BIN" setup.py build_ext --inplace
)
```

## Checkpoints

| Model | Path |
| --- | --- |
| SparseDriveV2 | `checkpoints/sparsedrivev2/sparsedrive_small_b2d_stage2.pth` |
| BridgeDrive | `checkpoints/bridgedrive/model_BridgeDrive_m2_k60_0030.pth` |
| Pedestrian detection_ft | `checkpoints/detection_ft/pedestrian/` |
| Static detection_ft | `checkpoints/detection_ft/static/` |

```bash
sha256sum -c checkpoints/SHA256SUMS
```

## Evaluation

Configuration: [configs/pipeline.yaml](configs/pipeline.yaml). Routes: [200 episodes](configs/episodes/claude_opus5_full200.json). API model: `claude-opus-5`.

```bash
read -r -s -p 'Anthropic API key: ' ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY

bash scripts/run.sh --dry-run

# Single route.
GPUS_SMI="0" MAX_ROUTES=1 bash scripts/run.sh

# All routes, one CARLA instance per GPU.
GPUS_SMI="0 1" bash scripts/run.sh
```

| Option | Default |
| --- | --- |
| `GPUS_SMI` | `0` (nvidia-smi indices) |
| `MAX_ROUTES` | `0` (all) |
| `OUT_DIR` | `output/run_<timestamp>/` (new directory) |
| `BASE_PORT` / `TM_BASE` | `5600` / `5700` |
| `AGENTICDRIVING_SIM_TIME_LIMIT_SEC` | `180` (diagnostic cutoff; `0` for full routes) |

Results: `res/`. Predictions, videos, and VLM records: `episodes/`. Logs: `logs/`.
