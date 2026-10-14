#!/usr/bin/env bash
# SparseDriveV2 + BridgeDrive detection_ft + Claude Opus 5.
set -euo pipefail
set +x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
case "${1:-}" in
  --help|-h)
    echo 'Usage: bash scripts/run.sh [--dry-run]'
    echo 'Required: FAIL2DRIVE_ROOT, CARLA_ROOT, SPARSEDRIVEV2_PYTHON_BIN, BRIDGEDRIVE_PYTHON_BIN, EVALUATOR_PYTHON_BIN'
    echo 'For execution: ANTHROPIC_API_KEY. Options: GPUS_SMI (0), MAX_ROUTES (0=all), OUT_DIR, BASE_PORT (5600), TM_BASE (5700).'
    exit 0 ;;
  ''|--dry-run) ;;
  *) echo 'Unknown argument; use --help' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Too many arguments' >&2; exit 2; }
export AGENTICDRIVING_ROOT="$RELEASE_ROOT"
: "${FAIL2DRIVE_ROOT:?Set FAIL2DRIVE_ROOT to the Fail2Drive checkout}"
: "${CARLA_ROOT:?Set CARLA_ROOT to the CARLA simulator installation}"
export FAIL2DRIVE_ROOT="$(cd "$FAIL2DRIVE_ROOT" && pwd)"
export CARLA_ROOT="$(cd "$CARLA_ROOT" && pwd)"
export LEADERBOARD_ROOT="$FAIL2DRIVE_ROOT/leaderboard"
export SCENARIO_RUNNER_ROOT="$FAIL2DRIVE_ROOT/scenario_runner"
export PYTHONDONTWRITEBYTECODE=1
export BRIDGEDRIVE_FINETUNED_DETECTION=1
export BRIDGEDRIVE_DETECTION_CODE="$AGENTICDRIVING_ROOT/detection"
export PYTHONPATH="$AGENTICDRIVING_ROOT:$CARLA_ROOT/PythonAPI/carla:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
: "${SPARSEDRIVEV2_PYTHON_BIN:?Set the SparseDriveV2 Python executable}"
: "${BRIDGEDRIVE_PYTHON_BIN:?Set the LEAD Python executable}"
: "${EVALUATOR_PYTHON_BIN:?Set the Fail2Drive Python executable}"
for name in SPARSEDRIVEV2_PYTHON_BIN BRIDGEDRIVE_PYTHON_BIN EVALUATOR_PYTHON_BIN; do
  resolved="$(command -v "${!name}")" || { echo "Executable not found: $name" >&2; exit 2; }
  export "$name=$(realpath "$resolved")"
done
export CONFIG="$RELEASE_ROOT/configs/pipeline.yaml"
export EPISODES_JSON="${EPISODES_JSON:-$RELEASE_ROOT/configs/episodes/claude_opus5_full200.json}"
export AGENTICDRIVING_SIM_TIME_LIMIT_SEC="${AGENTICDRIVING_SIM_TIME_LIMIT_SEC:-180}"
export OUT_DIR="${OUT_DIR:-$RELEASE_ROOT/output/run_$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="$(realpath -m "$OUT_DIR")"
export EPISODES_JSON="$(realpath "$EPISODES_JSON")"
if [[ -n "${ROUTES_DIR:-}" ]]; then export ROUTES_DIR="$(realpath "$ROUTES_DIR")"; fi
export MAX_ROUTES="${MAX_ROUTES:-0}"
export EVALUATOR_PYTHON="$EVALUATOR_PYTHON_BIN"
export GPUS_SMI="${GPUS_SMI:-0}"
export BASE_PORT="${BASE_PORT:-5600}"
export TM_BASE="${TM_BASE:-5700}"
"$EVALUATOR_PYTHON_BIN" "$SCRIPT_DIR/runtime.py" prepare "${1:---prepare}"
[[ "${1:-}" != --dry-run ]] || exit 0
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before execution}"
export ANTHROPIC_API_KEY
export CONFIG="$OUT_DIR/config.yaml"
export EPISODES_JSON="$OUT_DIR/episodes_requested.json"
export MAX_ROUTES=0
exec bash "$SCRIPT_DIR/run_routes.sh"
