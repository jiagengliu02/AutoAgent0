#!/usr/bin/env bash
# Run the release pipeline in isolated CARLA/evaluator slots.
set -uo pipefail

HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
# Anchor to a guaranteed-existing dir: if launched from a deleted/overwritten CWD,
# an empty sys.path entry makes getcwd() fail and every child python aborts at import.
cd "$HERE"
: "${AGENTICDRIVING_ROOT:?Run scripts/run.sh}"

# ----------------------------- configuration --------------------------------
GPUS_SMI="${GPUS_SMI:-0}"          # nvidia-smi indices; one CARLA + one shard each
BASE_PORT="${BASE_PORT:-5600}"             # shard i -> rpc BASE_PORT + i*5 (start_carla.sh's rule)
TM_BASE="${TM_BASE:-5700}"                 # shard i -> tm  TM_BASE  + i*5
QUALITY="${QUALITY:-Low}"
CARLA_ROOT="${CARLA_ROOT:?Set CARLA_ROOT}"
EVALUATOR_PYTHON="${EVALUATOR_PYTHON_BIN:?Set evaluator Python}"
EPISODES_JSON="${EPISODES_JSON:-$AGENTICDRIVING_ROOT/configs/episodes/claude_opus5_full200.json}"
ROUTES_DIR="${ROUTES_DIR:-$FAIL2DRIVE_ROOT/fail2drive_split}"
CONFIG="${CONFIG:-$AGENTICDRIVING_ROOT/configs/pipeline.yaml}"
AGENT_FILE="${AGENT_FILE:-$AGENTICDRIVING_ROOT/adapters/carla_agent/agenticdriving_agent.py}"
MAX_ROUTES="${MAX_ROUTES:-0}"              # 0 = all; >0 = only first N episodes
EPISODE_TIMEOUT_SEC="${EPISODE_TIMEOUT_SEC:-0}"  # 0 = disabled; wall-clock limit per episode
EPISODE_TIMEOUT_GRACE_SEC="${EPISODE_TIMEOUT_GRACE_SEC:-1800}"
# -----------------------------------------------------------------------------

[[ "$EPISODE_TIMEOUT_SEC" =~ ^[0-9]+$ ]] || {
    echo "[release] EPISODE_TIMEOUT_SEC must be a non-negative integer, got: $EPISODE_TIMEOUT_SEC" >&2
    exit 2
}
[[ "$EPISODE_TIMEOUT_GRACE_SEC" =~ ^[0-9]+$ ]] || {
    echo "[release] EPISODE_TIMEOUT_GRACE_SEC must be a non-negative integer, got: $EPISODE_TIMEOUT_GRACE_SEC" >&2
    exit 2
}
[[ -x "$EVALUATOR_PYTHON" ]] || {
    echo "[release] evaluator python is not executable: $EVALUATOR_PYTHON" >&2
    exit 2
}

# This script's own process group. Cleanup guards against ever signalling it, which
# would kill the run itself instead of its children.
MY_PGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"

# ------------------------------ kill helpers --------------------------------
# Kill a process group. Refuses to signal our own group.
kill_pgid() {
    local pgid="$1" sig="${2:-TERM}"
    [[ -n "$pgid" && "$pgid" != "$MY_PGID" ]] || return 0
    kill -"$sig" -- "-$pgid" 2>/dev/null || true
    return 0
}

# Kill everything one run dir recorded: evaluators (+ their SparseDriveV2 worker
# children), then the batch-lifetime workers, then CARLA.
kill_run_dir() {
    local dir="$1"
    local evals="$dir/eval.pgids"
    local carla="$dir/carla.pgids"
    local pid pgid

    if [[ -s "$evals" ]]; then
        echo "[release] stop: evaluators + SparseDriveV2 workers from $(basename "$dir")"
        while read -r pid pgid; do
            kill_pgid "${pgid:-}" TERM
        done < "$evals"
        sleep 3
        while read -r pid pgid; do
            kill_pgid "${pgid:-}" KILL
            [[ -n "${pid:-}" ]] && kill -KILL "$pid" 2>/dev/null
        done < "$evals"
    fi

    if [[ -s "$carla" ]]; then
        echo "[release] stop: CARLA servers from $(basename "$dir")"
        PID_FILE="$carla" bash "$FAIL2DRIVE_ROOT/start_carla.sh" stop || true
    fi
    return 0
}

: "${OUT_DIR:?Run scripts/run.sh to prepare an output directory}"
PID_FILE="$OUT_DIR/carla.pgids"          # CARLA server pgids (written by start_carla.sh)
EVAL_PGIDS_FILE="$OUT_DIR/eval.pgids"    # "<pid> <pgid>" per evaluator we launch
MAIN_LOG="$OUT_DIR/run.log"
mkdir -p "$OUT_DIR"/{res,logs,cfg,episodes}
: > "$EVAL_PGIDS_FILE"

# Everything below is echoed to both stdout and the run log.
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "[release] out_dir=$OUT_DIR"
echo "[release] episodes=$EPISODES_JSON"
echo "[release] config=$CONFIG"
echo "[release] episode_timeout_sec=$EPISODE_TIMEOUT_SEC (grace=${EPISODE_TIMEOUT_GRACE_SEC}s)"

port_open() { timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/$1" >/dev/null 2>&1; }

# ------------------------- resolve episode -> route xml ---------------------
mapfile -t NAMES < <("$EVALUATOR_PYTHON" - "$EPISODES_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
for name in cfg["episodes"]:
    print(name)
PY
)
[[ ${#NAMES[@]} -gt 0 ]] || { echo "[release] no episodes in $EPISODES_JSON" >&2; exit 2; }

if [[ "$MAX_ROUTES" -gt 0 && ${#NAMES[@]} -gt $MAX_ROUTES ]]; then
    NAMES=("${NAMES[@]:0:$MAX_ROUTES}")
    echo "[release] MAX_ROUTES=$MAX_ROUTES -> limiting to first ${#NAMES[@]} episodes"
fi

ROUTES=()
missing=0
for name in "${NAMES[@]}"; do
    xml="$ROUTES_DIR/${name}.xml"
    if [[ ! -f "$xml" ]]; then
        echo "[release] MISSING route: $xml" >&2
        missing=1
        continue
    fi
    ROUTES+=("$xml")
done
[[ $missing -eq 0 ]] || { echo "[release] aborting: some routes are missing" >&2; exit 2; }
NROUTES=${#ROUTES[@]}
echo "[release] $NROUTES episodes resolved"
if [[ "$NROUTES" -eq 0 ]]; then
    echo "[release] nothing pending; all requested episodes already have scores"
    exit 0
fi

# --------------- nvidia-smi index -> vulkan adapter index -------------------
# Match by GPU UUID: vulkaninfo lists a bare UUID, nvidia-smi prefixes it "GPU-".
# Resolved live so a driver/enumeration change cannot silently point CARLA at the
# llvmpipe CPU device (which segfaults ~60s in).
read -r -a SMI_ARR <<< "$GPUS_SMI"
mapfile -t VK_ARR < <("$EVALUATOR_PYTHON" - "${SMI_ARR[@]}" <<'PY'
import re, subprocess, sys

want = sys.argv[1:]

smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
    capture_output=True, text=True, check=True).stdout
smi_uuid = {}
for line in smi.strip().splitlines():
    idx, uuid = [p.strip() for p in line.split(",")]
    smi_uuid[idx] = uuid.removeprefix("GPU-").lower()

vk = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True).stdout
# Blocks look like:  GPU0:\n ... deviceName = ...\n ... deviceUUID = <uuid>
vk_uuid = {}
cur = None
for line in vk.splitlines():
    m = re.match(r"\s*GPU(\d+):", line)
    if m:
        cur = m.group(1)
        continue
    m = re.match(r"\s*deviceUUID\s*=\s*(\S+)", line)
    if m and cur is not None:
        vk_uuid[m.group(1).strip().lower()] = cur
        cur = None

for idx in want:
    uuid = smi_uuid.get(idx)
    if uuid is None:
        sys.exit(f"nvidia-smi has no GPU {idx}")
    adapter = vk_uuid.get(uuid)
    if adapter is None:
        sys.exit(f"no vulkan adapter matches nvidia-smi GPU {idx} (uuid {uuid})")
    print(adapter)
PY
)
if [[ ${#VK_ARR[@]} -ne ${#SMI_ARR[@]} ]]; then
    echo "[release] failed to map nvidia-smi indices -> vulkan adapters; is vulkaninfo installed?" >&2
    exit 1
fi

NUM_SHARDS=${#SMI_ARR[@]}
for ((i = 0; i < NUM_SHARDS; i++)); do
    echo "[release] shard $i: nvidia-smi GPU ${SMI_ARR[$i]} -> vulkan adapter ${VK_ARR[$i]} | rpc=$((BASE_PORT + i * 5)) tm=$((TM_BASE + i * 5))"
done

# ---------------------------- port sanity check -----------------------------
for ((i = 0; i < NUM_SHARDS; i++)); do
    for p in $((BASE_PORT + i * 5)) $((TM_BASE + i * 5)); do
        if port_open "$p"; then
            echo "[release] port $p is already in use; pick another BASE_PORT/TM_BASE" >&2
            exit 2
        fi
    done
done

# ------------------------------- teardown -----------------------------------
# Everything this script starts is killed by stop_all, which is wired to EXIT so it
# runs on EVERY exit path: normal finish, Ctrl-C, TERM, or an early `exit N` above.
# Installed BEFORE the first CARLA launch so an interrupt during the ~60s staggered
# startup cannot leak servers.
#
# Only ever touches things this run created (recorded pgids + our own ports), never
# other jobs on this shared box.
STOPPED=0

stop_all() {
    local rc=$?
    [[ "$STOPPED" == "1" ]] && return 0
    STOPPED=1
    trap - EXIT INT TERM

    if [[ ! -s "$EVAL_PGIDS_FILE" && ! -s "$PID_FILE" ]]; then
        return $rc   # nothing was ever started (early exit) -- stay quiet
    fi

    echo
    echo "[release] teardown: killing everything this run started..."
    kill_run_dir "$OUT_DIR"
    echo "[release] teardown: done"
    return $rc
}
trap 'stop_all; exit 130' INT TERM
trap stop_all EXIT

# ------------------------------- start CARLA --------------------------------
# start_carla.sh assigns slot i -> BASE_PORT + i*5, which is exactly our shard
# numbering, so passing the vulkan adapters in shard order lines the servers up.
echo "[release] starting $NUM_SHARDS CARLA server(s)..."
CARLA_ROOT="$CARLA_ROOT" GPUS="${VK_ARR[*]}" BASE_PORT="$BASE_PORT" QUALITY="$QUALITY" \
    PID_FILE="$PID_FILE" bash "$FAIL2DRIVE_ROOT/start_carla.sh" || {
        echo "[release] CARLA failed to start; see $FAIL2DRIVE_ROOT/carla_logs" >&2
        exit 1
    }

# --------------------- agent / orchestrator env (as single) -----------------
export AGENTICDRIVING_ENABLED=1
export AGENTICDRIVING_REDESIGN_CANDIDATE_BUDGET="${AGENTICDRIVING_REDESIGN_CANDIDATE_BUDGET:-10}"
export AGENTICDRIVING_MAX_REDESIGN_ATTEMPTS="${AGENTICDRIVING_MAX_REDESIGN_ATTEMPTS:-3}"
export AGENTICDRIVING_FALLBACK_MODE="${AGENTICDRIVING_FALLBACK_MODE:-hold}"
export AGENTICDRIVING_VERIFIER_TYPE="${AGENTICDRIVING_VERIFIER_TYPE:-pdms}"
export AGENTICDRIVING_VERIFIER_THRESHOLD="${AGENTICDRIVING_VERIFIER_THRESHOLD:-0.5}"

# ----------------------- per-shard config generation ------------------------
# The yaml pins `device: cuda:7` and that value is copied verbatim into
# SPARSEDRIVEV2_DEVICE (yaml wins over env), so each shard needs its own copy.
# graphics_adapter is unused by this repo but rewritten too, so the file is not
# lying about which device the shard's CARLA is on.
make_shard_cfg() {
    local shard=$1 smi=$2 vk=$3
    local out="$OUT_DIR/cfg/shard${shard}.yaml"
    sed -E -e "s|^([[:space:]]*device:).*|\1 cuda:${smi}|" \
           -e "s|^([[:space:]]*graphics_adapter:).*|\1 ${vk}|" \
           "$CONFIG" > "$out"
    echo "$out"
}

# ------------------------------ one shard worker ----------------------------
run_shard() {
    local shard=$1; shift
    local routes=("$@")
    local port=$((BASE_PORT + shard * 5))
    local tm=$((TM_BASE + shard * 5))
    local smi=${SMI_ARR[$shard]}
    local vk=${VK_ARR[$shard]}
    local cfg; cfg=$(make_shard_cfg "$shard" "$smi" "$vk")
    local total=${#routes[@]} n=0 failed=0
    echo "[shard $shard] gpu(smi)=$smi rpc=$port tm=$tm | $total episodes"


    for route in "${routes[@]}"; do
        n=$((n + 1))
        local name; name=$(basename "$route" .xml)
        local rid=${name##*_}
        echo "[shard $shard $n/$total] START $name"

        # Per-episode run dir. The yaml reads it via ${oc.env:AGENTICDRIVING_RUN_DIR},
        # resolved per process, so carla.output_dir + verifier.debug_log land here.
        export AGENTICDRIVING_RUN_DIR="$OUT_DIR/episodes/$name"
        mkdir -p "$AGENTICDRIVING_RUN_DIR"

        local result="$OUT_DIR/res/${rid}_res.json"
        local live="$OUT_DIR/logs/${rid}_live.txt"
        local log="$OUT_DIR/logs/${rid}_shard${shard}.log"
        rm -f "$result" "$result.simulation_timeout.json"

        # Put the town into sync mode before the evaluator attaches sensors: mirrors
        # the fail2drive shard preflight and avoids fixed_delta_seconds races.
        local town
        town="$(grep -oE 'town="[^"]+"' "$route" | head -1 | sed -E 's/town="([^"]+)"/\1/' || true)"
        "$EVALUATOR_PYTHON" - "$port" "$tm" "$town" "$route" >> "$log" 2>&1 <<'PY' || return 1
import os
import sys
import time
import xml.etree.ElementTree as ET

import carla

port = int(sys.argv[1])
tm_port = int(sys.argv[2])
town = sys.argv[3]
route_path = sys.argv[4]
client = carla.Client("localhost", port)
client.set_timeout(120)
world = client.get_world()
if town and world.get_map().name.split("/")[-1] != town:
    world = client.load_world(town)
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
settings.tile_stream_distance = float(os.environ.get("CARLA_TILE_STREAM_DISTANCE", "650"))
settings.actor_active_distance = float(os.environ.get("CARLA_ACTOR_ACTIVE_DISTANCE", "650"))
world.apply_settings(settings)

# Creating the Traffic Manager and issuing its first RPC can exceed the
# evaluator's hard-coded 2 s client timeout immediately after eight CARLA
# servers start together. Warm that endpoint with this 120 s preflight client
# so the evaluator's set_random_device_seed call is no longer the cold-start
# request. Keep TM synchronous to match the world/evaluator contract.
traffic_manager = client.get_trafficmanager(tm_port)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_random_device_seed(0)

# Warm the large-map tiles around the route start before RouteScenario spawns
# the hero. This is opt-in beyond the historical single tick so targeted
# reruns can preload adjacent road/collision tiles without changing defaults.
start = ET.parse(route_path).find(".//waypoints/position")
if start is not None:
    x = float(start.attrib["x"])
    y = float(start.attrib["y"])
    z = float(start.attrib.get("z", 0.0))
    world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(x=x, y=y, z=z + 50.0),
            carla.Rotation(pitch=-90.0),
        )
    )
warmup_ticks = int(os.environ.get("CARLA_ROUTE_PREFLIGHT_TICKS", "1"))
warmup_sleep = float(os.environ.get("CARLA_WARMUP_SLEEP_SEC", "0.05"))
for _ in range(max(1, warmup_ticks)):
    world.tick()
    if warmup_sleep > 0 and warmup_ticks > 1:
        time.sleep(warmup_sleep)
print(
    f"[preflight] map={world.get_map().name.split('/')[-1]} sync=True "
    f"tile_stream_distance={settings.tile_stream_distance} "
    f"actor_active_distance={settings.actor_active_distance} "
    f"route_start_warmup_ticks={max(1, warmup_ticks)} tm_port={tm_port} warmed=True",
    flush=True,
)
PY

        # setsid puts the evaluator in its own session/process group, so teardown can
        # signal the group and take its SparseDriveV2 worker child with it. A plain
        # `kill` on the shard subshell would leave that worker orphaned on the GPU.
        # This shell has no job control, so a background child is not a process-group
        # leader -> setsid exec()s in place and $! IS the new group leader.
        local evaluator_cmd=(
            "$EVALUATOR_PYTHON" "$HERE/runtime.py" evaluate "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py"
            --port="$port"
            --traffic-manager-port="$tm"
            --repetitions 1
            --routes "$route"
            --agent "$AGENT_FILE"
            --agent-config "$cfg"
            --checkpoint "$result"
            --debug-checkpoint "$live"
        )
        if [[ "$EPISODE_TIMEOUT_SEC" -gt 0 ]]; then
            # SIGINT is intentional: Leaderboard handles it by stopping the scenario,
            # calling agent.destroy() (which flushes buffered videos), computing route
            # completion and writing the checkpoint. --foreground makes timeout signal
            # only the evaluator, not its planner/ffmpeg children; those must stay alive
            # while destroy() finishes. Only if graceful cleanup itself hangs do we
            # escalate after the generous video-finalization grace period.
            setsid timeout --foreground --signal=INT \
                --kill-after="${EPISODE_TIMEOUT_GRACE_SEC}s" \
                "${EPISODE_TIMEOUT_SEC}s" "${evaluator_cmd[@]}" >> "$log" 2>&1 &
        else
            setsid "${evaluator_cmd[@]}" >> "$log" 2>&1 &
        fi
        local pid=$!
        local pgid
        pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
        # Guard: if setsid ever forked instead of exec'ing, pgid would be OUR group and
        # killing it would take down the run. Record only a genuinely new group.
        if [[ -n "$pgid" && "$pgid" != "$MY_PGID" ]]; then
            echo "$pid $pgid" >> "$EVAL_PGIDS_FILE"
        else
            echo "$pid" >> "$EVAL_PGIDS_FILE"
        fi
        wait "$pid"
        local rc=$?
        [[ "$rc" -eq 0 ]] || failed=1
        if [[ "$EPISODE_TIMEOUT_SEC" -gt 0 && ( "$rc" -eq 124 || "$rc" -eq 137 ) ]]; then
            echo "[shard $shard $n/$total] TIMEOUT $name after ${EPISODE_TIMEOUT_SEC}s (exit=$rc)"
            # Graceful timeout has returned only after agent.destroy() completed, so
            # the video is closed before residual workers are reaped. If the emergency
            # kill path fired, this also guarantees no planner child is left on the GPU.
            kill_pgid "${pgid:-}" TERM
            sleep 1
            kill_pgid "${pgid:-}" KILL
            # Graceful SIGINT normally produced a partial score. If cleanup had to be
            # killed, synthesize a valid zero-score timeout record so the batch can
            # resume and aggregate deterministically instead of leaving Started JSON.
            "$EVALUATOR_PYTHON" "$HERE/runtime.py" timeout \
                --result "$result" --route-name "$name" \
                --timeout-sec "$EPISODE_TIMEOUT_SEC" >> "$log" 2>&1 || \
                echo "[shard $shard] WARNING: failed to finalize timeout result for $name"
        fi
        # Consolidate every saved recovery/semantic VLM answer into one highly
        # readable episode-level JSON. Runtime logging normally creates it live;
        # this backfill also covers early crashes and evaluators using older
        # already-imported code during an in-place development run.
        "$EVALUATOR_PYTHON" -m agenticdriving.vlm.summary \
            "$AGENTICDRIVING_RUN_DIR" >> "$log" 2>&1 || \
            echo "[shard $shard] WARNING: VLM summary backfill failed for $name"
        echo "[shard $shard $n/$total] DONE  $name (exit=$rc)"
    done
    echo "[shard $shard] finished its $total episodes."
    return "$failed"
}

# ------------------- split episodes across shards + launch ------------------
echo "[release] splitting $NROUTES episodes across $NUM_SHARDS shards"
SHARD_PIDS=()
for ((s = 0; s < NUM_SHARDS; s++)); do
    subset=()
    for ((i = 0; i < NROUTES; i++)); do
        [[ $((i % NUM_SHARDS)) -eq $s ]] && subset+=("${ROUTES[$i]}")
    done
    [[ ${#subset[@]} -eq 0 ]] && continue
    run_shard "$s" "${subset[@]}" &
    SHARD_PIDS+=("$!")
done

BATCH_FAILED=0
for shard_pid in "${SHARD_PIDS[@]}"; do
    wait "$shard_pid" || BATCH_FAILED=1
done
echo "[release] all shards done."

# --------------------------------- summary ----------------------------------
# Printed before teardown; stop_all runs after this via the EXIT trap.
echo
echo "[release] ===== summary ====="
"$EVALUATOR_PYTHON" - "$OUT_DIR/res" "${NAMES[@]}" <<'PY'
import json, os, sys
res_dir, names = sys.argv[1], sys.argv[2:]
for name in names:
    rid = name.rsplit("_", 1)[-1]
    path = os.path.join(res_dir, f"{rid}_res.json")
    try:
        rec = json.load(open(path))["_checkpoint"]["records"][0]
        score = rec.get("scores", {}).get("score_composed")
        status = rec.get("status")
        score = "n/a" if score is None else f"{score:.2f}"
        print(f"  {name:<46} score={score:>6}  {status}")
    except Exception as exc:
        print(f"  {name:<46} NO RESULT ({type(exc).__name__})")
PY
echo
echo "[release] results:  $OUT_DIR/res"
echo "[release] logs:     $OUT_DIR/logs"
echo "[release] episodes: $OUT_DIR/episodes"
echo "[release] done."
exit "$BATCH_FAILED"
