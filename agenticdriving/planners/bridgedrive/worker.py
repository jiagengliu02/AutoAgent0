"""BridgeDrive worker subprocess: pickle-framed request loop.

Protocol (8-byte little-endian length prefix + pickle blob per message, the
same framing as the sparsedrivev2 worker — transcribed from
agenticdriving/planners/sparsedrivev2/worker.py):

  parent -> worker:
    {"init": {"global_plan", "global_plan_world_coord", "town",
              "checkpoint_dir"}}                      (once, first message)
    {"obs": {"input_data": {...}}, "info": {...}}     (per tick, full forward)
    {"tick": {"obs": {...}, "info": {...}}}           (stateful tick only,
                                                       judge-duty stride)
    {"message": "Done"}                               (shutdown)
  worker -> parent:
    {"ready": True}                                   (after init)
    {"ticked": True, "warmup": bool}                  (after a tick message)
    {"last_debug"}
    {"error": "<traceback>", ...}                     (on exception)

Run inside the LEAD venv with LEAD + CARLA PythonAPI on PYTHONPATH (the
BridgeDrivePerceptionCarlaService arranges the environment).
"""
from __future__ import annotations

import os
import pickle
import struct
import sys
import traceback

_LEN = struct.Struct("<Q")


def _read_message(stream):
    header = stream.read(_LEN.size)
    if not header or len(header) < _LEN.size:
        return None
    (length,) = _LEN.unpack(header)
    payload = stream.read(length)
    if len(payload) < length:
        return None
    return pickle.loads(payload)


def _write_message(stream, message) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_LEN.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def main() -> int:
    # Keep the protocol stream clean: dup stdout for framing, route stray
    # prints (LEAD logs freely) to stderr.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    sys.stdout = sys.stderr

    runtime = None
    stdin = sys.stdin.buffer
    while True:
        message = _read_message(stdin)
        if message is None or message.get("message") == "Done":
            return 0
        try:
            if "init" in message:
                init = message["init"]
                # Deferred import: LEAD + torch load only on demand.
                from agenticdriving.planners.bridgedrive.runtime import (
                    BridgeDriveRuntime,
                )

                runtime = BridgeDriveRuntime(
                    init.get("checkpoint_dir")
                    or os.environ["BRIDGEDRIVE_CHECKPOINT_DIR"]
                )
                runtime.init_episode(
                    init["global_plan"],
                    init["global_plan_world_coord"],
                    init["town"],
                )
                _write_message(protocol_out, {"ready": True})
                continue

            if runtime is None:
                raise RuntimeError("first message must be an init message")
            if "tick" in message:
                # Stateful preprocessing only — no forward. Judge-duty stride
                # support; planner duty never sends this.
                ack = runtime.tick_only(
                    message["tick"]["obs"], message["tick"]["info"]
                )
                _write_message(protocol_out, ack)
                continue
            last_debug = runtime.process(
                message["obs"], message["info"]
            )
            _write_message(
                protocol_out,
                {
                    "last_debug": last_debug,
                },
            )
        except Exception:  # noqa: BLE001 - forwarded to the parent verbatim
            _write_message(
                protocol_out,
                {
                    "error": traceback.format_exc(),
                    "last_debug": None,
                },
            )


if __name__ == "__main__":
    sys.exit(main())
