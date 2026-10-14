"""Pickle-framed SparseDriveV2 worker for the CARLA adapter.

stdin/stdout carry binary protocol messages. All normal logging is redirected to
stderr so library prints cannot corrupt the protocol stream.
"""
from __future__ import annotations

import logging
import os
import pickle
import struct
import sys
import traceback
from typing import Any


def _read_message(pipe) -> Any:
    header = pipe.read(8)
    if len(header) != 8:
        raise EOFError
    size = struct.unpack("<Q", header)[0]
    payload = bytearray()
    while len(payload) < size:
        chunk = pipe.read(size - len(payload))
        if not chunk:
            raise EOFError
        payload.extend(chunk)
    return pickle.loads(payload)


def _write_message(pipe, message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    pipe.write(struct.pack("<Q", len(payload)))
    pipe.write(payload)
    pipe.flush()


def main() -> int:
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    sys.stdout = sys.stderr
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("sparsedrivev2_worker")

    try:
        from agenticdriving.planners.sparsedrivev2.inference import SparseDriveV2Inference

        inference = SparseDriveV2Inference(
            repo_root=os.environ.get("SPARSEDRIVEV2_REPO_ROOT", "").strip() or None,
            checkpoint_path=os.environ.get("SPARSEDRIVEV2_CKPT", "").strip() or None,
            config_path=os.environ.get("SPARSEDRIVEV2_CONFIG", "").strip() or None,
            device=os.environ.get("SPARSEDRIVEV2_DEVICE", "cuda"),
        )
        inference.setup()
        log.info("SparseDriveV2 worker ready")
    except Exception:
        log.error("SparseDriveV2 worker setup failed")
        log.error(traceback.format_exc())
        return 1

    stdin = sys.stdin.buffer
    try:
        while True:
            try:
                message = _read_message(stdin)
            except EOFError:
                break
            if isinstance(message, dict) and message.get("message") == "Done":
                break
            try:
                obs = message["obs"]
                info = message["info"]
                proposals, scores = inference.process(obs=obs, info=info)
                _write_message(
                    protocol_out,
                    {
                        "proposals": proposals,
                        "scores": scores,
                        "last_debug": inference.last_debug,
                        "num_poses": int(inference.num_poses),
                    },
                )
            except Exception:
                log.error("SparseDriveV2 worker frame failed")
                log.error(traceback.format_exc())
                _write_message(
                    protocol_out,
                    {
                        "error": traceback.format_exc(),
                        "last_debug": getattr(inference, "last_debug", {}),
                        "num_poses": int(getattr(inference, "num_poses", 0) or 0),
                    },
                )
    finally:
        try:
            inference.finalize()
        except AttributeError:
            pass
        except Exception:
            log.exception("SparseDriveV2 worker shutdown failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
