"""Claude Messages API adapter for the recovery VLM interface (no SDK/GPU)."""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Sequence

from PIL import Image

from agenticdriving.vlm.parsing import try_parse_json


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the API credential to a redirect target.
        return None


class ClaudeTrajectorySelector:
    """One HTTP attempt per infer_prompt; retries are owned by the caller."""

    def __init__(
        self, *, model_id: str, max_new_tokens: int,
        api_key_env: str = "ANTHROPIC_API_KEY", effort: str = "low",
        enable_thinking: bool = False,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.api_key_env = api_key_env
        self.effort = effort
        self.enable_thinking = enable_thinking
        self.device = "api"
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported Claude effort")
        if not enable_thinking and effort not in {"low", "medium", "high"}:
            raise ValueError("Disabled Claude thinking requires effort <= high")
        self._api_key()  # Validate locally; initialization makes no paid request.

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Missing Claude API credential: set {self.api_key_env}")
        return key

    def infer_prompt(
        self, image_paths: Sequence[Path], prompt: str, *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, object]:
        content = []
        for path in image_paths:
            with Image.open(path) as img:
                media_type = Image.MIME.get(img.format)
            if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
                raise ValueError("Unsupported Claude image format")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type,
                           "data": base64.b64encode(Path(path).read_bytes()).decode("ascii")},
            })
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.model_id,
            "max_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "adaptive" if self.enable_thinking else "disabled"},
            "output_config": {"effort": self.effort},
        }
        if system_prompt:
            payload["system"] = system_prompt
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "anthropic-version": "2023-06-01", "x-api-key": self._api_key()},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.build_opener(_NoRedirect()).open(
                request, timeout=timeout_sec if timeout_sec is not None else 60.0,
            ) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not copy provider bodies, request headers, or credentials to logs.
            raise RuntimeError(f"Claude API HTTP {exc.code} (no automatic retry)") from None
        raw = "\n".join(
            str(block.get("text", "")) for block in result.get("content", [])
            if block.get("type") == "text"
        )
        usage = result.get("usage") or {}
        prompt_tokens = sum(int(usage.get(key, 0) or 0) for key in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
        ))
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        return {
            "raw_output": raw, "parsed_output": try_parse_json(raw),
            "elapsed_sec": time.monotonic() - started,
            "prompt": prompt, "system_prompt": system_prompt,
            "token_usage": {"prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens},
            "provider_usage": usage, "stop_reason": result.get("stop_reason"),
        }
