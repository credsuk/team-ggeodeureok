from __future__ import annotations

import json
from typing import Any
import urllib.error
import urllib.request


class OllamaClient:
    """Small, dependency-free client for the locally running Ollama chat API."""

    def __init__(self, base_url: str, model: str, timeout: int = 240, keep_alive: str = "5m"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive

    def chat_json(self, system: str, user: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema or "json",
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.1},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"로컬 Ollama에 연결할 수 없습니다 ({self.base_url}). Ollama 실행 상태와 모델 '{self.model}'을 확인하세요."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama 응답이 JSON이 아닙니다.") from exc
        content = result.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("로컬 모델이 요청된 JSON 형식으로 답하지 않았습니다.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("로컬 모델의 JSON 응답이 객체 형식이 아닙니다.")
        return parsed
