# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from __future__ import annotations

import os
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import ChatMessage, GenerationRequest, OnlineModelConfig, OnlineModelEngine


class MessageIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class GenerateIn(BaseModel):
    messages: list[MessageIn] = Field(min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


class GenerateOut(BaseModel):
    text: str


def engine_from_env() -> OnlineModelEngine:
    config = OnlineModelConfig(
        model_id=os.getenv(
            "LLAMA_MODEL_ID", os.getenv("LLAMA_REMOTE_MODEL", OnlineModelConfig.model_id)
        ),
        system_prompt=os.getenv("LLAMA_SYSTEM_PROMPT", OnlineModelConfig.system_prompt),
        backend=os.getenv("LLAMA_ONLINE_BACKEND", OnlineModelConfig.backend),
        device_map=os.getenv("LLAMA_DEVICE_MAP", OnlineModelConfig.device_map),
        torch_dtype=os.getenv("LLAMA_TORCH_DTYPE", OnlineModelConfig.torch_dtype),
        trust_remote_code=os.getenv("LLAMA_TRUST_REMOTE_CODE", "false").lower()
        in {"1", "true", "yes"},
        openai_base_url=os.getenv(
            "LLAMA_OPENAI_BASE_URL", OnlineModelConfig.openai_base_url
        ),
        api_key=os.getenv("LLAMA_API_KEY"),
        request_timeout_seconds=int(os.getenv("LLAMA_REQUEST_TIMEOUT_SECONDS", "60")),
    )
    return OnlineModelEngine(config=config)


def create_app(engine: OnlineModelEngine | None = None) -> FastAPI:
    app = FastAPI(
        title="Llama Online Model Server",
        description="Serve a customized Llama chat model behind HTTP endpoints.",
        version="0.1.0",
    )
    model_engine = engine or engine_from_env()

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "Llama Online Model Server",
            "health": "/health",
            "chat": "/v1/chat/completions",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": model_engine.config.backend}

    @app.post("/generate", response_model=GenerateOut)
    def generate(request: GenerateIn) -> GenerateOut:
        messages = [ChatMessage(role=item.role, content=item.content) for item in request.messages]
        try:
            text = model_engine.generate(
                GenerationRequest(
                    messages=messages,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )
            )
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return GenerateOut(text=text)

    @app.post("/v1/chat/completions")
    def chat_completions(request: GenerateIn) -> dict[str, object]:
        output = generate(request)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_engine.config.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output.text},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "models.online.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
