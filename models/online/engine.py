# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Literal, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """A single chat message accepted by the online model engine."""

    role: Role
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    """Generation controls shared by the HTTP API and direct Python use."""

    messages: Sequence[ChatMessage]
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass(frozen=True)
class OnlineModelConfig:
    """Runtime configuration for an online customized Llama model."""

    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    system_prompt: str = "You are a helpful, concise assistant."
    backend: Literal["mock", "transformers", "openai"] = "mock"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    openai_base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    request_timeout_seconds: int = 60


@dataclass
class OnlineModelEngine:
    """Small serving engine that can run in mock mode or load a HF model lazily."""

    config: OnlineModelConfig = field(default_factory=OnlineModelConfig)
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def generate(self, request: GenerationRequest) -> str:
        """Generate an assistant response for a chat-style request."""

        messages = self._with_default_system_prompt(request.messages)
        if self.config.backend == "mock":
            return self._mock_response(messages)
        if self.config.backend == "openai":
            return self._openai_compatible_response(messages, request)
        return self._transformers_response(messages, request)

    def _with_default_system_prompt(
        self, messages: Sequence[ChatMessage]
    ) -> list[ChatMessage]:
        if messages and messages[0].role == "system":
            return list(messages)
        return [ChatMessage(role="system", content=self.config.system_prompt), *messages]

    def _mock_response(self, messages: Sequence[ChatMessage]) -> str:
        last_user_message = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        return (
            "Mock online Llama model response. Configure LLAMA_ONLINE_BACKEND="
            "openai or transformers and LLAMA_MODEL_ID to serve real weights. Last user "
            f"message: {last_user_message}"
        )

    def _load_transformers(self) -> tuple[object, object]:
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                trust_remote_code=self.config.trust_remote_code,
            )
        return self._tokenizer, self._model

    def _transformers_response(
        self, messages: Sequence[ChatMessage], request: GenerationRequest
    ) -> str:
        tokenizer, model = self._load_transformers()
        serializable_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        inputs = tokenizer.apply_chat_template(
            serializable_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

    def _openai_compatible_response(
        self, messages: Sequence[ChatMessage], request: GenerationRequest
    ) -> str:
        if not self.config.api_key:
            raise RuntimeError(
                "LLAMA_API_KEY is required when LLAMA_ONLINE_BACKEND=openai."
            )

        endpoint = f"{self.config.openai_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model_id,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        http_request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(
                http_request, timeout=self.config.request_timeout_seconds
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible backend returned HTTP {error.code}: {detail}"
            ) from error

        return response_payload["choices"][0]["message"]["content"].strip()
