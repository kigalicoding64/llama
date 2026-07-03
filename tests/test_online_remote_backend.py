import json

import models.online.engine as engine_module
from models.online import ChatMessage, GenerationRequest, OnlineModelConfig, OnlineModelEngine


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": "Remote Llama response"}}]}
        ).encode("utf-8")


def test_openai_compatible_backend_posts_chat_completion(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(engine_module, "urlopen", fake_urlopen)
    engine = OnlineModelEngine(
        OnlineModelConfig(
            backend="openai",
            model_id="custom-llama",
            openai_base_url="https://models.example.test/v1/",
            api_key="secret",
        )
    )

    response = engine.generate(
        GenerationRequest(messages=[ChatMessage(role="user", content="Hello")])
    )

    assert response == "Remote Llama response"
    assert captured["url"] == "https://models.example.test/v1/chat/completions"
    assert captured["timeout"] == 60
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["payload"]["model"] == "custom-llama"
    assert captured["payload"]["messages"][-1] == {"role": "user", "content": "Hello"}
