from models.online import ChatMessage, GenerationRequest, OnlineModelConfig, OnlineModelEngine


def test_mock_engine_uses_custom_system_prompt_and_last_user_message():
    engine = OnlineModelEngine(
        OnlineModelConfig(system_prompt="You are a support bot for my store.")
    )

    response = engine.generate(
        GenerationRequest(messages=[ChatMessage(role="user", content="Where is my order?")])
    )

    assert "Mock online Llama model response" in response
    assert "Where is my order?" in response
