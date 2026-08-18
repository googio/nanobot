from nanobot.agent.turn_usage import TurnUsageAccumulator


def test_turn_usage_accumulates_requests_and_keeps_latest_context() -> None:
    usage = TurnUsageAccumulator()

    usage.record({"prompt_tokens": 100, "completion_tokens": 12, "cached_tokens": 70})
    usage.record({"prompt_tokens": 240, "completion_tokens": 18, "cached_tokens": 180})

    assert usage.snapshot() == {
        "prompt_tokens": 340,
        "completion_tokens": 30,
        "cached_tokens": 250,
        "request_count": 2,
        "context_tokens": 240,
    }


def test_turn_usage_marks_mixed_provider_and_estimated_requests() -> None:
    usage = TurnUsageAccumulator()

    usage.record({"prompt_tokens": 10, "completion_tokens": 3})
    usage.record({"prompt_tokens": 20, "completion_tokens": 4, "estimated_tokens": 24})

    assert usage.snapshot() == {
        "prompt_tokens": 30,
        "completion_tokens": 7,
        "estimated_tokens": 24,
        "request_count": 2,
        "context_tokens": 20,
    }


def test_turn_usage_omits_context_until_provider_reports_input() -> None:
    usage = TurnUsageAccumulator()

    usage.record({"completion_tokens": 5})

    assert usage.snapshot() == {
        "prompt_tokens": 0,
        "completion_tokens": 5,
        "request_count": 1,
    }
