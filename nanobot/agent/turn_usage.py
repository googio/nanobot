"""Accumulate model usage for one agent turn without losing request semantics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TurnUsageAccumulator:
    """Collect per-request usage into one stable turn-level snapshot.

    Provider prompt tokens are additive for billing/work accounting, while the
    latest prompt size describes current context occupancy.  Keeping both
    behind this interface prevents callers from accidentally treating the sum
    of every tool-loop request as the active context size.
    """

    _totals: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    _request_count: int = 0
    _context_tokens: int | None = None

    def record(self, usage: dict[str, int]) -> None:
        """Record one physical model request, including requests with no usage."""
        self._request_count += 1
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is not None and prompt_tokens >= 0:
            self._context_tokens = prompt_tokens
        for key, value in usage.items():
            if key in {"context_tokens", "request_count"}:
                continue
            if value < 0:
                continue
            self._totals[key] = self._totals.get(key, 0) + value

    def snapshot(self) -> dict[str, int]:
        """Return a backward-compatible usage mapping for persistence/wire use."""
        result = dict(self._totals)
        result["request_count"] = self._request_count
        if self._context_tokens is not None:
            result["context_tokens"] = self._context_tokens
        return result
