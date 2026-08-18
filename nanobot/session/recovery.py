"""Durable, side-effect-safe recovery for interrupted WebUI turns.

The coordinator owns restart policy.  AgentLoop only exposes checkpoint
materialization and an admission hook, so transport code never has to guess
whether an interrupted tool call is safe to replay.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.outbound_events import (
    RecoveryStateEvent,
    SessionUpdatedEvent,
    outbound_message_for_event,
)
from nanobot.bus.queue import MessageBus
from nanobot.session import turn_continuation
from nanobot.session.keys import UNIFIED_SESSION_KEY, last_channel_from_metadata
from nanobot.session.manager import Session, SessionManager
from nanobot.webui.metadata import WEBUI_TURN_METADATA_KEY

RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
PENDING_USER_TURN_KEY = "pending_user_turn"
RECOVERY_METADATA_KEY = "webui_recovery"
RECOVERY_INBOUND_METADATA_KEY = "_webui_recovery_id"
PROVIDER_STATE_CHECKPOINT_VERSION_KEY = "provider_state_checkpoint_version"
PROVIDER_STATE_CHECKPOINT_VERSION = "v1"

_RECOVERY_STATUSES = frozenset({"resuming", "awaiting_user", "recovered", "failed"})
_AUTO_CONTINUE_PHASES = frozenset({"tools_completed"})
_UNCERTAIN_TOOL_PHASES = frozenset({"awaiting_tools"})


class RecoveryActionError(ValueError):
    """A stale or malformed recovery action from an authenticated WebUI."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class RecoveryAdmission(Protocol):
    """Narrow AgentLoop boundary for queued recovery validation."""

    async def admit(self, message: InboundMessage) -> bool: ...


def _checkpoint_message_key(message: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        message.get("role"),
        message.get("content"),
        message.get("tool_call_id"),
        message.get("name"),
        message.get("tool_calls"),
        message.get("reasoning_content"),
        message.get("thinking_blocks"),
    )


def restore_runtime_checkpoint(session: Session) -> bool:
    """Materialize the durable checkpoint exactly once and clear it.

    Pending tool calls become explicit interrupted tool results.  They are
    never executed here.  Provider-native state is retained only for the two
    checkpoint shapes known to be synchronized with persisted history.
    """
    checkpoint = cast(object, session.metadata.get(RUNTIME_CHECKPOINT_KEY))
    if not isinstance(checkpoint, dict):
        return False
    data = cast(dict[str, Any], checkpoint)
    assistant = cast(object, data.get("assistant_message"))
    completed = cast(Iterable[object], data.get("completed_tool_results") or [])
    pending = cast(Iterable[object], data.get("pending_tool_calls") or [])

    restored: list[dict[str, Any]] = []
    if isinstance(assistant, dict):
        row = dict(cast(dict[str, Any], assistant))
        row.setdefault("timestamp", datetime.now().isoformat())
        restored.append(row)
    for value in completed:
        if isinstance(value, dict):
            row = dict(cast(dict[str, Any], value))
            row.setdefault("timestamp", datetime.now().isoformat())
            restored.append(row)
    for value in pending:
        if not isinstance(value, dict):
            continue
        tool_call = cast(dict[str, Any], value)
        function = cast(dict[str, Any], tool_call.get("function") or {})
        restored.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "name": function.get("name") or "tool",
                "content": "Error: Task interrupted before this tool finished.",
                "timestamp": datetime.now().isoformat(),
                "_recovery_interrupted": True,
            }
        )

    overlap = 0
    for size in range(min(len(session.messages), len(restored)), 0, -1):
        if all(
            _checkpoint_message_key(left) == _checkpoint_message_key(right)
            for left, right in zip(session.messages[-size:], restored[:size])
        ):
            overlap = size
            break
    session.messages.extend(restored[overlap:])

    assistant_data = cast(dict[str, Any], assistant) if isinstance(assistant, dict) else None
    synchronized = (
        data.get(PROVIDER_STATE_CHECKPOINT_VERSION_KEY)
        == PROVIDER_STATE_CHECKPOINT_VERSION
    )
    phase = data.get("phase")
    exact_final = (
        phase == "final_response"
        and assistant_data is not None
        and assistant_data.get("role") == "assistant"
        and not data.get("completed_tool_results")
        and not data.get("pending_tool_calls")
    )
    exact_tools = (
        phase == "tools_completed"
        and assistant_data is not None
        and assistant_data.get("role") == "assistant"
        and not data.get("pending_tool_calls")
    )
    if not (synchronized and (exact_final or exact_tools)):
        session.provider_state = None

    session.metadata.pop(PENDING_USER_TURN_KEY, None)
    session.metadata.pop(RUNTIME_CHECKPOINT_KEY, None)
    session.updated_at = datetime.now()
    return True


def restore_pending_interruption(session: Session, *, superseded: bool = False) -> bool:
    """Close a persisted user-only turn without pretending it was answered."""
    if not session.metadata.get(PENDING_USER_TURN_KEY):
        return False
    if session.messages and session.messages[-1].get("role") == "user":
        content = (
            "Task recovery was superseded by a newer message."
            if superseded
            else "Error: Task interrupted before a response was generated."
        )
        session.messages.append(
            {
                "role": "assistant",
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "_recovery_interrupted": True,
            }
        )
        session.provider_state = None
        session.updated_at = datetime.now()
    session.metadata.pop(PENDING_USER_TURN_KEY, None)
    return True


def append_recovery_interruption(session: Session, *, superseded: bool = False) -> None:
    """Close a restored partial turn whose last durable row is not the user message."""
    session.messages.append(
        {
            "role": "assistant",
            "content": (
                "Task recovery was superseded by a newer message."
                if superseded
                else "Error: Task recovery was interrupted before completion."
            ),
            "timestamp": datetime.now().isoformat(),
            "_recovery_interrupted": True,
        }
    )
    session.provider_state = None
    session.updated_at = datetime.now()


def recovery_state_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a sanitized recovery state suitable for the WebSocket wire."""
    value = metadata.get(RECOVERY_METADATA_KEY) if metadata else None
    if not isinstance(value, dict):
        return None
    state = cast(dict[str, Any], value)
    status = state.get("status")
    recovery_id = state.get("recovery_id")
    if status not in _RECOVERY_STATUSES or not isinstance(recovery_id, str):
        return None
    payload: dict[str, Any] = {"status": status, "recovery_id": recovery_id}
    for key in ("reason", "message"):
        item = state.get(key)
        if isinstance(item, str) and item:
            payload[key] = item
    attempts = state.get("attempts")
    if isinstance(attempts, int) and attempts >= 0:
        payload["attempts"] = attempts
    return payload


@dataclasses.dataclass(slots=True)
class RecoveryCoordinator:
    """Classify, announce, and gate durable WebUI turn recovery."""

    sessions: SessionManager
    bus: MessageBus
    unified_session: bool = False

    async def scan(self) -> None:
        """Recover every interrupted WebUI session once at gateway startup."""
        for item in self.sessions.list_sessions():
            key = item.get("key")
            if not isinstance(key, str):
                continue
            session = self.sessions.get_or_create(key)
            route = self._websocket_route(session)
            if route is None:
                continue
            try:
                await self._recover_session(session, route[1])
            except Exception:
                logger.exception("failed to recover interrupted WebUI session {}", session.key)
                recovery_id = self._recovery_id(session)
                failed = self._set_state(
                    session,
                    status="failed",
                    recovery_id=recovery_id,
                    attempts=self._recovery_attempts(session),
                    reason="recovery_failed",
                    message="The interrupted task could not be recovered automatically.",
                )
                self.sessions.save(session)
                await self._publish(route[1], failed)

    async def admit(self, message: InboundMessage) -> bool:
        """Reject stale queued recoveries and let new user input supersede them."""
        recovery_id = message.metadata.get(RECOVERY_INBOUND_METADATA_KEY)
        if isinstance(recovery_id, str):
            session = self.sessions.get_or_create(message.session_key)
            state = recovery_state_from_metadata(session.metadata)
            return bool(
                state
                and state["status"] == "resuming"
                and state["recovery_id"] == recovery_id
            )
        if message.channel != "websocket":
            return True
        session = self.sessions.get_or_create(message.session_key)
        state = recovery_state_from_metadata(session.metadata)
        if state and state["status"] in {"resuming", "awaiting_user", "failed"}:
            restore_runtime_checkpoint(session)
            if not restore_pending_interruption(session, superseded=True):
                append_recovery_interruption(session, superseded=True)
            recovered = self._set_state(
                session,
                status="recovered",
                recovery_id=cast(str, state["recovery_id"]),
                attempts=cast(int, state.get("attempts", 0)),
                reason="superseded",
                message="The interrupted task was replaced by your newer message.",
            )
            self.sessions.save(session)
            await self._publish(message.chat_id, recovered)
        return True

    async def turn_completed(self, session_key: str) -> None:
        """Resolve a resuming state after the recovered turn commits."""
        session = self.sessions.get_or_create(session_key)
        state = recovery_state_from_metadata(session.metadata)
        if not state or state["status"] != "resuming":
            return
        route = self._websocket_route(session)
        if route is None:
            return
        recovered = self._set_state(
            session,
            status="recovered",
            recovery_id=cast(str, state["recovery_id"]),
            attempts=cast(int, state.get("attempts", 0)),
            reason="continued",
            message="The interrupted task was recovered.",
        )
        self.sessions.save(session)
        await self._publish(route[1], recovered)

    async def handle_action(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply an authenticated continue/dismiss operation."""
        chat_id = payload.get("chat_id")
        recovery_id = payload.get("recovery_id")
        if not isinstance(chat_id, str) or not chat_id:
            raise RecoveryActionError("missing chat_id")
        if not isinstance(recovery_id, str) or not recovery_id:
            raise RecoveryActionError("missing recovery_id")
        session = self.sessions.get_or_create(self._session_key(chat_id))
        state = recovery_state_from_metadata(session.metadata)
        if not state or state["recovery_id"] != recovery_id:
            raise RecoveryActionError("recovery state is stale", status=409)

        if action == "dismiss":
            restore_runtime_checkpoint(session)
            restore_pending_interruption(session)
            next_state = self._set_state(
                session,
                status="recovered",
                recovery_id=recovery_id,
                attempts=cast(int, state.get("attempts", 0)),
                reason="dismissed",
                message="The interrupted task was dismissed.",
            )
            self.sessions.save(session)
            await self._publish(chat_id, next_state)
            return next_state
        if action != "continue":
            raise RecoveryActionError("unknown recovery action")
        if state["status"] not in {"awaiting_user", "failed"}:
            raise RecoveryActionError("recovery is not waiting for confirmation", status=409)
        next_state = self._set_state(
            session,
            status="resuming",
            recovery_id=recovery_id,
            attempts=cast(int, state.get("attempts", 0)) + 1,
            reason="user_confirmed",
            message="Continuing the interrupted task from saved context.",
            resume_message_count=len(session.messages),
        )
        self.sessions.save(session)
        await self._publish(chat_id, next_state)
        await self._queue_continuation(session, chat_id, next_state)
        return next_state

    async def _recover_session(self, session: Session, chat_id: str) -> None:
        checkpoint_value = cast(object, session.metadata.get(RUNTIME_CHECKPOINT_KEY))
        checkpoint = (
            cast(dict[str, Any], checkpoint_value)
            if isinstance(checkpoint_value, dict)
            else None
        )
        pending = session.metadata.get(PENDING_USER_TURN_KEY) is True
        state = recovery_state_from_metadata(session.metadata)
        if not pending and checkpoint is None:
            if state and state["status"] == "resuming":
                resume_count = self._resume_message_count(session)
                if resume_count is not None and len(session.messages) > resume_count:
                    next_state = self._set_state(
                        session,
                        status="recovered",
                        recovery_id=cast(str, state["recovery_id"]),
                        attempts=cast(int, state.get("attempts", 0)),
                        reason="committed",
                        message="The interrupted task had already completed.",
                    )
                else:
                    next_state = self._set_state(
                        session,
                        status="awaiting_user",
                        recovery_id=cast(str, state["recovery_id"]),
                        attempts=cast(int, state.get("attempts", 1)),
                        reason="loop_guard",
                        message=(
                            "Recovery was interrupted again. "
                            "Continue only after reviewing the task."
                        ),
                    )
                self.sessions.save(session)
                await self._publish(chat_id, next_state)
            return
        if state and state["status"] in {"awaiting_user", "failed"}:
            await self._publish(chat_id, state)
            return
        if state and state["status"] == "resuming":
            restore_runtime_checkpoint(session)
            restore_pending_interruption(session)
            waiting = self._set_state(
                session,
                status="awaiting_user",
                recovery_id=cast(str, state["recovery_id"]),
                attempts=cast(int, state.get("attempts", 1)),
                reason="loop_guard",
                message="Recovery was interrupted again. Continue only after reviewing the task.",
            )
            self.sessions.save(session)
            await self._publish(chat_id, waiting)
            return

        recovery_id = uuid4().hex
        phase = checkpoint.get("phase") if checkpoint is not None else None
        pending_calls = checkpoint.get("pending_tool_calls") if checkpoint is not None else None
        if phase == "final_response":
            restore_runtime_checkpoint(session)
            recovered = self._set_state(
                session,
                status="recovered",
                recovery_id=recovery_id,
                attempts=0,
                reason="answer_restored",
                message="The completed answer was restored after restart.",
            )
            self.sessions.save(session)
            await self._publish(chat_id, recovered, session_updated=True)
            return
        if phase in _UNCERTAIN_TOOL_PHASES or pending_calls:
            restore_runtime_checkpoint(session)
            waiting = self._set_state(
                session,
                status="awaiting_user",
                recovery_id=recovery_id,
                attempts=0,
                reason="tool_state_unknown",
                message="A tool may have been interrupted. It will not run again without confirmation.",
            )
            self.sessions.save(session)
            await self._publish(chat_id, waiting, session_updated=True)
            return
        if phase in _AUTO_CONTINUE_PHASES:
            restore_runtime_checkpoint(session)
        resuming = self._set_state(
            session,
            status="resuming",
            recovery_id=recovery_id,
            attempts=1,
            reason="restart",
            message="Continuing the interrupted task from saved context.",
            resume_message_count=len(session.messages),
        )
        self.sessions.save(session)
        await self._publish(chat_id, resuming)
        await self._queue_continuation(session, chat_id, resuming)

    async def _queue_continuation(
        self,
        session: Session,
        chat_id: str,
        state: Mapping[str, Any],
    ) -> None:
        recovery_id = cast(str, state["recovery_id"])
        await self.bus.publish_inbound(
            InboundMessage(
                channel="websocket",
                sender_id="system:recovery",
                chat_id=chat_id,
                content=(
                    "Continue the interrupted request from the saved conversation context. "
                    "Do not repeat completed work or mention the restart unless it affects the answer."
                ),
                metadata={
                    "webui": True,
                    "_wants_stream": True,
                    WEBUI_TURN_METADATA_KEY: f"recovery:{recovery_id}",
                    RECOVERY_INBOUND_METADATA_KEY: recovery_id,
                    turn_continuation.INTERNAL_CONTINUATION_META: True,
                    turn_continuation.SKIP_USER_PERSIST_META: True,
                },
                session_key_override=session.key,
                require_existing_session=True,
            )
        )

    @staticmethod
    def _recovery_id(session: Session) -> str:
        state = recovery_state_from_metadata(session.metadata)
        if state is not None:
            return cast(str, state["recovery_id"])
        return uuid4().hex

    @staticmethod
    def _recovery_attempts(session: Session) -> int:
        state = recovery_state_from_metadata(session.metadata)
        if state is None:
            return 0
        return cast(int, state.get("attempts", 0))

    @staticmethod
    def _resume_message_count(session: Session) -> int | None:
        raw_value = cast(object, session.metadata.get(RECOVERY_METADATA_KEY))
        value = cast(dict[str, Any], raw_value) if isinstance(raw_value, dict) else None
        if value is None:
            return None
        count = value.get("resume_message_count")
        return count if isinstance(count, int) and count >= 0 else None

    async def _publish(
        self,
        chat_id: str,
        state: Mapping[str, Any],
        *,
        session_updated: bool = False,
    ) -> None:
        await self.bus.publish_outbound(
            outbound_message_for_event(
                channel="websocket",
                chat_id=chat_id,
                event=RecoveryStateEvent(
                    status=cast(str, state["status"]),
                    recovery_id=cast(str, state["recovery_id"]),
                    reason=cast(str | None, state.get("reason")),
                    message=cast(str | None, state.get("message")),
                    attempts=cast(int, state.get("attempts", 0)),
                ),
            )
        )
        if session_updated:
            await self.bus.publish_outbound(
                outbound_message_for_event(
                    channel="websocket",
                    chat_id=chat_id,
                    event=SessionUpdatedEvent(scope="thread"),
                )
            )

    @staticmethod
    def _set_state(
        session: Session,
        *,
        status: str,
        recovery_id: str,
        attempts: int,
        reason: str,
        message: str,
        resume_message_count: int | None = None,
    ) -> dict[str, Any]:
        state = {
            "status": status,
            "recovery_id": recovery_id,
            "attempts": max(0, attempts),
            "reason": reason,
            "message": message,
            "updated_at": datetime.now().isoformat(),
        }
        if resume_message_count is not None:
            state["resume_message_count"] = max(0, resume_message_count)
        session.metadata[RECOVERY_METADATA_KEY] = state
        session.updated_at = datetime.now()
        return state

    def _session_key(self, chat_id: str) -> str:
        return UNIFIED_SESSION_KEY if self.unified_session else f"websocket:{chat_id}"

    @staticmethod
    def _websocket_route(session: Session) -> tuple[str, str] | None:
        if session.key.startswith("websocket:"):
            chat_id = session.key.split(":", 1)[1]
            return ("websocket", chat_id) if chat_id else None
        if session.key == UNIFIED_SESSION_KEY:
            route = last_channel_from_metadata(session.metadata)
            if route and route[0] == "websocket":
                return route
        return None
