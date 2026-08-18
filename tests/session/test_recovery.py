from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.bus.outbound_events import RecoveryStateEvent, SessionUpdatedEvent
from nanobot.bus.queue import MessageBus
from nanobot.session.manager import Session, SessionManager
from nanobot.session.recovery import (
    PENDING_FOLLOWUPS_KEY,
    PENDING_USER_TURN_KEY,
    RECOVERY_METADATA_KEY,
    RUNTIME_CHECKPOINT_KEY,
    RecoveryActionError,
    RecoveryCoordinator,
    acknowledge_pending_followups,
    pending_followups,
    record_pending_followup,
)


def _persist(manager: SessionManager, session: Session) -> None:
    session.metadata["webui"] = True
    manager.save(session)


def _coordinator(workspace: Path) -> tuple[RecoveryCoordinator, MessageBus, SessionManager]:
    bus = MessageBus()
    sessions = SessionManager(workspace)
    return RecoveryCoordinator(sessions, bus), bus, sessions


@pytest.mark.asyncio
async def test_restart_before_model_call_auto_continues_once(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.messages.append({"role": "user", "content": "finish this"})
    session.metadata[PENDING_USER_TURN_KEY] = True
    _persist(sessions, session)

    coordinator, bus, restarted = _coordinator(tmp_path)
    await coordinator.scan()

    queued = bus.inbound.get_nowait()
    assert queued.chat_id == "chat"
    assert queued.metadata["_webui_recovery_id"]
    restored = restarted.get_or_create("websocket:chat")
    assert restored.metadata[PENDING_USER_TURN_KEY] is True
    assert restored.metadata[RECOVERY_METADATA_KEY]["status"] == "resuming"
    assert restored.metadata[RECOVERY_METADATA_KEY]["attempts"] == 1


@pytest.mark.asyncio
async def test_live_turn_followup_survives_restart_until_it_is_committed(tmp_path: Path) -> None:
    """A message injected mid-turn is not lost between checkpoints."""
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    _persist(sessions, session)
    followup_id = record_pending_followup(
        session,
        InboundMessage(
            channel="websocket",
            sender_id="user",
            chat_id="chat",
            content="also check the logs",
            metadata={"webui": True},
        ),
    )
    assert followup_id is not None
    sessions.save(session)

    coordinator, bus, restarted = _coordinator(tmp_path)
    await coordinator.scan()

    queued = bus.inbound.get_nowait()
    assert queued.content == "also check the logs"
    assert queued.metadata["_recovery_followup_id"] == followup_id
    restored = restarted.get_or_create("websocket:chat")
    assert len(pending_followups(restored)) == 1

    acknowledge_pending_followups(restored, [followup_id])
    assert PENDING_FOLLOWUPS_KEY not in restored.metadata


@pytest.mark.asyncio
async def test_completed_tools_resume_after_persisted_results(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.messages.append({"role": "user", "content": "inspect"})
    session.metadata[PENDING_USER_TURN_KEY] = True
    session.metadata[RUNTIME_CHECKPOINT_KEY] = {
        "phase": "tools_completed",
        "assistant_message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "function": {"name": "read_file"}}],
        },
        "completed_tool_results": [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "read_file",
                "content": "saved result",
            }
        ],
        "pending_tool_calls": [],
    }
    _persist(sessions, session)

    coordinator, bus, restarted = _coordinator(tmp_path)
    await coordinator.scan()

    assert bus.inbound.get_nowait().chat_id == "chat"
    restored = restarted.get_or_create("websocket:chat")
    assert restored.messages[-1]["content"] == "saved result"
    assert RUNTIME_CHECKPOINT_KEY not in restored.metadata


@pytest.mark.asyncio
async def test_uncertain_tool_is_never_replayed_automatically(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.messages.append({"role": "user", "content": "send it"})
    session.metadata[PENDING_USER_TURN_KEY] = True
    session.metadata[RUNTIME_CHECKPOINT_KEY] = {
        "phase": "awaiting_tools",
        "assistant_message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "function": {"name": "send_email"}}],
        },
        "completed_tool_results": [],
        "pending_tool_calls": [
            {"id": "call-1", "function": {"name": "send_email"}}
        ],
    }
    _persist(sessions, session)

    coordinator, bus, restarted = _coordinator(tmp_path)
    await coordinator.scan()

    assert bus.inbound.empty()
    restored = restarted.get_or_create("websocket:chat")
    assert restored.metadata[RECOVERY_METADATA_KEY]["status"] == "awaiting_user"
    assert restored.metadata[RECOVERY_METADATA_KEY]["reason"] == "tool_state_unknown"
    assert restored.messages[-1]["_recovery_interrupted"] is True
    event = bus.outbound.get_nowait().event
    assert isinstance(event, RecoveryStateEvent)
    assert event.status == "awaiting_user"


@pytest.mark.asyncio
async def test_unknown_checkpoint_never_auto_continues(tmp_path: Path) -> None:
    """Malformed or newer checkpoint phases fail closed pending confirmation."""
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.messages.append({"role": "user", "content": "deploy it"})
    session.metadata[PENDING_USER_TURN_KEY] = True
    session.metadata[RUNTIME_CHECKPOINT_KEY] = {"phase": "future_phase"}
    _persist(sessions, session)

    coordinator, bus, restarted = _coordinator(tmp_path)
    await coordinator.scan()

    assert bus.inbound.empty()
    restored = restarted.get_or_create("websocket:chat")
    assert restored.metadata[RECOVERY_METADATA_KEY]["status"] == "awaiting_user"
    assert restored.metadata[RECOVERY_METADATA_KEY]["reason"] == "checkpoint_unknown"


@pytest.mark.asyncio
async def test_final_answer_is_restored_without_model_call(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.messages.append({"role": "user", "content": "answer"})
    session.metadata[PENDING_USER_TURN_KEY] = True
    session.metadata[RUNTIME_CHECKPOINT_KEY] = {
        "phase": "final_response",
        "assistant_message": {"role": "assistant", "content": "already finished"},
        "completed_tool_results": [],
        "pending_tool_calls": [],
    }
    _persist(sessions, session)

    coordinator, bus, restarted = _coordinator(tmp_path)
    await coordinator.scan()

    assert bus.inbound.empty()
    restored = restarted.get_or_create("websocket:chat")
    assert restored.messages[-1]["content"] == "already finished"
    first = bus.outbound.get_nowait().event
    second = bus.outbound.get_nowait().event
    assert isinstance(first, RecoveryStateEvent) and first.status == "recovered"
    assert isinstance(second, SessionUpdatedEvent)


@pytest.mark.asyncio
async def test_repeated_restart_enters_loop_guard(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.messages.append({"role": "user", "content": "continue"})
    session.metadata[PENDING_USER_TURN_KEY] = True
    _persist(sessions, session)

    first, first_bus, _ = _coordinator(tmp_path)
    await first.scan()
    assert first_bus.inbound.qsize() == 1

    second, second_bus, restarted = _coordinator(tmp_path)
    await second.scan()

    assert second_bus.inbound.empty()
    state = restarted.get_or_create("websocket:chat").metadata[RECOVERY_METADATA_KEY]
    assert state["status"] == "awaiting_user"
    assert state["reason"] == "loop_guard"


@pytest.mark.asyncio
async def test_new_user_message_supersedes_queued_recovery(tmp_path: Path) -> None:
    coordinator, bus, sessions = _coordinator(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.messages.append({"role": "user", "content": "old request"})
    session.metadata[PENDING_USER_TURN_KEY] = True
    _persist(sessions, session)
    await coordinator.scan()
    queued_recovery = bus.inbound.get_nowait()

    newer = InboundMessage(
        channel="websocket",
        sender_id="user",
        chat_id="chat",
        content="new request",
    )
    assert await coordinator.admit(newer) is True
    assert await coordinator.admit(queued_recovery) is False
    restored = sessions.get_or_create("websocket:chat")
    assert restored.messages[-1]["_recovery_interrupted"] is True
    assert restored.metadata[RECOVERY_METADATA_KEY]["reason"] == "superseded"


@pytest.mark.asyncio
async def test_new_user_message_cancels_active_recovery_task(tmp_path: Path) -> None:
    coordinator, bus, sessions = _coordinator(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.metadata[RECOVERY_METADATA_KEY] = {
        "status": "resuming",
        "recovery_id": "active",
        "attempts": 1,
    }
    _persist(sessions, session)

    started = asyncio.Event()

    async def _active_recovery() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(_active_recovery())
    await started.wait()
    coordinator.register_recovery_task("websocket:chat", task)

    newer = InboundMessage(
        channel="websocket",
        sender_id="user",
        chat_id="chat",
        content="new request",
    )
    assert await coordinator.admit(newer) is True
    assert task.cancelled()
    restored = sessions.get_or_create("websocket:chat")
    assert restored.metadata[RECOVERY_METADATA_KEY]["reason"] == "superseded"
    assert restored.messages[-1]["_recovery_interrupted"] is True
    assert bus.outbound.qsize() == 1


@pytest.mark.asyncio
async def test_recovery_action_rejects_stale_page_and_continues_current_state(
    tmp_path: Path,
) -> None:
    coordinator, bus, sessions = _coordinator(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.metadata[RECOVERY_METADATA_KEY] = {
        "status": "awaiting_user",
        "recovery_id": "current",
        "attempts": 0,
    }
    _persist(sessions, session)

    with pytest.raises(RecoveryActionError, match="stale"):
        await coordinator.handle_action(
            "continue",
            {"chat_id": "chat", "recovery_id": "old"},
        )

    result = await coordinator.handle_action(
        "continue",
        {"chat_id": "chat", "recovery_id": "current"},
    )
    assert result["status"] == "resuming"
    assert bus.inbound.get_nowait().metadata["_webui_recovery_id"] == "current"


@pytest.mark.asyncio
async def test_persisted_completion_wins_over_stale_resuming_marker(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.messages.extend(
        [
            {"role": "user", "content": "work"},
            {"role": "assistant", "content": "done"},
        ]
    )
    session.metadata[RECOVERY_METADATA_KEY] = {
        "status": "resuming",
        "recovery_id": "recovery",
        "attempts": 1,
        "resume_message_count": 1,
    }
    _persist(sessions, session)

    coordinator, bus, restarted = _coordinator(tmp_path)
    await coordinator.scan()

    assert bus.inbound.empty()
    state = restarted.get_or_create("websocket:chat").metadata[RECOVERY_METADATA_KEY]
    assert state["status"] == "recovered"
    assert state["reason"] == "committed"


@pytest.mark.asyncio
async def test_dismiss_does_not_queue_work(tmp_path: Path) -> None:
    coordinator, bus, sessions = _coordinator(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.metadata[RECOVERY_METADATA_KEY] = {
        "status": "awaiting_user",
        "recovery_id": "current",
        "attempts": 0,
    }
    _persist(sessions, session)

    result = await coordinator.handle_action(
        "dismiss",
        {"chat_id": "chat", "recovery_id": "current"},
    )

    assert result["status"] == "recovered"
    assert bus.inbound.empty()


@pytest.mark.asyncio
async def test_scan_failure_is_visible_instead_of_aborting_other_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, bus, sessions = _coordinator(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.metadata[PENDING_USER_TURN_KEY] = True
    _persist(sessions, session)

    async def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(RecoveryCoordinator, "_recover_session", fail)
    await coordinator.scan()

    state = sessions.get_or_create("websocket:chat").metadata[RECOVERY_METADATA_KEY]
    assert state["status"] == "failed"
    assert isinstance(bus.outbound.get_nowait().event, RecoveryStateEvent)


@pytest.mark.asyncio
async def test_bus_remains_quiet_after_recovered_state(tmp_path: Path) -> None:
    coordinator, bus, sessions = _coordinator(tmp_path)
    session = sessions.get_or_create("websocket:chat")
    session.metadata[RECOVERY_METADATA_KEY] = {
        "status": "recovered",
        "recovery_id": "done",
        "attempts": 1,
    }
    _persist(sessions, session)

    await coordinator.scan()

    await asyncio.sleep(0)
    assert bus.inbound.empty()
    assert bus.outbound.empty()
