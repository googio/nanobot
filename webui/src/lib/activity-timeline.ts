import type { UIMessage } from "@/lib/types";

/** The conversation projection deliberately has only two surfaces per turn:
 * one activity container and one final answer. The transport may emit many
 * rows, but callers never need to know how those rows are grouped. */
export type TurnUnit =
  | {
      type: "activity";
      messages: UIMessage[];
      turnLatencyMs?: number;
      startedAtMs?: number;
    }
  | { type: "message"; message: UIMessage };

export function isReasoningOnlyAssistant(message: UIMessage): boolean {
  if (message.role !== "assistant" || message.kind === "trace") return false;
  if (message.activityKind === "model" || message.content.trim().length > 0) return false;
  return !!(message.reasoning?.length || message.reasoningStreaming || message.isStreaming);
}

export function isAgentActivityMember(message: UIMessage): boolean {
  return isReasoningOnlyAssistant(message) || message.kind === "trace" || message.activityKind === "model";
}

export function hasPendingAgentActivity(messages: UIMessage[]): boolean {
  const last = messages.at(-1);
  if (!last || !isAgentActivityMember(last)) return false;
  if (last.isStreaming || last.reasoningStreaming) return true;

  const lastTurnId = last.turnId;
  const previous = messages.at(-2);
  // A trace without a visible answer is an unfinished turn on replay. Once a
  // final assistant answer exists after it, the activity is simply history.
  return !previous
    || previous.role !== "assistant"
    || isAgentActivityMember(previous)
    || previous.turnId !== lastTurnId;
}

/**
 * Project raw gateway rows into the stable conversation shape:
 *
 *   user → [one live/completed activity surface] → [one final answer]
 *
 * Assistant text that is followed by another activity is a model snippet. It
 * remains visible in the activity timeline, but never creates a second answer
 * bubble. This is the same causal model used by Codex-style transcripts.
 */
export function normalizeActivityTimeline(
  messages: UIMessage[],
): TurnUnit[] {
  const units: TurnUnit[] = [];
  let turnMessages: UIMessage[] = [];
  let activeTurnId: string | undefined;
  let activeTurnStartedAtMs: number | undefined;

  const flushTurn = () => {
    if (!turnMessages.length) {
      activeTurnId = undefined;
      activeTurnStartedAtMs = undefined;
      return;
    }

    const ordered = orderMessagesByTurnSeq(turnMessages);
    const lastActivityIndex = ordered.reduce(
      (index, message, current) => isRawActivity(message) ? current : index,
      -1,
    );
    const answerIndices = ordered
      .map((message, index) => ({ message, index }))
      .filter(({ message }) => isAssistantAnswer(message));
    const finalAnswerIndex = answerIndices.at(-1)?.index;
    // A replay can deliver a completed answer before a late trace row. Keep
    // that answer visible, but place the late activity in the single activity
    // surface before it. A still-streaming answer followed by activity is an
    // intermediate model segment and stays compact inside that surface.
    const hasFinalAnswer = finalAnswerIndex !== undefined
      && (finalAnswerIndex > lastActivityIndex || ordered[finalAnswerIndex].isStreaming !== true);

    const activity: UIMessage[] = [];
    const answers: UIMessage[] = [];
    ordered.forEach((message, index) => {
      if (isRawActivity(message)) {
        activity.push(message);
      } else if (isAssistantAnswer(message)) {
        if (message.reasoning?.trim() || message.reasoningStreaming) {
          activity.push(reasoningOnlyMessageFromAnswer(message));
        }
        if (hasFinalAnswer && index === finalAnswerIndex) {
          answers.push(stripInlineReasoning(message));
        } else {
          activity.push(modelActivitySnippet(message));
        }
      } else {
        activity.push(message);
      }
    });

    if (activity.length) {
      units.push({
        type: "activity",
        messages: activity,
        turnLatencyMs: activityTurnLatencyMs(activity, ordered),
        startedAtMs: activeTurnStartedAtMs,
      });
    }
    if (answers.length) {
      units.push({ type: "message", message: mergeAssistantAnswers(answers) });
    }

    turnMessages = [];
    activeTurnId = undefined;
    activeTurnStartedAtMs = undefined;
  };

  for (const message of messages) {
    if (message.role === "user") {
      flushTurn();
      units.push({ type: "message", message });
      activeTurnId = message.turnId;
      activeTurnStartedAtMs = validCreatedAtMs(message.createdAt);
      continue;
    }
    if (message.turnId && activeTurnId && message.turnId !== activeTurnId) flushTurn();
    if (message.turnId) activeTurnId = message.turnId;
    turnMessages.push(message);
  }

  flushTurn();
  return units;
}

function isRawActivity(message: UIMessage): boolean {
  return isAgentActivityMember(message);
}

function isAssistantAnswer(message: UIMessage): boolean {
  return message.role === "assistant" && message.kind !== "trace" && message.content.trim().length > 0;
}

function orderMessagesByTurnSeq(messages: UIMessage[]): UIMessage[] {
  if (messages.length < 2 || !messages.every((message) => Number.isFinite(message.turnSeq))) {
    return messages;
  }
  return messages
    .map((message, index) => ({ message, index }))
    .sort((left, right) => (left.message.turnSeq! - right.message.turnSeq!) || (left.index - right.index))
    .map(({ message }) => message);
}

function mergeAssistantAnswers(answers: UIMessage[]): UIMessage {
  const first = answers[0];
  const last = answers.at(-1)!;
  const media = answers.flatMap((message) => message.media ?? []);
  const images = answers.flatMap((message) => message.images ?? []);
  const merged: UIMessage = {
    ...first,
    ...last,
    id: first.id,
    content: answers.map((message) => message.content.trim()).filter(Boolean).join("\n\n"),
    createdAt: first.createdAt,
    isStreaming: answers.some((message) => message.isStreaming),
  };
  if (media.length) merged.media = media;
  else delete merged.media;
  if (images.length) merged.images = images;
  else delete merged.images;
  return merged;
}

function modelActivitySnippet(message: UIMessage): UIMessage {
  return {
    ...stripInlineReasoning(message),
    id: `${message.id}-activity`,
    activityKind: "model",
    turnPhase: "activity",
    isStreaming: false,
  };
}

function reasoningOnlyMessageFromAnswer(message: UIMessage): UIMessage {
  return {
    id: `${message.id}-reasoning`,
    role: "assistant",
    content: "",
    createdAt: message.createdAt,
    reasoning: message.reasoning,
    reasoningStreaming: message.reasoningStreaming,
    isStreaming: message.reasoningStreaming,
    activitySegmentId: message.activitySegmentId,
    latencyMs: message.latencyMs,
    turnId: message.turnId,
    turnPhase: "reasoning",
    turnSeq: message.turnSeq,
  };
}

function stripInlineReasoning(message: UIMessage): UIMessage {
  const next = { ...message };
  delete next.reasoning;
  delete next.reasoningStreaming;
  return next;
}

function validCreatedAtMs(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function activityTurnLatencyMs(activityMessages: UIMessage[], allMessages: UIMessage[]): number | undefined {
  for (let index = allMessages.length - 1; index >= 0; index -= 1) {
    const latency = allMessages[index].latencyMs;
    if (isValidLatency(latency)) return latency;
  }
  for (let index = activityMessages.length - 1; index >= 0; index -= 1) {
    const latency = activityMessages[index].latencyMs;
    if (isValidLatency(latency)) return latency;
  }
  return undefined;
}

function isValidLatency(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}
