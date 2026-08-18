import type { UIMessage } from "@/lib/types";

export type TurnUnit =
  | {
      type: "activity";
      messages: UIMessage[];
      turnLatencyMs?: number;
      startedAtMs?: number;
    }
  | { type: "message"; message: UIMessage };

interface NormalizeActivityTimelineOptions {
  preserveTrailingActivity?: boolean;
}

export function isReasoningOnlyAssistant(message: UIMessage): boolean {
  if (message.role !== "assistant" || message.kind === "trace") return false;
  if (message.content.trim().length > 0) return false;
  return !!(message.reasoning?.length || message.reasoningStreaming || message.isStreaming);
}

export function isAgentActivityMember(message: UIMessage): boolean {
  return isReasoningOnlyAssistant(message) || message.kind === "trace";
}

export function hasPendingAgentActivity(messages: UIMessage[]): boolean {
  if (messages.length === 0) return false;
  const last = messages[messages.length - 1];
  if (!isAgentActivityMember(last)) return false;

  let trailingStart = messages.length - 1;
  while (
    trailingStart > 0
    && isAgentActivityMember(messages[trailingStart - 1])
  ) {
    trailingStart -= 1;
  }

  const trailing = messages.slice(trailingStart);
  if (trailing.some((message) => message.isStreaming || message.reasoningStreaming)) {
    return true;
  }

  const previous = messages[trailingStart - 1];
  if (!previous || previous.role !== "assistant" || isAgentActivityMember(previous)) {
    return true;
  }

  const trailingTurnIds = new Set(
    trailing
      .map((message) => message.turnId)
      .filter((turnId): turnId is string => typeof turnId === "string" && turnId.length > 0),
  );
  if (!previous.turnId) return trailingTurnIds.size > 0;
  return trailingTurnIds.size > 0 && !trailingTurnIds.has(previous.turnId);
}

export function normalizeActivityTimeline(
  messages: UIMessage[],
  options: NormalizeActivityTimelineOptions = {},
): TurnUnit[] {
  const units: TurnUnit[] = [];
  let turnMessages: UIMessage[] = [];
  let activeTurnId: string | undefined;
  let activeTurnStartedAtMs: number | undefined;

  const flushTurn = (flushOptions: NormalizeActivityTimelineOptions = {}) => {
    if (turnMessages.length === 0) {
      activeTurnId = undefined;
      return;
    }

    const turnUnits: TurnUnit[] = [];
    const turnStartedAtMs = activeTurnStartedAtMs;
    const orderedTurnMessages = orderMessagesByTurnSeq(turnMessages);
    const visibleMessages = visibleMessagesForTurn(orderedTurnMessages);
    let visibleIndex = 0;
    let activityMessages: UIMessage[] = [];

    const flushActivityMessages = () => {
      if (!activityMessages.length) return;
      pushActivityUnits(
        turnUnits,
        activityMessages,
        visibleMessages.slice(visibleIndex),
        turnStartedAtMs,
      );
      activityMessages = [];
    };

    for (const message of orderedTurnMessages) {
      if (isAgentActivityMember(message)) {
        activityMessages.push(message);
        continue;
      }

      if (assistantHasInlineReasoning(message)) {
        activityMessages.push(reasoningOnlyMessageFromAnswer(message));
        flushActivityMessages();
        turnUnits.push({ type: "message", message: stripInlineReasoning(message) });
        visibleIndex += 1;
        continue;
      }

      flushActivityMessages();
      turnUnits.push({ type: "message", message });
      visibleIndex += 1;
    }

    flushActivityMessages();
    units.push(...normalizeCompletedTurnUnits(consolidateAssistantAnswers(turnUnits), flushOptions));
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

    if (message.turnId && activeTurnId && message.turnId !== activeTurnId) {
      flushTurn();
    }
    if (message.turnId) {
      activeTurnId = message.turnId;
    }
    turnMessages.push(message);
  }

  flushTurn(options);
  return units;
}

/** Keep one visible assistant answer per user turn.
 *
 * Providers may emit text before and after tool calls.  The protocol keeps
 * those segments separate for auditability, while the conversation view
 * projects them into one answer so timestamps and actions are not repeated.
 */
function consolidateAssistantAnswers(turnUnits: TurnUnit[]): TurnUnit[] {
  const answerIndices: number[] = [];
  for (let index = 0; index < turnUnits.length; index += 1) {
    const unit = turnUnits[index];
    if (unit.type === "message" && unit.message.role === "assistant") {
      answerIndices.push(index);
    }
  }
  if (answerIndices.length < 2) return turnUnits;

  const answers = answerIndices.map((index) => {
    const unit = turnUnits[index];
    return unit.type === "message" ? unit.message : null;
  }).filter((message): message is UIMessage => message !== null);
  const first = answers[0];
  const last = answers[answers.length - 1];
  const content = answers
    .map((message) => message.content.trim())
    .filter(Boolean)
    .join("\n\n");
  const media = answers.flatMap((message) => message.media ?? []);
  const images = answers.flatMap((message) => message.images ?? []);
  const merged: UIMessage = {
    ...first,
    ...last,
    id: first.id,
    content,
    createdAt: first.createdAt,
    isStreaming: answers.some((message) => message.isStreaming),
  };
  if (media.length) merged.media = media;
  else delete merged.media;
  if (images.length) merged.images = images;
  else delete merged.images;

  const insertAt = answerIndices[answerIndices.length - 1];
  const answerIndexSet = new Set(answerIndices);
  return turnUnits.flatMap((unit, index) => {
    if (index === insertAt) return [{ type: "message", message: merged } as TurnUnit];
    return answerIndexSet.has(index) ? [] : [unit];
  });
}

function orderMessagesByTurnSeq(messages: UIMessage[]): UIMessage[] {
  if (
    messages.length < 2
    || !messages.every((message) => Number.isFinite(message.turnSeq))
  ) {
    return messages;
  }
  return messages
    .map((message, index) => ({ message, index }))
    .sort((left, right) => {
      const bySeq = (left.message.turnSeq ?? 0) - (right.message.turnSeq ?? 0);
      return bySeq || left.index - right.index;
    })
    .map(({ message }) => message);
}

function normalizeCompletedTurnUnits(
  turnUnits: TurnUnit[],
  options: NormalizeActivityTimelineOptions,
): TurnUnit[] {
  if (options.preserveTrailingActivity || turnUnits.length < 2) return turnUnits;
  if (turnUnits[turnUnits.length - 1]?.type !== "activity") return turnUnits;

  let trailingStart = turnUnits.length - 1;
  while (trailingStart > 0 && turnUnits[trailingStart - 1]?.type === "activity") {
    trailingStart -= 1;
  }

  const previous = turnUnits[trailingStart - 1];
  if (
    !previous
    || previous.type !== "message"
    || previous.message.role !== "assistant"
  ) {
    return turnUnits;
  }

  return [
    ...turnUnits.slice(0, trailingStart - 1),
    ...turnUnits.slice(trailingStart),
    previous,
  ];
}

function visibleMessagesForTurn(messages: UIMessage[]): UIMessage[] {
  const visibleMessages: UIMessage[] = [];
  for (const message of messages) {
    if (isAgentActivityMember(message)) continue;
    visibleMessages.push(assistantHasInlineReasoning(message) ? stripInlineReasoning(message) : message);
  }
  return visibleMessages;
}

function validCreatedAtMs(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function pushActivityUnits(
  units: TurnUnit[],
  activityMessages: UIMessage[],
  visibleMessages: UIMessage[],
  startedAtMs?: number,
) {
  let runMessages: UIMessage[] = [];
  let runBucket: "file" | "other" | undefined;
  let runSegmentId: string | undefined;

  const flushRun = () => {
    if (!runMessages.length) return;
    units.push({
      type: "activity",
      messages: runMessages,
      turnLatencyMs: activityTurnLatencyMs(runMessages, visibleMessages),
      startedAtMs,
    });
    runMessages = [];
    runBucket = undefined;
    runSegmentId = undefined;
  };

  for (const message of activityMessages) {
    const bucket = isFileEditActivityMessage(message) ? "file" : "other";
    const segmentId = message.activitySegmentId;
    const segmentChanged =
      bucket === "file"
      && runBucket === "file"
      && !!runSegmentId
      && !!segmentId
      && runSegmentId !== segmentId;
    if ((runBucket && bucket !== runBucket) || segmentChanged) {
      flushRun();
    }
    runBucket = bucket;
    if (segmentId) runSegmentId = segmentId;
    runMessages.push(message);
  }

  flushRun();
}

function isFileEditActivityMessage(message: UIMessage): boolean {
  return message.kind === "trace" && !!message.fileEdits?.length;
}

function assistantHasInlineReasoning(message: UIMessage): boolean {
  return (
    message.role === "assistant"
    && message.kind !== "trace"
    && message.content.trim().length > 0
    && (!!message.reasoning?.trim() || !!message.reasoningStreaming)
  );
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

function activityTurnLatencyMs(activityMessages: UIMessage[], visibleMessages: UIMessage[]): number | undefined {
  for (let i = visibleMessages.length - 1; i >= 0; i -= 1) {
    const latency = visibleMessages[i].latencyMs;
    if (isValidLatency(latency)) return latency;
  }
  for (let i = activityMessages.length - 1; i >= 0; i -= 1) {
    const latency = activityMessages[i].latencyMs;
    if (isValidLatency(latency)) return latency;
  }
  return undefined;
}

function isValidLatency(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}
