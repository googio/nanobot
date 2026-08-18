import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { RecoveryNotice } from "@/components/thread/RecoveryNotice";

const INTERRUPTED = {
  status: "awaiting_user" as const,
  recovery_id: "recovery-1",
  reason: "tool_state_uncertain",
};

describe("RecoveryNotice", () => {
  it("hides the internal resuming state after Continue is accepted", async () => {
    const onContinue = vi.fn().mockResolvedValue(undefined);

    render(
      <RecoveryNotice
        state={INTERRUPTED}
        onContinue={onContinue}
        onDismiss={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(onContinue).toHaveBeenCalledOnce();
  });

  it("keeps the notice visible when Continue fails", async () => {
    const onContinue = vi.fn().mockRejectedValue(new Error("offline"));

    render(
      <RecoveryNotice
        state={INTERRUPTED}
        onContinue={onContinue}
        onDismiss={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Recovery action failed");
    });
  });
});
