"""Termination exception: parallel to :class:`SuspensionRequest`.

The two control-flow exceptions raised by system tools to exit the loop cleanly:

- :class:`~parsimony_agents.agent.failure.suspension.SuspensionRequest` —
  agent wants to pause for user input (run is *suspended*, may resume).
- :class:`TerminationRequest` — agent has decided it cannot finish (run *terminates*
  with a :class:`~parsimony_agents.agent.events.Handoff` event).

Both are caught at the tool-execution boundary in the new loop. Cancellation
(``asyncio.CancelledError``) wins over either — if a cancel fires during tool
execution, the suspension/termination is suppressed and ``RunCancelled`` is emitted.
"""

from __future__ import annotations


class TerminationRequest(Exception):
    """Raised by the ``return_unable`` system tool to terminate the run.

    Carries the structured blockers + rationale the agent has produced. The loop's
    tool-execution phase catches this, yields :class:`~parsimony_agents.agent.events.Handoff`,
    sets ``state.done = True``, and exits.
    """

    def __init__(self, *, blockers: list[str], rationale: str):
        if not blockers:
            raise ValueError("return_unable requires a non-empty blockers list")
        if not rationale:
            raise ValueError("return_unable requires a non-empty rationale")
        super().__init__(rationale)
        self.blockers = list(blockers)
        self.rationale = rationale


__all__ = ["TerminationRequest"]
