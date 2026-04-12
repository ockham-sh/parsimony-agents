"""Terminal chat: interactive conversation with the agent.

A REPL-style loop that lets you ask questions, see streamed responses
with tool progress, and maintain multi-turn context across messages.

Prerequisites::

    pip install parsimony-agents[display]
    export FRED_API_KEY="..."               # free: https://fred.stlouisfed.org/docs/api/api_key.html
    export FMP_API_KEY="..."                # https://site.financialmodelingprep.com/developer/docs

Run::

    python -m parsimony_agents.examples.terminal_chat

Type "exit" or press Ctrl+D to quit. Press Ctrl+C to cancel a response.

For the single-shot display, see ``quickstart.py``.
For raw event access, see ``event_stream.py``.
"""

from __future__ import annotations

import asyncio
import signal
import sys

from parsimony.connectors import build_connectors_from_env

from parsimony_agents import Agent, stream_to_display

try:
    from rich.console import Console

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ANSI escape to move cursor up one line and clear it
_CLEAR_LINE = "\033[A\033[2K"


async def main() -> None:
    try:
        connectors = build_connectors_from_env()
    except ValueError as e:
        print(f"Missing environment variable: {e}")
        return

    agent = Agent(
        model="gemini/gemini-3-flash-preview",
        connectors=connectors,
    )

    console = Console(width=100, highlight=False) if HAS_RICH else None

    if console:
        console.print()
        console.print("  [bold]Parsimony Agent[/] — Terminal Chat")
        console.print("  [dim]Type your question and press Enter. 'exit' or Ctrl+D to quit.[/]")
    else:
        print("\nParsimony Agent — Terminal Chat")
        print("Type your question and press Enter. 'exit' or Ctrl+D to quit.")

    loop = asyncio.get_running_loop()
    ctx = None

    while True:
        try:
            print()
            if console:
                console.print(
                    "  [bold bright_blue]❯[/] ",
                    end="",
                )
                user_input = input("").strip()
            else:
                user_input = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        # Clear the input line — the banner will show the question
        sys.stdout.write(_CLEAR_LINE)
        sys.stdout.flush()

        task = asyncio.create_task(
            stream_to_display(agent, user_input, ctx=ctx, console=console)
        )
        loop.add_signal_handler(signal.SIGINT, task.cancel)

        try:
            result = await task
            ctx = result.context
        except asyncio.CancelledError:
            print("\n[Cancelled]\n")
        finally:
            loop.remove_signal_handler(signal.SIGINT)


if __name__ == "__main__":
    asyncio.run(main())
