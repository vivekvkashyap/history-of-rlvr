"""
Log router for split-pane tmux display.

Routes inference and trainer log messages to separate log files so that
tmux panes running ``tail -F`` on each file show a clean, dedicated view.

Uses ``rich.console.Console(file=...)`` to write Rich-markup output
(colors, bold, etc.) that renders properly in the tmux panes.
"""

import os
import time
from typing import Optional

from rich.console import Console


def _ts() -> str:
    """Return a short HH:MM:SS timestamp."""
    return time.strftime("%H:%M:%S")


class LogRouter:
    """
    Writes timestamped, Rich-formatted log lines to two separate files:
      - ``{log_dir}/inference.log``  (vLLM generation, weight sync)
      - ``{log_dir}/trainer.log``    (loss, reward, training metrics)

    The files are created/truncated on ``start()`` and flushed after
    every write so that ``tail -F`` picks up lines immediately.
    """

    def __init__(self, log_dir: str = "outputs/grpo/logs"):
        self.log_dir = log_dir
        self._inf_file = None
        self._tr_file = None
        self._inf_console: Optional[Console] = None
        self._tr_console: Optional[Console] = None

    # ════════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ════════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """Create log directory and open log files."""
        os.makedirs(self.log_dir, exist_ok=True)

        inf_path = os.path.join(self.log_dir, "inference.log")
        tr_path = os.path.join(self.log_dir, "trainer.log")

        # Truncate files so tail -F starts fresh
        self._inf_file = open(inf_path, "w", buffering=1)  # line-buffered
        self._tr_file = open(tr_path, "w", buffering=1)

        # Rich consoles that write colored output to the files
        self._inf_console = Console(
            file=self._inf_file, force_terminal=True, width=120,
        )
        self._tr_console = Console(
            file=self._tr_file, force_terminal=True, width=120,
        )

        self._inf_console.print(
            f"[bold cyan]═══ Inference Log Started ({_ts()}) ═══[/bold cyan]\n"
        )
        self._tr_console.print(
            f"[bold green]═══ Trainer Log Started ({_ts()}) ═══[/bold green]\n"
        )

    def stop(self) -> None:
        """Close log files."""
        if self._inf_console:
            self._inf_console.print(
                f"\n[bold cyan]═══ Inference Log Ended ({_ts()}) ═══[/bold cyan]"
            )
        if self._tr_console:
            self._tr_console.print(
                f"\n[bold green]═══ Trainer Log Ended ({_ts()}) ═══[/bold green]"
            )
        if self._inf_file:
            self._inf_file.close()
            self._inf_file = None
        if self._tr_file:
            self._tr_file.close()
            self._tr_file = None
        self._inf_console = None
        self._tr_console = None

    # ════════════════════════════════════════════════════════════════════
    #  Public API
    # ════════════════════════════════════════════════════════════════════

    def log_inference(self, message: str) -> None:
        """Write a timestamped line to inference.log (supports Rich markup)."""
        if self._inf_console:
            self._inf_console.print(f"[dim]{_ts()}[/dim]  {message}")

    def log_trainer(self, message: str) -> None:
        """Write a timestamped line to trainer.log (supports Rich markup)."""
        if self._tr_console:
            self._tr_console.print(f"[dim]{_ts()}[/dim]  {message}")
