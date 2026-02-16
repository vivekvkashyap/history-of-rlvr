"""
Tmux launcher for GRPO training with split panes.

Creates a tmux session with three panes:
  - Pane 0 (top):    Inference logs  (tail -F logs/inference.log)
  - Pane 1 (middle): Trainer logs    (tail -F logs/trainer.log)
  - Pane 2 (bottom): Main process    (python -m experiments.sync_rl.main ...)

Usage:
    python -m experiments.sync_rl.launch [training args...]

Example:
    python -m experiments.sync_rl.launch --use_lora true --lora_rank 16 --report_to wandb

Tmux controls:
    Ctrl+B  ↑/↓    switch between panes
    Ctrl+B  [      enter scroll mode (q to exit)
    Ctrl+B  d      detach (training keeps running)
    tmux attach -t grpo-rl   re-attach
"""

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    """Run a command, raise on failure."""
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def tmux_exists() -> bool:
    """Check if tmux is installed."""
    try:
        subprocess.run(
            ["tmux", "-V"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def session_exists(session: str) -> bool:
    proc = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def find_available_session(base: str) -> str:
    """Find a session name that doesn't conflict."""
    if not session_exists(base):
        return base
    idx = 2
    while True:
        candidate = f"{base}-{idx}"
        if not session_exists(candidate):
            return candidate
        idx += 1


def main() -> None:
    if not tmux_exists():
        raise SystemExit(
            "tmux is not installed.  Install with: sudo apt install tmux"
        )

    # All CLI args are forwarded to python -m experiments.sync_rl.main
    forwarded_args = sys.argv[1:]

    # Determine output_dir from args (look for --output_dir flag)
    output_dir = "outputs/grpo"
    for i, arg in enumerate(forwarded_args):
        if arg == "--output_dir" and i + 1 < len(forwarded_args):
            output_dir = forwarded_args[i + 1]
            break

    log_dir = os.path.join(output_dir, "logs")
    cwd = Path.cwd()

    # Create log directory and truncate old log files so panes start clean
    os.makedirs(log_dir, exist_ok=True)
    inf_log = os.path.join(log_dir, "inference.log")
    tr_log = os.path.join(log_dir, "trainer.log")
    for p in (inf_log, tr_log):
        open(p, "w").close()  # truncate any previous content

    session = find_available_session("grpo-rl")

    # ── Create tmux session ────────────────────────────────────────────
    # Start detached session with bash
    run(["tmux", "new-session", "-d", "-s", session, "-n", "RL", "-c", str(cwd)])

    # Split into 3 panes (top / middle / bottom)
    run(["tmux", "split-window", "-v", "-t", f"{session}:RL.0", "-c", str(cwd)])
    run(["tmux", "split-window", "-v", "-t", f"{session}:RL.1", "-c", str(cwd)])
    run(["tmux", "select-layout", "-t", f"{session}:RL", "even-vertical"])

    # ── Pane titles ────────────────────────────────────────────────────
    run(["tmux", "select-pane", "-t", f"{session}:RL.0", "-T", "Inference"])
    run(["tmux", "select-pane", "-t", f"{session}:RL.1", "-T", "Trainer"])
    run(["tmux", "select-pane", "-t", f"{session}:RL.2", "-T", "Main"])

    # ── Pane border styling ────────────────────────────────────────────
    run(["tmux", "set-option", "-t", session, "-g", "pane-border-status", "top"])
    run([
        "tmux", "set-option", "-t", session, "-g",
        "pane-border-format", " #{pane_title} ",
    ])
    run([
        "tmux", "set-window-option", "-t", f"{session}:RL",
        "pane-border-status", "top",
    ])

    # ── Send commands to each pane ─────────────────────────────────────

    # Pane 0 (top) – Inference log
    # -n 0: don't show existing lines; -F: follow even across truncation
    inf_tail_cmd = (
        f'while [ ! -s {inf_log} ]; do sleep 0.5; done; '
        f'clear; tail -n 0 -F {inf_log}'
    )
    run(["tmux", "send-keys", "-t", f"{session}:RL.0", inf_tail_cmd, "C-m"])

    # Pane 1 (middle) – Trainer log
    tr_tail_cmd = (
        f'while [ ! -s {tr_log} ]; do sleep 0.5; done; '
        f'clear; tail -n 0 -F {tr_log}'
    )
    run(["tmux", "send-keys", "-t", f"{session}:RL.1", tr_tail_cmd, "C-m"])

    # Pane 2 (bottom) – Main training process
    # Build the training command with env vars forwarded
    args_str = " ".join(forwarded_args)
    python = sys.executable  # respect virtualenv

    # Forward CUDA env vars so the tmux pane has them
    env_prefix_parts = []
    for var in ("CUDA_HOME", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(var)
        if val is not None:
            env_prefix_parts.append(f"{var}={val}")
    env_prefix = " ".join(env_prefix_parts)
    if env_prefix:
        env_prefix += " "

    train_cmd = f"{env_prefix}{python} -m experiments.sync_rl.main {args_str}"
    run(["tmux", "send-keys", "-t", f"{session}:RL.2", train_cmd, "C-m"])

    # Focus the Trainer pane by default
    run(["tmux", "select-pane", "-t", f"{session}:RL.1"])

    # ── Attach ─────────────────────────────────────────────────────────
    if sys.stdout.isatty():
        print(f"Attaching to tmux session '{session}'...")
        print("  Ctrl+B ↑/↓  switch panes")
        print("  Ctrl+B [    scroll mode (q to exit)")
        print("  Ctrl+B d    detach (training continues)")
        os.execvp("tmux", ["tmux", "attach-session", "-t", session])
    else:
        print(f"tmux session '{session}' created (non-interactive, not attaching).")
        print(f"  Attach with: tmux attach -t {session}")


if __name__ == "__main__":
    main()
