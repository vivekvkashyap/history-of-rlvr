"""
Tmux launcher for async GRPO training with split panes.

Creates a tmux session with three panes stacked vertically:
  - Pane 0 (top):    Inference  (vLLM server + inference log)
  - Pane 1 (middle): Main       (orchestrator / training process)
  - Pane 2 (bottom): Trainer    (trainer log)

The launcher starts the vLLM server first and waits for it to become
healthy before launching the training process.

Usage:
    python -m rl.async_rl.launch [training args...]

Example:
    python -m rl.async_rl.launch --use_lora true --lora_rank 16 --report_to wandb

    # Override server GPU and port:
    python -m rl.async_rl.launch --vllm_server_port 8001

Tmux controls:
    Ctrl+B  ↑/↓    switch between panes
    Ctrl+B  [       enter scroll mode (q to exit)
    Ctrl+B  d       detach (training keeps running)
    tmux attach -t async-grpo-rl   re-attach
"""

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    """Run a command, raise on failure."""
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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

    # All CLI args are forwarded to python -m rl.async_rl.main
    forwarded_args = sys.argv[1:]

    # ── Extract key params from args ───────────────────────────────────
    output_dir = "outputs/async_grpo"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    vllm_port = "8000"
    vllm_host = "0.0.0.0"
    vllm_gpu_id = "1"  # default: inference on GPU 1

    for i, arg in enumerate(forwarded_args):
        if arg == "--output_dir" and i + 1 < len(forwarded_args):
            output_dir = forwarded_args[i + 1]
        elif arg == "--model_name" and i + 1 < len(forwarded_args):
            model_name = forwarded_args[i + 1]
        elif arg == "--vllm_server_port" and i + 1 < len(forwarded_args):
            vllm_port = forwarded_args[i + 1]
        elif arg == "--vllm_server_host" and i + 1 < len(forwarded_args):
            vllm_host = forwarded_args[i + 1]

    log_dir = os.path.join(output_dir, "logs")
    # Project root (parent of src/)
    cwd = Path(__file__).resolve().parent.parent.parent.parent

    # Create log directory and truncate old log files
    os.makedirs(log_dir, exist_ok=True)
    tr_log = os.path.join(log_dir, "trainer.log")
    open(tr_log, "w").close()

    session = find_available_session("async-grpo-rl")

    # ── Create tmux session ────────────────────────────────────────────
    # 3 panes stacked vertically (one on top of each other), equally sized
    run(["tmux", "new-session", "-d", "-s", session, "-n", "RL", "-c", str(cwd)])

    # Split pane 0 → top 33% (pane 0) + bottom 67% (pane 1)
    run(["tmux", "split-window", "-v", "-t", f"{session}:RL.0", "-p", "67", "-c", str(cwd)])
    # Split pane 1 → middle 50% (pane 1) + bottom 50% (pane 2)
    run(["tmux", "split-window", "-v", "-t", f"{session}:RL.1", "-p", "50", "-c", str(cwd)])

    # Set equal thirds now
    run(["tmux", "select-layout", "-t", f"{session}:RL", "even-vertical"])

    # Re-equalize panes when the terminal window is resized,
    # but only if no pane is zoomed (so Ctrl+B z still works)
    run([
        "tmux", "set-hook", "-t", session,
        "client-resized",
        f"if-shell -F '#{{window_zoomed_flag}}' true "
        f"'select-layout -t {session}:RL even-vertical'",
    ])

    # ── Pane titles ────────────────────────────────────────────────────
    run(["tmux", "select-pane", "-t", f"{session}:RL.0", "-T", "Inference"])
    run(["tmux", "select-pane", "-t", f"{session}:RL.1", "-T", "Main"])
    run(["tmux", "select-pane", "-t", f"{session}:RL.2", "-T", "Trainer"])

    # ── Pane border styling (colored per pane) ──────────────────────────
    run(["tmux", "set-option", "-t", session, "-g", "pane-border-status", "top"])
    # Color the border line: Inference=cyan, Main=green, Trainer=magenta
    border_fmt = (
        "#{?#{==:#{pane_index},0},"
        "#[fg=cyan bold] #{pane_title} #[default],"
        "#{?#{==:#{pane_index},1},"
        "#[fg=green bold] #{pane_title} #[default],"
        "#[fg=magenta bold] #{pane_title} #[default]}}"
    )
    run([
        "tmux", "set-option", "-t", session, "-g",
        "pane-border-format", border_fmt,
    ])
    run([
        "tmux", "set-window-option", "-t", f"{session}:RL",
        "pane-border-status", "top",
    ])

    # ── Forward env vars ───────────────────────────────────────────────
    env_prefix_parts = []
    for var in ("CUDA_HOME", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(var)
        if val is not None:
            env_prefix_parts.append(f"{var}={val}")
    env_prefix = " ".join(env_prefix_parts)
    if env_prefix:
        env_prefix += " "

    python = sys.executable  # respect virtualenv

    # Write pane commands to temp scripts so the shell doesn't echo
    # ugly one-liners into the tmux panes.
    script_dir = os.path.join(log_dir, ".scripts")
    os.makedirs(script_dir, exist_ok=True)

    # ── Pane 0 (top): Inference – vLLM server ─────────────────────────
    server_script = os.path.join(script_dir, "server.sh")
    with open(server_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("clear\n")
        f.write(
            f"CUDA_VISIBLE_DEVICES={vllm_gpu_id} "
            f"{python} -m history_of_rlvr.rl.async_rl.server "
            f"--model {model_name} "
            f"--host {vllm_host} "
            f"--port {vllm_port} "
            f"--tensor-parallel-size 1 "
            f"--gpu-memory-utilization 0.5 "
            f"--dtype bfloat16 "
            f"--max-model-len 1536 "
            f"--enforce-eager "
            f"--disable-log-stats\n"
        )
    os.chmod(server_script, 0o755)
    run(["tmux", "send-keys", "-t", f"{session}:RL.0", f"bash {server_script}", "C-m"])

    # ── Pane 1 (middle): Main – training process (orchestrator) ───────
    args_str = " ".join(forwarded_args)
    main_script = os.path.join(script_dir, "main.sh")
    with open(main_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("clear\n")
        f.write(f'echo "Waiting for vLLM server at {vllm_host}:{vllm_port}..."\n')
        f.write(
            f"while ! curl -s http://{vllm_host}:{vllm_port}/health "
            f"> /dev/null 2>&1; do sleep 2; done\n"
        )
        f.write('echo "Server is up! Starting training..."\n')
        f.write("sleep 1\n")
        f.write(f"{env_prefix}{python} -m history_of_rlvr.rl.async_rl.main {args_str}\n")
    os.chmod(main_script, 0o755)
    run(["tmux", "send-keys", "-t", f"{session}:RL.1", f"bash {main_script}", "C-m"])

    # ── Pane 2 (bottom): Trainer log ──────────────────────────────────
    trainer_script = os.path.join(script_dir, "trainer.sh")
    with open(trainer_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("clear\n")
        f.write(f"while [ ! -s {tr_log} ]; do sleep 0.5; done\n")
        f.write("clear\n")
        f.write(f"tail -n 0 -F {tr_log}\n")
    os.chmod(trainer_script, 0o755)
    run(["tmux", "send-keys", "-t", f"{session}:RL.2", f"bash {trainer_script}", "C-m"])

    # Focus the Main pane by default
    run(["tmux", "select-pane", "-t", f"{session}:RL.1"])

    # ── Attach ─────────────────────────────────────────────────────────
    if sys.stdout.isatty():
        print(f"Attaching to tmux session '{session}'...")
        # print("  Ctrl+B ↑/↓  switch panes")
        # print("  Ctrl+B [    scroll mode (q to exit)")
        # print("  Ctrl+B d    detach (training continues)")
        os.execvp("tmux", ["tmux", "attach-session", "-t", session])
    else:
        print(
            f"tmux session '{session}' created "
            f"(non-interactive, not attaching)."
        )
        print(f"  Attach with: tmux attach -t {session}")


if __name__ == "__main__":
    main()
