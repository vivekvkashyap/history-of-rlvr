"""
Base Environment class and ``run()`` launcher for environment-based training.

Provides:
  - ``Environment``: abstract base class that every environment must subclass.
  - ``EnvironmentDataset``: thin ``torch.utils.data.Dataset`` wrapper.
  - ``run(env)``: one-call launcher that either starts a full tmux session
    (vLLM server + training) or runs the training loop directly.

Environments only need to implement three things:
  1. ``get_dataset()`` – load & format the dataset
  2. ``compute_rewards(completions, ground_truths)`` – score completions
  3. ``system_prompt`` / ``name`` – metadata

The ``run()`` function handles model loading, config parsing, algorithm
selection (GRPO from ``rl/algorithms/``), and async_rl orchestration.
"""

import abc
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


# ════════════════════════════════════════════════════════════════════════
#  Abstract base class
# ════════════════════════════════════════════════════════════════════════


class Environment(abc.ABC):
    """
    Base class for all training environments.

    Subclasses must set ``name`` and ``system_prompt`` as class attributes,
    and implement ``get_dataset()`` and ``compute_rewards()``.
    """

    # ── Required class attributes ─────────────────────────────────────
    name: str = ""
    """Short identifier used in output paths and tmux session names."""

    system_prompt: str = ""
    """System prompt prepended to every user question."""

    def __init__(self):
        """Initialize the environment. Tokenizer will be set later."""
        self.tokenizer = None

    # ── Dataset ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_dataset(self) -> List[Dict[str, str]]:
        """
        Load and return the dataset as a list of dicts.

        Each dict must have at least:
            - ``"prompt"``: the full formatted prompt string (ChatML)
            - ``"ground_truth"``: the expected answer string

        Optionally include ``"question"`` (raw question text) for logging.
        """
        ...

    # ── Reward function ────────────────────────────────────────────────

    @abc.abstractmethod
    def compute_rewards(
        self,
        completions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        """
        Score a batch of completions against their ground truths.

        Args:
            completions: model-generated completions (length B*G)
            ground_truths: expected answers, repeated G times per prompt

        Returns:
            list of float rewards, same length as ``completions``
        """
        ...

    # ── Evaluation helpers ────────────────────────────────────────────

    def get_eval_dataset(self) -> List[Dict[str, str]]:
        """
        Load the evaluation dataset.

        Override this to return a held-out test set. Default returns the
        same data as ``get_dataset()`` (which is the training set).
        """
        return self.get_dataset()

    def compute_reward_details(
        self,
        completion: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """
        Return per-component reward breakdown for a single completion.

        Override this in subclasses to provide detailed diagnostics
        (e.g. correctness, format compliance, etc.).

        Default: returns only the total reward.
        """
        total = self.compute_rewards([completion], [ground_truth])[0]
        return {"total": total}

    # ── Config overrides ─────────────────────────────────────────────

    def get_config_overrides(self) -> Dict[str, Any]:
        """
        Return config overrides for this environment.

        Override this method to customize training parameters per
        environment so you don't need to pass them via CLI every time.

        Keys must match ``AsyncGRPOConfig`` field names.
        CLI arguments still take highest precedence.

        Example::

            def get_config_overrides(self):
                return {
                    "learning_rate": 1e-5,
                    "max_steps": 1000,
                    "num_generations": 8,
                }
        """
        return {}

    # ── Prompt formatting helper ──────────────────────────────────────

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for proper chat template formatting."""
        self.tokenizer = tokenizer

    def format_prompt(self, question: str) -> str:
        """
        Build a prompt from a raw question string.

        If a tokenizer is set, uses its chat template via apply_chat_template().
        Otherwise, falls back to ChatML format.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        if self.tokenizer is not None:
            # Use the model's native chat template
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback to ChatML format
            return (
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )


# ════════════════════════════════════════════════════════════════════════
#  Dataset wrapper
# ════════════════════════════════════════════════════════════════════════


class EnvironmentDataset(Dataset):
    """
    Thin ``torch.utils.data.Dataset`` wrapper around the list returned by
    ``Environment.get_dataset()``.

    Provides ``__len__`` and ``__getitem__`` so it can be passed directly
    to the orchestrator and trainer.
    """

    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]


# ════════════════════════════════════════════════════════════════════════
#  Launcher
# ════════════════════════════════════════════════════════════════════════


def run(env: Environment) -> None:
    """
    Main entry point for running an environment.

    If ``--launch`` is present in ``sys.argv``, starts a full tmux session
    with vLLM server + training.  Otherwise, runs the training loop
    directly (assumes vLLM server is already running).

    All remaining CLI arguments are forwarded to ``AsyncGRPOConfig``.
    """
    args = sys.argv[1:]

    if "--launch" in args:
        args.remove("--launch")
        _tmux_launch(env, args)
    else:
        _train(env, args)


# ════════════════════════════════════════════════════════════════════════
#  Direct training (no tmux – server must be running)
# ════════════════════════════════════════════════════════════════════════


def _train(env: Environment, cli_args: List[str]) -> None:
    """
    Run the training loop directly.

    Parses ``AsyncGRPOConfig`` from *cli_args*, loads model + tokenizer,
    builds the environment dataset + reward function, and trains.
    """
    import logging
    import warnings

    import torch
    from rich.console import Console
    from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

    # Parent dir for cross-package imports (history_of_rlvr/)
    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    # Also add rl/ so that async_rl/main.py style imports work
    rl_parent = os.path.join(root, "rl")
    if rl_parent not in sys.path:
        sys.path.insert(0, rl_parent)

    from rl.async_rl.config import AsyncGRPOConfig
    from rl.async_rl.trainer import AsyncGRPOTrainer
    from rl.sync_rl.log_router import LogRouter

    # Suppress noisy logs
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for name in ("httpx", "openai", "httpcore", "transformers", "accelerate", "peft"):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*use_cache=True.*")
    warnings.filterwarnings("ignore", message=".*PAD/BOS/EOS.*")

    console = Console(force_terminal=True, highlight=False)

    # ── Parse config ──────────────────────────────────────────────────
    # 1. Collect environment-level config overrides
    env_overrides = env.get_config_overrides()

    # 2. Convert env overrides to CLI-style args, but only for keys
    #    that are NOT already explicitly set in cli_args.  This way
    #    CLI arguments always take highest precedence.
    cli_keys = set()
    for a in cli_args:
        if a.startswith("--"):
            cli_keys.add(a.lstrip("-"))

    override_args: List[str] = []
    for key, value in env_overrides.items():
        if key not in cli_keys:
            if isinstance(value, bool):
                override_args += [f"--{key}", str(value).lower()]
            else:
                override_args += [f"--{key}", str(value)]

    # 3. Parse: env overrides first, then cli_args (later values win)
    old_argv = sys.argv
    sys.argv = ["env_runner"] + override_args + cli_args
    parser = HfArgumentParser(AsyncGRPOConfig)
    (config,) = parser.parse_args_into_dataclasses()
    sys.argv = old_argv

    # 4. Log applied overrides
    if env_overrides:
        console.print()
        console.print(f"  [bold cyan]Environment config overrides ({env.name}):[/bold cyan]")
        for key, value in env_overrides.items():
            tag = "[dim](overridden by CLI)[/dim]" if key in cli_keys else ""
            console.print(f"    {key} = {value}  {tag}")

    algo_name = config.algorithm.upper()
    console.print()
    console.rule(f"{env.name} – Async {algo_name} Training")
    console.print()
    console.print(f"  Environment  {env.name}")
    console.print(f"  Model        {config.model_name}")
    if config.algorithm == "cispo":
        console.print(f"  CISPO        G={config.num_generations}  eps_high={config.cispo_epsilon_high}  eps_low={config.cispo_epsilon_low}")
    else:
        console.print(f"  GRPO         G={config.num_generations}  eps_lo={config.epsilon_lower}  eps_hi={config.epsilon_upper}  beta={config.beta}")
    console.print(f"  Train        lr={config.learning_rate}  steps={config.max_steps}  batch={config.batch_size}  micro={config.micro_batch_size}")
    console.print(f"  Server       {config.vllm_server_host}:{config.vllm_server_port}")
    console.print(f"  GPU          trainer=cuda:{config.trainer_gpu_id}")
    if config.use_lora:
        console.print(f"  LoRA         rank={config.lora_rank}  alpha={config.lora_alpha}")
    if config.inflight_weight_updates:
        console.print(f"  Inflight     ON  max_off_policy_steps={config.max_off_policy_steps}")
    else:
        console.print(f"  Inflight     OFF (legacy blocking sync)")
    if config.continuous_batching:
        console.print(f"  ContBatch    ON  pool_size={config.pool_size}")
    else:
        console.print(f"  ContBatch    OFF (batch-at-a-time)")
    console.print()

    # ── Build dataset ─────────────────────────────────────────────────
    console.print(f"Loading {env.name} dataset (initial load without tokenizer)...")
    # Initial load to get dataset size
    temp_data = env.get_dataset()
    console.print(f"{len(temp_data)} training examples found.")

    # ── Build eval dataset ────────────────────────────────────────────
    eval_data_raw = None
    if config.eval_steps > 0:
        console.print(f"Loading {env.name} eval dataset (split={config.eval_split})...")
        try:
            eval_data_raw = env.get_eval_dataset()
            console.print(f"{len(eval_data_raw)} eval examples found.")
        except Exception as e:
            console.print(f"[yellow]Warning: could not load eval dataset: {e}[/yellow]")
            console.print("[yellow]Evaluation will be disabled.[/yellow]")

    # ── Model + Tokenizer ─────────────────────────────────────────────
    console.print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # ── Set tokenizer in environment and reload datasets ──────────────
    console.print("Setting tokenizer and formatting prompts with model's chat template...")
    env.set_tokenizer(tokenizer)
    
    # Reload datasets with proper formatting
    dataset = EnvironmentDataset(env.get_dataset())
    eval_dataset = eval_data_raw if eval_data_raw is None else env.get_eval_dataset()
    console.print(f"Dataset formatted with {config.model_name} chat template.")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(f"cuda:{config.trainer_gpu_id}")

    # Align model config with tokenizer so Trainer doesn't emit
    # "PAD/BOS/EOS tokens differ" warnings
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    console.print("Model loaded.")

    # ── Log router ────────────────────────────────────────────────────
    log_dir = os.path.join(config.output_dir, "logs")
    log_router = LogRouter(log_dir=log_dir)
    log_router.start()

    # ── Trainer ────────────────────────────────────────────────────────
    console.print("Initialising trainer + NCCL communicator...")
    trainer = AsyncGRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        dataset=dataset,
        reward_fn=env.compute_rewards,
        eval_dataset=eval_dataset,
        reward_details_fn=env.compute_reward_details,
        log_router=log_router,
    )
    console.print("Trainer ready. Starting training...")
    console.rule("Training")
    console.print()

    # ── Train ─────────────────────────────────────────────────────────
    try:
        trainer.train()
    finally:
        log_router.stop()
    console.print()
    console.rule("Training Complete")


# ════════════════════════════════════════════════════════════════════════
#  Tmux launcher (starts vLLM server + training)
# ════════════════════════════════════════════════════════════════════════


def _run_cmd(cmd: list) -> None:
    """Run a command, raise on failure."""
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def _tmux_launch(env: Environment, forwarded_args: List[str]) -> None:
    """
    Launch a tmux session with 3 panes: vLLM server, training, trainer log.

    Adapted from ``rl/async_rl/launch.py`` but driven by an environment
    module instead of ``rl.async_rl.main``.
    """
    # ── Check tmux ────────────────────────────────────────────────────
    try:
        subprocess.run(
            ["tmux", "-V"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        raise SystemExit(
            "tmux is not installed.  Install with: sudo apt install tmux"
        )

    # ── Extract key params from forwarded args ────────────────────────
    output_dir = f"outputs/{env.name}_grpo"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    vllm_port = "8000"
    vllm_host = "0.0.0.0"
    vllm_gpu_id = "1"
    has_output_dir = False

    for i, arg in enumerate(forwarded_args):
        if arg == "--output_dir" and i + 1 < len(forwarded_args):
            output_dir = forwarded_args[i + 1]
            has_output_dir = True
        elif arg == "--model_name" and i + 1 < len(forwarded_args):
            model_name = forwarded_args[i + 1]
        elif arg == "--vllm_server_port" and i + 1 < len(forwarded_args):
            vllm_port = forwarded_args[i + 1]
        elif arg == "--vllm_server_host" and i + 1 < len(forwarded_args):
            vllm_host = forwarded_args[i + 1]

    # Inject --output_dir into forwarded args so the training process
    # writes logs to the same directory the Trainer pane tails.
    if not has_output_dir:
        forwarded_args = forwarded_args + ["--output_dir", output_dir]

    log_dir = os.path.join(output_dir, "logs")
    cwd = Path(__file__).parent.parent  # history_of_rlvr/

    # Create log directory and truncate old log files
    os.makedirs(log_dir, exist_ok=True)
    tr_log = os.path.join(log_dir, "trainer.log")
    open(tr_log, "w").close()

    # ── Find available session name ───────────────────────────────────
    base_session = f"{env.name}-grpo-rl"

    def _session_exists(name: str) -> bool:
        return subprocess.run(
            ["tmux", "has-session", "-t", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0

    session = base_session
    idx = 2
    while _session_exists(session):
        session = f"{base_session}-{idx}"
        idx += 1

    # ── Create tmux session with 3 panes ──────────────────────────────
    _run_cmd(["tmux", "new-session", "-d", "-s", session, "-n", "RL", "-c", str(cwd)])
    _run_cmd(["tmux", "split-window", "-v", "-t", f"{session}:RL.0", "-p", "67", "-c", str(cwd)])
    _run_cmd(["tmux", "split-window", "-v", "-t", f"{session}:RL.1", "-p", "50", "-c", str(cwd)])
    _run_cmd(["tmux", "select-layout", "-t", f"{session}:RL", "even-vertical"])

    _run_cmd([
        "tmux", "set-hook", "-t", session,
        "client-resized",
        f"if-shell -F '#{{window_zoomed_flag}}' true "
        f"'select-layout -t {session}:RL even-vertical'",
    ])

    # Pane titles
    _run_cmd(["tmux", "select-pane", "-t", f"{session}:RL.0", "-T", "Inference"])
    _run_cmd(["tmux", "select-pane", "-t", f"{session}:RL.1", "-T", "Main"])
    _run_cmd(["tmux", "select-pane", "-t", f"{session}:RL.2", "-T", "Trainer"])

    # Pane border styling
    _run_cmd(["tmux", "set-option", "-t", session, "-g", "pane-border-status", "top"])
    border_fmt = (
        "#{?#{==:#{pane_index},0},"
        "#[fg=cyan bold] #{pane_title} #[default],"
        "#{?#{==:#{pane_index},1},"
        "#[fg=green bold] #{pane_title} #[default],"
        "#[fg=magenta bold] #{pane_title} #[default]}}"
    )
    _run_cmd([
        "tmux", "set-option", "-t", session, "-g",
        "pane-border-format", border_fmt,
    ])
    _run_cmd([
        "tmux", "set-window-option", "-t", f"{session}:RL",
        "pane-border-status", "top",
    ])

    # ── Forward env vars ──────────────────────────────────────────────
    env_prefix_parts = []
    for var in ("CUDA_HOME", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(var)
        if val is not None:
            env_prefix_parts.append(f"{var}={val}")
    env_prefix = " ".join(env_prefix_parts)
    if env_prefix:
        env_prefix += " "

    python = sys.executable
    script_dir = os.path.join(log_dir, ".scripts")
    os.makedirs(script_dir, exist_ok=True)

    # ── Pane 0: vLLM server ───────────────────────────────────────────
    server_script = os.path.join(script_dir, "server.sh")
    with open(server_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("clear\n")
        f.write(
            f"CUDA_VISIBLE_DEVICES={vllm_gpu_id} "
            f"{python} -m rl.async_rl.server "
            f"--model {model_name} "
            f"--host {vllm_host} "
            f"--port {vllm_port} "
            f"--tensor-parallel-size 1 "
            f"--gpu-memory-utilization 0.9 "
            f"--dtype bfloat16 "
            f"--max-model-len 1536 "
            # f"--max-num-seqs 512 "
            f"--enforce-eager "
            f"--disable-log-stats\n"
        )
    os.chmod(server_script, 0o755)
    _run_cmd(["tmux", "send-keys", "-t", f"{session}:RL.0", f"bash {server_script}", "C-m"])

    # ── Pane 1: Training (using the environment module) ───────────────
    args_str = " ".join(forwarded_args)
    # The environment module is run as: python -m environments.<name>
    env_module = f"environments.{env.name}"
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
        f.write(f"{env_prefix}{python} -m {env_module} {args_str}\n")
    os.chmod(main_script, 0o755)
    _run_cmd(["tmux", "send-keys", "-t", f"{session}:RL.1", f"bash {main_script}", "C-m"])

    # ── Pane 2: Trainer log ───────────────────────────────────────────
    trainer_script = os.path.join(script_dir, "trainer.sh")
    with open(trainer_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("clear\n")
        f.write(f"while [ ! -s {tr_log} ]; do sleep 0.5; done\n")
        f.write("clear\n")
        f.write(f"tail -n 0 -F {tr_log}\n")
    os.chmod(trainer_script, 0o755)
    _run_cmd(["tmux", "send-keys", "-t", f"{session}:RL.2", f"bash {trainer_script}", "C-m"])

    # Focus the Main pane
    _run_cmd(["tmux", "select-pane", "-t", f"{session}:RL.1"])

    # ── Attach ────────────────────────────────────────────────────────
    if sys.stdout.isatty():
        print(f"Attaching to tmux session '{session}'...")
        os.execvp("tmux", ["tmux", "attach-session", "-t", session])
    else:
        print(
            f"tmux session '{session}' created "
            f"(non-interactive, not attaching)."
        )
        print(f"  Attach with: tmux attach -t {session}")
