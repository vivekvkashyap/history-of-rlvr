"""
vLLM inference server with NCCL-based weight synchronisation.

Runs a standard vLLM OpenAI-compatible API server extended with HTTP
endpoints that allow a training process to push updated model weights
into the running inference engine without restarting it.

Weight updates flow:
  1. Trainer calls POST /init_communicator to establish an NCCL group.
  2. For each parameter, trainer calls POST /update_named_param (metadata)
     and then broadcasts the tensor via NCCL.
  3. After all parameters are sent, trainer calls POST /reset_prefix_cache.

Based on the verifiers-rl inference server, made fully standalone
(no verifiers package dependency).

Usage:
    python -m rl.async_rl.server \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --port 8000 \\
        --tensor-parallel-size 1 \\
        --gpu-memory-utilization 0.5 \\
        --dtype bfloat16 \\
        --max-model-len 1536 \\
        --enforce-eager
"""

import asyncio
import os
import signal
from argparse import Namespace
from typing import Any, Sequence, cast

import torch
import uvloop
from fastapi import Request
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, set_ulimit

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# ── Concurrency control for weight updates ─────────────────────────────
MAX_CONCURRENT_WEIGHT_UPDATES = 10
weight_update_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WEIGHT_UPDATES)
background_tasks: set[asyncio.Task] = set()


# ════════════════════════════════════════════════════════════════════════
#  Worker extension (runs inside each vLLM worker process)
# ════════════════════════════════════════════════════════════════════════


class WeightSyncWorkerExtension:
    """
    vLLM worker extension that receives weight updates via NCCL.

    Installed by setting ``engine_args.worker_extension_cls`` to the
    fully-qualified class name.  Each vLLM worker process instantiates
    one copy of this class.
    """

    pynccl_comm = None   # NCCL communicator
    client_rank = None   # rank of the training process (broadcaster)
    device = None        # CUDA device for this worker

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """Establish the NCCL communication group with the trainer."""
        if self.pynccl_comm is not None:
            raise RuntimeError(
                "Weight update group already initialised. "
                "Call close_communicator first."
            )

        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size,
        )
        assert self.device is not None
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1  # trainer is last rank

    def update_named_param(
        self, name: str, dtype: str, shape: Sequence[int],
    ) -> None:
        """Receive a single named parameter via NCCL broadcast."""
        if self.pynccl_comm is None:
            raise RuntimeError(
                "Communicator not initialised. Call init_communicator first."
            )

        torch_dtype = getattr(torch, dtype.split(".")[-1])
        weight = torch.empty(shape, dtype=torch_dtype, device=self.device)
        client_rank = self.client_rank
        assert client_rank is not None
        self.pynccl_comm.broadcast(weight, src=client_rank)
        self.pynccl_comm.group.barrier()
        cast(Any, self).model_runner.model.load_weights(
            weights=[(name, weight)],
        )

    def close_communicator(self) -> None:
        """Tear down the NCCL communication group."""
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


# ════════════════════════════════════════════════════════════════════════
#  Server entry-point
# ════════════════════════════════════════════════════════════════════════


async def run_server(args: Namespace) -> None:
    sock_addr = (args.host or "0.0.0.0", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    # ── Helper: tracked background tasks ───────────────────────────────
    def create_background_task(coro):
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return task

    # ── Engine ─────────────────────────────────────────────────────────
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = (
        "rl.async_rl.server.WeightSyncWorkerExtension"
    )
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER,
    )
    app = build_app(args)

    # ── Custom endpoints ───────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size")
    async def get_world_size():
        return {
            "world_size": args.tensor_parallel_size * args.data_parallel_size,
        }

    @app.post("/init_communicator")
    async def init_communicator(request: Request):
        data = await request.json()
        host = data.get("host")
        port = data.get("port")
        world_size = data.get("world_size")
        create_background_task(
            engine.collective_rpc(
                "init_communicator", args=(host, port, world_size),
            )
        )
        return {"status": "ok"}

    @app.post("/update_named_param")
    async def update_named_param(request: Request):
        data = await request.json()
        name = data.get("name")
        dtype = data.get("dtype")
        shape = tuple(data.get("shape"))

        async def throttled_update():
            async with weight_update_semaphore:
                await engine.collective_rpc(
                    "update_named_param", args=(name, dtype, shape),
                )

        create_background_task(throttled_update())
        return {"status": "ok"}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache(request: Request):
        create_background_task(engine.reset_prefix_cache())
        return {"status": "ok"}

    @app.post("/get_num_background_tasks")
    async def get_num_background_tasks():
        return {"num_background_tasks": len(background_tasks)}

    @app.post("/close_communicator")
    async def close_communicator(request: Request):
        await engine.collective_rpc("close_communicator")
        return {"status": "ok"}

    # ── Start serving ──────────────────────────────────────────────────
    vllm_config = await engine.get_vllm_config()
    await init_app_state(engine, vllm_config, app.state, args)
    shutdown_task = await serve_http(
        app,
        sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )
    await shutdown_task

    # ── Cleanup ────────────────────────────────────────────────────────
    for task in background_tasks:
        task.cancel()
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

    sock.close()


def main():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-compatible server with weight synchronisation",
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args() or Namespace()
    validate_parsed_serve_args(args)
    print(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
