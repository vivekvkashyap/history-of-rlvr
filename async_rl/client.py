"""
vLLM client for NCCL-based weight synchronisation.

Connects to the vLLM inference server (see ``server.py``) and provides
methods to:
  1. Initialise an NCCL communication group with the server workers.
  2. Push updated model weights parameter-by-parameter via NCCL broadcast.
  3. Reset the KV cache after a weight update.

The client occupies the *last* rank in the NCCL group.  Server workers
occupy ranks 0 .. world_size-2.

Based on the verifiers-rl inference client, made fully standalone.
"""

import atexit
import logging
import time

import requests
import torch
from requests import ConnectionError
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)


class VLLMClient:
    """
    Client that talks to the async_rl vLLM server for weight updates.

    Usage:
        client = VLLMClient(host="0.0.0.0", port=8000)
        client.init_communicator()

        # After a training step:
        for name, param in model.named_parameters():
            client.update_named_param(name, param.data)
        client.reset_prefix_cache()
        while client.get_num_background_tasks() > 0:
            time.sleep(0.5)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        self.session = requests.Session()
        # Connection pooling for rapid requests
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3,
            pool_block=False,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.host = host
        self.server_port = port
        self.server_url = f"http://{self.host}:{self.server_port}"

        self.group_port = group_port
        self.check_server(connection_timeout)

    # ════════════════════════════════════════════════════════════════════
    #  Server health check
    # ════════════════════════════════════════════════════════════════════

    def check_server(
        self, total_timeout: float = 0.0, retry_interval: float = 2.0,
    ) -> None:
        """Wait for the vLLM server to become healthy."""
        url = f"{self.server_url}/health"
        start_time = time.time()

        while True:
            try:
                response = requests.get(url)
            except RequestException as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    logger.info("vLLM server is up!")
                    return
                last_error = ConnectionError(
                    f"Health check returned status {response.status_code}: "
                    f"{response.text}"
                )

            elapsed = time.time() - start_time
            if elapsed >= total_timeout:
                raise ConnectionError(
                    f"The vLLM server can't be reached at "
                    f"{self.host}:{self.server_port} after {total_timeout}s. "
                    f"Make sure the server is running "
                    f"(python -m async_rl.server ...)."
                ) from last_error

            logger.info(
                f"Server not up yet. Retrying in {retry_interval}s..."
            )
            time.sleep(retry_interval)

    # ════════════════════════════════════════════════════════════════════
    #  NCCL communicator lifecycle
    # ════════════════════════════════════════════════════════════════════

    def init_communicator(self) -> None:
        """
        Set up the NCCL communication group between this client
        (trainer) and the vLLM server workers.
        """
        # Get the server's world size (number of TP * DP workers)
        url = f"{self.server_url}/get_world_size"
        try:
            response = requests.get(url)
        except Exception as e:
            logger.error(f"Failed to get world size: {e}")
            raise

        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
            logger.info(f"vLLM world size: {vllm_world_size}")
        else:
            raise Exception(
                f"Request failed: {response.status_code}, {response.text}"
            )

        world_size = vllm_world_size + 1  # +1 for this client
        self.rank = vllm_world_size       # client is last rank
        logger.info(
            f"Client rank: {self.rank}, total world size: {world_size}"
        )

        # Tell the server to initialise its side of the NCCL group
        url = f"{self.server_url}/init_communicator"
        try:
            response = self.session.post(
                url,
                json={
                    "host": self.host,
                    "port": self.group_port,
                    "world_size": world_size,
                },
            )
        except Exception as e:
            logger.error(f"Failed to init communicator: {e}")
            raise

        if response.status_code != 200:
            raise Exception(
                f"Request failed: {response.status_code}, {response.text}"
            )

        # Brief delay to let the server finish socket setup
        time.sleep(0.1)

        # Create our side of the process group + NCCL communicator
        pg = StatelessProcessGroup.create(
            host=self.host,
            port=self.group_port,
            rank=self.rank,
            world_size=world_size,
        )
        device = 0
        logger.info(
            f"Initialising PyNcclCommunicator on device {device}, "
            f"rank {self.rank}, world_size {world_size}"
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device=device)
        atexit.register(self.close_communicator)

    # ════════════════════════════════════════════════════════════════════
    #  Weight update
    # ════════════════════════════════════════════════════════════════════

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        """
        Push a single named parameter to the vLLM server.

        Sends metadata via HTTP, then broadcasts the tensor via NCCL.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.server_url}/update_named_param"

        try:
            response = self.session.post(
                url,
                json={"name": name, "dtype": dtype, "shape": shape},
                timeout=300.0,
            )
        except Timeout:
            logger.error(
                f"Timeout waiting for server response for {name} after 300s"
            )
            raise Exception(f"Request timeout for {name} after 300s")
        except Exception as e:
            logger.error(f"Error sending request for {name}: {e}")
            raise

        if response.status_code != 200:
            raise Exception(
                f"Request failed: {response.status_code}, {response.text}"
            )

        # NCCL broadcast (this client is the source)
        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

    # ════════════════════════════════════════════════════════════════════
    #  Cache reset & background task tracking
    # ════════════════════════════════════════════════════════════════════

    def reset_prefix_cache(self) -> None:
        """Reset the vLLM KV cache after a weight update."""
        url = f"{self.server_url}/reset_prefix_cache"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(
                f"Request failed: {response.status_code}, {response.text}"
            )

    def get_num_background_tasks(self) -> int:
        """Get the number of in-flight background tasks on the server."""
        url = f"{self.server_url}/get_num_background_tasks"
        response = self.session.post(url)
        return response.json()["num_background_tasks"]

    def close_communicator(self) -> None:
        """Tear down the NCCL communication group."""
        url = f"http://{self.host}:{self.server_port}/close_communicator"
        try:
            response = self.session.post(url)
        except ConnectionError:
            pass
        else:
            if response.status_code != 200:
                raise Exception(
                    f"Request failed: {response.status_code}, {response.text}"
                )
