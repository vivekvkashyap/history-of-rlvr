"""
vLLM-based inference engine for GRPO training.

Generates G completions per prompt using vLLM for fast batched inference,
and supports syncing weights from the HuggingFace training model.

GPU placement:  vLLM is pinned to a specific GPU (``gpu_id``) so that the
training model can live on a different GPU.  This is done via
``torch.cuda.set_device`` before vLLM initialisation.
"""

import os
import logging
from typing import List, Tuple
from dataclasses import dataclass

# Enable pickle-based serialization so that apply_model() can send
# the weight-loading function to the vLLM V1 worker subprocess.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# Temporary file used to pass weights from the training process to the
# vLLM worker subprocess (V1 engine runs the model in a separate process).
_WEIGHT_SYNC_PATH = "/tmp/_vllm_weight_sync.pt"


def _load_weights_from_file(model):
    """
    Module-level function (must be picklable) that loads weights from a
    temporary file into the vLLM model.  Called inside the vLLM worker
    process via ``llm.apply_model()``.
    """
    import torch as _torch
    weights = _torch.load(
        _WEIGHT_SYNC_PATH, map_location="cpu", weights_only=True,
    )
    model.load_weights(weights.items())


@dataclass
class InferenceConfig:
    """Configuration subset relevant to inference."""
    model_name: str
    num_generations: int
    max_new_tokens: int
    temperature: float
    top_p: float
    gpu_memory_utilization: float
    dtype: str
    max_model_len: int
    gpu_id: int = 1  # which GPU vLLM should use


class VLLMInferenceEngine:
    """
    Wraps vLLM's LLM for batched multi-completion generation.

    Usage:
        engine = VLLMInferenceEngine(config)
        completions, token_ids, logprobs = engine.generate(prompts)
        # ... train the HuggingFace model ...
        engine.update_weights(hf_model)
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.num_generations = config.num_generations
        self.gpu_id = config.gpu_id

        self.sampling_params = SamplingParams(
            n=config.num_generations,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
            logprobs=1,  # return logprobs for generated tokens
        )

        # ── Pin vLLM to the requested GPU ──────────────────────────────
        # vLLM V1 runs the model in a spawned subprocess.  The subprocess
        # inherits CUDA_VISIBLE_DEVICES and defaults to cuda:0 within its
        # own CUDA context.  To place the model on a specific physical GPU
        # we temporarily restrict CUDA_VISIBLE_DEVICES to just that GPU
        # before the LLM is created (and the subprocess is spawned).
        # The main process has already initialised CUDA, so its context
        # is unaffected by the env-var change.
        orig_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        logger.info(f"Creating vLLM engine on physical GPU {self.gpu_id}")

        self.llm = LLM(
            model=config.model_name,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            trust_remote_code=True,
            enforce_eager=True,  # needed for weight updates
            enable_prefix_caching=False,
        )

        # Restore CUDA_VISIBLE_DEVICES so the training process keeps
        # seeing all GPUs.
        if orig_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig_cvd
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def generate(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], List[List[int]], List[List[float]]]:
        """
        Generate G completions per prompt.

        Args:
            prompts: list of B formatted prompt strings

        Returns:
            completions: list of B * G completion strings
                         (first G belong to prompt 0, next G to prompt 1, ...)
            completion_token_ids: list of B * G token id lists
            completion_logprobs:  list of B * G lists of per-token log probs
                                  (from the vLLM sampling policy)
        """
        outputs = self.llm.generate(prompts, self.sampling_params)

        completions: List[str] = []
        completion_token_ids: List[List[int]] = []
        completion_logprobs: List[List[float]] = []

        # vLLM returns outputs in same order as input prompts.
        # Each output has .outputs list of length n (= num_generations).
        for request_output in outputs:
            for gen in request_output.outputs:
                completions.append(gen.text)
                token_ids = list(gen.token_ids)
                completion_token_ids.append(token_ids)

                # Extract per-token logprobs for the actually-sampled tokens
                if gen.logprobs is not None:
                    lps: List[float] = []
                    for step_logprobs, tid in zip(gen.logprobs, token_ids):
                        if step_logprobs is not None and tid in step_logprobs:
                            lps.append(step_logprobs[tid].logprob)
                        else:
                            lps.append(0.0)
                    completion_logprobs.append(lps)
                else:
                    completion_logprobs.append([0.0] * len(token_ids))

        return completions, completion_token_ids, completion_logprobs

    def update_weights(self, state_dict: dict) -> None:
        """
        Sync weights into the vLLM model via ``apply_model``.

        vLLM V1 runs the model in a subprocess, so we save the state dict
        to a temporary file and load it inside the worker process.
        """
        torch.save(state_dict, _WEIGHT_SYNC_PATH)
        self.llm.apply_model(_load_weights_from_file)
        logger.info("Weights synced to vLLM via apply_model")
