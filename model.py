import os
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from PIL import Image, ImageDraw

from config import HF_CACHE_ROOT, MODE, MODEL_ID
from logging_setup import log

if TYPE_CHECKING:
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline


def resolve_snapshot_path(model_id: str) -> str:
    """
    Resolve the local snapshot path for a cached model.

    Args:
        model_id: The model name from Hugging Face
            (e.g., 'distilbert/distilbert-base-uncased-finetuned-sst-2-english')

    Returns:
        The full path to the cached model snapshot
    """
    if "/" not in model_id:
        raise ValueError(f"model_id '{model_id}' must be in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    # Read the snapshot hash from refs/main
    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    # Fall back to first available snapshot
    if os.path.isdir(snapshots_dir):
        versions = [
            d
            for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if versions:
            versions.sort()
            return os.path.join(snapshots_dir, versions[0])

    raise RuntimeError(f"Cached model not found: {model_id}")


def download_snapshot(model_id: str) -> str:
    """
    Download the model into the local Hugging Face cache, returning the snapshot path.

    A no-op when the files are already cached. FLUX.1-dev is gated, so this needs
    the license accepted on Hugging Face and `hf auth login` done once.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_id,
        # Single-file checkpoints at the repo root, unused by the diffusers pipeline
        ignore_patterns=["flux1-dev.safetensors", "ae.safetensors"],
    )


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MockPipeline:
    """
    Stand-in for FluxPipeline with the same call shape but no weights.

    Returns an image whose colour is derived from the seed, so the same seed gives
    the same image.
    """

    def __call__(self, prompt: str, *, height: int, width: int, generator, **_):
        colour = random.Random(generator.initial_seed()).choices(range(256), k=3)
        image = Image.new("RGB", (width, height), tuple(colour))
        ImageDraw.Draw(image).text((16, 16), prompt, fill="white")
        return SimpleNamespace(images=[image])


def get_model() -> "FluxPipeline | MockPipeline":
    log.info(f"Loading model in '{MODE}' mode")
    if MODE == "mock":
        return MockPipeline()

    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

    if MODE == "deployed":
        path, device = resolve_snapshot_path(MODEL_ID), "cuda"
    else:
        path, device = download_snapshot(MODEL_ID), pick_device()

    return FluxPipeline.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        local_files_only=MODE == "deployed",
    ).to(device)
