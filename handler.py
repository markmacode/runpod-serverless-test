import logging
import os
import random
import tempfile
from logging import handlers
from pathlib import Path
from typing import Any, cast

import runpod
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from runpod.serverless.utils import rp_upload
from runpod.serverless.utils.rp_cleanup import clean
from runpod.serverless.utils.rp_validator import validate

MODEL_ID = "black-forest-labs/FLUX.1-dev"
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

# Force offline mode to prevent accidental downloads
os.environ["HF_HUB_OFFLINE"] = "1"

schema = {
    "prompt": {
        "type": str,
        "required": True,
    },
    "seed": {
        "type": int,
        "required": False,
        "default": None,
    },
}


def setup_logger(log_level=logging.DEBUG):
    """
    Configures and returns a logger that writes to the console.

    This function should be called once when the worker initializes.
    """
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - [Request: %(request_id)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get the root logger
    logger = logging.getLogger("runpod_worker")
    logger.setLevel(log_level)

    # This handler sends logs to standard output, which Runpod captures as worker logs.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    # Add the console handler to the logger
    # Check if handlers are already added to avoid duplication on hot reloads
    if not handlers:
        logger.addHandler(console_handler)

    return logger


# Set up the logger when the script is first loaded by the worker.
log = setup_logger(log_level=logging.DEBUG)
log = logging.LoggerAdapter(log, {"request_id": "N/A"})
log.info("Logger initialized. Ready to process jobs.")


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


def get_model() -> FluxPipeline:
    cache_path = resolve_snapshot_path(MODEL_ID)
    return FluxPipeline.from_pretrained(
        cache_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")


def get_seed(input: dict) -> int:
    """
    Get the seed for the image generation.

    Args:
        input: The endpoint input dict which contains the optional seed.

    Returns:
        The provided seed in the input, or a new random seed.
    """
    seed = input.get("seed")
    if seed is not None:
        return seed
    return random.randint(0, 2**32 - 1)


model = get_model()


def handler(event: dict[str, Any]) -> Any:
    """
    This function processes incoming requests to your Serverless endpoint.

    Args:
        event (dict): Contains the input data and request metadata

    Returns:
       Any: The result to be returned to the client
    """
    log.info("Worker start")

    event_id = event["id"]
    staging = Path(tempfile.mkdtemp(prefix=f"{event_id}-"))

    try:
        log.info("Validating input")
        validated_input = validate(event["input"], schema)

        if "errors" in validated_input:
            return {"error": validated_input["errors"]}

        input = cast(dict, validated_input["validated_input"])
        prompt = input["prompt"]
        log.debug(f"Received prompt: {prompt}")

        log.info("Inference start")
        seed = get_seed(input)
        image = model(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        log.info("Inference done")

        log.info("Image saving")
        path = staging / "image.png"
        image.save(path, format="PNG")
        log.info("Image saved")

        log.info("Image upload start")
        image_url = rp_upload.upload_image(event_id, str(path))
        log.info("Image upload done")

        return {
            "output": {
                "image_url": image_url,
                "seed": seed,
            }
        }
    except Exception as e:
        log.exception("An error occurred during image generation")
        return {"error": str(e)}
    finally:
        log.info("Cleaning storage")
        clean(folder_list=[str(staging.absolute())])


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
