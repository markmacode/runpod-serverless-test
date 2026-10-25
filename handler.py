from config import MODEL_ID
import logging
import random
import tempfile
from pathlib import Path
from typing import Any, cast

import runpod
import torch
from runpod.serverless.utils import rp_upload
from runpod.serverless.utils.rp_validator import validate

from logging_setup import log
from model import get_model

# The schema defaults for the generation params are None because the validator skips
# constraints for any value that has the same type as its default. The real defaults
# are applied in the handler.
GENERATION_DEFAULTS = {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 3.5,
}


def bounded(low: float, high: float, multiple_of: int | None = None):
    """
    Build a schema constraint that keeps a numeric param within inclusive bounds.

    The generation params drive GPU time, and so cost, which is why they are bounded.
    Missing and wrong-typed values pass here, as the validator reports wrong types itself.
    """

    def constraint(value) -> bool:
        if not isinstance(value, (int, float)):
            return True
        if multiple_of is not None and value % multiple_of != 0:
            return False
        return low <= value <= high

    return constraint


schema = {
    "prompt": {
        "type": str,
        "required": True,
    },
    "seed": {
        "type": int,
        "required": False,
        "default": None,
        "constraints": bounded(0, 2**32 - 1),
    },
    # FLUX packs 16x16 pixel patches, so dimensions must divide evenly
    "width": {
        "type": int,
        "required": False,
        "default": None,
        "constraints": bounded(256, 1536, multiple_of=16),
    },
    "height": {
        "type": int,
        "required": False,
        "default": None,
        "constraints": bounded(256, 1536, multiple_of=16),
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": None,
        "constraints": bounded(1, 100),
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": None,
        "constraints": bounded(0, 20),
    },
}



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


# Loaded under __main__ below, not at import. Runpod runs its heartbeat in a child
# process, and with Python 3.14's default start method on Linux (forkserver) the child
# re-imports this file. Loading here would put a second copy of the model on the GPU.
model = None


def handler(event: dict[str, Any]) -> Any:
    """
    This function processes incoming requests to your Serverless endpoint.

    Args:
        event (dict): Contains the input data and request metadata

    Returns:
       Any: The result to be returned to the client
    """
    event_id = event["id"]
    job_log = logging.LoggerAdapter(log, {"request_id": event_id})
    job_log.info("Job start")

    try:
        job_log.info("Validating input")
        validated_input = validate(event["input"], schema)

        if "errors" in validated_input:
            return {"error": validated_input["errors"]}

        input = cast(dict, validated_input["validated_input"])
        prompt = input["prompt"]
        job_log.debug(f"Received prompt: {prompt}")

        job_log.info("Inference start")
        seed = get_seed(input)
        params = {
            key: input[key] if input[key] is not None else default
            for key, default in GENERATION_DEFAULTS.items()
        }
        
        if model is None:
            return {"error": f"Model {MODEL_ID} was not loaded correctly."}
            
        image = model(
            prompt,
            **params,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        job_log.info("Inference done")

        # upload_image takes a file path, so stage the image on disk first
        with tempfile.TemporaryDirectory(prefix=f"{event_id}-") as staging:
            path = Path(staging) / "image.png"
            image.save(path, format="PNG")
            job_log.info("Image upload start")
            image_url = rp_upload.upload_image(event_id, str(path))
            job_log.info("Image upload done")

        # Runpod wraps the return value in "output" itself
        return {"image_url": image_url, "seed": seed}
    except Exception as e:
        job_log.exception("An error occurred during image generation")
        return {"error": str(e)}


if __name__ == "__main__":
    model = get_model()
    runpod.serverless.start({"handler": handler})
