import os

MODEL_ID = "black-forest-labs/FLUX.1-dev"
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

# How the worker gets its model, selected with the HANDLER_MODE env var:
#   deployed: real model from the Runpod cache volume, offline, CUDA only (default,
#             so the container needs no extra config).
#   local:    real model from the local Hugging Face cache, downloading it if missing,
#             on the best available device. Use small generation params in the request
#             (see `just run`), as full size is very slow without a datacenter GPU.
#   mock:     stand-in pipeline with no weights, for exercising the handler logic.
MODE = os.environ.get("HANDLER_MODE", "deployed")
if MODE not in ("deployed", "local", "mock"):
    raise ValueError(f"HANDLER_MODE must be deployed, local or mock, got '{MODE}'")

# Must be set before huggingface_hub is imported, which happens when the model loads.
# model.py imports it lazily, so importing this module first is enough.
if MODE == "deployed":
    os.environ["HF_HUB_OFFLINE"] = "1"

if MODE == "local" or MODE == "mock":
    from dotenv import load_dotenv
    load_dotenv()
