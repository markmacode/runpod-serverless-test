# Run the automated tests against a stand-in model (seconds, no GPU or download)
test:
    uv run pytest

# Run the handler once on test_input.json with a stand-in model
run-mock:
    HANDLER_MODE=mock uv run handler.py

# Run the handler once with the real model locally (needs `just download` first)
run:
    HANDLER_MODE=local uv run handler.py --test_input '{"input": {"prompt": "A dice on top of a hill", "width": 512, "height": 512, "num_inference_steps": 20}}'

# Download FLUX.1-dev into the local Hugging Face cache (~33GB, gated: needs `hf auth login`)
download:
    uv run hf download black-forest-labs/FLUX.1-dev --exclude flux1-dev.safetensors ae.safetensors

build:
    docker build --platform linux/amd64 --tag markbromell/runpod-serverless-test .

push:
    docker push markbromell/runpod-serverless-test
