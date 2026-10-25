# FLUX Image Generation on Runpod Serverless

A [Runpod serverless](https://docs.runpod.io/serverless/overview) worker that turns a text prompt into an image with [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev). It validates the request, runs FLUX on the GPU, uploads the PNG to an S3-compatible bucket, and returns a link to the image along with the seed used, so the result can be reproduced.

```jsonc
// Request
{ "input": { "prompt": "A dice on top of a hill", "seed": 42 } }

// Response
{ "status": "COMPLETED", "output": { "image_url": "https://<bucket>/...png", "seed": 42 } }
```

## API

| Field                 | Type   | Default  | Allowed                  | Notes                                                     |
| --------------------- | ------ | -------- | ------------------------ | --------------------------------------------------------- |
| `prompt`              | string | required |                          | What to generate.                                         |
| `seed`                | int    | random   | 0 to 2³²−1               | Always returned, so any image can be regenerated exactly. |
| `width`               | int    | 1024     | 256–1536, multiple of 16 | FLUX works on 16×16 pixel patches.                        |
| `height`              | int    | 1024     | 256–1536, multiple of 16 |                                                           |
| `num_inference_steps` | int    | 50       | 1–100                    | More steps take longer and give finer detail.             |
| `guidance_scale`      | float  | 3.5      | 0–20                     | How closely the image follows the prompt.                 |

Invalid input, such as an out-of-range value, a wrong type or an unknown field, fails fast with a list of errors, before any GPU time is spent:

```jsonc
{ "status": "FAILED", "error": ["width does not meet the constraints."] }
```

## How it works

```
handler.py        Request entrypoint: validate → generate → upload → respond
model.py          Loads the pipeline for the current mode (real FLUX or a mock)
config.py         HANDLER_MODE, model ID and cache location
logging_setup.py  Console logging, tagged with the Runpod job ID
tests/            Automated tests, run against the mock pipeline
```

The model loads once, when the worker starts, and is reused for every request. `HANDLER_MODE` controls where the model comes from:

| Mode                 | Model                                                                                                                | Used for                                           |
| -------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `deployed` (default) | Real FLUX from Runpod's cached model volume, offline, CUDA                                                           | Production. The container needs no extra config.   |
| `local`              | Real FLUX, downloaded to your local Hugging Face cache, on CUDA, Apple Silicon (MPS) or CPU                          | Occasional checks against the real model.          |
| `mock`               | A stand-in pipeline with no weights. It returns a coloured image derived from the seed, with the prompt drawn on it. | Tests and day-to-day development. Runs in seconds. |

### Design decisions

- **Mock pipeline for development.** FLUX.1-dev is about 33GB and needs a large GPU, but almost all of this worker's logic sits around the model: validation, seeding, saving, upload and error handling. The mock has the same call signature, so that logic runs in seconds on a laptop, and the tests need no GPU or download.
- **The model is read from Runpod's cache, offline.** `deployed` mode reads the model from the cached model volume with `HF_HUB_OFFLINE=1`. That avoids downloading tens of GB on every cold start, and a misconfigured worker fails at startup instead of quietly downloading the model.
- **Bounded generation parameters.** Clients can tune size, steps and guidance, but within limits. Serverless GPU time is billed per second, so a request for 10,000 steps or an 8K image would be expensive.
- **Reproducible builds.** Dependencies are pinned in `uv.lock`, and the Docker image installs from that lockfile, so the deployed worker runs the same package versions as local development.

## Running locally

> [!IMPORTANT]
> This project requires [uv](https://docs.astral.sh/uv/) and Python 3.14. This project also has `just` as a dev dependency to help with common CLI commands, so after `uv sync` you can run it as `uv run just <recipe>` (or `just <recipe>` if it is accesible in your `PATH`). You can run `just --list` to see every recipe and what they do.

```sh
uv sync
just test        # run the automated tests
just run-mock    # run the handler once on test_input.json, using the mock pipeline
```

`run-mock` uses Runpod's local test mode: it runs the job in `test_input.json` through the handler and prints the result. If no bucket is configured, the image is written to `simulated_uploaded/`. To upload to a real bucket instead, copy `.env.sample` to `.env` and fill in the credentials.

### Running the real model locally

```sh
uv run hf auth login   # FLUX.1-dev is gated: accept the license on Hugging Face first
just download          # ~33GB into ~/.cache/huggingface
just run               # 512×512, 20 steps
```

FLUX.1-dev needs about 33GB of memory in bf16, so this is only practical on a large-memory machine or a local NVIDIA GPU. For most purposes, the mock pipeline and a deployed endpoint cover more.

## Deploying to Runpod

1. **Build and push the image.** Change the image tag in the `justfile` to your own registry first.
   ```sh
   just build
   just push
   ```
2. **Create a serverless endpoint** in the Runpod console from that Docker image.
3. **Choose a GPU with at least 48GB of VRAM.** The weights alone are about 33GB in bf16.
4. **Set the endpoint's cached model** to `black-forest-labs/FLUX.1-dev`. Because the model is gated, this needs a Hugging Face token with access to it. Runpod mounts the cached model at `/runpod-volume/huggingface-cache`, where the worker looks for it.
5. **Set the bucket environment variables** `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID` and `BUCKET_SECRET_ACCESS_KEY`. Without them, images are saved on the worker's disk, and the client can't reach them. Uploaded images are returned as presigned URLs, valid for 7 days.

Leave `HANDLER_MODE` unset, since `deployed` is the default.

### Calling the endpoint

```sh
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "A dice on top of a hill", "seed": 42}}'
```

`/runsync` waits for the result. For longer jobs, use `/run`, which returns a job ID, and then poll `/status/<JOB_ID>`.

## Notes from building this

### Observations

- Delay time has a big difference (200x - 300x faster, from 30+ seconds down to around 100ms) on an active instance vs a cold start. This is even with model caching enabled. If there was a steady stream of expected traffic, it would be worth setting the minimum active workers to something greater than `0`, or increasing the idle timout to something greater than the default `5` seconds.
- The docs don't give much guidance on a testing strategy with a mock model, so that you can isolate testing on the handler logic. A lot of local machines (mine included) wouldn't be viable for running the full weight models for quick handler testing. I implemented this in my own way, but documenting a best practice could help with local testing for users.
- I was getting OOM errors before on initialisation, so before any request was even sent or handled. The [model caching tutorial](https://docs.runpod.io/tutorials/serverless/model-caching-text) and [handler function best practices](https://docs.runpod.io/serverless/workers/handler-functions#handler-function-best-practices) loads the model at module level (this is what I initially tried), but this breaks on Python 3.14.
  - TODO: I am confirming this on a Python 3.11 image that follows the docs instead of my solution.
  - The Runpod SDK runs its [heartbeat in a child process](https://github.com/runpod/runpod-python/blob/397c37d3d2a04aaa2eb3c8a83cdc359e230bcb6e/runpod/serverless/modules/rp_ping.py#L83-L87), and Python 3.14 [changed Linux's default start method](https://docs.python.org/3/whatsnew/3.14.html#multiprocessing) from `fork` to `forkserver`. The forkserver is a fresh Python process that [imports the main module](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods) before forking children, so it ran `handler.py` again and loaded a second copy of the model. From to docs: *"Make sure that the main module can be safely imported by a new Python interpreter without causing unintended side effects"*
  - With FLUX this needed roughly 64GB on a 44GB GPU and crashed the worker with `CUDA out of memory`. The tutorial uses Python 3.11 and a small model, so the problem never shows there. Loading the model under `if __name__ == "__main__":` keeps the "load once at startup" behaviour and works on any Python version.

### Documentation discrepancies

In [Step 8: Deploy your worker using the Runpod console](https://docs.runpod.io/serverless/quickstart#step-8-deploy-your-worker-using-the-runpod-console), steps 2 and 3 don't match the current console:

- There is no **New Endpoint** button. It's now **Get started**.
- There is no **Import from Docker Registry** button. It's now **Deploy from a Docker image**.
