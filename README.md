# Runpod Serverless Application

This repo is for a [Runpod serverless](https://docs.runpod.io/serverless/overview) application that will generate images from a prompt
using the [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) model.

## Observations

- Sending the initial request to the live endpoint with just "Hello world" triggered the initialisation of 3 GPUs. The max workers is set to 3, so it initialised to that limit.

## Documentation Discrepancies

[Step 8: Deploy your worker using the Runpod console](https://docs.runpod.io/serverless/quickstart#step-8-deploy-your-worker-using-the-runpod-console):
this section in the documentation for step 2 and 3 is inaccurate.

- There is no **New Endpoint** button, instead there is a **Get started** button.
- There is no **Import from Docker Registry** button, instead there is a **Deploy from a Docker image** button.
