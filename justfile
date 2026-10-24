test:
    uv run handler.py

build:
    docker build --platform linux/amd64 --tag markbromell/runpod-serverless-test .

push:
    docker push markbromell/runpod-serverless-test
