FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

WORKDIR /

# Install the exact versions from uv.lock into the system Python, so the image matches
# local development. --no-install-project skips installing this repo as a package.
COPY pyproject.toml uv.lock /
RUN UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen --no-dev --no-install-project --no-cache

# Copy your handler files
COPY handler.py config.py logging_setup.py model.py /

# Start the container
CMD ["python3", "-u", "handler.py"]
