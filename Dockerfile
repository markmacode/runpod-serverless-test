FROM python:3.14-slim

WORKDIR /

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    torch \
    diffusers

# Copy your handler file
COPY handler.py /

# Start the container
CMD ["python3", "-u", "handler.py"]
