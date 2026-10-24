import logging

log = logging.getLogger("runpod_worker")
log.setLevel(logging.DEBUG)
# The root logger has its own handler under Runpod, which would print every line twice
log.propagate = False

# Runpod captures stdout as the worker logs. Per-job logs set request_id through a
# LoggerAdapter, and everything else, like startup, falls back to the default.
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - [Request: %(request_id)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        defaults={"request_id": "N/A"},
    )
)
log.addHandler(console_handler)

log.info("Logger initialized. Ready to process jobs.")
