import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from runpod.serverless.utils.rp_validator import validate

import handler
from handler import bounded, schema
from model import MockPipeline


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """Give the handler the stand-in pipeline, which it otherwise loads under __main__."""
    monkeypatch.setattr(handler, "model", MockPipeline())


@pytest.fixture(autouse=True)
def uploads(monkeypatch):
    """
    Replace the upload with one that keeps each image in memory, so tests never
    reach a bucket (local and mock modes load .env, which may hold real credentials)
    or write to simulated_uploaded/.
    """
    uploaded = []

    def fake_upload(job_id, path):
        with Image.open(path) as image:
            uploaded.append({"path": Path(path), "image": image.copy()})
        return f"https://bucket.test/{job_id}/image.png"

    monkeypatch.setattr(handler.rp_upload, "upload_image", fake_upload)
    return uploaded


@pytest.fixture
def model_calls(monkeypatch):
    """Record the keyword arguments each model call receives."""
    calls = []
    model = handler.model

    def spy(prompt, **kwargs):
        calls.append(kwargs)
        return model(prompt, **kwargs)

    monkeypatch.setattr(handler, "model", spy)
    return calls


def run(job_input: dict) -> dict:
    return handler.handler({"id": "test-job", "input": job_input})


# --- Input validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"guidance_scale": 3.5},
        {"guidance_scale": 0},
        {"guidance_scale": 20},
        {"width": 256, "height": 1536},
        {"num_inference_steps": 1},
        {"num_inference_steps": 100},
        {"seed": 0},
        {"seed": 2**32 - 1},
    ],
)
def test_accepts_values_within_bounds(params):
    assert "errors" not in validate({"prompt": "x", **params}, schema)


@pytest.mark.parametrize(
    "params",
    [
        {"width": 500},  # not a multiple of 16
        {"width": 1552},
        {"height": 240},
        {"num_inference_steps": 0},
        {"num_inference_steps": 101},
        {"guidance_scale": -0.5},
        {"guidance_scale": 20.5},
        {"seed": -1},
        {"seed": 2**32},
    ],
)
def test_rejects_values_out_of_bounds(params, uploads):
    result = run({"prompt": "x", **params})

    (key,) = params
    assert result == {"error": [f"{key} does not meet the constraints."]}
    assert uploads == []


def test_rejects_wrong_type():
    result = run({"prompt": "x", "guidance_scale": "high"})

    assert "guidance_scale should be <class 'float'> type" in result["error"][0]


def test_rejects_missing_prompt():
    assert run({}) == {"error": ["prompt is a required input."]}


def test_rejects_unknown_input():
    result = run({"prompt": "x", "steps": 20})

    assert result == {"error": ["Unexpected input. steps is not a valid input option."]}


def test_validator_skips_constraints_when_value_matches_default_type():
    """
    Documents why the generation params default to None in the schema: Runpod's
    validator skips constraints for any value with the same type as its default.
    If this fails, the validator has been fixed and the real defaults can move back
    into the schema.
    """
    quirky_schema = {
        "steps": {
            "type": int,
            "required": False,
            "default": 50,
            "constraints": lambda v: v <= 100,
        }
    }

    assert "errors" not in validate({"steps": 5000}, quirky_schema)


def test_bounded_leaves_missing_and_wrong_typed_values_to_the_validator():
    constraint = bounded(0, 10)

    assert constraint(None)
    assert constraint("not a number")


def test_bounded_checks_multiple_of():
    constraint = bounded(0, 100, multiple_of=16)

    assert constraint(32)
    assert not constraint(40)


# --- Handler ------------------------------------------------------------------------


def test_returns_image_url_and_seed():
    result = run({"prompt": "A dice on top of a hill", "seed": 42})

    # Runpod wraps this in "output" itself, so it mustn't be nested here
    assert result == {"image_url": "https://bucket.test/test-job/image.png", "seed": 42}


def test_generates_a_seed_when_none_given():
    seed = run({"prompt": "x"})["seed"]

    assert 0 <= seed <= 2**32 - 1


def test_applies_defaults_for_missing_params(model_calls, uploads):
    run({"prompt": "x"})

    (call,) = model_calls
    assert call["width"] == 1024
    assert call["height"] == 1024
    assert call["num_inference_steps"] == 50
    assert call["guidance_scale"] == 3.5
    assert uploads[0]["image"].size == (1024, 1024)


def test_passes_request_params_to_model(model_calls, uploads):
    run(
        {
            "prompt": "x",
            "width": 512,
            "height": 256,
            "num_inference_steps": 20,
            # Falsy, so this also checks it isn't mistaken for missing
            "guidance_scale": 0,
        }
    )

    (call,) = model_calls
    assert call["num_inference_steps"] == 20
    assert call["guidance_scale"] == 0
    assert uploads[0]["image"].size == (512, 256)


def test_same_seed_gives_same_image(uploads):
    run({"prompt": "x", "seed": 7})
    run({"prompt": "x", "seed": 7})
    run({"prompt": "x", "seed": 8})

    first, repeat, other = (upload["image"].tobytes() for upload in uploads)
    assert first == repeat
    assert first != other


def test_removes_staged_image_after_upload(uploads):
    run({"prompt": "x"})

    assert not uploads[0]["path"].exists()


def test_returns_error_when_inference_fails(monkeypatch, uploads):
    def broken_model(prompt, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(handler, "model", broken_model)

    assert run({"prompt": "x"}) == {"error": "CUDA out of memory"}
    assert uploads == []


def test_importing_handler_does_not_load_model():
    """
    Runpod's heartbeat runs in a child process that re-imports handler.py (forkserver
    on Linux, spawn on macOS). Loading the model at import put a second copy of FLUX
    on the GPU and ran it out of memory.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import handler; print(handler.model)"],
        cwd=Path(__file__).parent.parent,
        env=os.environ | {"HANDLER_MODE": "mock"},
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "None"
    assert "Loading model" not in result.stderr
