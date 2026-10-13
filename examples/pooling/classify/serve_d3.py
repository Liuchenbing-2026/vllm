# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve d3 using its pinned request renderer and native vLLM pooling.

Place the official v3.1.0 checkpoint directory on PYTHONPATH. Export a separate
vLLM checkpoint view with prepare_d3.py before starting this example.
"""

import argparse
import json
import math
import threading
from pathlib import Path
from typing import Any

import torch
import uvicorn
from d3_runtime import D3, answer_codes, load_processor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoTokenizer

from vllm import LLM


class D3RequestRenderer(D3):
    """Reuse official validation and rendering without loading its HF backbone."""

    def __init__(self, checkpoint: str, max_length: int):
        self.root = Path(checkpoint)
        self.config = json.loads((self.root / "decision_config.json").read_text())
        if self.config.get("format_id") != "d3-code-readout-v1":
            raise ValueError("Unexpected d3 request format")
        if self.config.get("prompt") != "d3":
            raise ValueError("Unexpected d3 prompt family")
        self.prompt = self.config["prompt"]
        self.temperature = float(self.config["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be positive and finite")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.codes, self.token_ids = answer_codes(self.tokenizer)
        if (
            self.codes != self.config["codes"]
            or self.token_ids != self.config["token_ids"]
        ):
            raise ValueError("Checkpoint answer codes disagree with its tokenizer")
        self.processor = load_processor(self.root)
        self.max_length = max_length
        self.permutation_average = False
        self.model_name = "d3-v3.1.0"
        self.image_unavailable = None
        self.video_unavailable = "This serving example supports text and images only"


class VLLMD3:
    def __init__(self, checkpoint: str, **kwargs):
        self.renderer = D3RequestRenderer(checkpoint, kwargs.get("max_model_len", 8192))
        self.llm = LLM(
            model=checkpoint,
            runner="pooling",
            pooler_config={"task": "classify", "use_activation": False},
            dtype="bfloat16",
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            mm_processor_kwargs={"min_pixels": 65536, "max_pixels": 1638400},
            limit_mm_per_prompt={"image": 8, "video": 0},
            **kwargs,
        )

    def system_one(self, *, state, questions, images=None, videos=None):
        prepared = self.renderer.prepare(state, questions, images, videos)
        keys = prepared.runnable
        prompts = []
        for key in keys:
            if prepared.images:
                prompts.append(
                    {
                        "prompt": prepared.texts[key],
                        "multi_modal_data": {"image": prepared.images},
                    }
                )
            else:
                prompts.append({"prompt_token_ids": prepared.sequences[key]})
        probabilities = {}
        tokens = 0
        if prompts:
            outputs = self.llm.encode(prompts, pooling_task="classify", use_tqdm=False)
            for key, output in zip(keys, outputs, strict=True):
                logits = output.outputs.data.float().cpu()
                if logits.shape != (len(self.renderer.codes),):
                    raise RuntimeError("Unexpected d3 readout shape")
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite d3 readout")
                count = len(prepared.questions[key].keys)
                probabilities[key] = (
                    (logits[:count] / self.renderer.temperature).softmax(-1).tolist()
                )
                tokens += len(output.prompt_token_ids)
        return self.renderer.respond(prepared, probabilities, tokens)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    state: Any
    questions: dict[str, dict[str, Any]]
    images: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_images(self):
        if any(not value.startswith("data:image/") for value in self.images):
            raise ValueError("HTTP images must be image data URLs")
        return self


def create_app(model: VLLMD3) -> FastAPI:
    app = FastAPI()
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/systemone")
    def decide(request: DecisionRequest):
        try:
            with lock:
                images = model.renderer.load_images(request.images, strict=True)
                return model.system_one(
                    state=request.state, questions=request.questions, images=images
                )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--port", type=int, default=18531)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--compilation-config", type=json.loads, default=None)
    args = parser.parse_args()
    model = VLLMD3(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=8,
        gpu_memory_utilization=0.90,
        enforce_eager=args.enforce_eager,
        compilation_config=args.compilation_config,
    )
    uvicorn.run(create_app(model), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
