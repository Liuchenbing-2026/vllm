# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve an exported Decider checkpoint using the pinned official prompt code.

Add the original checkpoint's source/src directory to PYTHONPATH.
"""

import argparse
import json
import math
import threading
from pathlib import Path
from typing import Any, Literal

import torch
import uvicorn
from autojev.model import answer, decision_messages, open_image, options
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from transformers import AutoProcessor

from vllm import LLM


class DecisionQuestion(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str = "Choose the best matching option."
    criteria: dict[str, Any] | list[Any] | None = None

    @model_validator(mode="after")
    def validate_criteria(self):
        if self.type == "choice":
            if (
                not isinstance(self.criteria, dict)
                or not 1 <= len(self.criteria) <= 255
            ):
                raise ValueError("choice requires 1 to 255 ordered criteria")
        elif self.type == "score":
            if (
                not isinstance(self.criteria, list)
                or not 2 <= len(self.criteria) <= 255
            ):
                raise ValueError("score requires 2 to 255 ordered criteria")
        elif self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError("noul criteria must be an object")
        return self


class DecisionRequest(BaseModel):
    state: Any
    questions: dict[str, DecisionQuestion]
    images: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_request(self):
        if not self.questions:
            raise ValueError("questions must be nonempty")
        if any(not value.startswith("data:image/") for value in self.images):
            raise ValueError("HTTP images must be image data URLs")
        return self


class VLLMDecider:
    def __init__(self, checkpoint: str, **kwargs):
        config = json.loads((Path(checkpoint) / "decision_config.json").read_text())
        self.codes = config["codes"]
        self.temperature = config["temperature"]
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid calibration temperature")
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.processor.image_processor.size = {
            "shortest_edge": 65536,
            "longest_edge": 262144,
        }
        self.llm = LLM(
            model=checkpoint,
            runner="pooling",
            pooler_config={"task": "classify", "use_activation": False},
            dtype="bfloat16",
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            mm_processor_kwargs={"min_pixels": 65536, "max_pixels": 262144},
            limit_mm_per_prompt={"image": 8, "video": 0},
            **kwargs,
        )

    def predict(self, rows):
        prompts = []
        for row in rows:
            text = self.processor.apply_chat_template(
                decision_messages(row, self.codes),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt = {"prompt": text}
            if row.get("images"):
                prompt["multi_modal_data"] = {
                    "image": [open_image(value) for value in row["images"]]
                }
            prompts.append(prompt)
        outputs = self.llm.encode(prompts, pooling_task="classify", use_tqdm=False)
        probabilities = []
        for row, output in zip(rows, outputs, strict=True):
            count = len(options(row["question"])[0])
            logits = output.outputs.data.float().cpu()
            if logits.shape != (len(self.codes),) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid Decider readout")
            probabilities.append(
                (logits[:count] / self.temperature).softmax(-1).tolist()
            )
        return probabilities, sum(len(output.prompt_token_ids) for output in outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18210)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--compilation-config", type=json.loads)
    args = parser.parse_args()
    engine = VLLMDecider(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=8,
        gpu_memory_utilization=0.85,
        enforce_eager=args.enforce_eager,
        compilation_config=args.compilation_config,
    )
    app = FastAPI()
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/systemone")
    def decide(request: DecisionRequest):
        rows = [
            {
                "state": request.state,
                "question": q.model_dump(),
                "images": request.images,
            }
            for q in request.questions.values()
        ]
        try:
            with lock:
                probabilities, tokens = engine.predict(rows)
            return {
                "answers": {
                    key: answer(row["question"], p)
                    for key, row, p in zip(
                        request.questions, rows, probabilities, strict=True
                    )
                },
                "usage": {"input_tokens": tokens},
            }
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
