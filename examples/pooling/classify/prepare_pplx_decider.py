# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a local vLLM view without modifying or copying backbone weights."""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    config = json.loads((source / "config.json").read_text())
    decision = json.loads((source / "decision_config.json").read_text())
    if decision["attention_mode"] != "noncausal_full_attention":
        raise ValueError("Unexpected attention contract")
    if decision["pooling"] != "last":
        raise ValueError("Unexpected pooling contract")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        if not (source / name).is_file():
            raise FileNotFoundError(name)
    head = load_file(str(source / "readout.safetensors"))["weight"]
    if head.shape != (len(decision["codes"]), config["text_config"]["hidden_size"]):
        raise ValueError("Decision readout shape does not match the checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    for file in source.iterdir():
        if file.is_file() and file.name not in {
            "config.json",
            "model.safetensors.index.json",
            "readout.safetensors",
        }:
            (args.output / file.name).symlink_to(file)
    config.update(
        architectures=["PplxDeciderForSequenceClassification"],
        num_labels=len(decision["codes"]),
        decision_attention_mode=decision["attention_mode"],
        decision_pooling=decision["pooling"],
        is_causal=False,
    )
    config["text_config"]["is_causal"] = False
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file({"readout.weight": head}, str(args.output / "decision-head.safetensors"))
    index["weight_map"]["readout.weight"] = "decision-head.safetensors"
    index["metadata"]["total_size"] += head.numel() * head.element_size()
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
