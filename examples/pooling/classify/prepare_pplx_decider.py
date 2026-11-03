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
    if "attention_mode" not in decision and decision.get("format_version") != 1:
        raise ValueError("Only legacy v1 checkpoints may omit the attention mode")
    attention = decision.get("attention_mode", "causal")
    if attention not in {"causal", "noncausal_full_attention"}:
        raise ValueError("Unexpected attention contract")
    if decision.get("pooling", "last") != "last":
        raise ValueError("Unexpected pooling contract")
    joint_path = source / "joint_config.json"
    joint = json.loads(joint_path.read_text()) if joint_path.exists() else None
    if joint is not None and (
        joint.get("version") != 3
        or joint.get("template") != "joint-question-inline-readout-v1"
        or joint.get("marker") != "<|fim_suffix|>"
        or joint.get("readout_offset") != 0
        or attention != "noncausal_full_attention"
    ):
        raise ValueError("Unsupported joint readout protocol")
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
        architectures=[
            "PplxDeciderForTokenClassification"
            if joint
            else "PplxDeciderForSequenceClassification"
        ],
        num_labels=len(decision["codes"]),
        decision_attention_mode=attention,
        decision_pooling="inline_marker" if joint else "last",
        is_causal=attention == "causal",
    )
    config["text_config"]["is_causal"] = config["is_causal"]
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file({"readout.weight": head}, str(args.output / "decision-head.safetensors"))
    index["weight_map"]["readout.weight"] = "decision-head.safetensors"
    index["metadata"]["total_size"] += head.numel() * head.element_size()
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
