# Laya decision model

`LayaForDecision` serves `convaiinnovations/laya` as a pooling model. Its
ModernBERT encoder uses bidirectional full and sliding attention; the decision
head scores the option marker tokens. It does not generate text.

## Prerequisites

`LayaForDecision` targets vLLM `main` (post-`v0.32.0`); older release builds
such as `v0.32.0rc1` do not include this model architecture or newer core
APIs. On Ascend, pair this tree with a matching `vllm-ascend` `main` checkout
whose `.github/vllm-main-verified.commit` aligns with the vLLM baseline.

Build and install the selected source revisions using the platform's documented
installation procedure. Resolve dependencies from those revisions with the
platform's constraints, and check the installed packages with `uv pip check`.
Verify the package versions and runtime import paths before serving; mounting
new source files over an older installed package does not establish a compatible
installation. Do not override version metadata to bypass compatibility checks.

## Prepare the checkpoint

The original repository stores encoder configuration and tokenizer files in
subdirectories. Create a serving view without modifying the weights. The
following script expects an already downloaded checkpoint in `./laya` and
creates `./laya-serving`:

```python
import json
from pathlib import Path

from transformers import AutoTokenizer

root = Path("laya").resolve()
target = Path("laya-serving").resolve()
target.mkdir(exist_ok=True)
settings = json.loads((root / "rl_agent_config.json").read_text())
config = json.loads((root / "encoder/config.json").read_text())
tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
config.update({
    key: settings[key]
    for key in ("head_layers", "temperature_by_options")
})
config.update(
    architectures=["LayaForDecision"],
    decision_temperature=settings["temperature"],
    dtype="bfloat16",
    n_act=len(settings["act_costs"]) + 1,
    mask_token_id=tokenizer.mask_token_id,
    type_token_ids={
        kind: tokenizer(kind + " question:", add_special_tokens=False)["input_ids"][0]
        for kind in ("choice", "score", "noul")
    },
)
(target / "config.json").write_text(json.dumps(config, indent=2))
for source in [root / "model.safetensors", *(root / "tokenizer").iterdir()]:
    destination = target / source.name
    if source.is_file() and not destination.exists():
        destination.symlink_to(source)
```

`decision_temperature` preserves Laya's per-question-type calibration. Transformers
5 removes the generation field `temperature` when loading model configurations.
Recreate older prepared configurations with this recipe to retain calibration.

## Serve and query

```bash
vllm serve ./laya-serving --served-model-name laya --runner pooling \
  --pooler-config '{"task":"classify","use_activation":true}' \
  --dtype bfloat16 --max-model-len 512 --max-num-batched-tokens 512 \
  --max-num-seqs 1 --no-enable-prefix-caching --no-enable-chunked-prefill \
  --enforce-eager
```

The example above uses a 512-token budget; `--max-model-len` and
`--max-num-batched-tokens` may be raised up to the encoder configuration's
`max_position_embeddings` (keeping `--max-num-batched-tokens` at least
`--max-model-len` while chunked prefill is disabled).

Use the checkpoint's `rl_common.build_sequence` to construct prompt token IDs
for one question, including its type and option markers. Send those IDs as
`input` to `/classify`. The returned vector contains the option probabilities
followed by the action probabilities. The decision head runs in FP32; the
encoder can run in BF16. Preserve the checkpoint's per-type and per-option
count temperatures when comparing against its original `RLAgent` reference.

Graph execution depends on the platform's encoder attention implementation.
Validate the selected framework revisions and execution mode against the original
`RLAgent` reference before deployment, including short and long prompts. Results
from another revision or execution mode do not validate the installed combination.
