# d3 decision pooling

This example targets `vllm-sr/d3` revision
`cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75` (27B) and
`vllm-sr/d3-flash` revision
`60962e45fac468330988f8409d13f43c19195615` (Qwen3.5-9B).
Select the matching pinned checkpoint directory; hidden dimensions and the
readout shape are read from the checkpoint rather than assumed from its size.
The NPU deployment targets vLLM v0.30.0 with vLLM-Ascend v0.30.0rc2.
The Qwen3.8-27B backbone uses bidirectional full-attention layers, causal
recurrent layers and a separate FP32 decision readout. It returns candidate
logits from the last input position without generating text.

Download and hash-check the original checkpoint, including its official Python
request renderer, tokenizer, processor and `decision_config.json`. Keep this
directory intact. Create a separate checkpoint view so the independent readout
is included in the weight index:

```bash
python examples/pooling/classify/prepare_d3.py /models/d3-v3.1.0 /models/d3-vllm
export PYTHONPATH="/models/d3-v3.1.0${PYTHONPATH:+:$PYTHONPATH}"
python examples/pooling/classify/serve_d3.py \
  --model /models/d3-vllm --enforce-eager --port 18531
```

The view symlinks backbone shards without copying or changing their tensors.
The server reuses the official renderer for choice, yes/no and score requests,
candidate ordering, validation and result aggregation. FP32 logits are restricted
to each question's candidates before temperature scaling and softmax.

```bash
curl --fail http://127.0.0.1:18531/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"The parcel arrived damaged.","questions":{"route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"returns":null,"billing":null}}}}'
```

HTTP images must be image data URLs. The example binds to loopback and supports
text and images; video serving is not implemented here. Hardware acceptance,
graph replay, full-suite accuracy and performance are separate checks. A model
registration or successful server start alone does not establish those results.
