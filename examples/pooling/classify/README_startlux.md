# StartLux-35B-A3B text decisions on vLLM v0.26.0

This backport adds native decision pooling for StartLux-Decision-35B-A3B.
It reuses Qwen3.5 MoE and returns FP32 A–Z logits from the last input token.
The serving example uses the checkpoint's prompt renderer, candidate mapping,
wide-choice handling and calibrated temperatures to produce `/v1/systemone`
responses. No decoding step or new Ascend model implementation is required.

| Component | Fixed version |
| --- | --- |
| vLLM base | `v0.26.0`, `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vLLM branch | `startlux_35b_v026_npu_eager` |
| vLLM-Ascend | `cf0baa38dfb2aef4faf6baaaa97beb4ac974ae35`, unchanged source |
| StartLux source | `0e7a2e81b9c92756e26d8edd843a44d50e362669` |
| Model | `startlux-models/StartLux-Decision-35B-A3B` |
| Mode | BF16, eager, TP=2, PP=1, text input |

The v0.26.0 tag and Ascend's `vllm-main-verified.commit` are different revisions.
This branch explicitly targets the released tag, not the verified main snapshot.
Graph execution, images, LoRA and pipeline parallelism are not validated by this
backport. The HTTP example rejects images explicitly.

## Prepare an isolated environment

Use a new container with the assigned Ascend devices and an existing CANN 9.1.0,
PyTorch 2.10.0 and torch-npu 2.10.0.post4 environment. The tested base image is:

```text
quay.nju.edu.cn/ascend/vllm-ascend@sha256:243bf20fad2f4f6de56aa1b041bcdfdbe5bb9d485f3190f40f1c5d0f418fc469
```

The image contains vLLM v0.26.0 and an earlier Ascend checkout. Fix the Ascend
source revision explicitly; do not treat the image package version as proof of
its source revision. Its `csrc`, `cmake`, `CMakeLists.txt`, `setup.py` and
`pyproject.toml` match the target Ascend revision, so the existing native
artifacts can be reused for this Python-only backport.

Create the container on the host, substituting your model and work directories.
The example uses allocated devices 0 and 1. Do not select occupied devices.

```bash
docker run -d --name startlux-35b-v026 --network host --shm-size 16g \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /path/to/work:/workspace/startlux \
  -v /path/to/models:/models:ro \
  quay.nju.edu.cn/ascend/vllm-ascend@sha256:243bf20fad2f4f6de56aa1b041bcdfdbe5bb9d485f3190f40f1c5d0f418fc469 \
  sleep infinity
docker exec --privileged -it startlux-35b-v026 bash
```

Inside the container, with `uv` installed, use isolated source checkouts and
reuse this exact image's compatible native artifacts:

```bash
cd /workspace/startlux
uv venv --system-site-packages .venv
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

git clone --branch startlux_35b_v026_npu_eager \
  https://github.com/Liuchenbing-2026/vllm.git vllm
cp /vllm-workspace/vllm/vllm/_version.py vllm/vllm/_version.py

git clone https://github.com/vllm-project/vllm-ascend.git vllm-ascend
git -C vllm-ascend checkout cf0baa38dfb2aef4faf6baaaa97beb4ac974ae35
cp -a /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom \
  /vllm-workspace/vllm-ascend/vllm_ascend/lib \
  /vllm-workspace/vllm-ascend/vllm_ascend/*.so vllm-ascend/vllm_ascend/

git clone https://github.com/StartLuxLabs/StartLux-Decision.git source
git -C source checkout 0e7a2e81b9c92756e26d8edd843a44d50e362669
export PYTHONPATH="$PWD/vllm:$PWD/vllm-ascend:$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)'
```

Both printed import paths must resolve to the isolated checkouts. The image's
installed package metadata is retained; source revisions above identify the
code being executed. No package upgrade or native rebuild was used in validation.

Mount the complete model directory read-only at `/models/StartLux-Decision-35B-A3B`.
It must include the original tokenizer, chat template and `decision_config.json`;
no model configuration or weights need to be edited.

## Start the service

Run only after the two assigned cards are free. The device IDs below are
examples; substitute the IDs allocated to your container.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTHONPATH="/workspace/startlux/vllm:/workspace/startlux/vllm-ascend:/workspace/startlux/source${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_DETERMINISTIC=true
export LCCL_DETERMINISTIC=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0
export OMP_NUM_THREADS=4
cd /workspace/startlux/vllm
/workspace/startlux/.venv/bin/python \
  examples/pooling/classify/serve_startlux_decision.py \
  --model /models/StartLux-Decision-35B-A3B \
  --tensor-parallel-size 2 --max-model-len 8192 \
  --port 18326 --enforce-eager
```

The server binds to loopback. In the tested host environment, `docker exec --privileged` was needed for driver access; a successful import without device
access does not validate NPU execution.

```bash
curl http://127.0.0.1:18326/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"Paris is the capital of France.","questions":{"capital":{"type":"choice","instructions":"What is the capital of France?","criteria":{"Paris":null,"Berlin":null}}}}'
```

## Validate

Run the regression tests using the configured environment. If the image lacks
the test-only imports, install them into the isolated virtual environment with
`uv pip install pytest tblib anthropic`:

```bash
cd /workspace/startlux/vllm
/workspace/startlux/.venv/bin/python -m pytest \
  tests/models/multimodal/pooling/test_startlux_decision.py -q
```

The tests cover last-token selection, raw FP32 candidate logits, tied/untied
head loading, missing-head rejection and the older Qwen3.5 constructor's
missing tokenizer attribute. The native model registration is also included
in `tests/models/registry.py` for the existing model-loading tests.

For the official seven-suite evaluation and optional latency measurement:

```bash
cd /workspace/startlux/source
/workspace/startlux/.venv/bin/python eval/suites.py predict \
  --endpoint http://127.0.0.1:18326 --out /workspace/results/startlux-35b-v026
/workspace/startlux/.venv/bin/python eval/suites.py score \
  /workspace/results/startlux-35b-v026 --json /workspace/results/seven-metrics.json
/workspace/startlux/.venv/bin/python eval/latency.py \
  http://127.0.0.1:18326/v1/systemone 200
```

Full evaluation and latency commands are provided for reproduction; their
presence does not imply full-suite or performance acceptance on this branch.

## Validation on this branch

Measured with the pinned sources above on two Ascend 910B4-1 devices:

| Check | Result |
| --- | --- |
| Repository pre-commit checks | Passed for all changed files |
| Focused unit tests | 3 passed |
| Real BF16 TP=2 model and HTTP protocol | Passed choice, yes/no, score and reversed-choice requests |
| Fixed first 8 rows from each of seven suites | 56 requests, 88 decisions |
| Argmax agreement with the saved v0.30 reference | 88/88 |
| Strict absolute probability tolerance of 0.001 | Failed: 66/88 decisions exceed tolerance; maximum difference 0.032216 |
| Full seven-suite accuracy and latency | Not measured on this backport |

The cross-version subset is a regression probe, not an estimate of full-suite
accuracy or proof of numerical equivalence. The saved reference used a different
parallel configuration; this comparison does not isolate the source of the
probability differences. The tolerance was retained and its failure recorded.
The v0.30 full-suite score must not be attributed to this v0.26 branch.
