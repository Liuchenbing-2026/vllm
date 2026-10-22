# EAGLE-4 engine hooks on the v0.30.0rc2 image baseline

| Scope | Current evidence | Remaining work |
| --- | --- | --- |
| Speculative method configuration and factory | 17 CPU checks pass with real imported modules | Full model construction |
| Runner verification hooks | 12 CPU checks pass; the unchanged baseline fails the original 8 | Real accelerator execution and graph capture |
| GDN and Mamba2 extension dispatch | 8 CPU checks pass with substituted device kernels | Numerical kernels, metadata combinations and device validation |
| Tree draft seeding | 8 CPU cases pass: one-step prefill and unsupported-width rejection | Accelerator generation and remaining plugin interfaces |
| Original EAGLE plugin registration | No missing hooks; registration succeeds on the modified checkout | Speculator/runtime compatibility, Ascend integration |
| Model quality, stability and performance | Not tested | Matched target/draft evaluation and baseline comparison |

This is an incremental development checkpoint, **not a completed EAGLE-4 port**.
The tests import the real configuration, factory, runner and layer modules; CPU
runner/layer tests replace hardware operations with recording doubles. They do
not certify accelerator arithmetic, full graph capture, model accuracy or CI.

## Fixed versions

| Component | Version |
| --- | --- |
| Base image | `quay.io/ascend/vllm-ascend:v0.30.0rc2` |
| ARM64 image manifest | `sha256:7859b75177884252d2b703eff727831ccab4e33a8bde0a9b4ac7563facc98b33` |
| vLLM base | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Task branch | `feat/eagle4-rc2-engine-hooks-20261022` |
| vLLM-Ascend image base | `8fde6d4e335fe300e2b949aab0e31659cce0811d` |
| EAGLE source baseline | `eagle-hq/EAGLE-4@f71e245a45acc16fd4ebf2b58ce59eba9b2738bf` |
| EAGLE candidate portability checkpoint | `d97f6cbb03b6b5f3a1359f5b7e5e35ddebae6faa` |
| Image runtime | CANN 9.2.0, torch 2.10.0+cpu, torch-npu 2.10.0.post7, Transformers 5.14.1 |
| Local CPU test runtime | Python 3.12, torch 2.10.0, pytest 9.1.1, Transformers 5.14.1, Pydantic 2.14.1 |

The image baseline is unchanged. Record the delivery commit using
`git rev-parse HEAD` after checking out the task branch. Runtime experiments
must bind that SHA, the plugin SHA and the Ascend SHA; a branch name alone is
not an immutable runtime identity.

## Changes and provenance

The generic registry and tree verification contracts are adapted from Apache-2.0
vLLM extension work in
[eagle-hq/vllm at dfa3fb1](https://github.com/eagle-hq/vllm/tree/dfa3fb1e45b07996a3b973c2a599dbd9172681e0).
They are backported to the fixed 0.30 source rather than replacing it with 0.32.
Existing SPDX notices are retained. EAGLE model/algorithm implementation remains
in its separately licensed external plugin; no model implementation is copied
into vLLM-Ascend.

- Configuration accepts explicitly registered names, preserves built-in names,
  includes plugin options in the computation hash, invokes configuration and
  lookahead callbacks, and selects Model Runner V2.
- The factory resolves the registered speculator lazily. Auxiliary target hidden
  states follow the registered method's declared trait.
- The runner calls the verification-input hook before forward/replay, chooses an
  optional rejection sampler, and invokes accepted-path processing before state
  commitment and the next draft. Default hooks keep the chain path.
- GDN and Mamba2 builders carry a linear-attention verification extension to the
  layer. Speculative rows use its recurrence; other rows keep their original
  kernels. Mamba2 ReplaySSM and prefix-cache layouts are rejected for this path.
- The AR loop can seed a tree with one prefill while preserving the configured
  decode width and graphs. This rc2 checkpoint rejects widths other than one or
  the configured maximum. Auxiliary-state combination uses the model capability.
- The plugin registry is lazily initialized process-local cached state. Its
  lifecycle and cross-process registration remain review points; it is not
  request state.

## Executed CPU verification

The virtual environment was created with `uv`; project-wide GPU test fixtures
are deliberately excluded by `--noconftest`. No source extraction or fake
replacement of the runner module is used.

```bash
.venv/bin/python -m pytest --noconftest \
  tests/v1/spec_decode/test_method_registry.py -q
.venv/bin/python -m pytest --noconftest \
  tests/v1/spec_decode/test_method_registry_runner.py -q
.venv/bin/python -m pytest --noconftest \
  tests/v1/spec_decode/test_rc2_verify_dispatch.py -q
.venv/bin/python -m pytest --noconftest \
  tests/v1/spec_decode/test_rc2_linear_verify.py -q
.venv/bin/python -m pytest --noconftest \
  tests/v1/spec_decode/test_rc2_mamba_verify.py -q
.venv/bin/python -m pytest --noconftest \
  tests/v1/spec_decode/test_rc2_draft_prefill.py -q
```

There are **45 distinct passing CPU cases**, accumulated across targeted runs;
this is not a claim of one full green project test run. GDN and Mamba2 fixture
failures were corrected and only affected cases rerun. Ruff 0.14.0 and syntax
checks pass. The local checkout has no compiled `vllm._C`; those operations were
not invoked by the CPU tests.

Original plugin registration was also exercised through its public entry point:

```bash
PYTHONPATH=/path/to/pinned/EAGLE-4 .venv/bin/python - <<'PY'
from eagle_harness.vllm.adapter import missing_hooks
from eagle_harness.vllm.plugin import register
from vllm.v1.spec_decode.registry import get_speculative_method
assert not missing_hooks()
assert register()
assert get_speculative_method("eagle_harness") is not None
PY
```

A warning that 0.30 is outside the plugin's tested 0.32 version range is retained.
Registration success does not resolve later runtime signature or hardware issues.

For the unchanged-baseline negative control, run the same runner tests from an
isolated checkout of `ced6857` with `PYTHONPATH` pointing there and
`--import-mode=importlib`. Check `vllm.__file__` in the log. Pytest's default
prepend mode can otherwise import the modified checkout through the test path.

## Failure history and next checks

| Finding | Resolution or remaining limitation |
| --- | --- |
| Original image plugin registration rejected 8 hooks | Modified local source now passes the original interface check; image deployment not yet repeated |
| Initial negative-control invocation imported the changed checkout | Excluded that result; fixed import isolation gives 8 baseline failures |
| GDN test refactor retained a removed module-global name | Fixture now explicitly passes metadata; assertions unchanged |
| Mamba2 test fixture lacked rc2-required metadata/cache fields | Filled the rc2 fixture fields; assertions unchanged |
| Sparse checkout omitted pre-commit helper scripts | Restored original scripts; initial failure retained, no hook bypass |
| Dummy FULL graph fixture requested missing attention metadata | Now asserts the existing rejection; no production relaxation |
| Plugin references 0.32 `target_dependent_ar` and proposal arguments | Compatibility work remains; not a runnable service checkpoint |
| Hardware kernels were substituted in CPU dispatch tests | Actual NPU kernel, graph, HCCL and model tests remain required |

Next: reconcile draft-loop interfaces, connect the Ascend factory/runner and
NPU attention/state kernels, then validate a matched official target/draft in an
authorized device window. SGLang has an independent engine adapter and is not
validated by these vLLM tests.

No working model-service, load-test or full-quality command has yet been verified
for this checkpoint. Those entry points and raw evidence will be added after
actual execution. Do not substitute synthetic primitive checks for them.
