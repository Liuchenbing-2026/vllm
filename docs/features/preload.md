# Preload model weights

`vllm preload` keeps model weights in device memory in a separate daemon.
Engines using `--load-format ipc_cache` map those weights through device IPC,
avoiding disk loading and host-to-device weight copies on restart.
Python imports, KV cache initialization, compilation and graph capture still run.

```bash
vllm preload /path/to/model --tensor-parallel-size 2
# In a second terminal, after the daemon reports READY:
vllm serve /path/to/model --tensor-parallel-size 2 \
  --load-format ipc_cache --model-loader-extra-config '{"fallback":false}'
```

Keep the preload process alive across engine restarts. Stop engines before
stopping preload: zero-copy tensors depend on the producer's allocation.
Both processes must run as the same user on the same physical devices and use
matching checkpoint, revision, dtype, quantization, TP layout and runtime options.
IPC communication is local to a host. Do not expose its Unix socket to untrusted
users. The default socket directory is private to the current user.

| Option | Behavior |
| --- | --- |
| `--weight-cache-socket-dir /path` | Select the daemon socket directory; set the engine's `socket_dir` extra config to the same path. |
| `mode: "zero_copy"` | Default: share the daemon's device allocations across restarts. |
| `mode: "copy"` | Clone into engine-owned memory, then release the daemon's cached weights. This consumes the cache. |
| `fallback: true` | Default: load from disk if the cache is unavailable or incompatible. |
| `fallback: false` | Fail instead of hiding a cache miss; use for validation and benchmarking. |

This version supports tensor and expert parallelism; the daemon rejects pipeline
and data parallelism. Its distributed rendezvous port must differ from the engine's.
The original `python -m vllm.model_executor.model_loader.weight_cache.daemon
--model ...` launcher remains available.

CUDA and ROCm use the existing PyTorch CUDA IPC transport. Out-of-tree platforms
must explicitly provide device IPC export/import and any daemon initialization.
Unsupported platforms fail before fallback; device tensors are never silently
serialized through CPU memory. Consult the platform's documentation for supported
weight layouts, allocator settings and hardware validation.
