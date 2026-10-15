# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI compatibility and fail-closed device transport contracts."""

import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.entrypoints.cli.preload import PreloadSubcommand
from vllm.model_executor.model_loader.weight_cache import daemon, protocol
from vllm.platforms.interface import Platform, PlatformEnum
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.mark.parametrize("model_args", [["/checkpoint"], ["--model", "/checkpoint"]])
def test_preload_model_arguments(model_args):
    parser = FlexibleArgumentParser()
    command = PreloadSubcommand()
    command.subparser_init(parser.add_subparsers())
    args = parser.parse_args(["preload", *model_args, "--tensor-parallel-size", "2"])
    with patch.object(daemon.EngineArgs, "from_cli_args") as create:
        create.side_effect = RuntimeError("stop before device initialization")
        with pytest.raises(RuntimeError, match="stop before device"):
            command.cmd(args)
        parsed = create.call_args.args[0]
        assert parsed.model == "/checkpoint"
        assert parsed.tensor_parallel_size == 2


@pytest.mark.parametrize("mode", ["pipeline", "data"])
def test_preload_rejects_unsupported_parallelism(mode):
    config = SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1)
    setattr(config, f"{mode}_parallel_size", 2)
    with pytest.raises(ValueError, match=f"{mode} parallelism is not supported"):
        daemon._reject_unsupported_parallelism(config)


def test_cpu_entries_preserve_values_and_views():
    tensor = torch.arange(24).reshape(4, 6)[1:, 2:]
    entry = protocol.TensorEntry.from_tensor(tensor, "buffer")
    assert entry.ipc_args is None
    torch.testing.assert_close(entry.rebuild(0), tensor, atol=0, rtol=0)


def test_unsupported_device_does_not_copy_to_cpu():
    platform = Platform()
    platform._enum = PlatformEnum.CPU
    platform.device_type = "cpu"
    platform.device_name = "cpu"
    with (
        patch.object(protocol, "current_platform", platform),
        pytest.raises(protocol.UnsupportedPlatformForIPCError),
    ):
        protocol.TensorEntry.from_tensor(torch.empty(4, device="meta"), "param")


@pytest.mark.parametrize("kind", [PlatformEnum.CUDA, PlatformEnum.ROCM])
def test_existing_cuda_platform_support(kind):
    platform = Platform()
    platform._enum = kind
    assert platform.supports_weight_cache()


def test_cuda_rebuild_remaps_only_local_ordinal():
    args = tuple(range(15))
    with patch("torch.multiprocessing.reductions.rebuild_cuda_tensor") as rebuild:
        Platform.rebuild_weight_cache_tensor(args, 3)
    expected = list(args)
    expected[6] = 3
    assert rebuild.call_args.args == tuple(expected)
    assert args[6] == 6


def test_daemon_release_drops_compilation_layer_owners():
    from vllm.model_executor.layers import rotary_embedding

    model = torch.nn.Linear(2, 2)
    reference = weakref.ref(model)
    rope = torch.nn.Linear(2, 2)
    rope_reference = weakref.ref(rope)
    rope_cache = {("test",): rope}
    cache = daemon.WeightCacheDaemon.__new__(daemon.WeightCacheDaemon)
    cache.tp_rank = 0
    cache.model = model
    cache.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"layer": model})
    )
    del model, rope
    with (
        patch("torch.accelerator.synchronize"),
        patch("torch.accelerator.empty_cache"),
        patch.object(rotary_embedding, "_ROPE_DICT", rope_cache),
        patch.object(daemon, "send_msg") as send,
    ):
        cache._handle_release(None)
    assert reference() is None
    assert rope_reference() is None
    send.assert_called_once_with(None, {"status": "ok"})
