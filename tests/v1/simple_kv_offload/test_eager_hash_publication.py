# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused regressions for the vLLM-side offload compatibility patch."""

from types import SimpleNamespace

import pytest
from vllm.config import KVTransferConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler


# Like test_scheduler.py, these CPU-only tests create no accelerator resources.
@pytest.mark.skip_global_cleanup
def test_derived_cpu_config_preserves_coordinator_role_and_capacity():
    gpu = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4096, layers=["probe"], layer_stride=4096, block_stride=256
            )
        ],
        kv_cache_groups=[],
    )
    gpu.kv_transfer_config = KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both"
    )
    cpu = SimpleCPUOffloadScheduler._derive_cpu_config(gpu, 8192)
    assert cpu.num_blocks == 32
    assert cpu.kv_cache_tensors[0].size == 8192
    assert cpu.kv_transfer_config is gpu.kv_transfer_config


def make_selector(eagle=True):
    sched = SimpleCPUOffloadScheduler.__new__(SimpleCPUOffloadScheduler)
    hashes = [bytes([i]) * 32 for i in range(5)]
    sched._gpu_block_pool = BlockPool(32, enable_caching=True, hash_block_size=16)
    sched.cpu_block_pool = BlockPool(32, enable_caching=True, hash_block_size=16)
    sched.cpu_coordinator = SimpleNamespace(
        enable_partial_hash_hits=False,
        single_type_managers=[
            SimpleNamespace(use_eagle=eagle, has_positionally_stable_blocks=True)
        ],
    )
    sched.cpu_kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(prefix_cacheable=True))
        ]
    )
    sched.group_block_sizes = (16,)
    sched.block_size = 64
    sched.hash_block_size = 16
    sched.enable_kv_cache_events = False
    sched._in_flight_store_gpu_blocks = set()
    state = SimpleNamespace(
        request=SimpleNamespace(
            num_computed_tokens=80, num_output_placeholders=0, block_hashes=hashes
        ),
        num_stored_blocks=[0],
        pending_unhashed_blocks={},
    )
    blocks = sched._gpu_block_pool.get_new_blocks(5)
    return sched, state, blocks


def publish(sched, state, blocks):
    for i, (block, h) in enumerate(zip(blocks, state.request.block_hashes)):
        sched._gpu_block_pool._insert_block_hash(
            make_block_hash_with_group_id(h, 0), block, num_tokens=(i + 1) * 16
        )


@pytest.mark.parametrize(("eagle", "count"), [(True, 5), (False, 4)])
@pytest.mark.skip_global_cleanup
def test_eager_store_retains_confirmed_verifier_block_past_lcm(eagle, count):
    """A 64-token prefix needs its confirmed 16-token EAGLE lookahead page."""
    sched, state, blocks = make_selector(eagle)
    publish(sched, state, blocks)
    ids = [b.block_id for b in blocks]
    selected, advanced, _ = sched._select_eager_blocks_to_store(state, (ids,))
    assert selected == ids[:count]
    assert advanced == [count]


@pytest.mark.skip_global_cleanup
def test_eager_retries_late_hash_publication_after_physical_ids_change():
    sched, state, blocks = make_selector()
    old_ids = [b.block_id for b in blocks]
    selected, advanced, _ = sched._select_eager_blocks_to_store(state, (old_ids,))
    assert selected == []
    state.num_stored_blocks = advanced
    assert state.pending_unhashed_blocks[0] == set(range(5))
    # Recycled SWA block tables can retain IDs which now belong to other data.
    sched._gpu_block_pool.free_blocks(blocks)
    new_blocks = sched._gpu_block_pool.get_new_blocks(5)
    publish(sched, state, new_blocks)
    selected, advanced, _ = sched._select_eager_blocks_to_store(state, (old_ids,))
    assert selected == [b.block_id for b in new_blocks]
    assert not state.pending_unhashed_blocks[0]
    assert advanced == [0]
    sched._in_flight_store_gpu_blocks.update(selected)
    assert sched._select_eager_blocks_to_store(state, (old_ids,))[0] == []
