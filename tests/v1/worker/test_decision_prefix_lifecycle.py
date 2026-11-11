"""Request lifecycle tests; MM extraction is mocked, no model/device execution."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.v1.worker.gpu.model_states.default import DefaultModelState


def state(model_ranges=None):
    s = object.__new__(DefaultModelState)
    s.mm_prefix_ranges = {}
    s.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(), get_sliding_window=lambda: None
    )
    s.model = SimpleNamespace()
    if model_ranges is not None:
        s.model.get_request_prefix_ranges = lambda req: model_ranges.get(req.req_id, [])
    s.rope_state = None
    s.prompt_embeds_state = None
    return s


@pytest.fixture
def mm_ranges():
    with patch(
        "vllm.v1.worker.gpu.model_states.default.compute_mm_prefix_ranges",
        side_effect=lambda features, **kwargs: {0: features[0]},
    ):
        yield


def add(s, req_id, features=()):
    s.add_request(0, SimpleNamespace(req_id=req_id, mm_features=list(features)))


def test_text_ranges_without_encoder_cache(mm_ranges):
    s = state({"a": [(0, 4)]})
    s.encoder_cache = None  # Later PP ranks need no encoder cache for ranges.
    add(s, "a")
    assert s.get_mm_prefix_ranges(SimpleNamespace(req_ids=["a"])) == {0: [(0, 4)]}


def test_reordered_batch_and_missing_request(mm_ranges):
    s = state({"a": [(0, 4)], "b": [(0, 7)]})
    add(s, "a")
    add(s, "b")
    assert s.get_mm_prefix_ranges(SimpleNamespace(req_ids=["b", "missing", "a"])) == {
        0: [(0, 7)],
        1: [],
        2: [(0, 4)],
    }


def test_remove_and_reuse_request_drops_stale_range(mm_ranges):
    ranges = {"a": [(0, 4)]}
    s = state(ranges)
    add(s, "a")
    s.remove_request("a")
    assert s.mm_prefix_ranges == {}
    ranges.clear()
    add(s, "a")
    assert s.get_mm_prefix_ranges(SimpleNamespace(req_ids=["a"])) == {0: []}


def test_native_multimodal_ranges_preserved(mm_ranges):
    s = state()
    add(s, "a", [(2, 9)])
    assert s.get_mm_prefix_ranges(SimpleNamespace(req_ids=["a"])) == {0: [(2, 9)]}


def test_model_range_precedence(mm_ranges):
    s = state({"a": [(0, 10)]})
    add(s, "a", [(2, 9)])
    assert s.get_mm_prefix_ranges(SimpleNamespace(req_ids=["a"])) == {0: [(0, 10)]}


def test_disabled_path():
    s = state()
    s.mm_prefix_ranges = None
    assert s.get_mm_prefix_ranges(SimpleNamespace(req_ids=["a"])) is None
