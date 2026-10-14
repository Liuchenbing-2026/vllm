# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the segment-level tokenizer cache.

The tests use a stub tokenizer rather than a real checkpoint: the property under
test is the *concatenation identity* of the cache, and a stub lets us assert on
the boundary cases (a pre-tokenizer that merges across a boundary, a tokenizer
without added tokens, ``add_special_tokens`` that is not a no-op) that a
downloaded model only exercises by accident.
"""

import re

import pytest

from vllm.renderers.tokenizer_cache import (
    IncrementalTokenizerCache,
    TokenizerCacheConfig,
    cache_for,
    maybe_attach,
    token_ids_from_cache,
)

ADDED = ("<|im_end|>", "<|im_start|>")


class StubTokenizer:
    """Tokenizes like an HF fast tokenizer: added tokens are matched before the
    pre-tokenizer, so no merge can span one, and each piece gets a stable id."""

    def __init__(
        self,
        added: tuple[str, ...] = ADDED,
        *,
        adds_specials: bool = False,
    ):
        self._added = list(added)
        self.all_special_tokens = list(added)
        self._adds_specials = adds_specials
        self._vocab: dict[str, int] = {}

    def get_added_vocab(self) -> dict[str, int]:
        return dict.fromkeys(self._added, 0)

    def _pieces(self, text: str) -> list[str]:
        if not self._added:
            return [text] if text else []
        pattern = re.compile(
            "|".join(re.escape(t) for t in sorted(self._added, key=len, reverse=True))
        )
        pieces: list[str] = []
        last = 0
        for match in pattern.finditer(text):
            if match.start() > last:
                pieces.append(text[last : match.start()])
            pieces.append(match.group(0))
            last = match.end()
        if last < len(text):
            pieces.append(text[last:])
        return pieces

    def _id_of(self, piece: str) -> int:
        return self._vocab.setdefault(piece, len(self._vocab) + 1)

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs):
        ids = [self._id_of(p) for p in self._pieces(text)]
        if add_special_tokens and self._adds_specials:
            ids = [0, *ids, 1]
        return {"input_ids": ids}


class MergingStubTokenizer(StubTokenizer):
    """Ids depend on the whole string, so segment-wise encoding differs.

    This is what a Metaspace/SentencePiece pre-tokenizer that prepends a space
    to every segment looks like from the outside.
    """

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs):
        ids = [self._id_of(p) for p in self._pieces(text)]
        return {"input_ids": [len(text), *ids]}


def test_config_from_additional_config_and_env(monkeypatch):
    config = type("C", (), {"additional_config": {"tokenizer_cache_gb": 2}})()
    assert TokenizerCacheConfig.from_vllm_config(config).capacity_gb == 2

    monkeypatch.setenv("VLLM_TOKENIZER_CACHE_GB", "1.5")
    assert TokenizerCacheConfig.from_vllm_config(None).capacity_gb == 1.5

    for bad in ("nonsense", -1):
        config = type("C", (), {"additional_config": {"tokenizer_cache_gb": bad}})()
        assert TokenizerCacheConfig.from_vllm_config(config).capacity_gb == 0


def test_disabled_without_added_tokens():
    tokenizer = StubTokenizer(added=())
    assert maybe_attach(tokenizer, capacity_gb=1) is None
    cache = IncrementalTokenizerCache(tokenizer, 1)
    assert not cache.enabled
    assert cache.disabled_reason is not None


def test_disabled_when_identity_breaks():
    cache = IncrementalTokenizerCache(MergingStubTokenizer(), 1)
    assert not cache.enabled
    assert "identity" in (cache.disabled_reason or "")


def test_no_op_when_capacity_is_zero(monkeypatch):
    monkeypatch.delenv("VLLM_TOKENIZER_CACHE_GB", raising=False)
    tokenizer = StubTokenizer()
    assert maybe_attach(tokenizer, capacity_gb=0) is None
    assert maybe_attach(tokenizer) is None  # no config, no env
    assert cache_for(tokenizer) is None


def test_encode_matches_tokenizer_and_caches_segments():
    tokenizer = StubTokenizer()
    cache = maybe_attach(tokenizer, capacity_gb=1)
    assert cache is not None
    assert cache_for(tokenizer) is cache

    text = "hello<|im_end|>world<|im_start|>again"
    expected = list(tokenizer(text)["input_ids"])

    assert cache.encode(text) == expected
    assert cache.stat().hits == 0
    assert cache.stat().total == 3  # three segments, all cold

    assert cache.encode(text) == expected
    assert cache.stat().hits == 3

    # A longer prompt reuses the segments of the shorter one and pays for one.
    longer = text + "<|im_end|>tail"
    assert cache.encode(longer) == list(tokenizer(longer)["input_ids"])
    assert cache.stat().hits == 5
    assert cache.stat().total == 10
    assert cache.stat().hit_ratio == pytest.approx(0.5)


def test_is_eligible_follows_add_special_tokens():
    # ``add_special_tokens=True`` is a no-op for this stub: both are served.
    cache = IncrementalTokenizerCache(StubTokenizer(), 1)
    assert cache.enabled
    assert cache.is_eligible(add_special_tokens=True)
    assert cache.is_eligible(add_special_tokens=False)

    # Here it is not a no-op, so only the False variant may be served.
    adding = IncrementalTokenizerCache(StubTokenizer(adds_specials=True), 1)
    assert adding.enabled
    assert adding.is_eligible(add_special_tokens=False)
    assert not adding.is_eligible(add_special_tokens=True)


@pytest.mark.parametrize("capacity_gb", [0.0001, 0.01])
def test_oversize_segment_is_returned_but_not_cached(capacity_gb):
    tokenizer = StubTokenizer()
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb)
    text = "x" * 1_000_000
    assert cache.encode(text) == list(tokenizer(text)["input_ids"])
    assert cache.stat().hits == 0


def test_chat_fast_path_probes_once_then_reuses():
    tokenizer = StubTokenizer()
    cache = maybe_attach(tokenizer, capacity_gb=1)
    assert cache is not None
    text = "system<|im_end|>user<|im_start|>answer"
    expected = list(tokenizer(text)["input_ids"])

    calls: list[bool] = []

    def render(*, tokenize: bool = True):
        calls.append(tokenize)
        return list(tokenizer(text)["input_ids"]) if tokenize else text

    assert token_ids_from_cache(cache, render) == expected
    # The probe renders once without ids and once with them; the fast path then
    # renders the text it caches.
    assert calls == [False, True, False]
    assert token_ids_from_cache(cache, render) == expected
    assert calls == [False, True, False, False]  # no further probes

    # ``tokenize=False`` never consults the cache.
    assert token_ids_from_cache(cache, render, tokenize=False) is None
    assert calls == [False, True, False, False]


def test_chat_fast_path_disabled_when_template_disagrees():
    tokenizer = StubTokenizer()
    cache = maybe_attach(tokenizer, capacity_gb=1)
    assert cache is not None

    def render(*, tokenize: bool = True):
        return list(tokenizer("something else")["input_ids"]) if tokenize else "first"

    assert token_ids_from_cache(cache, render) is None
    assert not cache.verify_chat_path(render)
