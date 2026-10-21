# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import random
from dataclasses import dataclass

import pytest
import torch
from transformers import Qwen3VLConfig

from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.video_prune.evs import (
    compute_mrope_for_media,
    recompute_mrope_positions,
)
from vllm.utils.func_utils import supports_kw


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    # _get_mrope_input_positions returns CPU tensors (via torch.from_numpy).
    # Ensure the default device is CPU so the rest of the test tensors match.
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


IMAGE_TOKEN_ID = 999
VIDEO_TOKEN_ID = 888
VISION_START_TOKEN_ID = 777
VISION_END_TOKEN_ID = 778


@dataclass
class DummyVisionConfig:
    spatial_merge_size: int = 1


@dataclass
class DummyConfig:
    image_token_id: int = IMAGE_TOKEN_ID
    video_token_id: int = VIDEO_TOKEN_ID
    vision_start_token_id: int = VISION_START_TOKEN_ID
    vision_end_token_id: int = VISION_END_TOKEN_ID
    vision_config: DummyVisionConfig = dataclasses.field(
        default_factory=DummyVisionConfig
    )


def make_video_embedding(
    t, h, w, interleave_text_tokens: tuple[int, int], video_pruning_rate: float = 0.0
):
    """Make a video embedding for a given video size and pruning rate.

    Args:
        t: Number of frames.
        h: Number of rows.
        w: Number of columns.
        interleave_text_tokens: Tuple of minimum and maximum number of text tokens to
            interleave with the video.
        video_pruning_rate: Pruning rate for the video.

    Returns:
        Tuple of (unpruned_tokens_sequence, pruned_tokens_sequence, retention_mask)

    """
    unpruned_tokens_sequence = []
    population = list(range(1, 100))

    for _ in range(t):
        num_prefix_tokens = random.randint(
            interleave_text_tokens[0], interleave_text_tokens[1]
        )

        prefix_tokens = random.choices(population, k=num_prefix_tokens)
        vision_tokens = (
            [VISION_START_TOKEN_ID] + [VIDEO_TOKEN_ID] * h * w + [VISION_END_TOKEN_ID]
        )

        unpruned_tokens_sequence.extend(prefix_tokens)
        unpruned_tokens_sequence.extend(vision_tokens)

    unpruned_tokens_sequence = torch.tensor(unpruned_tokens_sequence, dtype=torch.long)
    video_token_mask = unpruned_tokens_sequence == VIDEO_TOKEN_ID

    pruning_mask = torch.bernoulli(video_token_mask.float() * video_pruning_rate).bool()
    # Sanity check that we don't prune what should not be pruned.
    assert not pruning_mask[~video_token_mask].any()

    retention_mask = ~pruning_mask
    pruned_tokens_sequence = unpruned_tokens_sequence[retention_mask]
    return unpruned_tokens_sequence, pruned_tokens_sequence, retention_mask


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize("grid_thw", [[3, 8, 7], [128, 10, 12]])
@pytest.mark.parametrize("num_prefix_tokens", [1, 11])
@pytest.mark.parametrize("num_suffix_tokens", [0, 7])
@pytest.mark.parametrize("video_pruning_rate", [0, 0.25, 0.75])
@pytest.mark.parametrize("interleave_text_tokens", [(0, 0), (1, 4)])
def test_match_qwen3vl_mrope_evs_on(
    spatial_merge_size: int,
    num_prefix_tokens: int,
    grid_thw: tuple[int, int, int],
    num_suffix_tokens: int,
    video_pruning_rate: float,
    interleave_text_tokens: tuple[int, int],
):
    hf_config = DummyConfig()
    hf_config.vision_config.spatial_merge_size = spatial_merge_size

    t, h, w = grid_thw
    population = list(range(1, 100))
    prefix_tokens = random.choices(population, k=num_prefix_tokens)
    suffix_tokens = random.choices(population, k=num_suffix_tokens)

    video_tokens, video_tokens_pruned, retention_mask = make_video_embedding(
        t,
        h // spatial_merge_size,
        w // spatial_merge_size,
        interleave_text_tokens=interleave_text_tokens,
        video_pruning_rate=video_pruning_rate,
    )
    assert len(video_tokens) == len(retention_mask)

    input_tokens = prefix_tokens + video_tokens.tolist() + suffix_tokens
    input_tokens_pruned = prefix_tokens + video_tokens_pruned.tolist() + suffix_tokens

    whole_sequence_retention_mask = torch.cat(
        [
            torch.ones(len(prefix_tokens), dtype=torch.bool),
            retention_mask,
            torch.ones(len(suffix_tokens), dtype=torch.bool),
        ],
        dim=0,
    )

    # Build the GT mrope for unpruned input.
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,  # HACK.
                ),
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=0, length=len(input_tokens)),
    )
    expected_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=[mm_feature],
        config=hf_config,
    )

    # Compute mrope for a video-only media (unpruned).
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,  # HACK.
                ),
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=0, length=video_tokens.numel()),
    )
    video_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=video_tokens.tolist(),
        mm_features=[mm_feature],
        config=hf_config,
    )
    video_mrope = video_mrope.permute(1, 0)  # [N, 3]
    hidden_size = 16

    is_video_embed = torch.isin(
        video_tokens_pruned, torch.tensor([VIDEO_TOKEN_ID], dtype=torch.long)
    )

    expanded_positions = torch.full(
        (len(video_tokens_pruned), 5),
        fill_value=-100,
        device=video_mrope.device,
        dtype=torch.long,
    )
    expanded_positions[is_video_embed, :3] = video_mrope[retention_mask][is_video_embed]
    expanded_positions[~is_video_embed, :3] = video_mrope[retention_mask][
        ~is_video_embed
    ]

    is_vision_start = video_tokens_pruned == VISION_START_TOKEN_ID
    expanded_positions[..., 3] = is_vision_start
    expanded_positions[..., 4] = is_video_embed

    # Check that all positions were filled, since we initialized them as negative.
    assert (expanded_positions >= 0).all()

    video_embeddings = torch.empty(
        (len(video_tokens_pruned), hidden_size), device=video_mrope.device
    )

    video_embeddings = torch.cat(
        [
            video_embeddings,
            expanded_positions.float(),
        ],
        dim=1,
    )
    multimodal_embeddings = [video_embeddings]

    expected_mrope_masked = expected_mrope[:, whole_sequence_retention_mask]

    # Initialize computed_mrope with sequential positions for all prefix tokens
    computed_mrope = torch.empty((3, len(input_tokens_pruned)), dtype=torch.long)
    computed_mrope[:, 0 : len(prefix_tokens)] = expected_mrope[
        :, 0 : len(prefix_tokens)
    ]

    # Paranoia check that computed_mrope is wrong.
    assert not torch.equal(computed_mrope, expected_mrope_masked)

    _, actual_mrope, _ = Qwen3VLForConditionalGeneration._recompute_mrope_positions(
        input_ids=input_tokens_pruned,
        multimodal_embeddings=multimodal_embeddings,
        mrope_positions=computed_mrope,
        num_computed_tokens=len(prefix_tokens),
        vision_start_token_id=hf_config.vision_start_token_id,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
    )

    assert torch.equal(actual_mrope, expected_mrope_masked)


def _make_pruned_video_positions(empty_frame: bool):
    config = Qwen3VLConfig(
        vision_config={"spatial_merge_size": 1},
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
    )
    frame_prefix = [2] * 6 + [VISION_START_TOKEN_ID]
    frame_suffix = [VISION_END_TOKEN_ID]
    full_tokens = (frame_prefix + [VIDEO_TOKEN_ID] * 4 + frame_suffix) * 4
    grid = MultiModalKwargsItem(
        {
            "video_grid_thw": MultiModalFieldElem(
                data=torch.tensor([4, 2, 2]), field=None
            )
        }
    )
    feature = MultiModalFeatureSpec(
        data=grid,
        modality="video",
        identifier="video",
        mm_position=PlaceholderRange(offset=0, length=len(full_tokens)),
    )
    full_positions, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=full_tokens, mm_features=[feature], config=config
    )
    counts = [4, 0, 3, 3] if empty_frame else [4, 2, 2, 2]
    mask = torch.tensor(
        [
            keep
            for count in counts
            for keep in [True] * 7 + [i < count for i in range(4)] + [True]
        ]
    )
    retained_tokens = torch.tensor(full_tokens)[mask]
    positions = torch.cat(
        [
            full_positions[:, mask],
            retained_tokens.eq(VISION_START_TOKEN_ID).view(1, -1),
            retained_tokens.eq(VIDEO_TOKEN_ID).view(1, -1),
        ],
        dim=0,
    ).long()
    placeholder = (
        frame_prefix
        + [VIDEO_TOKEN_ID] * sum(counts)
        + frame_suffix
        + (frame_prefix + frame_suffix) * 3
    )
    return config, grid, placeholder, positions


@pytest.mark.parametrize("prefix_len", [3, 20])
@pytest.mark.parametrize("outer_markers", [False, True])
@pytest.mark.parametrize("num_videos", [1, 2])
@pytest.mark.parametrize("empty_frame", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_pruned_video_uses_placeholder_ranges(
    prefix_len, outer_markers, num_videos, empty_frame, chunked
):
    """Pruned frame metadata must stay inside its prompt placeholder.

    New processors retain outer vision markers, which do not belong to the
    video replacement. Exact ranges also preserve text, empty frames and a
    continuation starting on the first video token.
    """
    config, grid, placeholder, mm_pos = _make_pruned_video_positions(empty_frame)
    tokens = [1] * prefix_len
    features = []
    for index in range(num_videos):
        if outer_markers:
            tokens.append(VISION_START_TOKEN_ID)
        start = len(tokens)
        tokens += placeholder
        features.append(
            MultiModalFeatureSpec(
                data=grid,
                modality="video",
                identifier=f"video-{index}",
                mm_position=PlaceholderRange(offset=start, length=len(placeholder)),
            )
        )
        if outer_markers:
            tokens.append(VISION_END_TOKEN_ID)
        tokens += [1] * 3

    expected = torch.empty((3, len(tokens)), dtype=torch.long)
    previous_end, base = 0, 0
    for feature in features:
        start = feature.mm_position.offset
        expected[:, previous_end:start] = torch.arange(start - previous_end) + base
        base += start - previous_end
        expected[:, start : start + mm_pos.shape[1]] = mm_pos[:3] + base
        base += int(mm_pos[:3].max()) + 1
        previous_end = start + mm_pos.shape[1]
    expected[:, previous_end:] = torch.arange(len(tokens) - previous_end) + base
    actual, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=tokens, mm_features=features, config=config
    )
    if chunked:
        split = int(mm_pos[4].nonzero()[0])
        calls = [
            (
                feature.mm_position.offset + offset,
                [mm_pos[:, lower:upper]],
                [feature.mm_position],
            )
            for feature in features
            for offset, lower, upper in [(0, 0, split), (split, split, mm_pos.shape[1])]
        ]
    else:
        calls = [(0, [mm_pos] * num_videos, [f.mm_position for f in features])]
    recompute = Qwen3VLForConditionalGeneration._recompute_mrope_positions
    for computed, position_blocks, placeholders in calls:
        kwargs = {}
        if supports_kw(recompute, "multimodal_placeholders", allow_var_kwargs=False):
            kwargs["multimodal_placeholders"] = placeholders
        embeds = [
            torch.cat([torch.zeros((m.shape[1], 8)), m.T.float()], dim=1)
            for m in position_blocks
        ]
        _, actual, delta = recompute(
            input_ids=torch.tensor(tokens),
            multimodal_embeddings=embeds,
            mrope_positions=actual,
            num_computed_tokens=computed,
            vision_start_token_id=VISION_START_TOKEN_ID,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            **kwargs,
        )
    assert torch.equal(actual, expected)
    assert delta == int(expected.max()) + 1 - len(tokens)


@pytest.mark.parametrize("metadata_channels", [4, 5])
@pytest.mark.parametrize("chunked", [False, True])
def test_image_position_metadata_preserves_legacy_semantics(metadata_channels, chunked):
    """The Qwen2.5 four-channel and Qwen3 five-channel image grids stay exact."""
    tokens = torch.tensor(
        [1, VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * 4 + [VISION_END_TOKEN_ID, 1]
    )
    mm_pos = compute_mrope_for_media(torch.tensor([1, 2, 2]), 1).T
    if metadata_channels == 5:
        mm_pos = torch.cat([mm_pos, torch.zeros((1, 4), dtype=torch.long)])
    state = torch.arange(len(tokens)).repeat(3, 1)
    expected = torch.tensor(
        [[0, 1, 2, 2, 2, 2, 4, 5], [0, 1, 2, 2, 3, 3, 4, 5], [0, 1, 2, 3, 2, 3, 4, 5]]
    )
    parts = [(0, 0, 2), (4, 2, 4)] if chunked else [(0, 0, 4)]
    for computed, lower, upper in parts:
        state, delta = recompute_mrope_positions(
            tokens,
            [mm_pos[:, lower:upper]],
            state,
            computed,
            VISION_START_TOKEN_ID,
            IMAGE_TOKEN_ID,
            VIDEO_TOKEN_ID,
            [PlaceholderRange(offset=2, length=4)],
        )
    assert torch.equal(state, expected)
    assert delta == -2
    if not chunked:
        legacy, legacy_delta = recompute_mrope_positions(
            tokens,
            [mm_pos],
            torch.arange(len(tokens)).repeat(3, 1),
            0,
            VISION_START_TOKEN_ID,
            IMAGE_TOKEN_ID,
            VIDEO_TOKEN_ID,
        )
        assert torch.equal(legacy, state)
        assert legacy_delta == delta


@pytest.mark.parametrize("metadata_channels", [4, 5])
def test_empty_position_metadata_preserves_prompt(metadata_channels):
    initial = torch.arange(8).repeat(3, 1)
    state, delta = recompute_mrope_positions(
        torch.ones(8, dtype=torch.long),
        [torch.empty((metadata_channels, 0), dtype=torch.long)],
        initial,
        0,
        VISION_START_TOKEN_ID,
        IMAGE_TOKEN_ID,
        VIDEO_TOKEN_ID,
        [PlaceholderRange(offset=2, length=4)],
    )
    assert torch.equal(state, initial)
    assert delta == 0


@pytest.mark.parametrize("chunked", [False, True])
def test_cohere_image_positions_accept_placeholder_ranges(chunked):
    from vllm.model_executor.models.cohere_compass import (
        CohereCompassForConditionalGeneration,
    )

    tokens = (
        [1, VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * 4 + [VISION_END_TOKEN_ID, 1]
    )
    mm_pos = compute_mrope_for_media(torch.tensor([1, 2, 2]), 1).T
    mm_pos = torch.cat([mm_pos, torch.zeros((1, 4), dtype=torch.long)])
    hidden = torch.arange(32).reshape(4, 8).float()
    state = torch.arange(len(tokens)).repeat(3, 1)
    parts = [(0, 0, 2), (4, 2, 4)] if chunked else [(0, 0, 4)]
    for computed, lower, upper in parts:
        embed = torch.cat(
            [hidden[lower:upper], mm_pos[:, lower:upper].T.float()], dim=1
        )
        cleaned, state, delta = (
            CohereCompassForConditionalGeneration._recompute_mrope_positions(
                input_ids=tokens,
                multimodal_embeddings=[embed],
                mrope_positions=state,
                num_computed_tokens=computed,
                vision_start_token_id=VISION_START_TOKEN_ID,
                image_token_id=IMAGE_TOKEN_ID,
                video_token_id=VIDEO_TOKEN_ID,
                multimodal_placeholders=[PlaceholderRange(offset=2, length=4)],
            )
        )
        assert torch.equal(cleaned[0], hidden[lower:upper])
    expected = torch.tensor(
        [[0, 1, 2, 2, 2, 2, 4, 5], [0, 1, 2, 2, 3, 3, 4, 5], [0, 1, 2, 3, 2, 3, 4, 5]]
    )
    assert torch.equal(state, expected)
    assert delta == -2


def test_video_lookahead_slice_preserves_target_positions():
    """A drafter's shifted gather uses its actual slice, not the target cursor."""
    _, _, placeholder, mm_pos = _make_pruned_video_positions(False)
    tokens = [1, 1, 1, VISION_START_TOKEN_ID] + placeholder + [VISION_END_TOKEN_ID, 1]
    start = 4
    state = torch.empty((3, len(tokens)), dtype=torch.long)
    state[:, :start] = torch.arange(start)
    state[:, start : start + mm_pos.shape[1]] = mm_pos[:3] + start
    tail = start + mm_pos.shape[1]
    state[:, tail:] = (
        torch.arange(len(tokens) - tail) + int(mm_pos[:3].max()) + start + 1
    )
    _, actual, _ = Qwen3VLForConditionalGeneration._recompute_mrope_positions(
        input_ids=torch.tensor(tokens),
        multimodal_embeddings=[
            torch.cat(
                [torch.zeros((mm_pos.shape[1] - 1, 8)), mm_pos[:, 1:].T.float()], dim=1
            )
        ],
        mrope_positions=state,
        num_computed_tokens=start + 1,
        vision_start_token_id=VISION_START_TOKEN_ID,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        multimodal_placeholders=[
            PlaceholderRange(offset=start, length=len(placeholder))
        ],
    )
    assert torch.equal(actual, state)
