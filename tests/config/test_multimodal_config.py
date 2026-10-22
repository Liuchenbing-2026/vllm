# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PreTrainedConfig

from tests.models.utils import build_model_context
from vllm.config.ec_transfer import ECRole, ECTransferConfig
from vllm.config.model import ModelConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.config.vllm import VllmConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.mark.parametrize(
    "model_id,limit_mm_per_prompt,expected",
    [
        ("Qwen/Qwen2-0.5B-Instruct", {}, False),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {}, True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 0, "video": 0}, False),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 0}, True),
    ],
)
@pytest.mark.core_model
def test_supports_multimodal_inputs(model_id, limit_mm_per_prompt, expected):
    """Test supports_multimodal_inputs returns correct boolean for various
    configs."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )
    assert ctx.model_config.supports_multimodal_inputs is expected


def test_mm_encoder_attn_backend_str_conversion():
    config = MultiModalConfig(mm_encoder_attn_backend="FLASH_ATTN")
    assert config.mm_encoder_attn_backend == AttentionBackendEnum.FLASH_ATTN


def test_mm_encoder_attn_backend_invalid():
    with pytest.raises(ValueError):
        MultiModalConfig(mm_encoder_attn_backend="not_a_backend")


def test_mm_hasher_algorithm_invalid():
    with pytest.raises(ValueError, match="mm_hasher_algorithm"):
        MultiModalConfig(mm_hasher_algorithm="md5")


def test_mm_encoder_attn_backend_hash_updates():
    base_hash = MultiModalConfig().compute_hash()
    overridden_hash = MultiModalConfig(
        mm_encoder_attn_backend=AttentionBackendEnum.FLASH_ATTN
    ).compute_hash()
    assert base_hash != overridden_hash


def test_language_model_only_does_not_affect_mm_hash():
    """language_model_only does not affect the ViT computation graph,
    so it should not change the multimodal config hash."""
    base_hash = MultiModalConfig().compute_hash()
    lm_only_hash = MultiModalConfig(language_model_only=True).compute_hash()
    assert base_hash == lm_only_hash


def test_language_model_only_affects_model_hash():
    """language_model_only affects the LM computation graph,
    so it should change the model config hash."""
    model = "llava-hf/llava-1.5-7b-hf"
    base_hash = ModelConfig(model).compute_hash()
    lm_only_hash = ModelConfig(model, language_model_only=True).compute_hash()
    assert base_hash != lm_only_hash


@pytest.mark.parametrize("backend_arg", ["video_backend", "backend"])
def test_use_gpu_video_backend_from_media_io_kwargs(backend_arg: str):
    config = MultiModalConfig(
        media_io_kwargs={"video": {backend_arg: "pynvvideocodec"}}
    )

    assert config.use_gpu_video_backend()


def test_mm_encoder_fp8_scale_path_requires_fp8():
    with pytest.raises(ValueError, match="mm_encoder_attn_dtype"):
        MultiModalConfig(mm_encoder_fp8_scale_path="/tmp/scales.json")


def test_mm_encoder_attn_dtype_hash_updates(tmp_path):
    scale_file = tmp_path / "scales.json"
    scale_file.write_text("{}")
    base_hash = MultiModalConfig().compute_hash()
    fp8_hash = MultiModalConfig(mm_encoder_attn_dtype="fp8").compute_hash()
    fp8_static_hash = MultiModalConfig(
        mm_encoder_attn_dtype="fp8",
        mm_encoder_fp8_scale_path=str(scale_file),
    ).compute_hash()
    assert base_hash != fp8_hash
    assert fp8_hash != fp8_static_hash


_MULTIMODAL_MODEL = "llava-hf/llava-1.5-7b-hf"
_TEXT_ONLY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _make_mm_prefix_model_config(
    model: str = _MULTIMODAL_MODEL,
    *,
    language_model_only: bool = False,
) -> ModelConfig:
    return ModelConfig(model, language_model_only=language_model_only)


def test_supports_multimodal_for_mm_prefix_before_multimodal_config():
    """A text-only model never builds a multimodal config, so the early
    return must neither clear mm_prefix nor write the sticky cache."""
    model_config = _make_mm_prefix_model_config(_TEXT_ONLY_MODEL)
    assert model_config.multimodal_config is None

    assert model_config._supports_multimodal_for_mm_prefix() is True
    assert not getattr(model_config, "_supports_multimodal_inputs_cache", None)


def test_language_model_only_disables_via_supports_multimodal_inputs():
    """language_model_only zeros all limits, so registry reports text-only."""
    model_config = _make_mm_prefix_model_config(language_model_only=True)

    assert model_config._supports_multimodal_for_mm_prefix() is False


def test_convertor_clears_mm_prefix_when_multimodal_disabled():
    hf_config = PreTrainedConfig(
        model_type="gemma3",
        architectures=["Gemma3ForConditionalGeneration"],
    )
    hf_config.is_mm_prefix_lm = True
    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.is_mm_prefix_lm(supports_multimodal=True) is True
    assert convertor.is_mm_prefix_lm(supports_multimodal=False) is False

    enabled = convertor.convert(supports_multimodal=True)
    disabled = convertor.convert(supports_multimodal=False)
    assert enabled.is_mm_prefix_lm is True
    assert disabled.is_mm_prefix_lm is False


def test_sticky_cache_survives_text_subconfig_regeneration():
    """with_hf_config deepcopies the cached decision onto text submodules."""
    model_config = _make_mm_prefix_model_config()
    assert model_config._supports_multimodal_for_mm_prefix() is True

    # `with_hf_config` deep-copies this config and swaps `hf_config` for a
    # text-only submodule (e.g. Gemma4ForCausalLM).
    text_config = copy.deepcopy(model_config)
    text_config.hf_config = model_config.hf_text_config
    assert text_config._supports_multimodal_for_mm_prefix() is True

    # Without the copied cache the submodule architecture has no registered
    # multimodal processor, so re-querying the registry wrongly settles on
    # text-only and would clear mm_prefix.
    del text_config._supports_multimodal_inputs_cache
    assert text_config._supports_multimodal_for_mm_prefix() is False


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        (None, None),
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        # Callers compare against a bare device type, so an indexed device and a
        # torch.device must normalise to the same thing as "cuda".
        ("cuda:1", "cuda"),
        (torch.device("cuda", 1), "cuda"),
    ],
)
def test_mm_processor_device_type_normalizes(device: object, expected: str | None):
    kwargs = {} if device is None else {"device": device}
    config = MultiModalConfig(mm_processor_kwargs=kwargs)
    assert config.get_mm_processor_device_type() == expected


def test_merge_mm_processor_kwargs_scoped_precedence():
    configured = {
        "size": {
            "shortest_edge": 64,
            "longest_edge": 512,
        },
        "do_resize": True,
        "padding": False,
        "images_kwargs": {
            "size": {"longest_edge": 1024},
            "do_resize": False,
        },
        "videos_kwargs": {
            "size": {"longest_edge": 2048},
        },
    }
    inference = {
        "size": {"shortest_edge": 128},
        "do_resize": True,
        "images_kwargs": {
            "size": {"shortest_edge": 256},
        },
    }

    config = MultiModalConfig(
        mm_processor_kwargs=configured,
        mm_device_do_normalize=False,
    )

    assert config.merge_mm_processor_kwargs(inference) == {
        "size": {
            "shortest_edge": 128,
            "longest_edge": 512,
        },
        "do_resize": True,
        "padding": False,
        "images_kwargs": {
            "size": {
                "shortest_edge": 256,
                "longest_edge": 1024,
            },
            "do_resize": True,
        },
        "videos_kwargs": {
            "size": {
                "shortest_edge": 128,
                "longest_edge": 2048,
            },
        },
    }


def test_merge_mm_processor_kwargs_preserves_scoped_siblings_and_precedence():
    config = MultiModalConfig(
        mm_processor_kwargs={
            "size": {
                "shortest_edge": 100,
                "longest_edge": 1000,
            },
            "videos_kwargs": {
                "fps": 2,
                "size": {"longest_edge": 1200},
            },
        },
        mm_device_do_normalize=False,
    )

    assert config.merge_mm_processor_kwargs(
        {
            "size": {"longest_edge": 1800},
            "videos_kwargs": {
                "size": {"shortest_edge": 200},
            },
        }
    ) == {
        "size": {
            "shortest_edge": 100,
            "longest_edge": 1800,
        },
        "videos_kwargs": {
            "fps": 2,
            "size": {
                "shortest_edge": 200,
                "longest_edge": 1800,
            },
        },
    }


def test_merge_mm_processor_kwargs_mapping_vs_replacement():
    config = MultiModalConfig(
        mm_processor_kwargs={
            "size": {"shortest_edge": 64},
            "images_kwargs": {"size": {"longest_edge": 1024}},
        },
        mm_device_do_normalize=False,
    )

    assert config.merge_mm_processor_kwargs({"size": 7}) == {
        "size": 7,
        "images_kwargs": {"size": 7},
    }
    assert config.merge_mm_processor_kwargs({"size": {}}) == {
        "size": {"shortest_edge": 64},
        "images_kwargs": {
            "size": {
                "shortest_edge": 64,
                "longest_edge": 1024,
            },
        },
    }


_EMPTY_IMAGE_SCOPE_KWARGS: tuple[dict[str, object], ...] = (
    {},
    {"images_kwargs": None},
    {"images_kwargs": {}},
)


@pytest.mark.parametrize("configured", _EMPTY_IMAGE_SCOPE_KWARGS)
@pytest.mark.parametrize("inference", _EMPTY_IMAGE_SCOPE_KWARGS)
def test_merge_mm_processor_kwargs_empty_scopes_are_absent(
    configured: dict[str, object],
    inference: dict[str, object],
):
    config = MultiModalConfig(
        mm_processor_kwargs=configured,
        mm_device_do_normalize=False,
    )

    assert config.merge_mm_processor_kwargs(inference) == {}


@pytest.mark.parametrize("empty_inference", _EMPTY_IMAGE_SCOPE_KWARGS)
def test_merge_mm_processor_kwargs_empty_inference_scope_preserves_config_merge(
    empty_inference: dict[str, object],
):
    config = MultiModalConfig(
        mm_processor_kwargs={
            "images_kwargs": {
                "size": {
                    "shortest_edge": 64,
                    "longest_edge": 512,
                },
            },
        },
        mm_device_do_normalize=False,
    )
    inference = {
        "size": {"shortest_edge": 128},
        **empty_inference,
    }

    assert config.merge_mm_processor_kwargs(inference) == {
        "size": {"shortest_edge": 128},
        "images_kwargs": {
            "size": {
                "shortest_edge": 128,
                "longest_edge": 512,
            },
        },
    }


@pytest.mark.parametrize("empty_config", _EMPTY_IMAGE_SCOPE_KWARGS)
def test_merge_mm_processor_kwargs_empty_config_scope_allows_inference_merge(
    empty_config: dict[str, object],
):
    config = MultiModalConfig(
        mm_processor_kwargs={
            "size": {"longest_edge": 512},
            **empty_config,
        },
        mm_device_do_normalize=False,
    )

    assert config.merge_mm_processor_kwargs(
        {
            "size": {"shortest_edge": 128},
            "images_kwargs": {"size": {"longest_edge": 1024}},
        }
    ) == {
        "size": {
            "shortest_edge": 128,
            "longest_edge": 512,
        },
        "images_kwargs": {
            "size": {
                "shortest_edge": 128,
                "longest_edge": 1024,
            },
        },
    }


@pytest.mark.parametrize("non_mapping_scope", [False, 0, "", [], 123])
def test_merge_mm_processor_kwargs_falsey_non_mapping_scopes_are_not_empty(
    non_mapping_scope: object,
):
    config = MultiModalConfig(
        mm_processor_kwargs={"images_kwargs": non_mapping_scope},
        mm_device_do_normalize=False,
    )

    for empty_inference in _EMPTY_IMAGE_SCOPE_KWARGS:
        assert config.merge_mm_processor_kwargs(empty_inference) == {
            "images_kwargs": non_mapping_scope
        }


@pytest.mark.parametrize(
    ("configured_scope", "inference_scope", "expected_scope"),
    [
        (123, {"size": {"shortest_edge": 128}}, {"size": {"shortest_edge": 128}}),
        ({"size": {"shortest_edge": 64}}, 123, 123),
        (123, 456, 456),
    ],
)
def test_merge_mm_processor_kwargs_non_mapping_scope_precedence(
    configured_scope: object,
    inference_scope: object,
    expected_scope: object,
):
    config = MultiModalConfig(
        mm_processor_kwargs={"images_kwargs": configured_scope},
        mm_device_do_normalize=False,
    )

    merged = config.merge_mm_processor_kwargs({"images_kwargs": inference_scope})

    assert merged["images_kwargs"] == expected_scope


def test_merge_mm_processor_kwargs_none_leaf_is_explicit():
    config = MultiModalConfig(mm_device_do_normalize=False)

    assert config.merge_mm_processor_kwargs({"images_kwargs": {"do_resize": None}}) == {
        "images_kwargs": {"do_resize": None}
    }


def test_merge_mm_processor_kwargs_normalize_and_ownership():
    configured: dict[str, object] = {
        "images_kwargs": {
            "size": {"shortest_edge": 64},
        },
    }
    inference: dict[str, object] = {
        "do_normalize": True,
        "do_rescale": True,
        "images_kwargs": {
            "size": {"longest_edge": 1024},
        },
    }
    configured_before = copy.deepcopy(configured)
    inference_before = copy.deepcopy(inference)

    config = MultiModalConfig(
        mm_processor_kwargs=configured,
        mm_device_do_normalize=True,
    )
    merged = config.merge_mm_processor_kwargs(inference)

    assert merged["do_normalize"] is False
    assert merged["do_rescale"] is False
    assert configured == configured_before
    assert inference == inference_before
    assert config.mm_processor_kwargs == configured_before

    images_kwargs = merged["images_kwargs"]
    assert isinstance(images_kwargs, dict)
    size = images_kwargs["size"]
    assert isinstance(size, dict)
    size["shortest_edge"] = 999

    assert configured == configured_before
    assert inference == inference_before
    assert config.mm_processor_kwargs == configured_before


def _validate_mm_processor_device(*, device: str, ec_role: ECRole | None) -> None:
    ec_config = (
        None
        if ec_role is None
        # `is_ec_producer`/`is_ec_consumer` are False without a connector, so an
        # unset one would make every role look like "no EC role at all".
        else ECTransferConfig(ec_connector="ECExampleConnector", ec_role=ec_role)
    )
    MultiModalConfig(
        mm_processor_kwargs={"device": device}
    ).validate_mm_processor_device(ec_config)


@pytest.mark.parametrize("ec_role", [None, "ec_producer"])
def test_bad_mm_processor_device_rejected(ec_role: ECRole | None):
    """A bad device must fail during startup, not mid-request.

    Rejected for every role, and on a CPU-only platform too, so a typo can never
    silently fall through to running the processor somewhere unintended.
    """
    with (
        patch("vllm.platforms.current_platform.device_type", "cpu"),
        pytest.raises(ValueError, match='Invalid "device" in mm_processor_kwargs'),
    ):
        _validate_mm_processor_device(device="not-a-device", ec_role=ec_role)


@pytest.mark.parametrize("ec_role", [None, "ec_consumer", "ec_both"])
def test_accelerator_mm_processor_rejected_outside_encoder_instance(
    ec_role: ECRole | None,
):
    """Only an encode-only EPD instance has the device to itself.

    Every other role runs the language model in the same process, so frontend
    accelerator work would contend with the forward pass and allocate outside
    the memory profiled for the KV cache.
    """
    with (
        patch("vllm.platforms.current_platform.device_type", "cuda"),
        pytest.raises(ValueError, match="also runs the language model"),
    ):
        _validate_mm_processor_device(device="cuda", ec_role=ec_role)


def test_accelerator_mm_processor_allowed_on_encoder_instance():
    with patch("vllm.platforms.current_platform.device_type", "cuda"):
        _validate_mm_processor_device(device="cuda", ec_role="ec_producer")


def test_cpu_mm_processor_needs_no_ec_role():
    """The gate only applies to the accelerator, so CPU stays unrestricted."""
    with patch("vllm.platforms.current_platform.device_type", "cuda"):
        _validate_mm_processor_device(device="cpu", ec_role=None)


@pytest.mark.parametrize(
    ("flag", "kwargs", "expected"),
    [
        # "auto" is only settled once the EC role is known, so it must leave no
        # device behind for `VllmConfig` to mistake for an explicit request.
        ("auto", None, None),
        (None, None, None),
        ("cpu", None, "cpu"),
        ("cuda", None, "cuda"),
        # Any explicit value other than "cpu" means "whatever this platform
        # calls its accelerator".
        ("gpu", None, "cuda"),
        # An explicit device in the kwargs wins over the convenience flag.
        ("cpu", {"device": "cuda"}, "cuda"),
        ("cuda", {"device": "cpu"}, "cpu"),
    ],
)
def test_fold_mm_processor_device(
    flag: str | None, kwargs: dict | None, expected: str | None
):
    with patch("vllm.platforms.current_platform.device_type", "cuda"):
        folded = MultiModalConfig.fold_mm_processor_device(kwargs, flag)

    config = MultiModalConfig(mm_processor_kwargs=folded or {})
    assert config.get_mm_processor_device_type() == expected


def _resolve_mm_processor_device(
    *,
    ec_role: ECRole | None,
    mm_tensor_ipc: str = "torch_shm",
    device: str | None = None,
) -> str | None:
    """Run the `auto` resolution and report where the processor ended up."""
    mm_config = MultiModalConfig(
        mm_processor_kwargs={} if device is None else {"device": device},
        mm_tensor_ipc=mm_tensor_ipc,
    )
    model_config = MagicMock(spec=ModelConfig)
    model_config.multimodal_config = mm_config
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.ec_transfer_config = (
        None
        if ec_role is None
        else ECTransferConfig(ec_connector="ECExampleConnector", ec_role=ec_role)
    )

    with patch("vllm.platforms.current_platform.device_type", "cuda"):
        VllmConfig._resolve_mm_processor_device(vllm_config)
    return mm_config.get_mm_processor_device_type()


def test_auto_mm_processor_device_uses_accelerator_on_encoder_instance():
    """The one deployment that has the device to itself and can hand it over."""
    assert _resolve_mm_processor_device(ec_role="ec_producer") == "cuda"


@pytest.mark.parametrize("ec_role", [None, "ec_consumer", "ec_both"])
def test_auto_mm_processor_device_stays_on_cpu_off_encoder_instance(
    ec_role: ECRole | None,
):
    """Every other role runs the language model in the same process."""
    assert _resolve_mm_processor_device(ec_role=ec_role) is None


def test_auto_mm_processor_device_needs_a_device_capable_transport():
    """Other transports serialize host bytes, so the output is copied back."""
    assert (
        _resolve_mm_processor_device(ec_role="ec_producer", mm_tensor_ipc="direct_rpc")
        is None
    )


def test_auto_mm_processor_device_leaves_an_explicit_request_alone():
    assert _resolve_mm_processor_device(ec_role="ec_producer", device="cpu") == "cpu"


@pytest.mark.parametrize(
    ("ec_role", "expected"),
    [
        (None, False),
        ("ec_consumer", False),
        ("ec_both", False),
        ("ec_producer", True),
    ],
)
def test_ec_producer_only_enables_mm_encoder_only(
    ec_role: ECRole | None, expected: bool
):
    mm_config = MultiModalConfig()
    model_config = MagicMock(spec=ModelConfig)
    model_config.multimodal_config = mm_config
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.ec_transfer_config = (
        None
        if ec_role is None
        else ECTransferConfig(ec_connector="ECExampleConnector", ec_role=ec_role)
    )

    VllmConfig._resolve_mm_encoder_only(vllm_config)

    assert mm_config.mm_encoder_only is expected


def test_vllm_config_runs_the_mm_processor_device_check():
    """Startup must reach the check; the rule itself is covered above.

    Guards the wiring, which no other test would notice going missing.
    """
    model_config = MagicMock(spec=ModelConfig)
    model_config.multimodal_config = MultiModalConfig(
        mm_processor_kwargs={"device": "cuda"}
    )
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.ec_transfer_config = None

    with (
        patch("vllm.platforms.current_platform.device_type", "cuda"),
        pytest.raises(ValueError, match="also runs the language model"),
    ):
        VllmConfig._validate_mm_processor_device(vllm_config)


def _resolve_mm_video_decode_device(
    *,
    ec_role: ECRole | None,
    mm_tensor_ipc: str = "torch_shm",
    device: str | None = None,
    video_kwargs: dict | None = None,
    torchcodec_available: bool = True,
) -> dict:
    """Run processor-device then video-decode-device resolution and report
    the resulting video media IO kwargs."""
    mm_config = MultiModalConfig(
        mm_processor_kwargs={} if device is None else {"device": device},
        mm_tensor_ipc=mm_tensor_ipc,
        media_io_kwargs={} if video_kwargs is None else {"video": dict(video_kwargs)},
    )
    model_config = MagicMock(spec=ModelConfig)
    model_config.multimodal_config = mm_config
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.ec_transfer_config = (
        None
        if ec_role is None
        else ECTransferConfig(ec_connector="ECExampleConnector", ec_role=ec_role)
    )

    with (
        patch("vllm.platforms.current_platform.device_type", "cuda"),
        patch(
            "vllm.utils.import_utils.check_torchcodec_available",
            side_effect=None if torchcodec_available else ImportError("torchcodec"),
        ),
    ):
        VllmConfig._resolve_mm_processor_device(vllm_config)
        VllmConfig._resolve_mm_video_decode_device(vllm_config)
    return mm_config.media_io_kwargs.get("video", {})


def test_auto_video_decode_uses_nvdec_on_encoder_instance():
    """The processor runs on the accelerator there, so decoded frames should
    stay on-device too."""
    assert _resolve_mm_video_decode_device(ec_role="ec_producer") == {
        "backend": "torchcodec",
        "device": "cuda",
    }


@pytest.mark.parametrize("ec_role", [None, "ec_consumer", "ec_both"])
def test_auto_video_decode_stays_on_cpu_off_encoder_instance(
    ec_role: ECRole | None,
):
    """The processor stays on CPU off encode-only instances, so video
    decoding should too."""
    assert _resolve_mm_video_decode_device(ec_role=ec_role) == {}


def test_auto_video_decode_follows_explicit_processor_device():
    assert _resolve_mm_video_decode_device(ec_role="ec_producer", device="cpu") == {}
    assert _resolve_mm_video_decode_device(ec_role="ec_producer", device="cuda") == {
        "backend": "torchcodec",
        "device": "cuda",
    }


def test_auto_video_decode_respects_explicit_media_io_kwargs():
    assert _resolve_mm_video_decode_device(
        ec_role="ec_producer", video_kwargs={"backend": "opencv"}
    ) == {"backend": "opencv"}
    assert _resolve_mm_video_decode_device(
        ec_role="ec_producer", video_kwargs={"device": "cpu"}
    ) == {"device": "cpu"}


def test_auto_video_decode_skipped_without_torchcodec():
    assert (
        _resolve_mm_video_decode_device(
            ec_role="ec_producer",
            torchcodec_available=False,
        )
        == {}
    )


@pytest.fixture(params=["gemma", "gemma2"])
def paligemma_prefix_model(tmp_path, request):
    from transformers import PaliGemmaConfig

    config = PaliGemmaConfig(
        text_config={
            "model_type": request.param,
            "sliding_window": 4 if request.param == "gemma2" else None,
            "query_pre_attn_scalar": 8,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 512,
            "max_position_embeddings": 512,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 16,
            "projection_dim": 32,
        },
        projection_dim=32,
        image_token_index=511,
    )
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PaliGemmaProcessor,
        PreTrainedTokenizerFast,
        SiglipImageProcessor,
    )

    tokenizer = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "<bos>": 1, "<eos>": 2, "caption": 3, "en": 4, "\n": 5},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    processor = PaliGemmaProcessor(
        image_processor=SiglipImageProcessor(
            size={"height": 32, "width": 32}, image_seq_length=4
        ),
        tokenizer=tokenizer,
    )
    processor.save_pretrained(tmp_path)
    config.image_token_index = processor.image_token_id
    config.architectures = ["PaliGemmaForConditionalGeneration"]
    config.save_pretrained(tmp_path)
    return ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        enforce_eager=True,
        dtype="float32",
        max_model_len=512,
    )


def _paligemma_prefix_config(model_config, **overrides):
    from vllm.config import CacheConfig, DeviceConfig, SchedulerConfig

    return VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(
            enable_prefix_caching=overrides.pop("enable_prefix_caching", False),
            kv_offloading_size=overrides.pop("kv_offloading_size", None),
            kv_offloading_backend=overrides.pop("kv_offloading_backend", "native"),
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=2,
            enable_chunked_prefill=overrides.pop("enable_chunked_prefill", False),
            long_prefill_token_threshold=overrides.pop(
                "long_prefill_token_threshold", 0
            ),
            long_prefill_token_threshold_adaptive=overrides.pop(
                "long_prefill_token_threshold_adaptive", False
            ),
            max_num_scheduled_tokens=overrides.pop("max_num_scheduled_tokens", None),
        ),
        **overrides,
    )


def test_paligemma_prefix_defaults_require_complete_prompt(paligemma_prefix_model):
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(model=paligemma_prefix_model.model, skip_tokenizer_init=True)
    args._set_default_chunked_prefill_and_prefix_caching_args(paligemma_prefix_model)
    assert args.enable_chunked_prefill is False
    assert args.enable_prefix_caching is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_chunked_prefill": True},
        {"enable_prefix_caching": True},
        {"long_prefill_token_threshold": 32},
        {"kv_offloading_size": 1},
        {"long_prefill_token_threshold_adaptive": True},
        {"max_num_scheduled_tokens": 128},
        {"kv_offloading_size": 1, "kv_offloading_backend": "lmcache"},
        {
            "kv_transfer_config": {
                "kv_connector": "ExampleConnector",
                "kv_role": "kv_consumer",
            }
        },
    ],
)
def test_paligemma_prefix_rejects_incomplete_context(paligemma_prefix_model, overrides):
    with pytest.raises(ValueError, match="PaliGemma"):
        _paligemma_prefix_config(paligemma_prefix_model, **overrides)


def test_paligemma_prefix_allows_complete_uncached_prompt(paligemma_prefix_model):
    config = _paligemma_prefix_config(paligemma_prefix_model)
    assert not config.scheduler_config.enable_chunked_prefill
    assert not config.cache_config.enable_prefix_caching


def test_paligemma_prefix_semantics_survive_language_model_view(paligemma_prefix_model):
    config = _paligemma_prefix_config(paligemma_prefix_model)
    child = config.with_hf_config(paligemma_prefix_model.hf_text_config)
    assert (
        child.model_config.hf_config.model_type
        == paligemma_prefix_model.hf_text_config.model_type
    )
    assert getattr(child.model_config, "is_full_prefix_lm", False)


def test_paligemma_prefix_attention_inherits_outer_window_contract(
    paligemma_prefix_model,
):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backends.flex_attention import FlexAttentionBackend

    config = _paligemma_prefix_config(paligemma_prefix_model)
    child = config.with_hf_config(paligemma_prefix_model.hf_text_config)
    with set_current_vllm_config(child):
        layer = Attention(
            num_heads=4,
            head_size=8,
            scale=8**-0.5,
            num_kv_heads=1,
            cache_config=child.cache_config,
            per_layer_sliding_window=4,
            attn_backend=FlexAttentionBackend,
            prefix="paligemma-prefix-test",
        )
    assert layer.mm_prefix_clamp_sliding_window


def test_paligemma_prefix_waits_for_encoder_then_schedules_whole_prompt(
    paligemma_prefix_model,
):
    from vllm.config import ModelConfig
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalKwargsItem,
        PlaceholderRange,
    )
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

    model = ModelConfig(
        model=paligemma_prefix_model.model,
        skip_tokenizer_init=False,
        enforce_eager=True,
        dtype="float32",
        max_model_len=512,
    )
    config = _paligemma_prefix_config(model)
    config.cache_config.num_gpu_blocks = 128
    kv_cache = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
                ),
            )
        ],
    )
    scheduler = Scheduler(
        config,
        kv_cache,
        StructuredOutputManager(config),
        block_size=16,
        log_stats=False,
    )
    scheduler.max_num_encoder_input_tokens = 0
    request = Request(
        request_id="prefix",
        prompt_token_ids=[1, 6, 6, 6, 6, 3, 4],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem({}),
                modality="image",
                identifier="offline-image",
                mm_position=PlaceholderRange(offset=1, length=4),
            )
        ],
    )
    scheduler.add_request(request)
    blocked = scheduler.schedule()
    assert blocked.num_scheduled_tokens.get("prefix", 0) == 0
    scheduler.max_num_encoder_input_tokens = 512
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {"prefix": request.num_prompt_tokens}


def test_paligemma_prefix_ranges_exclude_resumed_output():
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalKwargsItem,
        PlaceholderRange,
    )
    from vllm.v1.worker.gpu.attn_utils import compute_mm_prefix_ranges
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem({}),
        modality="image",
        identifier="offline-image",
        mm_position=PlaceholderRange(offset=1, length=4),
    )
    request = CachedRequestState(
        req_id="prefix",
        prompt_token_ids=[1, 6, 6, 6, 6, 3, 4],
        mm_features=[feature],
        sampling_params=None,
        generator=None,
        block_ids=([],),
        num_computed_tokens=0,
        output_token_ids=[7, 8, 9],
    )
    assert request.num_tokens > request.num_prompt_tokens
    assert compute_mm_prefix_ranges(
        [request.req_id],
        {request.req_id: request.mm_features},
        prompt_lens=[request.num_prompt_tokens],
    ) == {0: [(0, 6)]}


@pytest.mark.parametrize("clamp", [False, True])
@pytest.mark.parametrize("window", [None, 3])
def test_full_prefix_flex_mask_preserves_window_contract(clamp, window):
    from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

    n = 8
    prefix_len = 6
    metadata = FlexAttentionMetadata(
        causal=True,
        num_actual_tokens=n,
        max_query_len=n,
        query_start_loc=torch.tensor([0, n], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, n], dtype=torch.int32),
        max_seq_len=n,
        seq_lens=torch.tensor([n], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.arange(n),
        total_cache_tokens=n,
        block_size=n,
        max_possible_sequence_length=n,
        num_reqs=1,
        physical_to_logical=torch.tensor([[0]], dtype=torch.int32),
        decode_offset=torch.tensor([0], dtype=torch.int32),
        num_blocks_per_seq=torch.tensor([1], dtype=torch.int32),
        persistent_kv_indices=torch.zeros((1, 1, 1), dtype=torch.int32),
        persistent_kv_num_blocks=torch.zeros((1, 1), dtype=torch.int32),
        persistent_doc_ids=torch.zeros(n, dtype=torch.int32),
        doc_ids=torch.zeros(n, dtype=torch.int32),
        direct_build=False,
        sliding_window=window,
        mm_prefix_range={0: [(0, prefix_len - 1)]},
        mm_prefix_clamp_sliding_window=clamp,
    )
    mask = metadata.get_mask_mod()
    for q in range(n):
        for k in range(n):
            in_window = window is None or abs(q - k) < window
            expected = (k <= q and in_window) or (q < prefix_len and k < prefix_len)
            if clamp:
                expected &= in_window
            assert (
                bool(
                    mask(
                        torch.tensor(0),
                        torch.tensor(0),
                        torch.tensor(q),
                        torch.tensor(k),
                    )
                )
                == expected
            )


def test_full_prefix_ranges_allow_text_only_and_empty_requests():
    from vllm.v1.worker.gpu.attn_utils import compute_mm_prefix_ranges

    assert compute_mm_prefix_ranges(["text-only", "empty"], {}, prompt_lens=[7, 0]) == {
        0: [(0, 6)],
        1: [],
    }


@pytest.mark.parametrize("sliding_window,expected", [(None, [(1, 4)]), (2, [])])
def test_legacy_mm_prefix_ranges_keep_media_spans(sliding_window, expected):
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalKwargsItem,
        PlaceholderRange,
    )
    from vllm.v1.worker.gpu.attn_utils import compute_mm_prefix_ranges

    features = {
        "legacy": [
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem({}),
                modality="image",
                identifier="image",
                mm_position=PlaceholderRange(offset=1, length=4),
            ),
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem({}),
                modality="audio",
                identifier="audio",
                mm_position=PlaceholderRange(offset=5, length=3),
            ),
        ]
    }
    assert compute_mm_prefix_ranges(["legacy"], features, sliding_window) == {
        0: expected
    }
