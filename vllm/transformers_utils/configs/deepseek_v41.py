from typing import Any
from transformers import PretrainedConfig

class DeepseekV41TextConfig(PretrainedConfig):
    model_type = "deepseek_v4.1_text"
    def __init__(self, max_position_embeddings=1048576, **kwargs):
        # Transformers >=5 expects the normalized RoPE field to be present
        # before validation/standardization of rope_scaling.
        rope_parameters = kwargs.get("rope_parameters")
        if rope_parameters is None and kwargs.get("rope_scaling") is not None:
            rope_parameters = kwargs["rope_scaling"]
        if rope_parameters is not None:
            self.rope_parameters = rope_parameters
        self.max_position_embeddings = max_position_embeddings
        super().__init__(max_position_embeddings=max_position_embeddings, **kwargs)

class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v4.1_vision"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v4.1"
    sub_configs = {"text_config": DeepseekV41TextConfig, "vision_config": DeepseekV41VisionConfig}
    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = DeepseekV41TextConfig(**text_config)
        elif text_config is None:
            text_config = DeepseekV41TextConfig()
        self.text_config = text_config
        if isinstance(vision_config, dict):
            vision_config = DeepseekV41VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = DeepseekV41VisionConfig()
        self.vision_config = vision_config
        super().__init__(**kwargs)
