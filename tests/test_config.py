"""Property-based tests for configuration management.

# Feature: nexifuse-robust-expert
"""

from __future__ import annotations

import tempfile
import os
from dataclasses import fields

from hypothesis import given, settings, strategies as st

from nexifuse.config import (
    ConfigManager,
    PipelineConfig,
    TrainingConfig,
    TeacherConfig,
    ScraperConfig,
    ValidationConfig,
    InferenceConfig,
    _REQUIRED_SECTIONS,
)


# --- Hypothesis strategies ---

training_configs = st.builds(
    TrainingConfig,
    base_model=st.text(min_size=1, max_size=50),
    lora_rank=st.integers(min_value=1, max_value=128),
    lora_alpha=st.integers(min_value=1, max_value=256),
    lora_target_modules=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=7),
    batch_size=st.integers(min_value=1, max_value=8),
    gradient_accumulation=st.integers(min_value=1, max_value=64),
    learning_rate=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
    lr_scheduler=st.sampled_from(["cosine", "linear", "constant"]),
    warmup_steps=st.integers(min_value=0, max_value=1000),
    num_epochs=st.integers(min_value=1, max_value=20),
    max_seq_length=st.integers(min_value=512, max_value=131072),
    quantization=st.sampled_from(["nf4", "mxfp4"]),
    output_dir=st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
    adapter_output_dir=st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
)

teacher_configs = st.builds(
    TeacherConfig,
    model_name=st.text(min_size=1, max_size=50),
    endpoint=st.text(min_size=1, max_size=100),
    context_docs_dir=st.text(min_size=1, max_size=50),
    domains=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=6),
    include_cot=st.booleans(),
)

scraper_configs = st.builds(
    ScraperConfig,
    repos=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5),
    file_patterns=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
    phi_patterns=st.lists(st.text(min_size=1, max_size=40), min_size=1, max_size=5),
)

validation_configs = st.builds(
    ValidationConfig,
    eslint_config=st.text(min_size=1, max_size=50),
    fhir_schema_dir=st.text(min_size=1, max_size=50),
)

inference_configs = st.builds(
    InferenceConfig,
    model_name=st.text(min_size=1, max_size=50),
    backend=st.sampled_from(["ollama", "vllm"]),
    host=st.text(min_size=1, max_size=30),
    port=st.integers(min_value=1, max_value=65535),
)

pipeline_configs = st.builds(
    PipelineConfig,
    training=training_configs,
    data_factory=teacher_configs,
    scraper=scraper_configs,
    validation=validation_configs,
    inference=inference_configs,
)


# --- Property 28: Configuration validation catches missing fields ---
# **Validates: Requirements 10.3, 10.4**

def _config_to_raw_dict(config: PipelineConfig) -> dict:
    """Convert a PipelineConfig to the raw dict format that ConfigManager.validate expects."""
    from dataclasses import asdict
    return asdict(config)


@given(
    config=pipeline_configs,
    section=st.sampled_from(list(_REQUIRED_SECTIONS.keys())),
)
@settings(max_examples=100)
def test_property_28_validation_catches_missing_fields(config: PipelineConfig, section: str):
    """Property 28: Configuration validation catches missing fields.

    For any configuration dictionary with a required field removed,
    ConfigManager.validate SHALL return a non-empty error list that
    includes the name of the missing field.

    **Validates: Requirements 10.3, 10.4**
    """
    raw = _config_to_raw_dict(config)
    required_fields = _REQUIRED_SECTIONS[section]

    # Pick a field to remove from this section
    if not required_fields:
        return

    for field_name in required_fields:
        mutated = {k: dict(v) if isinstance(v, dict) else v for k, v in raw.items()}
        mutated[section] = dict(mutated[section])
        del mutated[section][field_name]

        errors = ConfigManager.validate(mutated)
        assert len(errors) > 0, (
            f"Expected validation errors when '{section}.{field_name}' is missing, got none"
        )
        # The error list should mention the missing field
        assert any(field_name in e for e in errors), (
            f"Expected error mentioning '{field_name}', got: {errors}"
        )


# --- Property 29: Configuration YAML round-trip ---
# **Validates: Requirements 10.6**

@given(config=pipeline_configs)
@settings(max_examples=100)
def test_property_29_configuration_yaml_round_trip(config: PipelineConfig):
    """Property 29: Configuration YAML round-trip.

    For any valid PipelineConfig object, serializing to YAML then
    deserializing SHALL produce an equivalent PipelineConfig object
    with all fields matching.

    **Validates: Requirements 10.6**
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        tmp_path = f.name

    try:
        ConfigManager.serialize(config, tmp_path)
        restored = ConfigManager.deserialize(tmp_path)

        # Compare all fields of each sub-config
        assert config.training == restored.training, (
            f"Training config mismatch:\n  original: {config.training}\n  restored: {restored.training}"
        )
        assert config.data_factory == restored.data_factory, (
            f"TeacherConfig mismatch:\n  original: {config.data_factory}\n  restored: {restored.data_factory}"
        )
        assert config.scraper == restored.scraper, (
            f"ScraperConfig mismatch:\n  original: {config.scraper}\n  restored: {restored.scraper}"
        )
        assert config.validation == restored.validation, (
            f"ValidationConfig mismatch:\n  original: {config.validation}\n  restored: {restored.validation}"
        )
        assert config.inference == restored.inference, (
            f"InferenceConfig mismatch:\n  original: {config.inference}\n  restored: {restored.inference}"
        )
    finally:
        os.unlink(tmp_path)
