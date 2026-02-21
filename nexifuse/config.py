"""Configuration management for the NexiFuse pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    """Training pipeline hyperparameters."""

    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    batch_size: int = 1
    gradient_accumulation: int = 16
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    num_epochs: int = 3
    max_seq_length: int = 131072
    quantization: str = "nf4"
    output_dir: str = "outputs"
    adapter_output_dir: str = "nexifuse_model_adapter"


@dataclass
class TeacherConfig:
    """Teacher model configuration for synthetic data generation."""

    model_name: str = "deepseek-r1:70b"
    endpoint: str = "http://localhost:11434/api/generate"
    context_docs_dir: str = "./docs"
    domains: list[str] = field(
        default_factory=lambda: [
            "hl7v2", "fhir_r4", "mirth", "ehr_api", "ihe", "dicom",
        ]
    )
    include_cot: bool = True


@dataclass
class ScraperConfig:
    """GitHub corpus scraper configuration."""

    repos: list[str] = field(
        default_factory=lambda: [
            "nextgenhealthcare/connect-examples",
            "nextgenhealthcare/connect",
            "kaur-ravneet/mirthsync",
        ]
    )
    file_patterns: list[str] = field(
        default_factory=lambda: ["*.js", "*.xml", "*.json"]
    )
    phi_patterns: list[str] = field(
        default_factory=lambda: [
            r"\d{3}-\d{2}-\d{4}",
            r"password\s*=\s*['\"]",
            r"api[_-]?key\s*=\s*['\"]",
        ]
    )


@dataclass
class ValidationConfig:
    """Validation engine configuration."""

    eslint_config: str = ".eslintrc.mirth.json"
    fhir_schema_dir: str = "./schemas/fhir-r4"


@dataclass
class InferenceConfig:
    """Inference server configuration."""

    model_name: str = "nexifuse-robust-expert"
    backend: str = "ollama"
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration combining all sub-configs."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_factory: TeacherConfig = field(default_factory=TeacherConfig)
    scraper: ScraperConfig = field(default_factory=ScraperConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# Required top-level sections and their required fields
_REQUIRED_SECTIONS: dict[str, list[str]] = {
    "training": [
        "base_model", "lora_rank", "lora_alpha", "batch_size",
        "gradient_accumulation", "learning_rate", "lr_scheduler",
        "warmup_steps", "num_epochs", "max_seq_length", "quantization",
        "output_dir", "adapter_output_dir",
    ],
    "data_factory": [
        "model_name", "endpoint", "context_docs_dir", "domains", "include_cot",
    ],
    "scraper": ["repos", "file_patterns", "phi_patterns"],
    "validation": ["eslint_config", "fhir_schema_dir"],
    "inference": ["model_name", "backend", "host", "port"],
}


class ConfigManager:
    """Loads, validates, serializes, and deserializes pipeline configuration."""

    @staticmethod
    def load(path: str) -> PipelineConfig:
        """Load and validate a YAML config file. Raises on missing required fields."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(p) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Configuration file must contain a YAML mapping, got {type(raw).__name__}")

        errors = ConfigManager.validate(raw)
        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return ConfigManager._dict_to_config(raw)

    @staticmethod
    def validate(config: dict) -> list[str]:
        """Return list of validation errors. Empty list means valid."""
        errors: list[str] = []

        if not isinstance(config, dict):
            return [f"Config must be a dict, got {type(config).__name__}"]

        for section, required_fields in _REQUIRED_SECTIONS.items():
            if section not in config:
                errors.append(f"Missing required section: '{section}'")
                continue
            sec = config[section]
            if not isinstance(sec, dict):
                errors.append(f"Section '{section}' must be a mapping, got {type(sec).__name__}")
                continue
            for fld in required_fields:
                if fld not in sec:
                    errors.append(f"Missing required field: '{section}.{fld}'")

        return errors

    @staticmethod
    def serialize(config: PipelineConfig, path: str) -> None:
        """Serialize a PipelineConfig to a YAML file."""
        data = asdict(config)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def deserialize(path: str) -> PipelineConfig:
        """Deserialize a YAML file to a PipelineConfig."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(p) as f:
            raw = yaml.safe_load(f)

        return ConfigManager._dict_to_config(raw)

    @staticmethod
    def _dict_to_config(raw: dict) -> PipelineConfig:
        """Convert a raw dict to a PipelineConfig."""
        training_data = raw.get("training", {})
        data_factory_data = raw.get("data_factory", {})
        scraper_data = raw.get("scraper", {})
        validation_data = raw.get("validation", {})
        inference_data = raw.get("inference", {})

        # Coerce numeric fields that YAML may parse as strings (e.g., 2e-4)
        if "learning_rate" in training_data:
            training_data["learning_rate"] = float(training_data["learning_rate"])
        if "port" in inference_data:
            inference_data["port"] = int(inference_data["port"])

        return PipelineConfig(
            training=TrainingConfig(**{
                k: v for k, v in training_data.items()
                if k in TrainingConfig.__dataclass_fields__
            }),
            data_factory=TeacherConfig(**{
                k: v for k, v in data_factory_data.items()
                if k in TeacherConfig.__dataclass_fields__
            }),
            scraper=ScraperConfig(**{
                k: v for k, v in scraper_data.items()
                if k in ScraperConfig.__dataclass_fields__
            }),
            validation=ValidationConfig(**{
                k: v for k, v in validation_data.items()
                if k in ValidationConfig.__dataclass_fields__
            }),
            inference=InferenceConfig(**{
                k: v for k, v in inference_data.items()
                if k in InferenceConfig.__dataclass_fields__
            }),
        )
