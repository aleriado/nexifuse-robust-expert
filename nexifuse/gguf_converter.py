"""LoRA merge + GGUF conversion pipeline using Unsloth's native methods.

Merges LoRA adapter into base weights, converts to GGUF format,
quantizes, and generates an Ollama Modelfile.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from nexifuse.config import PipelineConfig
from nexifuse.prompt_formatter import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def merge_and_save(
    config: PipelineConfig,
    adapter_path: str | Path | None = None,
    merged_output: str | Path = "outputs/merged_model",
) -> Path:
    """Merge LoRA adapter into base model and save as full HF model.

    Uses Unsloth's save_pretrained_merged for efficient merging.
    """
    tc = config.training
    adapter_path = Path(adapter_path) if adapter_path else Path(tc.adapter_output_dir)
    merged_output = Path(merged_output)

    from unsloth import FastLanguageModel

    logger.info("Loading adapter from %s for merge", adapter_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=tc.max_seq_length,
        load_in_4bit=(tc.quantization == "nf4"),
    )

    logger.info("Merging LoRA and saving to %s (16-bit)...", merged_output)
    model.save_pretrained_merged(
        str(merged_output),
        tokenizer,
        save_method="merged_16bit",
    )
    logger.info("Merged model saved to %s", merged_output)
    return merged_output


def save_gguf(
    config: PipelineConfig,
    adapter_path: str | Path | None = None,
    output_dir: str | Path = "outputs",
    quantization: str = "q4_k_m",
) -> Path:
    """Convert LoRA adapter directly to quantized GGUF using Unsloth.

    This is the preferred method — Unsloth handles the full pipeline:
    merge LoRA → convert to GGUF → quantize, all in one step.

    Args:
        config: Pipeline configuration.
        adapter_path: Path to LoRA adapter directory.
        output_dir: Directory to write GGUF files.
        quantization: GGUF quantization type (q4_k_m, q5_k_m, q8_0, f16, etc.)

    Returns:
        Path to the output directory containing GGUF files.
    """
    tc = config.training
    adapter_path = Path(adapter_path) if adapter_path else Path(tc.adapter_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from unsloth import FastLanguageModel

    logger.info("Loading adapter from %s", adapter_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=tc.max_seq_length,
        load_in_4bit=(tc.quantization == "nf4"),
    )

    logger.info("Saving GGUF with quantization: %s → %s", quantization, output_dir)
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=quantization,
    )

    # Find the generated GGUF file
    gguf_files = list(output_dir.glob("*.gguf"))
    if gguf_files:
        logger.info("GGUF file(s) created: %s", [f.name for f in gguf_files])
    else:
        logger.warning("No .gguf files found in %s — check Unsloth output", output_dir)

    return output_dir


def generate_modelfile(
    gguf_path: str | Path,
    output_path: str | Path = "outputs/Modelfile",
    model_name: str = "nexifuse-robust-expert",
    temperature: float = 0.1,
) -> Path:
    """Generate an Ollama Modelfile for the GGUF model."""
    gguf_path = Path(gguf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Llama 3 chat template matching the training format
    chat_template = (
        '{{- if .System }}<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        '{{ .System }}<|eot_id|>{{ end }}'
        '{{ range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n'
        '{{ .Content }}<|eot_id|>{{ end }}'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    )

    modelfile_content = f'''FROM {gguf_path.resolve()}

TEMPLATE """{chat_template}"""

SYSTEM """{SYSTEM_PROMPT}"""

PARAMETER temperature {temperature}
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER num_ctx 4096
'''

    output_path.write_text(modelfile_content, encoding="utf-8")
    logger.info("Modelfile written to %s", output_path)
    return output_path


def register_with_ollama(
    modelfile_path: str | Path = "outputs/Modelfile",
    model_name: str = "nexifuse-robust-expert",
) -> bool:
    """Register the model with Ollama using the generated Modelfile."""
    modelfile_path = Path(modelfile_path)

    logger.info("Registering model '%s' with Ollama...", model_name)
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            logger.info("Model '%s' registered successfully with Ollama", model_name)
            return True
        else:
            logger.error("Ollama create failed: %s", result.stderr)
            return False
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.error("Failed to register with Ollama: %s", exc)
        return False
