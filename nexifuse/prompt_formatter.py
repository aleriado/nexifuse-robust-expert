"""Prompt formatting for training and inference.

Formats training examples into ChatML/Llama-style templates
for fine-tuning and inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nexifuse.models import TrainingExample

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a healthcare integration expert specializing in Mirth Connect, "
    "HL7 v2, FHIR R4, and EHR API connectivity. You write production-grade "
    "integration code with proper error handling, security best practices, "
    "and compliance with healthcare data standards. When reasoning about "
    "complex integrations, think step by step before providing code."
)

# Llama 3 / DeepSeek-R1-Distill-Llama chat template
_LLAMA_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{output}<|eot_id|>"
)

# ChatML template (fallback)
_CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n{output}<|im_end|>"
)


def format_example(
    example: TrainingExample,
    template: str = "llama",
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Format a single training example into a chat template string.

    Args:
        example: The training example to format.
        template: Template style — "llama" or "chatml".
        system_prompt: System prompt to inject.

    Returns:
        Formatted prompt string.
    """
    # Build the output, optionally including CoT trace
    output = ""
    if example.cot_trace:
        output = f"<think>\n{example.cot_trace}\n</think>\n\n"
    output += example.output

    # Include input context if present
    instruction = example.instruction
    if example.input:
        instruction = f"{instruction}\n\n{example.input}"

    tmpl = _LLAMA_TEMPLATE if template == "llama" else _CHATML_TEMPLATE
    return tmpl.format(
        system=system_prompt,
        instruction=instruction,
        output=output,
    )


def format_dataset(
    input_path: str | Path,
    output_path: str | Path = "data/formatted/train.jsonl",
    template: str = "llama",
    system_prompt: str = SYSTEM_PROMPT,
    identity_paths: list[str | Path] | None = None,
) -> int:
    """Format an entire JSONL dataset into chat-templated training data.

    Each output line is a JSON object with a "text" field containing
    the formatted conversation, suitable for causal LM fine-tuning.
    If identity_paths are provided, those examples are formatted and
    written first (conversational/identity), then the main input_path.

    Args:
        input_path: JSONL file of TrainingExample records (validated code examples).
        output_path: Where to write formatted JSONL.
        template: Template style — "llama" or "chatml".
        system_prompt: System prompt to inject.
        identity_paths: Optional list of JSONL files with conversational/identity examples to prepend.

    Returns:
        Number of examples formatted.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def format_stream(path: Path) -> int:
        c = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    example = TrainingExample.from_dict(record)
                    formatted = format_example(example, template, system_prompt)
                    fout.write(json.dumps({"text": formatted}, ensure_ascii=False) + "\n")
                    c += 1
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.debug("Skipping malformed record in %s: %s", path, exc)
        return c

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        # Prepend identity/conversational examples
        for ip in identity_paths or []:
            p = Path(ip)
            if p.exists():
                n = format_stream(p)
                count += n
                logger.info("Formatted %d identity examples from %s", n, p)

        # Main validated dataset
        if input_path.exists():
            n = format_stream(input_path)
            count += n

    logger.info("Formatted %d examples → %s (template: %s)", count, output_path, template)
    return count
