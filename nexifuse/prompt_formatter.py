"""Prompt formatting for training and inference.

Formats training examples into ChatML/Llama-style templates
for fine-tuning and inference. Supports single-turn and multi-turn conversations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nexifuse.models import ConversationExample, ConversationTurn, TrainingExample

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are NexiFuse, a healthcare integration expert specializing in Mirth Connect, HL7 v2, "
    "FHIR R4, and EHR API connectivity. You write production-grade integration code with proper "
    "error handling, security best practices, and compliance with healthcare data standards. "
    "When reasoning about complex integrations, think step by step.\n\n"
    "CRITICAL RULES:\n"
    "1. Always identify yourself as \"NexiFuse\" when asked about your identity.\n"
    "2. Never reference APIs, libraries, or functions that do not exist.\n"
    "3. For debugging questions, always provide: symptoms, diagnosis, fix, and verification steps.\n"
    "4. For math questions, show all work step by step and verify your final answer.\n"
    "5. Include error handling (try/catch) in all code examples.\n"
    "6. Never claim to be ChatGPT, GPT-4, Gemini, or any other AI system."
)

# Llama 3 / DeepSeek-R1-Distill-Llama single-turn chat template
_LLAMA_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{output}<|eot_id|>"
)

# ChatML single-turn template (fallback)
_CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n{output}<|im_end|>"
)

# Multi-turn templates
_LLAMA_MULTITURN_HEADER = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|>"
)

_LLAMA_TURN = (
    "<|start_header_id|>{role}<|end_header_id|>\n\n"
    "{content}<|eot_id|>"
)

_CHATML_MULTITURN_HEADER = "<|im_start|>system\n{system}<|im_end|>\n"

_CHATML_TURN = "<|im_start|>{role}\n{content}<|im_end|>\n"


def format_example(
    example: TrainingExample,
    template: str = "llama",
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Format a single training example into a chat template string."""
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


def format_conversation(
    example: ConversationExample,
    template: str = "llama",
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Format a multi-turn conversation into a chat template string."""
    if template == "llama":
        parts = [_LLAMA_MULTITURN_HEADER.format(system=system_prompt)]
        for turn in example.turns:
            parts.append(_LLAMA_TURN.format(role=turn.role, content=turn.content))
    else:
        parts = [_CHATML_MULTITURN_HEADER.format(system=system_prompt)]
        for turn in example.turns:
            parts.append(_CHATML_TURN.format(role=turn.role, content=turn.content))
    return "".join(parts)


def format_dataset(
    input_path: str | Path,
    output_path: str | Path = "data/formatted/train.jsonl",
    template: str = "llama",
    system_prompt: str = SYSTEM_PROMPT,
    identity_paths: list[str | Path] | None = None,
    conversation_paths: list[str | Path] | None = None,
) -> int:
    """Format an entire JSONL dataset into chat-templated training data.

    Each output line is a JSON object with a "text" field containing
    the formatted conversation, suitable for causal LM fine-tuning.

    Handles both single-turn (TrainingExample) and multi-turn (ConversationExample)
    records by detecting whether a "turns" key is present.

    Args:
        input_path: JSONL file of TrainingExample records (validated code examples).
        output_path: Where to write formatted JSONL.
        template: Template style — "llama" or "chatml".
        system_prompt: System prompt to inject.
        identity_paths: Optional JSONL files with conversational/identity examples to prepend.
        conversation_paths: Optional JSONL files with multi-turn conversation examples.

    Returns:
        Number of examples formatted.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def format_mixed_stream(path: Path) -> int:
        """Format a JSONL file that may contain both single-turn and multi-turn records."""
        c = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Auto-detect record type
                    if "turns" in record and isinstance(record["turns"], list):
                        conv = ConversationExample.from_dict(record)
                        formatted = format_conversation(conv, template, system_prompt)
                    else:
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
                n = format_mixed_stream(p)
                count += n
                logger.info("Formatted %d identity examples from %s", n, p)

        # Main validated dataset (may contain both single-turn and multi-turn)
        if input_path.exists():
            n = format_mixed_stream(input_path)
            count += n

        # Additional conversation files
        for cp in conversation_paths or []:
            p = Path(cp)
            if p.exists():
                n = format_mixed_stream(p)
                count += n
                logger.info("Formatted %d conversation examples from %s", n, p)

    logger.info("Formatted %d examples → %s (template: %s)", count, output_path, template)
    return count
