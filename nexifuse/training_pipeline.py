"""Training pipeline using Unsloth + LoRA.

Loads a quantized base model, applies LoRA adapters, and fine-tunes
on the formatted training dataset. Supports SFT and optional DPO stages.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from nexifuse.config import PipelineConfig

logger = logging.getLogger(__name__)


def _is_multi_gpu_ddp() -> bool:
    """True if we are in a multi-process DDP run (e.g. accelerate launch --num_processes 8)."""
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return world_size > 1
    except ValueError:
        return False


def _check_gpu():
    """Log GPU info if available."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                mem = props.total_memory / (1024**3)
                logger.info("GPU %d: %s (%.1f GB)", i, name, mem)
        else:
            logger.warning("No CUDA GPU detected")
    except ImportError:
        logger.warning("PyTorch not installed")





def run_sft(
    config: PipelineConfig,
    train_data_path: str | Path = "data/formatted/train.jsonl",
) -> Path:
    """Run supervised fine-tuning with Unsloth + LoRA.

    Returns:
        Path to the saved LoRA adapter directory.
    """
    tc = config.training
    train_data_path = Path(train_data_path)
    output_dir = Path(tc.output_dir)
    adapter_dir = Path(tc.adapter_output_dir)

    _check_gpu()

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import torch

    logger.info("Loading base model: %s (4-bit: %s, max_seq: %d)",
                tc.base_model, tc.quantization == "nf4", tc.max_seq_length)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=tc.base_model,
        max_seq_length=tc.max_seq_length,
        load_in_4bit=(tc.quantization == "nf4"),
        dtype=None,
    )

    logger.info("Applying LoRA: r=%d, alpha=%d, targets=%s",
                tc.lora_rank, tc.lora_alpha, tc.lora_target_modules)

    # Unsloth gradient checkpointing + DDP causes "marked ready twice" (reentrant backward).
    # Disable it when running multi-GPU so DDP works.
    use_grad_ckpt = "unsloth" if not _is_multi_gpu_ddp() else False
    if _is_multi_gpu_ddp():
        logger.info("Multi-GPU DDP detected: disabling Unsloth gradient checkpointing for DDP compatibility")

    model = FastLanguageModel.get_peft_model(
        model,
        r=tc.lora_rank,
        lora_alpha=tc.lora_alpha,
        lora_dropout=0,
        target_modules=tc.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=use_grad_ckpt,
    )

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    logger.info("Loading training data from %s", train_data_path)
    dataset = load_dataset("json", data_files=str(train_data_path), split="train")
    logger.info("Dataset size: %d examples", len(dataset))

    # Determine FP16 vs BF16 based on GPU capability
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    logger.info("Precision: %s", "BF16" if use_bf16 else "FP16")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=tc.batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation,
        learning_rate=tc.learning_rate,
        lr_scheduler_type=tc.lr_scheduler,
        warmup_steps=tc.warmup_steps,
        num_train_epochs=tc.num_epochs,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    # ── Subclass SFTTrainer to fix Unsloth int-loss bug ────────────
    # Unsloth 2026.2.x patches training_step and calls loss.mean()
    # but compute_loss can return int(0) for fully-masked batches.
    # We override compute_loss to guarantee a tensor return.
    def _device_for_model(model):
        return next(model.parameters()).device

    def _to_loss_tensor(value, device):
        if isinstance(value, torch.Tensor):
            return value
        try:
            return torch.tensor(float(value), device=device, requires_grad=True)
        except (TypeError, ValueError):
            return torch.tensor(0.0, device=device, requires_grad=True)

    class SafeSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            device = _device_for_model(model)
            if return_outputs:
                loss, outputs = result
                if not isinstance(loss, torch.Tensor):
                    loss = _to_loss_tensor(loss, device)
                return loss, outputs
            else:
                if not isinstance(result, torch.Tensor):
                    result = _to_loss_tensor(result, device)
                return result
    # ───────────────────────────────────────────────────────────────

    trainer = SafeSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=tc.max_seq_length,
        packing=False,
    )

    logger.info("Starting SFT training (%d epochs)...", tc.num_epochs)
    train_result = trainer.train()

    # Log training metrics (all ranks)
    metrics = train_result.metrics
    logger.info("Training complete. Loss: %.4f, Runtime: %.1fs, Samples/sec: %.2f",
                metrics.get("train_loss", 0),
                metrics.get("train_runtime", 0),
                metrics.get("train_samples_per_second", 0))

    # Save LoRA adapter only on main process in DDP to avoid overwrites
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("LoRA adapter saved to %s", adapter_dir)

    return adapter_dir


# ---------------------------------------------------------------------------
# Reward function for GRPO
# ---------------------------------------------------------------------------

def compute_reward(output_text: str, domain: str | None = None, config: PipelineConfig | None = None) -> float:
    """Score model output using domain validators.

    Checks JavaScript bracket balance, XML well-formedness, HL7 v2 structure,
    and FHIR R4 validity.  Each passing check adds 0.25 (max 1.0).
    If the text does not look like any recognised format, returns a base score
    of 0.5 so that general-purpose answers are not penalised to zero.

    Returns:
        Float in [0.0, 1.0].
    """
    from nexifuse.validator import (
        validate_javascript,
        validate_xml,
        validate_hl7v2,
        validate_fhir_r4,
    )

    text = output_text.strip()
    if not text:
        return 0.0

    eslint_config = config.validation.eslint_config if config else None
    fhir_schema_dir = config.validation.fhir_schema_dir if config else None

    score = 0.0
    any_applicable = False

    # JavaScript bracket validation
    try:
        if any(ch in text for ch in "{}()[]"):
            any_applicable = True
            if validate_javascript(text, eslint_config).passed:
                score += 0.25
    except Exception:
        pass

    # XML well-formedness
    try:
        stripped = text.lstrip()
        if stripped.startswith("<") and ">" in stripped:
            any_applicable = True
            if validate_xml(text).passed:
                score += 0.25
    except Exception:
        pass

    # HL7 v2 validation
    try:
        if text.startswith("MSH|"):
            any_applicable = True
            if validate_hl7v2(text).passed:
                score += 0.25
    except Exception:
        pass

    # FHIR R4 validation
    try:
        if stripped.startswith("{"):
            import json as _json
            data = _json.loads(text)
            if isinstance(data, dict) and data.get("resourceType"):
                any_applicable = True
                if validate_fhir_r4(text, fhir_schema_dir).passed:
                    score += 0.25
    except Exception:
        pass

    # If no format-specific validator applied, give a neutral base score
    if not any_applicable:
        return 0.5

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# GRPO training
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir: Path) -> str | None:
    """Return the path of the latest checkpoint-* directory, or None."""
    if not output_dir.exists():
        return None
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=os.path.getmtime)
    if checkpoints:
        return str(checkpoints[-1])
    return None


def run_grpo(
    config: PipelineConfig,
    sft_adapter_path: str | Path,
    train_data_path: str | Path,
) -> Path:
    """Run GRPO (Group Relative Policy Optimization) training.

    Loads the SFT adapter, then performs GRPO using ``compute_reward`` to
    score sampled completions.  The updated adapter is saved to
    ``config.training.adapter_output_dir``.

    Args:
        config: Full pipeline configuration.
        sft_adapter_path: Path to the SFT LoRA adapter to initialise from.
        train_data_path: JSONL file with a ``"prompt"`` field per line.

    Returns:
        Path to the saved adapter directory.
    """
    tc = config.training
    gc = config.grpo
    sft_adapter_path = Path(sft_adapter_path)
    train_data_path = Path(train_data_path)
    output_dir = Path(tc.output_dir) / "grpo"
    adapter_dir = Path(tc.adapter_output_dir)

    _check_gpu()

    # -- Imports (trl may or may not ship GRPOTrainer) ----------------------
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    import torch

    try:
        from trl import GRPOTrainer, GRPOConfig as TRLGRPOConfig
    except ImportError as exc:
        raise ImportError(
            "trl >= 0.15 is required for GRPOTrainer. "
            "Install with: pip install trl>=0.15"
        ) from exc

    # -- Load model with existing SFT adapter -------------------------------
    logger.info("Loading base model with SFT adapter from %s", sft_adapter_path)
    # Use shorter seq length for GRPO to avoid dimension mismatch in chunked log-softmax
    grpo_seq_length = min(tc.max_seq_length, 1024)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(sft_adapter_path),
        max_seq_length=grpo_seq_length,
        load_in_4bit=(tc.quantization == "nf4"),
        dtype=None,
    )

    use_grad_ckpt = "unsloth" if not _is_multi_gpu_ddp() else False
    if _is_multi_gpu_ddp():
        logger.info("Multi-GPU DDP detected: disabling Unsloth gradient checkpointing")

    model = FastLanguageModel.get_peft_model(
        model,
        r=tc.lora_rank,
        lora_alpha=tc.lora_alpha,
        lora_dropout=0,
        target_modules=tc.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=use_grad_ckpt,
    )

    # -- Load training data -------------------------------------------------
    logger.info("Loading GRPO prompts from %s", train_data_path)
    dataset = load_dataset("json", data_files=str(train_data_path), split="train")
    # GRPO expects a "prompt" field — map from our data format
    if "prompt" not in dataset.column_names and "instruction" in dataset.column_names:
        dataset = dataset.rename_column("instruction", "prompt")
    elif "prompt" not in dataset.column_names and "text" in dataset.column_names:
        # Extract user prompt from formatted chat template
        import re
        def extract_prompt(example):
            match = re.search(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", example["text"], re.DOTALL)
            example["prompt"] = match.group(1).strip() if match else example["text"][:500]
            return example
        dataset = dataset.map(extract_prompt)
    logger.info("GRPO dataset size: %d prompts", len(dataset))

    # -- Precision ----------------------------------------------------------
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    # -- Resume support -----------------------------------------------------
    latest_ckpt = _find_latest_checkpoint(output_dir)
    if latest_ckpt:
        logger.info("Resuming GRPO from checkpoint: %s", latest_ckpt)

    # -- Reward wrapper (GRPOTrainer expects list[str] -> list[float]) ------
    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        return [compute_reward(text, config=config) for text in completions]

    # -- Training arguments -------------------------------------------------
    training_args = TRLGRPOConfig(
        output_dir=str(output_dir),
        max_steps=gc.max_steps,
        per_device_train_batch_size=gc.batch_size,
        gradient_accumulation_steps=gc.gradient_accumulation,
        learning_rate=gc.learning_rate,
        num_generations=gc.num_generations,
        max_completion_length=gc.max_completion_length,
        beta=gc.beta,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        args=training_args,
    )

    logger.info("Starting GRPO training (max_steps=%d) ...", gc.max_steps)
    trainer.train(resume_from_checkpoint=latest_ckpt)

    # -- Save adapter (main process only) -----------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("GRPO adapter saved to %s", adapter_dir)

    return adapter_dir


# ---------------------------------------------------------------------------
# SimPO training
# ---------------------------------------------------------------------------

def run_simpo(
    config: PipelineConfig,
    adapter_path: str | Path,
    preference_data_path: str | Path,
) -> Path:
    """Run SimPO (Simple Preference Optimization) training.

    Loads the current adapter, then trains on preference pairs using the
    SimPO loss.  The updated adapter is saved to
    ``config.training.adapter_output_dir``.

    Args:
        config: Full pipeline configuration.
        adapter_path: Path to the LoRA adapter to continue from.
        preference_data_path: JSONL with ``"prompt"``, ``"chosen"``, ``"rejected"``
            fields per line.

    Returns:
        Path to the saved adapter directory.
    """
    tc = config.training
    sc = config.simpo
    adapter_path = Path(adapter_path)
    preference_data_path = Path(preference_data_path)
    output_dir = Path(tc.output_dir) / "simpo"
    adapter_dir = Path(tc.adapter_output_dir)

    _check_gpu()

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    import torch

    # SimPO may be exposed as a dedicated trainer or via DPOTrainer with
    # loss_type="simpo".  Try the dedicated class first.
    SimPOTrainerCls = None
    SimPOConfigCls = None
    try:
        from trl import SimPOTrainer as _SimPOTrainer, SimPOConfig as _SimPOConfig
        SimPOTrainerCls = _SimPOTrainer
        SimPOConfigCls = _SimPOConfig
    except ImportError:
        pass

    if SimPOTrainerCls is None:
        try:
            from trl import DPOTrainer, DPOConfig
            SimPOTrainerCls = DPOTrainer
            SimPOConfigCls = DPOConfig
            logger.info("SimPOTrainer not available; falling back to DPOTrainer with loss_type='simpo'")
        except ImportError as exc:
            raise ImportError(
                "trl >= 0.12 is required for SimPO / DPO training. "
                "Install with: pip install trl>=0.12"
            ) from exc

    # -- Load model with adapter --------------------------------------------
    logger.info("Loading base model with adapter from %s", adapter_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=tc.max_seq_length,
        load_in_4bit=(tc.quantization == "nf4"),
        dtype=None,
    )

    use_grad_ckpt = "unsloth" if not _is_multi_gpu_ddp() else False
    if _is_multi_gpu_ddp():
        logger.info("Multi-GPU DDP detected: disabling Unsloth gradient checkpointing")

    model = FastLanguageModel.get_peft_model(
        model,
        r=tc.lora_rank,
        lora_alpha=tc.lora_alpha,
        lora_dropout=0,
        target_modules=tc.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=use_grad_ckpt,
    )

    # -- Load preference data -----------------------------------------------
    logger.info("Loading preference data from %s", preference_data_path)
    dataset = load_dataset("json", data_files=str(preference_data_path), split="train")
    logger.info("SimPO dataset size: %d preference pairs", len(dataset))

    # -- Precision ----------------------------------------------------------
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    # -- Resume support -----------------------------------------------------
    latest_ckpt = _find_latest_checkpoint(output_dir)
    if latest_ckpt:
        logger.info("Resuming SimPO from checkpoint: %s", latest_ckpt)

    # -- Build training config ----------------------------------------------
    simpo_kwargs: dict = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=sc.batch_size,
        gradient_accumulation_steps=sc.gradient_accumulation,
        learning_rate=sc.learning_rate,
        num_train_epochs=sc.num_epochs,
        beta=sc.beta,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    # Add SimPO-specific parameters
    try:
        # Dedicated SimPOConfig supports gamma directly
        if SimPOConfigCls.__name__ == "SimPOConfig":
            simpo_kwargs["gamma"] = sc.gamma
        else:
            # DPOConfig: specify loss_type and simpo_gamma
            simpo_kwargs["loss_type"] = "simpo"
            simpo_kwargs["simpo_gamma"] = sc.gamma
    except Exception:
        simpo_kwargs["loss_type"] = "simpo"

    training_args = SimPOConfigCls(**simpo_kwargs)

    trainer = SimPOTrainerCls(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting SimPO training (%d epoch(s)) ...", sc.num_epochs)
    trainer.train(resume_from_checkpoint=latest_ckpt)

    # -- Save adapter (main process only) -----------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("SimPO adapter saved to %s", adapter_dir)

    return adapter_dir
