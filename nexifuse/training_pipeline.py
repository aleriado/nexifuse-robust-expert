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



