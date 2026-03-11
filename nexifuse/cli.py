"""CLI entry points for the NexiFuse pipeline.

Usage:
    python -m nexifuse.cli <command> [options]

Commands:
    ingest      Ingest documentation from docs/ directory
    scrape      Scrape GitHub repos for training data
    generate    Generate synthetic training data via teacher model
    clean       Clean and deduplicate raw training data
    validate    Validate training examples
    dpo         Generate DPO preference pairs
    format      Format validated data into chat templates
    train       Run SFT fine-tuning
    train-dpo   Run DPO alignment (after SFT)
    merge       Merge LoRA adapter into base model
    convert     Convert merged model to GGUF
    quantize    Quantize GGUF model
    modelfile   Generate Ollama Modelfile
    serve       Start inference server
    pipeline    Run full pipeline (ingest → scrape → generate → clean → validate → format)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from nexifuse.config import ConfigManager

logger = logging.getLogger("nexifuse")


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_ingest(args):
    from nexifuse.doc_ingester import ingest_docs
    stats = ingest_docs(docs_dir=args.docs_dir, output_dir=args.output_dir)
    print(f"Ingestion complete: {stats}")


def cmd_scrape(args):
    config = ConfigManager.load(args.config)
    from nexifuse.scraper import scrape_repos
    examples = scrape_repos(
        config, output_path=args.output, repos_dir=args.repos_dir,
        use_teacher=not args.no_teacher,
    )
    print(f"Scraped {len(examples)} examples")


def cmd_generate(args):
    config = ConfigManager.load(args.config)
    from nexifuse.data_factory import generate_examples
    examples = generate_examples(
        config, output_path=args.output, docs_dir=args.docs_dir,
        num_per_domain=args.num_per_domain,
    )
    print(f"Generated {len(examples)} examples")


def cmd_generate_general(args):
    config = ConfigManager.load(args.config)
    from nexifuse.data_factory import generate_general_examples
    examples = generate_general_examples(
        config, output_path=args.output, num_per_category=args.num_per_category,
    )
    print(f"Generated {len(examples)} general examples")


def cmd_generate_conversations(args):
    config = ConfigManager.load(args.config)
    from nexifuse.data_factory import generate_conversations
    examples = generate_conversations(
        config, output_path=args.output,
        num_per_scenario_domain=args.num_per_scenario_domain,
    )
    print(f"Generated {len(examples)} conversation examples")


def cmd_clean(args):
    from nexifuse.data_cleaner import clean_data
    input_paths = args.input if args.input else sorted(str(p) for p in Path("data/raw").glob("*.jsonl"))
    if not input_paths:
        print("No JSONL files found in data/raw/")
        return
    print(f"Cleaning {len(input_paths)} files: {[Path(p).name for p in input_paths]}")
    stats = clean_data(input_paths, output_path=args.output, similarity_threshold=args.threshold)
    print(f"Cleaning: {stats.input_rows} → {stats.output_rows} rows")


def cmd_validate(args):
    config = ConfigManager.load(args.config)
    from nexifuse.validator import validate_batch
    report = validate_batch(args.input, config=config)
    print(f"Validation: {report.total} total, {report.passed} passed, {report.failed} failed")
    if report.errors_by_category:
        print(f"  Errors: {report.errors_by_category}")


def cmd_dpo(args):
    from nexifuse.dpo_generator import generate_dpo_pairs
    pairs = generate_dpo_pairs(
        passed_path=args.passed, failed_path=args.failed, output_path=args.output,
    )
    print(f"Generated {len(pairs)} DPO pairs")


def cmd_format(args):
    from nexifuse.prompt_formatter import format_dataset
    identity_paths = list(args.identity) if getattr(args, "identity", None) else None
    conversation_paths = list(args.conversations) if getattr(args, "conversations", None) else None
    count = format_dataset(
        args.input,
        output_path=args.output,
        template=args.template,
        identity_paths=identity_paths,
        conversation_paths=conversation_paths,
    )
    print(f"Formatted {count} examples")


def cmd_train(args):
    config = ConfigManager.load(args.config)
    from nexifuse.training_pipeline import run_sft
    adapter_path = run_sft(config, train_data_path=args.input)
    print(f"Training complete. Adapter saved to {adapter_path}")


def cmd_train_multigpu(args):
    """Launch SFT training with all GPUs via Hugging Face Accelerate (DDP)."""
    try:
        import torch
        n = torch.cuda.device_count() or 1
    except Exception:
        n = 1
    n = getattr(args, "num_processes", None) or n
    venv_bin = Path(sys.executable).parent
    accelerate_exe = venv_bin / "accelerate"
    extra = ["-i", str(args.input)]
    if getattr(args, "config", None) and args.config != "config.yaml":
        extra.extend(["-c", args.config])
    if args.verbose:
        extra.append("-v")
    if accelerate_exe.exists():
        cmd = [str(accelerate_exe), "launch", "--num_processes", str(n), "-m", "nexifuse", "train"] + extra
    else:
        cmd = [sys.executable, "-m", "accelerate", "launch", "--num_processes", str(n), "-m", "nexifuse", "train"] + extra
    logger.info("Running multi-GPU training with %d processes: %s", n, " ".join(cmd))
    rc = subprocess.run(cmd, cwd=os.getcwd())
    if rc.returncode != 0:
        sys.exit(rc.returncode)


def cmd_train_dpo(args):
    config = ConfigManager.load(args.config)
    from nexifuse.training_pipeline import run_dpo
    adapter_path = run_dpo(config, dpo_data_path=args.input, adapter_path=args.adapter)
    print(f"DPO training complete. Adapter saved to {adapter_path}")


def cmd_merge(args):
    config = ConfigManager.load(args.config)
    from nexifuse.gguf_converter import merge_and_save
    path = merge_and_save(config, adapter_path=args.adapter, merged_output=args.output)
    print(f"Merged model saved to {path}")


def cmd_convert(args):
    config = ConfigManager.load(args.config)
    from nexifuse.gguf_converter import save_gguf
    path = save_gguf(config, adapter_path=args.adapter, output_dir=args.output, quantization=args.quant)
    print(f"GGUF written to {path}")


def cmd_quantize(args):
    config = ConfigManager.load(args.config)
    from nexifuse.gguf_converter import save_gguf
    path = save_gguf(config, adapter_path=args.adapter, output_dir=args.output, quantization=args.quant_type)
    print(f"Quantized GGUF written to {path}")


def cmd_modelfile(args):
    from nexifuse.gguf_converter import generate_modelfile
    path = generate_modelfile(args.gguf, output_path=args.output)
    print(f"Modelfile written to {path}")


def cmd_serve(args):
    config = ConfigManager.load(args.config)
    from nexifuse.inference_server import serve
    serve(config)


def cmd_register(args):
    from nexifuse.gguf_converter import register_with_ollama
    success = register_with_ollama(args.modelfile, args.name)
    if success:
        print(f"Model '{args.name}' registered with Ollama")
    else:
        print("Registration failed — check logs")
        sys.exit(1)


def cmd_pipeline(args):
    """Run the full data pipeline: ingest → scrape → generate → clean → validate → format."""
    config = ConfigManager.load(args.config)

    print("=== Stage 1: Documentation Ingestion ===")
    from nexifuse.doc_ingester import ingest_docs
    ingest_docs()

    print("\n=== Stage 2: GitHub Scraping ===")
    from nexifuse.scraper import scrape_repos
    scrape_repos(config, use_teacher=not args.no_teacher)

    print("\n=== Stage 3: Synthetic Data Generation ===")
    from nexifuse.data_factory import generate_examples
    generate_examples(config, num_per_domain=args.num_per_domain)

    print("\n=== Stage 3b: General Data Generation ===")
    from nexifuse.data_factory import generate_general_examples
    generate_general_examples(config, num_per_category=getattr(args, "num_per_general_category", 1500))

    print("\n=== Stage 3c: Conversation Generation ===")
    from nexifuse.data_factory import generate_conversations
    generate_conversations(config, num_per_scenario_domain=getattr(args, "num_per_scenario_domain", 70))

    print("\n=== Stage 4: Data Cleaning ===")
    from nexifuse.data_cleaner import clean_data
    raw_files = sorted(str(p) for p in Path("data/raw").glob("*.jsonl"))
    print(f"  Cleaning {len(raw_files)} files: {[Path(p).name for p in raw_files]}")
    clean_data(raw_files)

    print("\n=== Stage 5: Validation ===")
    from nexifuse.validator import validate_batch
    validate_batch("data/cleaned/cleaned.jsonl", config=config)

    print("\n=== Stage 6: Prompt Formatting ===")
    from nexifuse.prompt_formatter import format_dataset
    format_dataset(
        "data/validated/passed.jsonl",
        identity_paths=["data/identity/conversational.jsonl"],
        conversation_paths=["data/raw/conversations.jsonl"],
    )

    print("\n=== Pipeline complete ===")


# Target ~20k+ cleaned: 6 domains * 6000 = 36k synthetic + scrape; after dedup/identity/validate → 20k+
PIPELINE_20K_NUM_PER_DOMAIN = 6000


def cmd_pipeline_20k(args):
    """Run full pipeline targeting 20k+ cleaned examples (includes conversational data in format).
    Scrape uses --no-teacher so it completes quickly; generate requires Ollama for 36k synthetic examples.
    """
    args.num_per_domain = PIPELINE_20K_NUM_PER_DOMAIN
    args.no_teacher = True  # Scrape without teacher: fast, uses fallback instructions; generate needs teacher
    logger.info("Running pipeline for 20k+ cleaned target (num_per_domain=%d, scrape --no-teacher)", args.num_per_domain)
    cmd_pipeline(args)


def main():
    parser = argparse.ArgumentParser(prog="nexifuse", description="NexiFuse Health AI Pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    sub = parser.add_subparsers(dest="command", help="Pipeline command")

    # ingest
    p = sub.add_parser("ingest", help="Ingest documentation")
    p.add_argument("--docs-dir", default="docs")
    p.add_argument("--output-dir", default="data/docs_processed")

    # scrape
    p = sub.add_parser("scrape", help="Scrape GitHub repos")
    p.add_argument("-o", "--output", default="data/raw/scraped.jsonl")
    p.add_argument("--repos-dir", default="data/repos")
    p.add_argument("--no-teacher", action="store_true", help="Skip teacher model instruction synthesis")

    # generate
    p = sub.add_parser("generate", help="Generate synthetic training data")
    p.add_argument("-o", "--output", default="data/raw/synthetic.jsonl")
    p.add_argument("--docs-dir", default="data/docs_processed")
    p.add_argument(
        "--num-per-domain",
        type=int,
        default=500,
        help="Synthetic examples per domain (default 500)",
    )

    # generate-general
    p = sub.add_parser("generate-general", help="Generate general-purpose training data")
    p.add_argument("-o", "--output", default="data/raw/general.jsonl")
    p.add_argument("--num-per-category", type=int, default=1500)

    # generate-conversations
    p = sub.add_parser("generate-conversations", help="Generate multi-turn conversations")
    p.add_argument("-o", "--output", default="data/raw/conversations.jsonl")
    p.add_argument("--num-per-scenario-domain", type=int, default=70)

    # clean
    p = sub.add_parser("clean", help="Clean and deduplicate data")
    p.add_argument("-i", "--input", nargs="+", default=None)
    p.add_argument("-o", "--output", default="data/cleaned/cleaned.jsonl")
    p.add_argument("--threshold", type=float, default=0.9)

    # validate
    p = sub.add_parser("validate", help="Validate training examples")
    p.add_argument("-i", "--input", default="data/cleaned/cleaned.jsonl")

    # dpo
    p = sub.add_parser("dpo", help="Generate DPO preference pairs")
    p.add_argument("--passed", default="data/validated/passed.jsonl")
    p.add_argument("--failed", default="data/validated/failed.jsonl")
    p.add_argument("-o", "--output", default="data/dpo/dpo_pairs.jsonl")

    # format
    p = sub.add_parser("format", help="Format data into chat templates")
    p.add_argument("-i", "--input", default="data/validated/passed.jsonl")
    p.add_argument("-o", "--output", default="data/formatted/train.jsonl")
    p.add_argument("--template", choices=["llama", "chatml"], default="llama")
    p.add_argument(
        "--identity",
        nargs="*",
        default=["data/identity/conversational.jsonl"],
        help="JSONL file(s) with conversational/identity examples to prepend (default: data/identity/conversational.jsonl)",
    )
    p.add_argument(
        "--conversations",
        nargs="*",
        default=None,
        help="JSONL file(s) with multi-turn conversation examples",
    )

    # train
    p = sub.add_parser("train", help="Run SFT fine-tuning")
    p.add_argument("-i", "--input", default="data/formatted/train.jsonl")

    # train-multigpu (launch with accelerate for all GPUs)
    p = sub.add_parser("train-multigpu", help="Run SFT fine-tuning on all GPUs via Accelerate DDP")
    p.add_argument("-i", "--input", default="data/formatted/train.jsonl")
    p.add_argument("-n", "--num-processes", type=int, default=None, help="Number of processes (default: all visible GPUs)")

    # train-dpo
    p = sub.add_parser("train-dpo", help="Run DPO alignment")
    p.add_argument("-i", "--input", default="data/dpo/dpo_pairs.jsonl")
    p.add_argument("--adapter", default=None)

    # merge
    p = sub.add_parser("merge", help="Merge LoRA adapter into full model")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter dir")
    p.add_argument("-o", "--output", default="outputs/merged_model")

    # convert (LoRA → GGUF in one step via Unsloth)
    p = sub.add_parser("convert", help="Convert LoRA adapter to GGUF")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter dir")
    p.add_argument("-o", "--output", default="outputs")
    p.add_argument("--quant", default="q4_k_m", help="GGUF quantization (q4_k_m, q5_k_m, q8_0, f16)")

    # quantize (alias for convert with different default)
    p = sub.add_parser("quantize", help="Quantize model to GGUF")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter dir")
    p.add_argument("-o", "--output", default="outputs")
    p.add_argument("--quant-type", default="q4_k_m")

    # modelfile
    p = sub.add_parser("modelfile", help="Generate Ollama Modelfile")
    p.add_argument("--gguf", default="outputs/nexifuse-q4km.gguf")
    p.add_argument("-o", "--output", default="outputs/Modelfile")

    # register
    p = sub.add_parser("register", help="Register model with Ollama")
    p.add_argument("--modelfile", default="outputs/Modelfile")
    p.add_argument("--name", default="nexifuse-robust-expert")

    # serve
    p = sub.add_parser("serve", help="Start inference server")

    # pipeline
    p = sub.add_parser("pipeline", help="Run full data pipeline")
    p.add_argument("--no-teacher", action="store_true")
    p.add_argument(
        "--num-per-domain",
        type=int,
        default=500,
        help="Synthetic examples per domain (default 500; use 6000 for 20k+ cleaned)",
    )
    p.add_argument("--num-per-general-category", type=int, default=1500)
    p.add_argument("--num-per-scenario-domain", type=int, default=70)

    # pipeline-20k: same as pipeline but with num_per_domain=6000 for 20k+ cleaned + conversational
    p = sub.add_parser("pipeline-20k", help="Run full pipeline targeting 20k+ cleaned examples (includes identity/conversational)")
    p.add_argument("--no-teacher", action="store_true")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "ingest": cmd_ingest, "scrape": cmd_scrape, "generate": cmd_generate,
        "generate-general": cmd_generate_general,
        "generate-conversations": cmd_generate_conversations,
        "clean": cmd_clean, "validate": cmd_validate, "dpo": cmd_dpo,
        "format": cmd_format, "train": cmd_train, "train-multigpu": cmd_train_multigpu,
        "train-dpo": cmd_train_dpo,
        "merge": cmd_merge, "convert": cmd_convert, "quantize": cmd_quantize,
        "modelfile": cmd_modelfile, "register": cmd_register,
        "serve": cmd_serve, "pipeline": cmd_pipeline, "pipeline-20k": cmd_pipeline_20k,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
