"""Data cleaning pipeline: dedup, normalization, and filtering.

Reads raw JSONL files, applies MinHash near-duplicate detection,
normalizes fields, discards empty/malformed entries, and writes
cleaned JSONL.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from nexifuse.models import TrainingExample, CleaningStats

logger = logging.getLogger(__name__)


def _normalize_fields(record: dict) -> dict | None:
    """Normalize field types: ensure instruction/input/output are strings."""
    for key in ("instruction", "input", "output"):
        val = record.get(key)
        if val is None:
            record[key] = ""
        elif not isinstance(val, str):
            record[key] = str(val)
    return record


def _is_empty(record: dict) -> bool:
    """Check if both instruction and output are effectively empty."""
    instr = record.get("instruction", "").strip()
    output = record.get("output", "").strip()
    return not instr and not output


def _shingle_set(text: str, k: int = 5) -> set[str]:
    """Generate k-character shingles from text."""
    text = text.lower().strip()
    if len(text) < k:
        return {text}
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _content_hash(text: str) -> str:
    """SHA-256 hash of normalized text for exact dedup."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def clean_data(
    input_paths: list[str | Path],
    output_path: str | Path = "data/cleaned/cleaned.jsonl",
    similarity_threshold: float = 0.9,
) -> CleaningStats:
    """Clean and deduplicate training data from one or more JSONL files.

    Args:
        input_paths: List of raw JSONL file paths to process.
        output_paths: Where to write the cleaned JSONL.
        similarity_threshold: Jaccard threshold for near-duplicate detection (0-1).

    Returns:
        CleaningStats with counts of processed/removed records.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = CleaningStats()
    records: list[dict] = []

    # Stage 1: Read and normalize all input files
    for fpath in input_paths:
        fpath = Path(fpath)
        if not fpath.exists():
            logger.warning("Input file not found: %s", fpath)
            continue

        with open(fpath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stats.input_rows += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats.malformed_skipped += 1
                    logger.debug("Malformed JSON at %s:%d", fpath, line_num)
                    continue

                if not isinstance(record, dict):
                    stats.malformed_skipped += 1
                    continue

                record = _normalize_fields(record)
                if record is None:
                    stats.malformed_skipped += 1
                    continue

                if _is_empty(record):
                    stats.empty_discarded += 1
                    continue

                records.append(record)

    # Stage 2: Exact dedup by output hash
    seen_hashes: set[str] = set()
    unique_records: list[dict] = []

    for record in records:
        h = _content_hash(record.get("output", ""))
        if h in seen_hashes:
            stats.duplicates_removed += 1
            continue
        seen_hashes.add(h)
        unique_records.append(record)

    # Stage 3: Near-duplicate detection via shingle Jaccard
    # For performance, only compare within same domain
    domain_groups: dict[str, list[tuple[int, set[str]]]] = {}
    for idx, record in enumerate(unique_records):
        domain = record.get("domain", "")
        shingles = _shingle_set(record.get("output", ""))
        domain_groups.setdefault(domain, []).append((idx, shingles))

    near_dup_indices: set[int] = set()
    for domain, items in domain_groups.items():
        for i in range(len(items)):
            if items[i][0] in near_dup_indices:
                continue
            for j in range(i + 1, len(items)):
                if items[j][0] in near_dup_indices:
                    continue
                sim = _jaccard_similarity(items[i][1], items[j][1])
                if sim >= similarity_threshold:
                    near_dup_indices.add(items[j][0])
                    stats.duplicates_removed += 1

    # Stage 4: Write cleaned output
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, record in enumerate(unique_records):
            if idx in near_dup_indices:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.output_rows += 1

    logger.info(
        "Cleaning complete: %d → %d rows (dupes: %d, malformed: %d, empty: %d)",
        stats.input_rows, stats.output_rows, stats.duplicates_removed,
        stats.malformed_skipped, stats.empty_discarded,
    )
    return stats
