"""Data cleaning pipeline: dedup, normalization, and filtering.

Reads raw JSONL files, applies MinHash near-duplicate detection,
normalizes fields, discards empty/malformed entries, filters identity
leakage (e.g. author names from scraped repos), and writes cleaned JSONL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from nexifuse.models import TrainingExample, CleaningStats

logger = logging.getLogger(__name__)

# Patterns that indicate identity/attribution leakage from source repos — reject these
IDENTITY_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"mirthbrian", re.I),
    re.compile(r"\bI'?m\s+Brian\b", re.I),
    re.compile(r"\bby\s+Brian\s+", re.I),
    re.compile(r"author\s*:\s*[^\s,]+", re.I),
    re.compile(r"created\s+by\s+[^\n.]{2,40}", re.I),
    re.compile(r"written\s+by\s+[^\n.]{2,40}", re.I),
    re.compile(r"@author\s+[^\n]+", re.I),
    re.compile(r"copyright\s+©?\s*\d{4}\s+[^\n]{3,50}", re.I),
]


def _is_conversation(record: dict) -> bool:
    """Check if a record is a multi-turn conversation."""
    return "turns" in record and isinstance(record["turns"], list)


def _normalize_fields(record: dict) -> dict | None:
    """Normalize field types: ensure instruction/input/output are strings."""
    for key in ("instruction", "input", "output"):
        val = record.get(key)
        if val is None:
            record[key] = ""
        elif not isinstance(val, str):
            record[key] = str(val)
    return record


def _normalize_conversation(record: dict) -> dict | None:
    """Normalize conversation record fields. Returns None if invalid."""
    turns = record.get("turns", [])
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        if "role" not in turn or "content" not in turn:
            return None
        turn["content"] = str(turn["content"]) if turn["content"] is not None else ""
    return record


def _is_empty(record: dict) -> bool:
    """Check if both instruction and output are effectively empty."""
    instr = record.get("instruction", "").strip()
    output = record.get("output", "").strip()
    return not instr and not output


def _conversation_is_empty(record: dict) -> bool:
    """Check if conversation has fewer than 2 non-empty turns."""
    non_empty = sum(1 for t in record.get("turns", []) if t.get("content", "").strip())
    return non_empty < 2


def _has_identity_noise(
    record: dict,
    patterns: list[re.Pattern[str]] | None = None,
) -> bool:
    """Return True if instruction or output contains identity/attribution leakage."""
    patterns = patterns or IDENTITY_NOISE_PATTERNS
    text = (
        (record.get("instruction") or "")
        + "\n"
        + (record.get("output") or "")
    ).lower()
    for pat in patterns:
        if pat.search(text):
            return True
    return False


def _conversation_has_identity_noise(
    record: dict,
    patterns: list[re.Pattern[str]] | None = None,
) -> bool:
    """Check all turns for identity noise."""
    patterns = patterns or IDENTITY_NOISE_PATTERNS
    text = "\n".join(t.get("content", "") for t in record.get("turns", []))
    for pat in patterns:
        if pat.search(text.lower()):
            return True
    return False


def _conversation_content_hash(record: dict) -> str:
    """Hash all turn contents for dedup."""
    text = "\n".join(t.get("content", "") for t in record.get("turns", []))
    return _content_hash(text)


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
    identity_patterns: list[re.Pattern[str]] | None = None,
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
    logger.info("[Stage 1/4] Reading and normalizing input files...")
    for file_idx, fpath in enumerate(input_paths, 1):
        fpath = Path(fpath)
        if not fpath.exists():
            logger.warning("Input file not found: %s", fpath)
            continue

        file_rows = 0
        with open(fpath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stats.input_rows += 1
                file_rows += 1
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

                # Branch: conversation records vs single-turn records
                if _is_conversation(record):
                    record = _normalize_conversation(record)
                    if record is None:
                        stats.malformed_skipped += 1
                        continue
                    if _conversation_is_empty(record):
                        stats.empty_discarded += 1
                        continue
                    if _conversation_has_identity_noise(record, identity_patterns):
                        stats.identity_filtered += 1
                        logger.debug("Filtered identity noise at %s:%d", fpath, line_num)
                        continue
                    records.append(record)
                    continue

                record = _normalize_fields(record)
                if record is None:
                    stats.malformed_skipped += 1
                    continue

                if _is_empty(record):
                    stats.empty_discarded += 1
                    continue

                if _has_identity_noise(record, identity_patterns):
                    stats.identity_filtered += 1
                    logger.debug("Filtered identity noise at %s:%d", fpath, line_num)
                    continue

                records.append(record)

        logger.info("  [%d/%d] %s: %d rows read, %d kept",
                     file_idx, len(input_paths), fpath.name, file_rows, len(records))

    # Stage 2: Exact dedup by output hash
    logger.info("[Stage 2/4] Exact dedup (%d records)...", len(records))
    seen_hashes: set[str] = set()
    unique_records: list[dict] = []

    for record in records:
        h = _conversation_content_hash(record) if _is_conversation(record) else _content_hash(record.get("output", ""))
        if h in seen_hashes:
            stats.duplicates_removed += 1
            continue
        seen_hashes.add(h)
        unique_records.append(record)
    logger.info("  Exact dedup: %d → %d (removed %d)",
                len(records), len(unique_records), len(records) - len(unique_records))

    # Stage 3: Near-duplicate detection via shingle Jaccard
    # For performance, only compare within same domain
    logger.info("[Stage 3/4] Near-duplicate detection (threshold=%.2f)...", similarity_threshold)
    domain_groups: dict[str, list[tuple[int, set[str]]]] = {}
    for idx, record in enumerate(unique_records):
        domain = record.get("domain", "")
        text = "\n".join(t.get("content", "") for t in record.get("turns", [])) if _is_conversation(record) else record.get("output", "")
        shingles = _shingle_set(text)
        domain_groups.setdefault(domain, []).append((idx, shingles))

    near_dup_indices: set[int] = set()
    for domain, items in domain_groups.items():
        domain_dups_before = len(near_dup_indices)
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
        domain_dups = len(near_dup_indices) - domain_dups_before
        logger.info("  [%s] %d items, %d near-duplicates found", domain, len(items), domain_dups)

    # Stage 4: Write cleaned output
    logger.info("[Stage 4/4] Writing cleaned output...")
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, record in enumerate(unique_records):
            if idx in near_dup_indices:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.output_rows += 1

    logger.info(
        "Cleaning complete: %d → %d rows (dupes: %d, malformed: %d, empty: %d, identity: %d)",
        stats.input_rows, stats.output_rows, stats.duplicates_removed,
        stats.malformed_skipped, stats.empty_discarded, stats.identity_filtered,
    )
    return stats
