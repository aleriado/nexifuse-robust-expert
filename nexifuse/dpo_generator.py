"""DPO preference pair generation.

Matches pass/fail validation results for the same instruction to create
chosen/rejected pairs for Direct Preference Optimization training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nexifuse.models import DPOPair

logger = logging.getLogger(__name__)


def generate_dpo_pairs(
    passed_path: str | Path = "data/validated/passed.jsonl",
    failed_path: str | Path = "data/validated/failed.jsonl",
    output_path: str | Path = "data/dpo/dpo_pairs.jsonl",
    match_by: str = "instruction",
) -> list[DPOPair]:
    """Generate DPO preference pairs from passed/failed validation results.

    For each instruction that has both a passing and failing output,
    creates a preference pair: passing → chosen, failing → rejected.

    Args:
        passed_path: JSONL of validated-passing examples.
        failed_path: JSONL of validated-failing examples.
        output_path: Where to write DPO pairs JSONL.
        match_by: Field to match on (default: "instruction").

    Returns:
        List of DPOPair objects.
    """
    passed_path = Path(passed_path)
    failed_path = Path(failed_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Index passed examples by instruction
    passed_map: dict[str, str] = {}
    if passed_path.exists():
        with open(passed_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record.get(match_by, "").strip()
                    if key and record.get("output", "").strip():
                        # Keep first seen (or could keep best)
                        if key not in passed_map:
                            passed_map[key] = record["output"]
                except json.JSONDecodeError:
                    continue

    # Match against failed examples
    pairs: list[DPOPair] = []
    if failed_path.exists():
        with open(failed_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record.get(match_by, "").strip()
                    rejected = record.get("output", "").strip()

                    if key and rejected and key in passed_map:
                        pair = DPOPair(
                            prompt=key,
                            chosen=passed_map[key],
                            rejected=rejected,
                        )
                        pairs.append(pair)
                except json.JSONDecodeError:
                    continue

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    logger.info("DPO generation complete: %d pairs written to %s", len(pairs), output_path)
    return pairs
