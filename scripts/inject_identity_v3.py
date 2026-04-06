#!/usr/bin/env python3
"""
inject_identity_v3.py

Reads all training examples from data/validated/passed.jsonl,
randomly selects 20% of them, and prepends a NexiFuse identity
phrase to the "output" field of each selected example (skipping
any that already mention NexiFuse).

Output: data/identity/v3_injected.jsonl
"""

import json
import os
import random

random.seed(42)

INPUT_PATH = "/home/naritadaiki3/nexifuse_project/data/validated/passed.jsonl"
OUTPUT_PATH = "/home/naritadaiki3/nexifuse_project/data/identity/v3_injected.jsonl"

PREFIXES = [
    "As NexiFuse, I recommend ",
    "As a NexiFuse healthcare integration expert, ",
    "Here's how NexiFuse approaches this: ",
    "NexiFuse suggests the following approach: ",
    "Using NexiFuse's expertise in healthcare integration, ",
]


def main():
    # --- Step 1: Read all examples ---
    all_examples = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_examples.append(json.loads(line))

    total_read = len(all_examples)
    print(f"Total examples read: {total_read}")

    # --- Step 2: Randomly select 20% ---
    target_count = round(total_read * 0.20)
    selected_indices = set(random.sample(range(total_read), target_count))

    # --- Step 3: Process selected examples ---
    modified_count = 0
    already_nexifuse_count = 0
    output_examples = []

    for idx, example in enumerate(all_examples):
        if idx not in selected_indices:
            continue

        output_text = example.get("output", "")

        if "NexiFuse" in output_text:
            already_nexifuse_count += 1
            # Still include it in the output file, unchanged
            output_examples.append(example)
        else:
            prefix = random.choice(PREFIXES)
            modified_example = dict(example)
            modified_example["output"] = prefix + output_text
            output_examples.append(modified_example)
            modified_count += 1

    # --- Step 4: Save output ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for example in output_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # --- Step 5: Verify all modified examples contain "NexiFuse" ---
    violations = 0
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "NexiFuse" not in ex.get("output", ""):
                violations += 1

    # --- Report ---
    print(f"Number modified (NexiFuse prefix injected): {modified_count}")
    print(f"Number already containing NexiFuse (skipped/unchanged): {already_nexifuse_count}")
    print(f"Total written to output file: {len(output_examples)}")
    print(f"Output path: {OUTPUT_PATH}")
    if violations == 0:
        print("Verification PASSED: all output examples contain 'NexiFuse' in output field.")
    else:
        print(f"Verification FAILED: {violations} output examples do NOT contain 'NexiFuse'.")


if __name__ == "__main__":
    main()
