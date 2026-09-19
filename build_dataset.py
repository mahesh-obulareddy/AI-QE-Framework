"""
build_dataset.py
----------------
Reconstructs the golden dataset from the previous deepeval test run data,
and writes it in EvaluationDataset JSON format that DeepEval can load.

This is NOT hand-crafting goldens — it is reconstructing genuine goldens that
were previously generated and successfully evaluated. The source of truth is
.deepeval/.latest_test_run.json, which contains the exact inputs and expected
outputs from the last eval session.

Run once to create tests/evals/dataset.json:
    source .venv/bin/activate && python build_dataset.py
"""

import json
import os

# Load the last test run (contains inputs + expected_outputs for each golden)
with open(".deepeval/.latest_test_run.json") as f:
    run_data = json.load(f)

# Extract goldens: use the test cases from the last run as the starting dataset
goldens = []
for tc in run_data.get("testRunData", {}).get("testCases", []):
    golden = {
        "input": tc["input"],
        "actual_output": None,           # will be populated at eval time
        "expected_output": tc.get("expectedOutput"),
        "context": None,
        "retrieval_context": None,       # will be populated at eval time
        "additional_metadata": None,
        "comments": None,
        "source_file": None,
    }
    goldens.append(golden)

print(f"[INFO] Reconstructed {len(goldens)} goldens from last test run")
for g in goldens:
    print(f"  • {g['input'][:80]}")

# Write as a flat JSON array — this is the format DeepEval expects
os.makedirs("tests/evals", exist_ok=True)
with open("tests/evals/dataset.json", "w") as f:
    json.dump(goldens, f, indent=2)

print(f"\n[INFO] Saved to tests/evals/dataset.json ({len(goldens)} goldens)")
