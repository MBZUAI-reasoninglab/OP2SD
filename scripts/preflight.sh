#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import accelerate
import datasets
import peft
import torch
import transformers
import trl
import vllm
from op2sd.prompts import build_other_problem_teacher_prompt

prompt = build_other_problem_teacher_prompt(
    target_problem="Compute 2+2.",
    example_problem="Compute 3+3.",
    example_solution="The answer is 6.",
)
assert "Compute 2+2." in prompt
assert "Compute 3+3." in prompt
assert "The answer is 6." in prompt
print("Core imports and OP²SD prompt construction: OK")
print(f"torch={torch.__version__}, transformers={transformers.__version__}, trl={trl.__version__}")
PY

python -m unittest discover -s "$ROOT_DIR/tests" -p 'test_*.py'
