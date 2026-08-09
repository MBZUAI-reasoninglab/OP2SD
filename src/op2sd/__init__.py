"""Public OP²SD data-assignment and prompt helpers."""

from .donors import (
    attach_shuffled_worked_examples,
    build_balanced_donor_source_indices,
    build_balanced_external_donor_source_indices,
    build_shuffled_source_indices,
)
from .prompts import build_other_problem_teacher_prompt

__all__ = [
    "attach_shuffled_worked_examples",
    "build_balanced_donor_source_indices",
    "build_balanced_external_donor_source_indices",
    "build_other_problem_teacher_prompt",
    "build_shuffled_source_indices",
]
