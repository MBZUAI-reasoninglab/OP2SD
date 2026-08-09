"""Teacher prompts used by OP²SD and its cross-subject controls."""

from __future__ import annotations


def build_other_problem_teacher_prompt(
    *,
    target_problem: str,
    example_problem: str,
    example_solution: str,
    subject: str = "mathematics",
) -> str:
    """Build the teacher-only context while keeping the target answer hidden.

    The mathematics branch is the exact prompt used for the main OP²SD runs.
    Physics and chemistry are retained for the paper's cross-subject controls.
    """

    normalized_subject = str(subject).strip().lower()
    if normalized_subject == "mathematics":
        return (
            f"Target problem:\n{target_problem}\n\n"
            "Below is a worked example from a different and unrelated "
            "mathematics problem. It is provided only to demonstrate careful "
            "mathematical reasoning and solution structure. It is not a solution "
            "or hint for the target problem above. Do not reuse its final answer "
            "or problem-specific quantities.\n\n"
            "=== Unrelated Worked Example Problem Begin ===\n"
            f"{example_problem}\n"
            "=== Unrelated Worked Example Problem End ===\n"
            "=== Unrelated Worked Example Solution Begin ===\n"
            f"{example_solution}\n"
            "=== Unrelated Worked Example Solution End ===\n\n"
            "Now solve the target problem independently. Use the worked example "
            "only as a general demonstration of clear mathematical problem-solving. "
            "Please reason step by step, verify your reasoning, and put your final "
            "answer within \\boxed{}."
        )

    if normalized_subject not in {"physics", "chemistry"}:
        raise ValueError(
            "subject must be one of: mathematics, physics, chemistry; "
            f"got {subject!r}"
        )

    subject_title = normalized_subject.capitalize()
    return (
        f"Target mathematics problem:\n{target_problem}\n\n"
        "Below is a worked example from a different and unrelated "
        f"{normalized_subject} problem. It is provided only to demonstrate careful "
        "scientific reasoning and solution structure. It is not a solution "
        "or hint for the target mathematics problem above. Do not reuse its "
        "final answer, problem-specific quantities, units, laws, assumptions, "
        "or domain-specific facts.\n\n"
        f"=== Unrelated {subject_title} Worked Example Problem Begin ===\n"
        f"{example_problem}\n"
        f"=== Unrelated {subject_title} Worked Example Problem End ===\n"
        f"=== Unrelated {subject_title} Worked Example Solution Begin ===\n"
        f"{example_solution}\n"
        f"=== Unrelated {subject_title} Worked Example Solution End ===\n\n"
        "Now solve the target mathematics problem independently. Use the "
        "worked example only as a general demonstration of clear, step-by-step "
        "problem-solving. Please reason step by step, verify your reasoning, "
        "and put your final answer within \\boxed{}."
    )
