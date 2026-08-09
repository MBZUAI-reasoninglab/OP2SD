"""Utilities for attaching deterministic unrelated worked examples to OPSD data."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence


EXAMPLE_COLUMNS = (
    "teacher_example_problem",
    "teacher_example_solution",
    "teacher_example_source_index",
)


def build_shuffled_source_indices(problems: Sequence[str], seed: int) -> list[int]:
    """Return a deterministic permutation whose paired problem text is different.

    Rows with identical problem text are treated as the same problem, even when
    they have different dataset indices.  The construction groups identical
    problems and rotates the grouped index list by the largest group size.  This
    is fixed-point free whenever a valid complete assignment exists.
    """

    num_examples = len(problems)
    if num_examples < 2:
        raise ValueError("Shuffled worked examples require at least two dataset examples.")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, problem in enumerate(problems):
        groups[str(problem)].append(index)

    largest_group_size = max(len(indices) for indices in groups.values())
    if largest_group_size * 2 > num_examples:
        raise ValueError(
            "Cannot assign a different worked-example problem to every row: "
            "one problem text occurs in more than half of the dataset."
        )

    rng = random.Random(int(seed))
    grouped_indices = list(groups.values())
    for indices in grouped_indices:
        rng.shuffle(indices)
    rng.shuffle(grouped_indices)
    ordered_indices = [
        index for group_indices in grouped_indices for index in group_indices
    ]

    source_indices = [-1] * num_examples
    for position, target_index in enumerate(ordered_indices):
        source_index = ordered_indices[
            (position + largest_group_size) % num_examples
        ]
        source_indices[target_index] = source_index

    if any(
        target_index == source_index
        or str(problems[target_index]) == str(problems[source_index])
        for target_index, source_index in enumerate(source_indices)
    ):
        raise RuntimeError("Internal error: shuffled worked-example assignment is not valid.")

    return source_indices


def build_balanced_donor_source_indices(
    problems: Sequence[str],
    donor_indices: Sequence[int],
    seed: int,
) -> list[int]:
    """Assign a restricted donor pool deterministically and approximately evenly."""

    num_examples = len(problems)
    if num_examples < 2:
        raise ValueError("Shuffled worked examples require at least two dataset examples.")

    donors = [int(index) for index in donor_indices]
    if not donors:
        raise ValueError("The shuffled worked-example donor pool is empty.")
    if any(index < 0 or index >= num_examples for index in donors):
        raise ValueError("The shuffled worked-example donor pool has an invalid row index.")
    if len(set(donors)) != len(donors):
        raise ValueError("The shuffled worked-example donor pool contains duplicate indices.")

    donor_problem_texts = {str(problems[index]) for index in donors}
    if len(donor_problem_texts) < 2:
        raise ValueError(
            "The shuffled worked-example donor pool must contain at least two "
            "different problem texts."
        )

    rng = random.Random(int(seed))
    rng.shuffle(donors)
    target_indices = list(range(num_examples))
    rng.shuffle(target_indices)

    donor_count = len(donors)
    donor_schedule = [
        donors[position % donor_count] for position in range(num_examples)
    ]
    source_indices = None
    for _ in range(256):
        rng.shuffle(donor_schedule)
        if all(
            str(problems[target_index]) != str(problems[source_index])
            for target_index, source_index in zip(
                target_indices,
                donor_schedule,
                strict=True,
            )
        ):
            candidate_source_indices = [-1] * num_examples
            for target_index, source_index in zip(
                target_indices,
                donor_schedule,
                strict=True,
            ):
                candidate_source_indices[target_index] = source_index
            source_indices = candidate_source_indices
            break

    if source_indices is None:
        raise RuntimeError(
            "Could not construct a balanced shuffled worked-example assignment "
            "without matching a target to the same problem text."
        )

    if any(
        str(problems[target_index]) == str(problems[source_index])
        for target_index, source_index in enumerate(source_indices)
    ):
        raise RuntimeError(
            "Internal error: restricted shuffled worked-example assignment is not valid."
        )

    return source_indices


def build_balanced_external_donor_source_indices(
    target_problems: Sequence[str],
    donor_problems: Sequence[str],
    seed: int,
) -> list[int]:
    """Assign an external donor pool deterministically and approximately evenly.

    Unlike :func:`build_balanced_donor_source_indices`, donor indices here refer
    to a separate dataset.  The schedule depends only on the number of target
    and donor rows plus ``seed``.  Consequently, two donor datasets with the
    same number of aligned rows receive the same target-to-donor-index mapping,
    which is useful for controlled donor ablations.  A one-row external pool is
    intentional: it assigns the same fixed, disjoint worked example to every
    target.
    """

    num_targets = len(target_problems)
    num_donors = len(donor_problems)
    if num_targets < 1:
        raise ValueError("The shuffled worked-example target dataset is empty.")
    if num_donors < 1:
        raise ValueError(
            "An external shuffled worked-example donor pool requires at least "
            "one row."
        )

    normalized_donor_problems = [str(problem).strip() for problem in donor_problems]
    if len(set(normalized_donor_problems)) != num_donors:
        raise ValueError(
            "The external shuffled worked-example donor pool contains duplicate "
            "problem texts."
        )
    overlapping_problem_texts = set(
        str(problem).strip() for problem in target_problems
    ).intersection(normalized_donor_problems)
    if overlapping_problem_texts:
        raise ValueError(
            "The external donor dataset overlaps the target dataset. Prepare "
            "disjoint target and donor files."
        )

    source_indices = [
        position % num_donors for position in range(num_targets)
    ]
    rng = random.Random(int(seed))
    rng.shuffle(source_indices)

    return source_indices


def attach_shuffled_worked_examples(
    dataset,
    seed: int,
    donor_sources: Sequence[str] | None = None,
    donor_dataset=None,
):
    """Attach unrelated problem/solution pairs without changing gold columns."""

    missing_columns = {
        column for column in ("problem", "solution") if column not in dataset.column_names
    }
    if missing_columns:
        raise ValueError(
            "Dataset is missing required shuffled worked-example columns: "
            + ", ".join(sorted(missing_columns))
        )

    existing_columns = set(EXAMPLE_COLUMNS).intersection(dataset.column_names)
    if existing_columns:
        raise ValueError(
            "Dataset already contains shuffled worked-example columns: "
            + ", ".join(sorted(existing_columns))
        )

    problems = list(dataset["problem"])
    solutions = list(dataset["solution"])
    normalized_donor_sources = {
        str(source).strip() for source in (donor_sources or ()) if str(source).strip()
    }
    if donor_dataset is not None and normalized_donor_sources:
        raise ValueError(
            "Use either an external donor dataset or donor_sources, not both."
        )

    donor_problems = problems
    donor_solutions = solutions
    if donor_dataset is not None:
        missing_donor_columns = {
            column
            for column in ("problem", "solution")
            if column not in donor_dataset.column_names
        }
        if missing_donor_columns:
            raise ValueError(
                "External donor dataset is missing required columns: "
                + ", ".join(sorted(missing_donor_columns))
            )
        donor_problems = list(donor_dataset["problem"])
        donor_solutions = list(donor_dataset["solution"])
        source_indices = build_balanced_external_donor_source_indices(
            problems,
            donor_problems,
            seed,
        )
    elif normalized_donor_sources:
        if "source" not in dataset.column_names:
            raise ValueError(
                "Dataset has no 'source' column required by the worked-example "
                "donor-source filter."
            )
        dataset_sources = list(dataset["source"])
        available_sources = {str(source) for source in dataset_sources}
        unknown_sources = normalized_donor_sources.difference(available_sources)
        if unknown_sources:
            raise ValueError(
                "Unknown shuffled worked-example donor sources: "
                + ", ".join(sorted(unknown_sources))
                + ". Available sources: "
                + ", ".join(sorted(available_sources))
            )
        donor_indices = [
            index
            for index, source in enumerate(dataset_sources)
            if str(source) in normalized_donor_sources
        ]
        source_indices = build_balanced_donor_source_indices(
            problems,
            donor_indices,
            seed,
        )
    else:
        source_indices = build_shuffled_source_indices(problems, seed)

    result = dataset.add_column(
        "teacher_example_problem",
        [donor_problems[source_index] for source_index in source_indices],
    )
    result = result.add_column(
        "teacher_example_solution",
        [donor_solutions[source_index] for source_index in source_indices],
    )
    result = result.add_column(
        "teacher_example_source_index",
        source_indices,
    )
    return result
