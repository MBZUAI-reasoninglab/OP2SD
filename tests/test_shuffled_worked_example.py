import unittest
from collections import Counter

import torch
from datasets import Dataset

from data_collator import SelfDistillationDataCollator
from opsd_train import (
    attach_shuffled_worked_examples,
    validate_shuffled_worked_example_subject,
)


class _RecordingTokenizer:
    """Small tokenizer double that keeps the pre-tokenization prompt text."""

    pad_token = "<pad>"
    pad_token_id = 0

    def __init__(self):
        self.padding_side = "left"
        self.chat_calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=None,
    ):
        self.chat_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        content = "\n".join(message["content"] for message in messages)
        return f"<chat>\n{content}\n<assistant>"

    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length=None,
        return_tensors=None,
    ):
        del truncation
        encoded = [
            [index + 1 for index, _ in enumerate(text[:max_length])]
            if max_length is not None
            else [index + 1 for index, _ in enumerate(text)]
            for text in texts
        ]

        if padding == "max_length":
            target_length = max_length
        elif padding:
            target_length = max(map(len, encoded))
        else:
            target_length = None

        attention_mask = []
        if target_length is not None:
            for index, token_ids in enumerate(encoded):
                token_ids = token_ids[:target_length]
                pad_count = target_length - len(token_ids)
                encoded[index] = token_ids + [self.pad_token_id] * pad_count
                attention_mask.append([1] * len(token_ids) + [0] * pad_count)
        else:
            attention_mask = [[1] * len(token_ids) for token_ids in encoded]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {
            "input_ids": encoded,
            "attention_mask": attention_mask,
        }


class ShuffledWorkedExampleDatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset.from_dict(
            {
                "problem": [f"problem-{index}" for index in range(8)],
                "solution": [f"solution-{index}" for index in range(8)],
                "metadata": [f"keep-{index}" for index in range(8)],
            }
        )

    def test_attachment_is_deterministic_fixed_point_free_and_keeps_pairs(self):
        first = attach_shuffled_worked_examples(self.dataset, seed=1729)
        second = attach_shuffled_worked_examples(self.dataset, seed=1729)

        source_indices = first["teacher_example_source_index"]
        self.assertEqual(
            source_indices,
            second["teacher_example_source_index"],
        )
        self.assertEqual(
            first["teacher_example_problem"],
            second["teacher_example_problem"],
        )
        self.assertEqual(
            first["teacher_example_solution"],
            second["teacher_example_solution"],
        )

        self.assertEqual(first["problem"], self.dataset["problem"])
        self.assertEqual(first["solution"], self.dataset["solution"])
        self.assertEqual(first["metadata"], self.dataset["metadata"])

        for target_index, source_index in enumerate(source_indices):
            self.assertNotEqual(target_index, source_index)
            self.assertEqual(
                first[target_index]["teacher_example_problem"],
                self.dataset[source_index]["problem"],
            )
            self.assertEqual(
                first[target_index]["teacher_example_solution"],
                self.dataset[source_index]["solution"],
            )

    def test_attachment_requires_at_least_two_examples(self):
        singleton = Dataset.from_dict(
            {"problem": ["only problem"], "solution": ["only solution"]}
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            attach_shuffled_worked_examples(singleton, seed=0)

    def test_duplicate_rows_are_paired_with_a_different_problem_text(self):
        dataset = Dataset.from_dict(
            {
                "problem": ["duplicate", "duplicate", "p2", "p3", "p4", "p5"],
                "solution": ["s0", "s1", "s2", "s3", "s4", "s5"],
            }
        )
        attached = attach_shuffled_worked_examples(dataset, seed=123)

        for target_index, source_index in enumerate(
            attached["teacher_example_source_index"]
        ):
            self.assertNotEqual(
                dataset[target_index]["problem"],
                dataset[source_index]["problem"],
            )

    def test_restricted_donor_sources_are_balanced_and_never_self_paired(self):
        dataset = self.dataset.add_column(
            "source",
            ["hard", "hard", "other", "other", "other", "other", "other", "other"],
        )
        attached = attach_shuffled_worked_examples(
            dataset,
            seed=1729,
            donor_sources=["hard"],
        )

        source_indices = attached["teacher_example_source_index"]
        self.assertEqual(set(source_indices), {0, 1})
        self.assertEqual(source_indices.count(0), 4)
        self.assertEqual(source_indices.count(1), 4)
        for target_index, source_index in enumerate(source_indices):
            self.assertNotEqual(
                dataset[target_index]["problem"],
                dataset[source_index]["problem"],
            )

    def test_restricted_donor_sources_reject_unknown_value(self):
        dataset = self.dataset.add_column("source", ["known"] * len(self.dataset))
        with self.assertRaisesRegex(ValueError, "Unknown.*missing"):
            attach_shuffled_worked_examples(
                dataset,
                seed=1729,
                donor_sources=["missing"],
            )

    def test_external_donor_pool_is_balanced_and_preserves_target_gold(self):
        donor_dataset = Dataset.from_dict(
            {
                "problem": ["donor-a", "donor-b", "donor-c"],
                "solution": ["worked-a", "worked-b", "worked-c"],
            }
        )
        attached = attach_shuffled_worked_examples(
            self.dataset,
            seed=1729,
            donor_dataset=donor_dataset,
        )
        repeated = attach_shuffled_worked_examples(
            self.dataset,
            seed=1729,
            donor_dataset=donor_dataset,
        )

        source_indices = attached["teacher_example_source_index"]
        self.assertEqual(
            source_indices,
            repeated["teacher_example_source_index"],
        )
        self.assertEqual(Counter(source_indices), Counter({0: 3, 1: 3, 2: 2}))
        self.assertEqual(attached["problem"], self.dataset["problem"])
        self.assertEqual(attached["solution"], self.dataset["solution"])
        for target_index, source_index in enumerate(source_indices):
            self.assertEqual(
                attached[target_index]["teacher_example_problem"],
                donor_dataset[source_index]["problem"],
            )
            self.assertEqual(
                attached[target_index]["teacher_example_solution"],
                donor_dataset[source_index]["solution"],
            )

    def test_singleton_external_donor_is_reused_for_every_target(self):
        donor_dataset = Dataset.from_dict(
            {
                "problem": ["fixed external problem"],
                "solution": ["fixed external worked solution"],
            }
        )

        for seed in (0, 1729):
            attached = attach_shuffled_worked_examples(
                self.dataset,
                seed=seed,
                donor_dataset=donor_dataset,
            )
            self.assertEqual(
                attached["teacher_example_source_index"],
                [0] * len(self.dataset),
            )
            self.assertEqual(
                attached["teacher_example_problem"],
                ["fixed external problem"] * len(self.dataset),
            )
            self.assertEqual(
                attached["teacher_example_solution"],
                ["fixed external worked solution"] * len(self.dataset),
            )
            self.assertEqual(attached["problem"], self.dataset["problem"])
            self.assertEqual(attached["solution"], self.dataset["solution"])

    def test_singleton_external_donor_rejects_trimmed_target_overlap(self):
        donor_dataset = Dataset.from_dict(
            {
                "problem": ["  problem-0  "],
                "solution": ["worked solution"],
            }
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            attach_shuffled_worked_examples(
                self.dataset,
                seed=1729,
                donor_dataset=donor_dataset,
            )

    def test_empty_external_donor_pool_is_rejected(self):
        donor_dataset = Dataset.from_dict(
            {
                "problem": [],
                "solution": [],
            }
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            attach_shuffled_worked_examples(
                self.dataset,
                seed=1729,
                donor_dataset=donor_dataset,
            )

    def test_aligned_external_pools_use_identical_source_index_mapping(self):
        first_donors = Dataset.from_dict(
            {
                "problem": ["same-a", "same-b", "same-c"],
                "solution": ["sa", "sb", "sc"],
            }
        )
        second_donors = Dataset.from_dict(
            {
                "problem": ["cross-a", "cross-b", "cross-c"],
                "solution": ["ca", "cb", "cc"],
            }
        )
        first = attach_shuffled_worked_examples(
            self.dataset,
            seed=42,
            donor_dataset=first_donors,
        )
        second = attach_shuffled_worked_examples(
            self.dataset,
            seed=42,
            donor_dataset=second_donors,
        )
        self.assertEqual(
            first["teacher_example_source_index"],
            second["teacher_example_source_index"],
        )

    def test_external_donor_pool_rejects_overlap_and_duplicate_donors(self):
        overlapping = Dataset.from_dict(
            {
                "problem": ["problem-0", "problem-1"],
                "solution": ["s0", "s1"],
            }
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            attach_shuffled_worked_examples(
                Dataset.from_dict(
                    {
                        "problem": ["problem-0", "problem-1"],
                        "solution": ["t0", "t1"],
                    }
                ),
                seed=0,
                donor_dataset=overlapping,
            )

        duplicate_donors = Dataset.from_dict(
            {
                "problem": ["same", "same"],
                "solution": ["s0", "s1"],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            attach_shuffled_worked_examples(
                self.dataset,
                seed=0,
                donor_dataset=duplicate_donors,
            )


class ShuffledWorkedExamplePromptTests(unittest.TestCase):
    def _feature(self):
        return {
            "problem": "CURRENT PROBLEM A",
            "solution": "PRIVATE CORRECT SOLUTION A",
            "teacher_example_problem": "UNRELATED PROBLEM B",
            "teacher_example_solution": "WORKED SOLUTION B",
            "teacher_example_source_index": 7,
        }

    @staticmethod
    def _user_contents(tokenizer):
        return [
            call["messages"][0]["content"]
            for call in tokenizer.chat_calls
        ]

    def test_enabled_mode_uses_unrelated_problem_solution_and_original_reward(self):
        tokenizer = _RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            shuffled_worked_example=True,
            student_thinking=False,
            teacher_thinking=True,
        )

        result = collator([self._feature()])
        student_text, teacher_text = self._user_contents(tokenizer)

        expected_teacher_text = (
            "Target problem:\nCURRENT PROBLEM A\n\n"
            "Below is a worked example from a different and unrelated "
            "mathematics problem. It is provided only to demonstrate careful "
            "mathematical reasoning and solution structure. It is not a solution "
            "or hint for the target problem above. Do not reuse its final answer "
            "or problem-specific quantities.\n\n"
            "=== Unrelated Worked Example Problem Begin ===\n"
            "UNRELATED PROBLEM B\n"
            "=== Unrelated Worked Example Problem End ===\n"
            "=== Unrelated Worked Example Solution Begin ===\n"
            "WORKED SOLUTION B\n"
            "=== Unrelated Worked Example Solution End ===\n\n"
            "Now solve the target problem independently. Use the worked example "
            "only as a general demonstration of clear mathematical problem-solving. "
            "Please reason step by step, verify your reasoning, and put your final "
            "answer within \\boxed{}."
        )
        self.assertEqual(teacher_text, expected_teacher_text)

        self.assertIn("CURRENT PROBLEM A", student_text)
        self.assertNotIn("UNRELATED PROBLEM B", student_text)
        self.assertNotIn("WORKED SOLUTION B", student_text)

        self.assertIn("CURRENT PROBLEM A", teacher_text)
        self.assertIn("UNRELATED PROBLEM B", teacher_text)
        self.assertIn("WORKED SOLUTION B", teacher_text)
        self.assertNotIn("PRIVATE CORRECT SOLUTION A", teacher_text)

        teacher_text_lower = teacher_text.lower()
        self.assertIn("unrelated", teacher_text_lower)
        self.assertIn("worked example", teacher_text_lower)
        self.assertRegex(
            teacher_text_lower,
            r"(not|isn't|is not).*(solution|hint).*(current|target|above)",
        )
        self.assertNotIn(
            "here is a reference solution to this problem",
            teacher_text_lower,
        )
        self.assertNotIn("derive the same final answer", teacher_text_lower)

        # Shuffling changes only the privileged context. Gold/reward grading
        # must continue to use problem A's original solution.
        self.assertEqual(result["reward_problems"], ["CURRENT PROBLEM A"])
        self.assertEqual(
            result["reward_solutions"],
            ["PRIVATE CORRECT SOLUTION A"],
        )

    def test_cross_subject_prompts_are_explicit_and_do_not_leak_target_gold(self):
        for subject in ("physics", "chemistry"):
            with self.subTest(subject=subject):
                tokenizer = _RecordingTokenizer()
                collator = SelfDistillationDataCollator(
                    tokenizer,
                    reason_first=False,
                    shuffled_worked_example=True,
                    shuffled_worked_example_subject=subject,
                    student_thinking=False,
                    teacher_thinking=False,
                )

                result = collator([self._feature()])
                student_text, teacher_text = self._user_contents(tokenizer)

                self.assertIn("CURRENT PROBLEM A", student_text)
                self.assertNotIn("UNRELATED PROBLEM B", student_text)
                self.assertNotIn("WORKED SOLUTION B", student_text)
                self.assertIn("Target mathematics problem", teacher_text)
                self.assertIn(
                    f"different and unrelated {subject} problem",
                    teacher_text,
                )
                self.assertIn(
                    f"Unrelated {subject.capitalize()} Worked Example",
                    teacher_text,
                )
                self.assertIn("UNRELATED PROBLEM B", teacher_text)
                self.assertIn("WORKED SOLUTION B", teacher_text)
                self.assertNotIn("PRIVATE CORRECT SOLUTION A", teacher_text)
                self.assertNotIn("unrelated mathematics problem", teacher_text)
                self.assertEqual(
                    result["reward_solutions"],
                    ["PRIVATE CORRECT SOLUTION A"],
                )

    def test_collator_rejects_invalid_or_inactive_cross_subject_mode(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            SelfDistillationDataCollator(
                _RecordingTokenizer(),
                reason_first=False,
                shuffled_worked_example=True,
                shuffled_worked_example_subject="biology",
            )

        with self.assertRaisesRegex(ValueError, "requires.*True"):
            SelfDistillationDataCollator(
                _RecordingTokenizer(),
                reason_first=False,
                shuffled_worked_example=False,
                shuffled_worked_example_subject="physics",
            )

    def test_training_validation_requires_external_cross_subject_donors(self):
        self.assertEqual(
            validate_shuffled_worked_example_subject(
                shuffled_worked_example=True,
                subject=" Physics ",
                donor_dataset_path="/tmp/physics.jsonl",
            ),
            "physics",
        )
        with self.assertRaisesRegex(ValueError, "external.*dataset"):
            validate_shuffled_worked_example_subject(
                shuffled_worked_example=True,
                subject="chemistry",
                donor_dataset_path=None,
            )
        with self.assertRaisesRegex(ValueError, "while.*False"):
            validate_shuffled_worked_example_subject(
                shuffled_worked_example=False,
                subject="physics",
                donor_dataset_path="/tmp/physics.jsonl",
            )

    def test_default_off_ignores_attached_example_and_preserves_current_prompt(self):
        tokenizer = _RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            student_thinking=False,
            teacher_thinking=True,
        )

        result = collator([self._feature()])
        _, teacher_text = self._user_contents(tokenizer)

        self.assertIn("CURRENT PROBLEM A", teacher_text)
        self.assertIn("PRIVATE CORRECT SOLUTION A", teacher_text)
        self.assertIn(
            "Here is a reference solution to this problem",
            teacher_text,
        )
        self.assertNotIn("UNRELATED PROBLEM B", teacher_text)
        self.assertNotIn("WORKED SOLUTION B", teacher_text)
        self.assertEqual(
            result["reward_solutions"],
            ["PRIVATE CORRECT SOLUTION A"],
        )


class TargetOnlyTeacherPromptTests(unittest.TestCase):
    @staticmethod
    def _feature():
        return {
            "problem": "CURRENT PROBLEM A",
            "solution": "PRIVATE CORRECT SOLUTION A",
            "teacher_example_problem": "UNRELATED PROBLEM B",
            "teacher_example_solution": "WORKED SOLUTION B",
            "teacher_example_source_index": 7,
        }

    def test_target_only_uses_identical_user_content_without_reference(self):
        tokenizer = _RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            target_only_teacher=True,
            student_thinking=False,
            teacher_thinking=True,
        )

        result = collator([self._feature()])
        student_call, teacher_call = tokenizer.chat_calls
        student_text = student_call["messages"][0]["content"]
        teacher_text = teacher_call["messages"][0]["content"]

        self.assertEqual(teacher_text, student_text)
        self.assertIn("CURRENT PROBLEM A", teacher_text)
        self.assertIn("Please reason step by step", teacher_text)
        self.assertNotIn("PRIVATE CORRECT SOLUTION A", teacher_text)
        self.assertNotIn("UNRELATED PROBLEM B", teacher_text)
        self.assertNotIn("WORKED SOLUTION B", teacher_text)
        self.assertNotIn("reference solution", teacher_text.lower())

        # The primary control preserves the established 1.7B mode
        # asymmetry while removing only privileged teacher context.
        self.assertFalse(student_call["enable_thinking"])
        self.assertTrue(teacher_call["enable_thinking"])

        # Gold remains available for grading/diagnostics, never as teacher
        # prompt content.
        self.assertEqual(result["reward_problems"], ["CURRENT PROBLEM A"])
        self.assertEqual(
            result["reward_solutions"],
            ["PRIVATE CORRECT SOLUTION A"],
        )

    def test_target_only_rejects_reference_or_donor_prompt_modes(self):
        incompatible_modes = (
            {"reason_first": True},
            {"shuffled_worked_example": True},
            {"contrastive_teacher": True},
            {"reverse_teacher_generation": True},
        )
        for mode in incompatible_modes:
            with self.subTest(mode=mode):
                collator_kwargs = {
                    "reason_first": False,
                    "target_only_teacher": True,
                }
                collator_kwargs.update(mode)
                with self.assertRaisesRegex(
                    ValueError,
                    "target_only_teacher=True is mutually exclusive",
                ):
                    SelfDistillationDataCollator(
                        _RecordingTokenizer(),
                        **collator_kwargs,
                    )


class AnswerOnlyTeacherPromptTests(unittest.TestCase):
    @staticmethod
    def _feature(answer=r"\frac{3}{7}"):
        return {
            "problem": "CURRENT PROBLEM A",
            "solution": (
                "PRIVATE REFERENCE REASONING THAT MUST NOT LEAK "
                r"\boxed{WRONG-SOLUTION-TAIL}"
            ),
            "Answer": answer,
        }

    def test_answer_only_exposes_gold_answer_but_not_reference_reasoning(self):
        tokenizer = _RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            answer_only_teacher=True,
            student_thinking=False,
            teacher_thinking=True,
        )

        result = collator([self._feature()])
        student_call, teacher_call = tokenizer.chat_calls
        student_text = student_call["messages"][0]["content"]
        teacher_text = teacher_call["messages"][0]["content"]

        self.assertIn("CURRENT PROBLEM A", student_text)
        self.assertNotIn(r"\frac{3}{7}", student_text)
        self.assertIn("CURRENT PROBLEM A", teacher_text)
        self.assertIn(r"\boxed{\frac{3}{7}}", teacher_text)
        self.assertNotIn("PRIVATE REFERENCE REASONING", teacher_text)
        self.assertNotIn("WRONG-SOLUTION-TAIL", teacher_text)
        self.assertFalse(student_call["enable_thinking"])
        self.assertTrue(teacher_call["enable_thinking"])
        self.assertEqual(
            result["reward_solutions"],
            [self._feature()["solution"]],
        )

    def test_answer_column_wins_and_outer_box_is_not_duplicated(self):
        tokenizer = _RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            answer_only_teacher=True,
        )

        collator([self._feature(answer=r"\[\boxed{\frac{5}{9}}\]")])
        teacher_text = tokenizer.chat_calls[1]["messages"][0]["content"]

        self.assertIn(r"\boxed{\frac{5}{9}}", teacher_text)
        self.assertNotIn(r"\boxed{\boxed{", teacher_text)
        self.assertNotIn("WRONG-SOLUTION-TAIL", teacher_text)

    def test_blank_answer_uses_explicit_target_only_fallback(self):
        tokenizer = _RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            answer_only_teacher=True,
            student_thinking=False,
            teacher_thinking=False,
        )

        collator([self._feature(answer="  ")])
        student_text = tokenizer.chat_calls[0]["messages"][0]["content"]
        teacher_text = tokenizer.chat_calls[1]["messages"][0]["content"]

        self.assertEqual(teacher_text, student_text)
        self.assertNotIn("PRIVATE REFERENCE REASONING", teacher_text)

    def test_answer_only_requires_explicit_answer_column(self):
        tokenizer = _RecordingTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            reason_first=False,
            answer_only_teacher=True,
        )
        feature = self._feature()
        del feature["Answer"]

        with self.assertRaisesRegex(ValueError, "explicit 'Answer' column"):
            collator([feature])

    def test_answer_only_rejects_other_teacher_prompt_modes(self):
        incompatible_modes = (
            {"target_only_teacher": True},
            {"reason_first": True},
            {"shuffled_worked_example": True},
            {"contrastive_teacher": True},
            {"reverse_teacher_generation": True},
        )
        for mode in incompatible_modes:
            with self.subTest(mode=mode):
                collator_kwargs = {
                    "reason_first": False,
                    "answer_only_teacher": True,
                }
                collator_kwargs.update(mode)
                with self.assertRaisesRegex(
                    ValueError,
                    "mutually exclusive",
                ):
                    SelfDistillationDataCollator(
                        _RecordingTokenizer(),
                        **collator_kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
