import torch

from op2sd.prompts import build_other_problem_teacher_prompt


def _strip_outer_answer_wrappers(answer):
    """Remove only wrappers around a gold answer; never inspect the solution trace."""
    text = str(answer).strip()
    while text:
        previous = text
        if len(text) >= 4 and (
            (text.startswith(r"\(") and text.endswith(r"\)"))
            or (text.startswith(r"\[") and text.endswith(r"\]"))
        ):
            text = text[2:-2].strip()
        elif len(text) >= 2 and text.startswith("$") and text.endswith("$"):
            text = text[1:-1].strip()
        else:
            for command in (r"\boxed", r"\fbox"):
                prefix = command + "{"
                if not text.startswith(prefix):
                    continue
                opening_index = len(command)
                depth = 0
                closing_index = None
                for index in range(opening_index, len(text)):
                    if text[index] == "{":
                        depth += 1
                    elif text[index] == "}":
                        depth -= 1
                        if depth == 0:
                            closing_index = index
                            break
                if closing_index == len(text) - 1:
                    text = text[opening_index + 1 : closing_index].strip()
                break
        if text == previous:
            break
    return text


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the problem (with chat template)
    Teacher: sees the configured teacher context (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        contrastive_teacher=False,
        reverse_teacher_generation=False,
        reverse_teacher_cache_required=False,
        mock_student_distillation=False,
        shuffled_worked_example=False,
        shuffled_worked_example_subject="mathematics",
        target_only_teacher=False,
        answer_only_teacher=False,
        student_thinking=False,
        teacher_thinking=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.contrastive_teacher = contrastive_teacher
        self.reverse_teacher_generation = reverse_teacher_generation
        self.reverse_teacher_cache_required = reverse_teacher_cache_required
        self.mock_student_distillation = mock_student_distillation
        self.shuffled_worked_example = shuffled_worked_example
        self.shuffled_worked_example_subject = str(
            shuffled_worked_example_subject
        ).strip().lower()
        self.target_only_teacher = target_only_teacher
        self.answer_only_teacher = answer_only_teacher
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking

        supported_worked_example_subjects = {
            "mathematics",
            "physics",
            "chemistry",
        }
        if (
            self.shuffled_worked_example_subject
            not in supported_worked_example_subjects
        ):
            raise ValueError(
                "shuffled_worked_example_subject must be one of: "
                + ", ".join(sorted(supported_worked_example_subjects))
            )
        if (
            not self.shuffled_worked_example
            and self.shuffled_worked_example_subject != "mathematics"
        ):
            raise ValueError(
                "A non-mathematics shuffled_worked_example_subject requires "
                "shuffled_worked_example=True."
            )

        if self.target_only_teacher:
            incompatible_modes = {
                "answer_only_teacher": self.answer_only_teacher,
                "reason_first": self.reason_first,
                "shuffled_worked_example": self.shuffled_worked_example,
                "contrastive_teacher": self.contrastive_teacher,
                "reverse_teacher_generation": self.reverse_teacher_generation,
            }
            enabled_incompatible_modes = [
                name for name, enabled in incompatible_modes.items() if enabled
            ]
            if enabled_incompatible_modes:
                raise ValueError(
                    "target_only_teacher=True is mutually exclusive with "
                    "reference- or donor-bearing teacher prompt modes: "
                    + ", ".join(enabled_incompatible_modes)
                )

        if self.answer_only_teacher:
            incompatible_modes = {
                "target_only_teacher": self.target_only_teacher,
                "reason_first": self.reason_first,
                "shuffled_worked_example": self.shuffled_worked_example,
                "contrastive_teacher": self.contrastive_teacher,
                "reverse_teacher_generation": self.reverse_teacher_generation,
            }
            enabled_incompatible_modes = [
                name for name, enabled in incompatible_modes.items() if enabled
            ]
            if enabled_incompatible_modes:
                raise ValueError(
                    "answer_only_teacher=True is mutually exclusive with "
                    "other teacher prompt modes: "
                    + ", ".join(enabled_incompatible_modes)
                )

        # Prompt for reasoning about the solution before teaching
        self.reason_first_prompt = (
            "\n\nThe reference reasoning above arrives at the correct answer. "
            "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
            "Do NOT use <think> tags. Do NOT derive your own solution. "
            "Simply analyze and explain the reference solution provided above.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )
        self.subtle_mistake_teacher_prompt = (
            "\n\nNow write a plausible-looking solution that intentionally contains a subtle "
            "mathematical mistake.\n\n"
            "The mistake should be difficult to notice at first glance. It may be a small "
            "algebraic slip, an unjustified assumption, a misapplied theorem, an incorrect "
            "case split, a hidden counting error, or a quiet change in a key condition.\n\n"
            "Do not announce that the solution is wrong. Present it naturally as if it were "
            "a normal complete solution. The flaw should be in the reasoning, not in the "
            "formatting. Put the final answer within \\boxed{}."
        )
        self.reverse_teacher_generation_prompt = (
            "\n\nRewrite the reference solution above as a concise, easy-to-understand solution. "
            "Keep only the essential steps needed to justify the answer, preserve the same final answer, "
            "and do not introduce a new method. Do not add extra checks, alternative derivations, "
            "or commentary after the final answer. End with the final answer in \\boxed{}."
        )

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] Contrastive teacher mode: {self.contrastive_teacher}")
        print(f"[DataCollator] Reverse teacher generation mode: {self.reverse_teacher_generation}")
        print(f"[DataCollator] Mock student mode: {self.mock_student_distillation}")
        print(f"[DataCollator] Shuffled worked-example mode: {self.shuffled_worked_example}")
        print(
            "[DataCollator] Shuffled worked-example subject: "
            f"{self.shuffled_worked_example_subject}"
        )
        print(f"[DataCollator] Target-only teacher mode: {self.target_only_teacher}")
        print(f"[DataCollator] Answer-only teacher mode: {self.answer_only_teacher}")

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        mock_student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode
        bad_teacher_prompts = []
        reverse_teacher_prompts = []
        reverse_teacher_cached_completion_ids = []
        reward_problems = []
        reward_solutions = []

        for feature in features:
            # Extract problem and solution from dataset
            # Handle different possible column names
            problem = feature["problem"]
            solution = feature["solution"]
            reward_problems.append(problem)
            reward_solutions.append(solution)

            # Student prompt: just the problem with instruction (matching evaluation format)
            student_user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            student_messages = [{"role": "user", "content": student_user_message}]

            # Apply chat template for student (matching evaluation)
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.student_thinking
            )
            student_prompts.append(student_prompt)

            if self.mock_student_distillation:
                mock_student_user_message = (
                    f"Problem: {problem}\n\n"
                    "Please reason step by step, and put your final answer within \\boxed{}. "
                    "Intentionally introduce one subtle mathematical error that is difficult to notice. "
                    "Present the solution naturally and do not reveal that it contains an intentional error."
                )
                mock_student_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": mock_student_user_message}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.student_thinking,
                )
                mock_student_prompts.append(mock_student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze the solution
                reasoning_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a correct reasoning to this problem:"
                    f"=== Reference Reasoning Start ===\n"
                    f"{solution}\n"
                    f"=== Reference Reasoning End ===\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                if self.target_only_teacher:
                    # Reference-free control: preserve the exact student user
                    # content while independently applying the teacher chat
                    # template (and therefore the configured teacher thinking
                    # mode). The gold solution remains available only for
                    # grading and diagnostics below.
                    teacher_user_message = student_user_message
                elif self.answer_only_teacher:
                    if "Answer" not in feature:
                        raise ValueError(
                            "answer_only_teacher=True requires the training "
                            "dataset's explicit 'Answer' column."
                        )
                    final_answer = _strip_outer_answer_wrappers(
                        feature.get("Answer", "")
                    )
                    if final_answer:
                        teacher_user_message = (
                            f"Problem: {problem}\n\n"
                            "The verified final answer to this problem is:\n"
                            "=== Reference Answer Begin ===\n"
                            f"\\boxed{{{final_answer}}}\n"
                            "=== Reference Answer End ===\n\n"
                            "Using only this final answer as the required result, "
                            "solve the problem independently. Please reason step by "
                            "step, verify your reasoning, and put your final answer "
                            "within \\boxed{}."
                        )
                    else:
                        # A tiny number of proof-only rows in the source corpus
                        # have no standalone Answer. Preserve the target stream
                        # rather than filtering and reshuffling every later
                        # exposure; these rows become an explicit target-only
                        # fallback and are counted during dataset validation.
                        teacher_user_message = student_user_message
                elif self.shuffled_worked_example:
                    required_example_fields = (
                        "teacher_example_problem",
                        "teacher_example_solution",
                    )
                    missing_example_fields = [
                        name for name in required_example_fields if name not in feature
                    ]
                    if missing_example_fields:
                        raise ValueError(
                            "Shuffled worked-example mode requires dataset fields: "
                            + ", ".join(missing_example_fields)
                        )
                    example_problem = feature["teacher_example_problem"]
                    example_solution = feature["teacher_example_solution"]
                    teacher_user_message = build_other_problem_teacher_prompt(
                        target_problem=problem,
                        example_problem=example_problem,
                        example_solution=example_solution,
                        subject=self.shuffled_worked_example_subject,
                    )
                else:
                    # Original teacher prompt (unchanged)
                    teacher_user_message = (
                        f"Problem: {problem}\n\n"
                        f"Here is a reference solution to this problem:\n"
                        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                        f"{self.transition_prompt}\n"
                        f"Please reason step by step, and put your final answer within \\boxed{{}}."
                    )
                teacher_messages = [{"role": "user", "content": teacher_user_message}]

                # Apply chat template for teacher
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.teacher_thinking
                )
                teacher_prompts.append(teacher_prompt)

            if self.contrastive_teacher:
                bad_teacher_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a reference solution to this problem:\n"
                    f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                    f"{self.subtle_mistake_teacher_prompt}"
                )
                bad_teacher_messages = [{"role": "user", "content": bad_teacher_user_message}]
                bad_teacher_prompt = self.tokenizer.apply_chat_template(
                    bad_teacher_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.teacher_thinking,
                )
                bad_teacher_prompts.append(bad_teacher_prompt)

            if self.reverse_teacher_generation:
                reverse_teacher_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a reference solution to this problem:\n"
                    f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                    f"{self.reverse_teacher_generation_prompt}"
                )
                reverse_teacher_messages = [{"role": "user", "content": reverse_teacher_user_message}]
                reverse_teacher_prompt = self.tokenizer.apply_chat_template(
                    reverse_teacher_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.teacher_thinking,
                )
                reverse_teacher_prompts.append(reverse_teacher_prompt)
                cached_token_ids = feature.get("reverse_teacher_completion_token_ids")
                if cached_token_ids is not None:
                    reverse_teacher_cached_completion_ids.append(list(cached_token_ids))

        # Tokenize WITHOUT padding first to get true lengths
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
            "reward_problems": reward_problems,
            "reward_solutions": reward_solutions,
        }

        if self.mock_student_distillation:
            mock_student_encoded_no_pad = self.tokenizer(
                mock_student_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            mock_student_prompt_lengths = [
                len(ids) for ids in mock_student_encoded_no_pad["input_ids"]
            ]
            max_mock_student_prompt_len = max(mock_student_prompt_lengths)
            mock_student_encoded = self.tokenizer(
                mock_student_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_mock_student_prompt_len,
                return_tensors="pt",
            )
            result.update(
                {
                    "mock_student_prompts": mock_student_encoded["input_ids"],
                    "mock_student_prompt_attention_mask": mock_student_encoded[
                        "attention_mask"
                    ],
                    "mock_student_prompt_length": max_mock_student_prompt_len,
                    "mock_student_prompt_lengths_per_example": torch.tensor(
                        mock_student_prompt_lengths
                    ),
                }
            )

        if self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Normal mode: tokenize teacher prompts
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        if self.contrastive_teacher:
            bad_teacher_encoded_no_pad = self.tokenizer(
                bad_teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            bad_teacher_prompt_lengths = [len(ids) for ids in bad_teacher_encoded_no_pad["input_ids"]]
            max_bad_teacher_prompt_len = max(bad_teacher_prompt_lengths)

            bad_teacher_encoded = self.tokenizer(
                bad_teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_bad_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "bad_teacher_prompts": bad_teacher_encoded["input_ids"],
                    "bad_teacher_prompt_attention_mask": bad_teacher_encoded["attention_mask"],
                    "bad_teacher_prompt_length": max_bad_teacher_prompt_len,
                    "bad_teacher_prompt_lengths_per_example": torch.tensor(bad_teacher_prompt_lengths),
                }
            )

        if self.reverse_teacher_generation:
            reverse_teacher_encoded_no_pad = self.tokenizer(
                reverse_teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reverse_teacher_prompt_lengths = [
                len(ids) for ids in reverse_teacher_encoded_no_pad["input_ids"]
            ]
            max_reverse_teacher_prompt_len = max(reverse_teacher_prompt_lengths)

            reverse_teacher_encoded = self.tokenizer(
                reverse_teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reverse_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "reverse_teacher_prompts": reverse_teacher_encoded["input_ids"],
                    "reverse_teacher_prompt_attention_mask": reverse_teacher_encoded["attention_mask"],
                    "reverse_teacher_prompt_length": max_reverse_teacher_prompt_len,
                    "reverse_teacher_prompt_lengths_per_example": torch.tensor(
                        reverse_teacher_prompt_lengths
                    ),
                }
            )

            if reverse_teacher_cached_completion_ids:
                if len(reverse_teacher_cached_completion_ids) != batch_size:
                    raise ValueError(
                        "Some but not all batch examples have reverse_teacher_completion_token_ids."
                    )
                max_cached_completion_len = max(
                    max(1, len(ids)) for ids in reverse_teacher_cached_completion_ids
                )
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id
                padded_cached_ids = []
                for ids in reverse_teacher_cached_completion_ids:
                    if not ids:
                        ids = [pad_token_id]
                    padded_cached_ids.append(
                        ids + [pad_token_id] * (max_cached_completion_len - len(ids))
                    )
                result["reverse_teacher_cached_completion_ids"] = torch.tensor(
                    padded_cached_ids, dtype=torch.long
                )
            elif self.reverse_teacher_cache_required:
                raise ValueError(
                    "reverse_teacher_cache_path was provided, but this batch has no "
                    "reverse_teacher_completion_token_ids. The cache column may have been "
                    "dropped before the data collator."
                )

        return result
