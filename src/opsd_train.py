import hashlib
import json
import os
from collections import Counter

import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer import OPSDTrainer
from op2sd.donors import attach_shuffled_worked_examples
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


SUPPORTED_SHUFFLED_WORKED_EXAMPLE_SUBJECTS = frozenset(
    {"mathematics", "physics", "chemistry"}
)


def validate_shuffled_worked_example_subject(
    *,
    shuffled_worked_example: bool,
    subject: str,
    donor_dataset_path: str | None,
) -> str:
    """Normalize and validate the donor subject without changing math defaults."""

    normalized_subject = str(subject).strip().lower()
    if normalized_subject not in SUPPORTED_SHUFFLED_WORKED_EXAMPLE_SUBJECTS:
        raise ValueError(
            "shuffled_worked_example_subject must be one of: "
            + ", ".join(sorted(SUPPORTED_SHUFFLED_WORKED_EXAMPLE_SUBJECTS))
        )
    if not shuffled_worked_example and normalized_subject != "mathematics":
        raise ValueError(
            "A non-mathematics shuffled_worked_example_subject was specified "
            "while shuffled_worked_example=False."
        )
    if (
        shuffled_worked_example
        and normalized_subject != "mathematics"
        and not str(donor_dataset_path or "").strip()
    ):
        raise ValueError(
            "Physics and chemistry worked-example subjects require an external "
            "shuffled_worked_example_donor_dataset_path. The internal donor "
            "sources belong to the mathematics training corpus."
        )
    return normalized_subject


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    train_dataset_name: str = field(
        default="siyanzhao/Openthoughts_math_30k_opsd",
        metadata={
            "help": "Hugging Face dataset name used for training when "
            "train_dataset_path is not set."
        },
    )
    train_dataset_split: str = field(
        default="train",
        metadata={"help": "Training split name."},
    )
    train_dataset_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional local JSON/JSONL training file. Rows must contain "
            "'problem' and 'solution'. This overrides train_dataset_name."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    target_only_teacher: bool = field(
        default=False,
        metadata={
            "help": "Reference-free control in which the fixed teacher receives the same target-problem "
            "user message as the student, without the current problem's solution or a donor example. "
            "The teacher and student still use their independently configured thinking modes. "
            "The dataset solution is retained only for grading and diagnostics."
        },
    )
    answer_only_teacher: bool = field(
        default=False,
        metadata={
            "help": "Gold-answer-only control in which the fixed teacher receives the target problem and "
            "the explicit final answer from the dataset's 'Answer' column, but no reference reasoning. "
            "The student still receives only the target problem. Rows with an empty Answer use the "
            "target-only prompt so the target stream and optimizer schedule remain unchanged."
        },
    )
    shuffled_worked_example: bool = field(
        default=False,
        metadata={
            "help": "Replace the current problem's privileged reference in the teacher prompt with a "
            "fixed worked problem/solution pair from a different dataset row. The pair is explicitly "
            "labeled as unrelated and is presented only as a general reasoning example. The student's "
            "prompt and the original solution used for reward/grading are unchanged."
        },
    )
    shuffled_worked_example_seed: int = field(
        default=1729,
        metadata={
            "help": "Seed for the deterministic, fixed-point-free assignment of unrelated worked examples."
        },
    )
    shuffled_worked_example_subject: str = field(
        default="mathematics",
        metadata={
            "help": "Subject of the unrelated worked-example donor prompt. "
            "Supported values are mathematics (the unchanged default prompt), "
            "physics, and chemistry. Physics and chemistry require a local "
            "external donor dataset path."
        },
    )
    shuffled_worked_example_donor_sources: str | None = field(
        default=None,
        metadata={
            "help": "Optional comma-separated source values restricting the unrelated worked-example "
            "donor pool, for example 'amc_aime,aops_forum'. Empty uses all dataset rows."
        },
    )
    shuffled_worked_example_donor_dataset_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional local JSON/JSONL file containing a separate worked-"
            "example donor pool. Rows must contain 'problem' and 'solution'. "
            "Mutually exclusive with shuffled_worked_example_donor_sources."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip each vocabulary-level divergence contribution to a maximum value. This can improve "
            "stability by preventing extremely high-loss stylistic tokens from dominating the training signal. "
            "Set to 0 for no clipping."
        },
    )
    divergence_objective: str = field(
        default="forward_kl",
        metadata={
            "help": "Token-level distillation objective. Use 'forward_kl' for KL(teacher || student), "
            "'reverse_kl' for KL(student || teacher), or 'generalized_jsd' to use the beta value directly. "
            "The paper's main OPSD setting uses forward_kl."
        },
    )
    contrastive_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use a contrastive bad teacher that writes plausible-looking solutions with subtle mistakes. "
            "The training loss subtracts contrastive_alpha times the bad-teacher distillation loss."
        },
    )
    contrastive_alpha: float = field(
        default=0.2,
        metadata={"help": "Weight for the contrastive bad-teacher loss when contrastive_teacher=True."},
    )
    contrastive_good_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight for the good-teacher loss when contrastive_teacher=True. "
            "Set to 0.0 to train only away from the bad teacher."
        },
    )
    reverse_teacher_generation: bool = field(
        default=False,
        metadata={
            "help": "Generate an additional fixed-teacher completion from the reference solution and subtract "
            "its token-level distillation loss from the normal OPSD loss."
        },
    )
    reverse_teacher_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the negative reverse-teacher-generation distillation loss."},
    )
    reverse_teacher_cache_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional JSONL cache of fixed-teacher completions for reverse_teacher_generation. "
            "Each line must contain index and token_ids or completion_token_ids."
        },
    )
    reward_guided_distillation: bool = field(
        default=False,
        metadata={
            "help": "Use GRPO-style binary outcome advantages to choose the distillation target. "
            "Correct rollouts move toward the privileged teacher; incorrect rollouts move toward the base model."
        },
    )
    reward_guided_alpha: float = field(
        default=1.0,
        metadata={"help": "Scale applied to the absolute GRPO-style advantage in reward_guided_distillation."},
    )
    reward_guided_advantage_epsilon: float = field(
        default=1e-6,
        metadata={"help": "Numerical epsilon for reward-guided group advantage normalization."},
    )
    wrong_boxed_only_distillation: bool = field(
        default=False,
        metadata={
            "help": "Apply the normal privileged-teacher OPSD loss only to rollouts with a boxed but incorrect "
            "answer. Correct boxed rollouts and rollouts without a boxed answer receive zero sample weight."
        },
    )
    mock_student_distillation: bool = field(
        default=False,
        metadata={
            "help": "Alternate normal on-policy student rollouts with on-policy mock-student rollouts prompted "
            "to contain a subtle error. Apply normal privileged-teacher OPSD only to boxed incorrect outputs, "
            "always scoring the trajectory under the normal student prompt."
        },
    )
    change_to_wrong_distillation: bool = field(
        default=False,
        metadata={
            "help": "Skip unboxed rollouts, keep boxed incorrect rollouts unchanged, and corrupt boxed correct "
            "rollouts by randomly replacing numbers and discourse connectives before applying normal OPSD."
        },
    )
    change_to_wrong_number_rate: float = field(
        default=0.1,
        metadata={"help": "Fraction of numeric occurrences replaced in a selected boxed rollout."},
    )
    change_to_wrong_connector_rate: float = field(
        default=0.1,
        metadata={"help": "Fraction of discourse connectives replaced in a selected boxed rollout."},
    )
    change_to_wrong_include_no_boxed: bool = field(
        default=False,
        metadata={
            "help": "Also use rollouts without a boxed answer after replacing numeric occurrences and discourse "
            "connectives. Boxed incorrect rollouts remain unchanged."
        },
    )
    localized_error_recovery_distillation: bool = field(
        default=False,
        metadata={
            "help": "For verified-correct boxed rollouts, replace exactly one numeric literal, discard the "
            "original suffix, regenerate a student continuation from the corrupted prefix, and apply "
            "privileged-teacher OPSD only to the regenerated continuation."
        },
    )
    localized_error_recovery_tokens: int = field(
        default=256,
        metadata={
            "help": "Maximum number of student continuation tokens generated after the localized numeric error."
        },
    )
    localized_error_recovery_min_prefix_tokens: int = field(
        default=32,
        metadata={
            "help": "Minimum number of completion-prefix tokens required before the injected numeric error."
        },
    )
    prefix_consistency_distillation: bool = field(
        default=False,
        metadata={
            "help": "Keep naturally generated correct- and wrong-boxed trajectories, truncate each at a fixed "
            "completion-token fraction, sample multiple student continuations from the unchanged prefix, and "
            "apply privileged-teacher OPSD only to those continuation tokens. Unboxed initial trajectories can "
            "optionally be included."
        },
    )
    prefix_consistency_fraction: float = field(
        default=0.75,
        metadata={
            "help": "Fraction of the initial completion tokens retained as the unchanged natural prefix."
        },
    )
    prefix_consistency_num_regenerations: int = field(
        default=3,
        metadata={
            "help": "Number of independent student continuations sampled from each natural prefix."
        },
    )
    prefix_consistency_suffix_tokens: int = field(
        default=256,
        metadata={
            "help": "Maximum number of tokens in each prefix-conditioned student continuation."
        },
    )
    prefix_consistency_include_no_boxed: bool = field(
        default=False,
        metadata={
            "help": "Also regenerate from initial trajectories with no valid parsed boxed answer. Invalid "
            "reference answers are still skipped."
        },
    )
    prefix_consistency_selection_mode: str = field(
        default="all",
        metadata={
            "help": "How regenerated suffixes are selected for OPSD. 'all' preserves the original behavior. "
            "'wrong_boxed_all' admits only initially wrong boxed trajectories and distills all K regenerated "
            "suffixes with equal weight, irrespective of their final-answer outcome. "
            "'wrong_boxed_mixed_all' additionally requires 0 < gold_count < K, then distills all K suffixes "
            "in every accepted mixed-outcome group."
        },
    )
    prefix_consistency_normalize_by_eligible: bool = field(
        default=False,
        metadata={
            "help": "Normalize prefix-consistency branch loss by the globally eligible prefix count rather "
            "than the full local batch size. Intended for wrong-boxed conditional-loss ablations."
        },
    )
    prefix_consistency_outcome_alpha: float = field(
        default=0.0,
        metadata={
            "help": "Coefficient for the centered binary outcome-advantage policy loss over regenerated "
            "suffixes. Zero preserves OPSD-only prefix-consistency training."
        },
    )
    wrong_answer_teacher_correction_distillation: bool = field(
        default=False,
        metadata={
            "help": "For boxed but incorrect student rollouts, ask the fixed privileged teacher to repeat the "
            "candidate up to its first error and continue with a corrected solution. Distill the teacher-generated "
            "corrected trajectory; correct boxed and unboxed student rollouts are skipped."
        },
    )
    wrong_answer_branch_contrastive: bool = field(
        default=False,
        metadata={
            "help": "For boxed but incorrect rollouts, use a fixed teacher to locate the first error and generate "
            "a corrected continuation from the exact student prefix. Optimize only a pairwise contrastive loss "
            "that ranks the corrected continuation above the student's wrong continuation."
        },
    )
    branch_contrastive_tokens: int = field(
        default=100,
        metadata={"help": "Maximum number of corrected and wrong continuation tokens used by branch contrastive loss."},
    )
    branch_contrastive_beta: float = field(
        default=1.0,
        metadata={"help": "Scale beta applied to the branch contrastive log-probability margin."},
    )
    branch_error_locator_max_tokens: int = field(
        default=256,
        metadata={"help": "Maximum fixed-teacher tokens used to identify the first incorrect excerpt."},
    )
    best_checkpoint_distillation: bool = field(
        default=False,
        metadata={
            "help": "Use outcome advantages (+1 correct boxed, -1 wrong boxed, +0.25 no boxed). "
            "Positive samples use the privileged teacher; negative samples use the best evaluated policy."
        },
    )
    best_checkpoint_eval_dataset: str = field(
        default="aime24",
        metadata={"help": "Dataset used to select the best checkpoint (currently aime24 only)."},
    )
    best_checkpoint_eval_val_n: int = field(
        default=12,
        metadata={"help": "Number of generations per evaluation problem for best-checkpoint selection."},
    )
    best_checkpoint_eval_max_new_tokens: int = field(
        default=32768,
        metadata={"help": "Generation limit for in-training best-checkpoint evaluation."},
    )
    best_checkpoint_eval_seed: int = field(
        default=12345,
        metadata={"help": "Fixed vLLM sampling seed used to compare checkpoints."},
    )
    best_checkpoint_baseline_score: float | None = field(
        default=None,
        metadata={
            "help": "Optional known base Avg@N percentage. If omitted, base is evaluated before training."
        },
    )
    best_checkpoint_independent_verification: bool = field(
        default=False,
        metadata={
            "help": "For best-checkpoint distillation, sample independent candidates per problem. Correct "
            "candidates use normal privileged-teacher OPSD; each incorrect candidate gets its own verification "
            "rollout on a context containing that candidate. A verification is distilled on that same context only "
            "when its boxed answer is correct; incorrect or unboxed verifications are skipped."
        },
    )
    verification_num_candidates: int = field(
        default=1,
        metadata={"help": "Number of independently graded student candidates sampled per problem."},
    )
    long_thought_base_penalty: bool = field(
        default=False,
        metadata={
            "help": "For long student rollouts, add a length-weighted JSD(student, base) penalty on the student "
            "prompt. The base distribution is computed by disabling LoRA adapters."
        },
    )
    long_thought_base_penalty_start: int = field(
        default=2048,
        metadata={"help": "Generated-token count at or below which the base penalty weight C is zero."},
    )
    long_thought_base_penalty_full: int = field(
        default=4096,
        metadata={
            "help": "Generated-token count at which the base penalty weight C reaches "
            "long_thought_base_penalty_weight."
        },
    )
    long_thought_base_penalty_weight: float = field(
        default=2.0,
        metadata={"help": "Maximum base penalty weight C at long_thought_base_penalty_full generated tokens."},
    )
    adaptive_completion_length: bool = field(
        default=False,
        metadata={
            "help": "Increase rollout max_new_tokens when the recent boxed-answer rate falls below a target. "
            "The length only increases during a run."
        },
    )
    adaptive_completion_target: float = field(
        default=0.3,
        metadata={"help": "Target boxed-answer rate used by adaptive_completion_length."},
    )
    adaptive_completion_window_steps: int = field(
        default=50,
        metadata={"help": "Number of optimizer steps per adaptive_completion_length decision window."},
    )
    adaptive_completion_length_increment: int = field(
        default=1024,
        metadata={"help": "Amount to increase max_new_tokens when adaptive_completion_length triggers."},
    )
    adaptive_max_completion_length: int = field(
        default=4096,
        metadata={"help": "Upper bound for adaptive_completion_length max_new_tokens."},
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    student_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the student during rollout. "
            "Default False (matches the main OPSD setup: student rolls out without <think>)."
        },
    )
    teacher_thinking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the teacher when scoring student tokens. "
            "Default True. Set to False for the matched non-thinking ablation (both nonthink)."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    divergence_objective = script_args.divergence_objective.lower().replace("-", "_")
    if divergence_objective == "forward_kl":
        training_args.beta = 0.0
    elif divergence_objective == "reverse_kl":
        training_args.beta = 1.0
    elif divergence_objective == "generalized_jsd":
        pass
    else:
        raise ValueError(
            "divergence_objective must be one of: forward_kl, reverse_kl, generalized_jsd. "
            f"Got: {script_args.divergence_objective}"
        )

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract a model name from either a local path or a Hugging Face ID.
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_config = (
            f"opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher argument
    script_args.shuffled_worked_example_subject = (
        validate_shuffled_worked_example_subject(
            shuffled_worked_example=script_args.shuffled_worked_example,
            subject=script_args.shuffled_worked_example_subject,
            donor_dataset_path=(
                script_args.shuffled_worked_example_donor_dataset_path
            ),
        )
    )
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )
    if script_args.reverse_teacher_generation and not script_args.fixed_teacher:
        raise ValueError("reverse_teacher_generation=True requires fixed_teacher=True.")
    if script_args.reverse_teacher_generation and script_args.use_tinker_loss:
        raise ValueError("reverse_teacher_generation=True is implemented for full-vocab distillation loss.")
    if script_args.reverse_teacher_cache_path and not script_args.reverse_teacher_generation:
        raise ValueError("reverse_teacher_cache_path requires reverse_teacher_generation=True.")
    if script_args.target_only_teacher:
        if not script_args.fixed_teacher:
            raise ValueError("target_only_teacher=True requires fixed_teacher=True.")
        incompatible_modes = {
            "answer_only_teacher": script_args.answer_only_teacher,
            "reason_first": script_args.reason_first,
            "shuffled_worked_example": script_args.shuffled_worked_example,
            "contrastive_teacher": script_args.contrastive_teacher,
            "reverse_teacher_generation": script_args.reverse_teacher_generation,
            "wrong_answer_teacher_correction_distillation": (
                script_args.wrong_answer_teacher_correction_distillation
            ),
            "wrong_answer_branch_contrastive": (
                script_args.wrong_answer_branch_contrastive
            ),
            "best_checkpoint_independent_verification": (
                script_args.best_checkpoint_independent_verification
            ),
        }
        enabled_incompatible_modes = [
            name for name, enabled in incompatible_modes.items() if enabled
        ]
        if enabled_incompatible_modes:
            raise ValueError(
                "target_only_teacher=True is mutually exclusive with "
                "reference- or donor-bearing teacher modes: "
                + ", ".join(enabled_incompatible_modes)
            )
    if script_args.answer_only_teacher:
        if not script_args.fixed_teacher:
            raise ValueError("answer_only_teacher=True requires fixed_teacher=True.")
        incompatible_modes = {
            "target_only_teacher": script_args.target_only_teacher,
            "reason_first": script_args.reason_first,
            "shuffled_worked_example": script_args.shuffled_worked_example,
            "contrastive_teacher": script_args.contrastive_teacher,
            "reverse_teacher_generation": script_args.reverse_teacher_generation,
            "wrong_answer_teacher_correction_distillation": (
                script_args.wrong_answer_teacher_correction_distillation
            ),
            "wrong_answer_branch_contrastive": (
                script_args.wrong_answer_branch_contrastive
            ),
            "best_checkpoint_independent_verification": (
                script_args.best_checkpoint_independent_verification
            ),
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
    if not script_args.shuffled_worked_example and (
        script_args.shuffled_worked_example_donor_dataset_path
        or script_args.shuffled_worked_example_donor_sources
    ):
        raise ValueError(
            "Worked-example donor data was specified while "
            "shuffled_worked_example=False. Refusing to silently run Gold OPSD."
        )
    if script_args.shuffled_worked_example:
        if not script_args.fixed_teacher:
            raise ValueError("shuffled_worked_example=True requires fixed_teacher=True.")
        if (
            script_args.shuffled_worked_example_donor_dataset_path
            and script_args.shuffled_worked_example_donor_sources
        ):
            raise ValueError(
                "shuffled_worked_example_donor_dataset_path is mutually exclusive "
                "with shuffled_worked_example_donor_sources."
            )
        incompatible_modes = {
            "answer_only_teacher": script_args.answer_only_teacher,
            "reason_first": script_args.reason_first,
            "contrastive_teacher": script_args.contrastive_teacher,
            "reverse_teacher_generation": script_args.reverse_teacher_generation,
        }
        enabled_incompatible_modes = [
            name for name, enabled in incompatible_modes.items() if enabled
        ]
        if enabled_incompatible_modes:
            raise ValueError(
                "shuffled_worked_example=True is mutually exclusive with teacher-prompt "
                "modes: "
                + ", ".join(enabled_incompatible_modes)
            )

    train_dataset_file_sha256 = None
    if script_args.train_dataset_path:
        script_args.train_dataset_path = os.path.abspath(
            os.path.expanduser(script_args.train_dataset_path)
        )
        if not os.path.isfile(script_args.train_dataset_path):
            raise FileNotFoundError(
                "Local training dataset does not exist: "
                f"{script_args.train_dataset_path}"
            )
        with open(script_args.train_dataset_path, "rb") as handle:
            train_dataset_file_sha256 = hashlib.sha256(
                handle.read()
            ).hexdigest()

    donor_dataset_file_sha256 = None
    if script_args.shuffled_worked_example_donor_dataset_path:
        script_args.shuffled_worked_example_donor_dataset_path = os.path.abspath(
            os.path.expanduser(
                script_args.shuffled_worked_example_donor_dataset_path
            )
        )
        if not os.path.isfile(
            script_args.shuffled_worked_example_donor_dataset_path
        ):
            raise FileNotFoundError(
                "External worked-example donor dataset does not exist: "
                f"{script_args.shuffled_worked_example_donor_dataset_path}"
            )
        with open(
            script_args.shuffled_worked_example_donor_dataset_path, "rb"
        ) as handle:
            donor_dataset_file_sha256 = hashlib.sha256(
                handle.read()
            ).hexdigest()

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_steps": training_args.max_steps,
                "training_seed": training_args.seed,
                "max_completion_length": training_args.max_completion_length,
                "save_steps": training_args.save_steps,
                "save_total_limit": training_args.save_total_limit,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "divergence_objective": divergence_objective,
                "jsd_token_clip": (
                    script_args.jsd_token_clip
                    if script_args.jsd_token_clip > 0
                    else None
                ),
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "target_only_teacher": script_args.target_only_teacher,
                "answer_only_teacher": script_args.answer_only_teacher,
                "teacher_context_mode": (
                    "target_only"
                    if script_args.target_only_teacher
                    else (
                        "target_answer_only"
                        if script_args.answer_only_teacher
                        else (
                            "unrelated_worked_example"
                            if script_args.shuffled_worked_example
                            else "target_reference"
                        )
                    )
                ),
                "student_thinking": script_args.student_thinking,
                "teacher_thinking": script_args.teacher_thinking,
                "train_dataset_name": script_args.train_dataset_name,
                "train_dataset_split": script_args.train_dataset_split,
                "train_dataset_path": script_args.train_dataset_path,
                "train_dataset_file_sha256": train_dataset_file_sha256,
                "shuffled_worked_example": script_args.shuffled_worked_example,
                "shuffled_worked_example_subject": (
                    script_args.shuffled_worked_example_subject
                    if script_args.shuffled_worked_example
                    else None
                ),
                "shuffled_worked_example_seed": (
                    script_args.shuffled_worked_example_seed
                    if script_args.shuffled_worked_example
                    else None
                ),
                "shuffled_worked_example_donor_sources": (
                    script_args.shuffled_worked_example_donor_sources
                    if script_args.shuffled_worked_example
                    else None
                ),
                "shuffled_worked_example_donor_dataset_path": (
                    script_args.shuffled_worked_example_donor_dataset_path
                    if script_args.shuffled_worked_example
                    else None
                ),
                "shuffled_worked_example_donor_dataset_file_sha256": (
                    donor_dataset_file_sha256
                    if script_args.shuffled_worked_example
                    else None
                ),
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "contrastive_teacher": script_args.contrastive_teacher,
                "contrastive_alpha": script_args.contrastive_alpha if script_args.contrastive_teacher else None,
                "contrastive_good_weight": (
                    script_args.contrastive_good_weight if script_args.contrastive_teacher else None
                ),
                "reverse_teacher_generation": script_args.reverse_teacher_generation,
                "reverse_teacher_weight": (
                    script_args.reverse_teacher_weight
                    if script_args.reverse_teacher_generation
                    else None
                ),
                "reverse_teacher_cache_path": script_args.reverse_teacher_cache_path,
                "reward_guided_distillation": script_args.reward_guided_distillation,
                "reward_guided_alpha": (
                    script_args.reward_guided_alpha
                    if script_args.reward_guided_distillation
                    else None
                ),
                "wrong_boxed_only_distillation": script_args.wrong_boxed_only_distillation,
                "mock_student_distillation": script_args.mock_student_distillation,
                "change_to_wrong_distillation": script_args.change_to_wrong_distillation,
                "change_to_wrong_number_rate": (
                    script_args.change_to_wrong_number_rate
                    if script_args.change_to_wrong_distillation
                    else None
                ),
                "change_to_wrong_connector_rate": (
                    script_args.change_to_wrong_connector_rate
                    if script_args.change_to_wrong_distillation
                    else None
                ),
                "change_to_wrong_include_no_boxed": (
                    script_args.change_to_wrong_include_no_boxed
                    if script_args.change_to_wrong_distillation
                    else None
                ),
                "localized_error_recovery_distillation": (
                    script_args.localized_error_recovery_distillation
                ),
                "localized_error_recovery_tokens": (
                    script_args.localized_error_recovery_tokens
                    if script_args.localized_error_recovery_distillation
                    else None
                ),
                "localized_error_recovery_min_prefix_tokens": (
                    script_args.localized_error_recovery_min_prefix_tokens
                    if script_args.localized_error_recovery_distillation
                    else None
                ),
                "prefix_consistency_distillation": (
                    script_args.prefix_consistency_distillation
                ),
                "prefix_consistency_fraction": (
                    script_args.prefix_consistency_fraction
                    if script_args.prefix_consistency_distillation
                    else None
                ),
                "prefix_consistency_num_regenerations": (
                    script_args.prefix_consistency_num_regenerations
                    if script_args.prefix_consistency_distillation
                    else None
                ),
                "prefix_consistency_suffix_tokens": (
                    script_args.prefix_consistency_suffix_tokens
                    if script_args.prefix_consistency_distillation
                    else None
                ),
                "prefix_consistency_include_no_boxed": (
                    script_args.prefix_consistency_include_no_boxed
                    if script_args.prefix_consistency_distillation
                    else None
                ),
                "prefix_consistency_selection_mode": (
                    script_args.prefix_consistency_selection_mode
                    if script_args.prefix_consistency_distillation
                    else None
                ),
                "prefix_consistency_normalize_by_eligible": (
                    script_args.prefix_consistency_normalize_by_eligible
                    if script_args.prefix_consistency_distillation
                    else None
                ),
                "prefix_consistency_outcome_alpha": (
                    script_args.prefix_consistency_outcome_alpha
                    if script_args.prefix_consistency_distillation
                    else None
                ),
                "wrong_answer_teacher_correction_distillation": (
                    script_args.wrong_answer_teacher_correction_distillation
                ),
                "wrong_answer_branch_contrastive": script_args.wrong_answer_branch_contrastive,
                "branch_contrastive_tokens": (
                    script_args.branch_contrastive_tokens
                    if script_args.wrong_answer_branch_contrastive
                    else None
                ),
                "branch_contrastive_beta": (
                    script_args.branch_contrastive_beta
                    if script_args.wrong_answer_branch_contrastive
                    else None
                ),
                "branch_error_locator_max_tokens": (
                    script_args.branch_error_locator_max_tokens
                    if script_args.wrong_answer_branch_contrastive
                    else None
                ),
                "best_checkpoint_distillation": script_args.best_checkpoint_distillation,
                "best_checkpoint_eval_dataset": (
                    script_args.best_checkpoint_eval_dataset
                    if script_args.best_checkpoint_distillation
                    else None
                ),
                "best_checkpoint_eval_val_n": (
                    script_args.best_checkpoint_eval_val_n
                    if script_args.best_checkpoint_distillation
                    else None
                ),
                "best_checkpoint_eval_max_new_tokens": (
                    script_args.best_checkpoint_eval_max_new_tokens
                    if script_args.best_checkpoint_distillation
                    else None
                ),
                "best_checkpoint_eval_seed": (
                    script_args.best_checkpoint_eval_seed
                    if script_args.best_checkpoint_distillation
                    else None
                ),
                "best_checkpoint_baseline_score": (
                    script_args.best_checkpoint_baseline_score
                    if script_args.best_checkpoint_distillation
                    else None
                ),
                "best_checkpoint_independent_verification": (
                    script_args.best_checkpoint_independent_verification
                    if script_args.best_checkpoint_distillation
                    else None
                ),
                "verification_num_candidates": (
                    script_args.verification_num_candidates
                    if script_args.best_checkpoint_independent_verification
                    else None
                ),
                "long_thought_base_penalty": script_args.long_thought_base_penalty,
                "long_thought_base_penalty_start": (
                    script_args.long_thought_base_penalty_start
                    if script_args.long_thought_base_penalty
                    else None
                ),
                "long_thought_base_penalty_full": (
                    script_args.long_thought_base_penalty_full
                    if script_args.long_thought_base_penalty
                    else None
                ),
                "long_thought_base_penalty_weight": (
                    script_args.long_thought_base_penalty_weight
                    if script_args.long_thought_base_penalty
                    else None
                ),
                "adaptive_completion_length": script_args.adaptive_completion_length,
                "adaptive_completion_target": (
                    script_args.adaptive_completion_target
                    if script_args.adaptive_completion_length
                    else None
                ),
                "adaptive_completion_window_steps": (
                    script_args.adaptive_completion_window_steps
                    if script_args.adaptive_completion_length
                    else None
                ),
                "adaptive_completion_length_increment": (
                    script_args.adaptive_completion_length_increment
                    if script_args.adaptive_completion_length
                    else None
                ),
                "adaptive_max_completion_length": (
                    script_args.adaptive_max_completion_length
                    if script_args.adaptive_completion_length
                    else None
                ),
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    if script_args.train_dataset_path:
        train_dataset_path = os.path.abspath(
            os.path.expanduser(script_args.train_dataset_path)
        )
        if not os.path.isfile(train_dataset_path):
            raise FileNotFoundError(
                f"Local training dataset does not exist: {train_dataset_path}"
            )
        train_dataset = load_dataset(
            "json",
            data_files={script_args.train_dataset_split: train_dataset_path},
            split=script_args.train_dataset_split,
        )
        train_dataset_identity = train_dataset_path
    else:
        train_dataset = load_dataset(
            script_args.train_dataset_name,
            split=script_args.train_dataset_split,
        )
        train_dataset_identity = (
            f"{script_args.train_dataset_name}:{script_args.train_dataset_split}"
        )

    missing_train_columns = {
        column
        for column in ("problem", "solution")
        if column not in train_dataset.column_names
    }
    if missing_train_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            + ", ".join(sorted(missing_train_columns))
        )

    if script_args.answer_only_teacher:
        if "Answer" not in train_dataset.column_names:
            raise ValueError(
                "answer_only_teacher=True requires an explicit 'Answer' "
                "column. It intentionally does not extract an answer from "
                "the reference solution because that could reintroduce an "
                "incorrect or multipart solution trace."
            )
        blank_answer_count = sum(
            not str(answer or "").strip() for answer in train_dataset["Answer"]
        )
        print(
            "Answer-only teacher dataset audit: "
            f"answer_column=Answer, rows={len(train_dataset)}, "
            f"nonempty_answers={len(train_dataset) - blank_answer_count}, "
            f"blank_answers={blank_answer_count}."
        )
        if blank_answer_count:
            print(
                "Rows with a blank Answer retain their position in the target "
                "stream and use the target-only teacher prompt. No reference "
                "solution text is used as a fallback."
            )

    # SFTTrainer performs its own preparatory tokenization before our
    # SelfDistillationDataCollator builds the actual student/teacher prompts.
    # Hub datasets used by OPSD normally contain a conversational `messages`
    # column, but minimal local JSONL datasets may contain only `problem` and
    # `solution`. In that case TRL falls back to its default `text` field and
    # otherwise fails with KeyError("text"). The resulting input_ids are not
    # used to construct the OPSD prompts; keep `problem` and `solution` intact
    # and provide the problem as the harmless preprocessing text.
    if (
        "messages" not in train_dataset.column_names
        and training_args.dataset_text_field not in train_dataset.column_names
    ):
        train_dataset = train_dataset.add_column(
            training_args.dataset_text_field,
            list(train_dataset["problem"]),
        )
        print(
            "Added TRL preprocessing column "
            f"'{training_args.dataset_text_field}' from local problem text."
        )

    print(
        "Loaded training dataset: "
        f"identity={train_dataset_identity}, rows={len(train_dataset)}, "
        f"fingerprint={getattr(train_dataset, '_fingerprint', None)}"
    )

    if script_args.shuffled_worked_example:
        source_dataset_fingerprint = getattr(train_dataset, "_fingerprint", None)
        donor_sources = [
            source.strip()
            for source in (
                script_args.shuffled_worked_example_donor_sources or ""
            ).split(",")
            if source.strip()
        ]
        donor_dataset = None
        donor_dataset_identity = None
        donor_dataset_fingerprint = None
        if script_args.shuffled_worked_example_donor_dataset_path:
            donor_dataset_path = os.path.abspath(
                os.path.expanduser(
                    script_args.shuffled_worked_example_donor_dataset_path
                )
            )
            if not os.path.isfile(donor_dataset_path):
                raise FileNotFoundError(
                    "External worked-example donor dataset does not exist: "
                    f"{donor_dataset_path}"
                )
            donor_dataset = load_dataset(
                "json",
                data_files={"train": donor_dataset_path},
                split="train",
            )
            donor_dataset_identity = donor_dataset_path
            donor_dataset_fingerprint = getattr(
                donor_dataset, "_fingerprint", None
            )
            print(
                "Loaded external worked-example donor dataset: "
                f"identity={donor_dataset_identity}, rows={len(donor_dataset)}, "
                f"fingerprint={donor_dataset_fingerprint}"
            )
        train_dataset = attach_shuffled_worked_examples(
            train_dataset,
            seed=script_args.shuffled_worked_example_seed,
            donor_sources=donor_sources,
            donor_dataset=donor_dataset,
        )
        source_indices = train_dataset["teacher_example_source_index"]
        donor_reuse_counts = Counter(source_indices)
        unique_donor_count = len(donor_reuse_counts)
        min_donor_reuse = min(donor_reuse_counts.values())
        max_donor_reuse = max(donor_reuse_counts.values())
        mapping_sha256 = hashlib.sha256(
            ",".join(str(source_index) for source_index in source_indices).encode(
                "utf-8"
            )
        ).hexdigest()
        print(
            "Attached deterministic unrelated worked examples: "
            f"rows={len(train_dataset)}, seed={script_args.shuffled_worked_example_seed}, "
            f"subject={script_args.shuffled_worked_example_subject}, "
            f"donor_sources={donor_sources or ['all']}, "
            f"external_donor_dataset={donor_dataset_identity}, "
            f"unique_donors={unique_donor_count}, "
            f"donor_reuse_range=[{min_donor_reuse}, {max_donor_reuse}], "
            "same_problem_pairs="
            f"{sum(str(target).strip() == str(donor).strip() for target, donor in zip(train_dataset['problem'], train_dataset['teacher_example_problem'], strict=True))}, "
            f"source_dataset_fingerprint={source_dataset_fingerprint}, "
            f"donor_dataset_fingerprint={donor_dataset_fingerprint}, "
            f"mapping_sha256={mapping_sha256}"
        )
        if os.environ.get("LOCAL_RANK", "0") == "0" and wandb.run is not None:
            wandb.config.update(
                {
                    "shuffled_worked_example_source_dataset_fingerprint": (
                        source_dataset_fingerprint
                    ),
                    "shuffled_worked_example_donor_dataset_identity": (
                        donor_dataset_identity
                    ),
                    "shuffled_worked_example_donor_dataset_fingerprint": (
                        donor_dataset_fingerprint
                    ),
                    "shuffled_worked_example_mapping_sha256": mapping_sha256,
                    "shuffled_worked_example_unique_donor_count": (
                        unique_donor_count
                    ),
                    "shuffled_worked_example_min_donor_reuse": min_donor_reuse,
                    "shuffled_worked_example_max_donor_reuse": max_donor_reuse,
                },
                allow_val_change=True,
            )

    if script_args.reverse_teacher_cache_path:
        cache_path = script_args.reverse_teacher_cache_path
        reverse_teacher_cache = {}
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                idx = int(record["index"])
                token_ids = record.get("token_ids")
                if token_ids is None:
                    token_ids = record.get("completion_token_ids")
                if token_ids is None:
                    raise ValueError(f"Missing token_ids in reverse teacher cache record index={idx}")
                reverse_teacher_cache[idx] = [int(token_id) for token_id in token_ids]

        def add_reverse_teacher_cache(example, idx):
            if idx not in reverse_teacher_cache:
                raise ValueError(
                    f"Reverse teacher cache is missing dataset index {idx}. "
                    f"Regenerate the cache with enough rows: {cache_path}"
                )
            return {"reverse_teacher_completion_token_ids": reverse_teacher_cache[idx]}

        train_dataset = train_dataset.map(
            add_reverse_teacher_cache,
            with_indices=True,
            desc=f"Attaching reverse teacher cache from {cache_path}",
        )
        print(
            f"Loaded reverse teacher cache: {len(reverse_teacher_cache)} rows from {cache_path}"
        )

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        shuffled_worked_example=script_args.shuffled_worked_example,
        shuffled_worked_example_subject=(
            script_args.shuffled_worked_example_subject
        ),
        target_only_teacher=script_args.target_only_teacher,
        answer_only_teacher=script_args.answer_only_teacher,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        contrastive_teacher=script_args.contrastive_teacher,
        contrastive_alpha=script_args.contrastive_alpha,
        contrastive_good_weight=script_args.contrastive_good_weight,
        reverse_teacher_generation=script_args.reverse_teacher_generation,
        reverse_teacher_weight=script_args.reverse_teacher_weight,
        reverse_teacher_cache_required=bool(script_args.reverse_teacher_cache_path),
        reward_guided_distillation=script_args.reward_guided_distillation,
        reward_guided_alpha=script_args.reward_guided_alpha,
        reward_guided_advantage_epsilon=script_args.reward_guided_advantage_epsilon,
        wrong_boxed_only_distillation=script_args.wrong_boxed_only_distillation,
        mock_student_distillation=script_args.mock_student_distillation,
        change_to_wrong_distillation=script_args.change_to_wrong_distillation,
        change_to_wrong_number_rate=script_args.change_to_wrong_number_rate,
        change_to_wrong_connector_rate=script_args.change_to_wrong_connector_rate,
        change_to_wrong_include_no_boxed=script_args.change_to_wrong_include_no_boxed,
        localized_error_recovery_distillation=(
            script_args.localized_error_recovery_distillation
        ),
        localized_error_recovery_tokens=script_args.localized_error_recovery_tokens,
        localized_error_recovery_min_prefix_tokens=(
            script_args.localized_error_recovery_min_prefix_tokens
        ),
        prefix_consistency_distillation=(
            script_args.prefix_consistency_distillation
        ),
        prefix_consistency_fraction=script_args.prefix_consistency_fraction,
        prefix_consistency_num_regenerations=(
            script_args.prefix_consistency_num_regenerations
        ),
        prefix_consistency_suffix_tokens=(
            script_args.prefix_consistency_suffix_tokens
        ),
        prefix_consistency_include_no_boxed=(
            script_args.prefix_consistency_include_no_boxed
        ),
        prefix_consistency_selection_mode=(
            script_args.prefix_consistency_selection_mode
        ),
        prefix_consistency_normalize_by_eligible=(
            script_args.prefix_consistency_normalize_by_eligible
        ),
        prefix_consistency_outcome_alpha=(
            script_args.prefix_consistency_outcome_alpha
        ),
        wrong_answer_teacher_correction_distillation=(
            script_args.wrong_answer_teacher_correction_distillation
        ),
        wrong_answer_branch_contrastive=script_args.wrong_answer_branch_contrastive,
        branch_contrastive_tokens=script_args.branch_contrastive_tokens,
        branch_contrastive_beta=script_args.branch_contrastive_beta,
        branch_error_locator_max_tokens=script_args.branch_error_locator_max_tokens,
        best_checkpoint_distillation=script_args.best_checkpoint_distillation,
        best_checkpoint_eval_dataset=script_args.best_checkpoint_eval_dataset,
        best_checkpoint_eval_val_n=script_args.best_checkpoint_eval_val_n,
        best_checkpoint_eval_max_new_tokens=script_args.best_checkpoint_eval_max_new_tokens,
        best_checkpoint_eval_seed=script_args.best_checkpoint_eval_seed,
        best_checkpoint_baseline_score=script_args.best_checkpoint_baseline_score,
        best_checkpoint_independent_verification=script_args.best_checkpoint_independent_verification,
        verification_num_candidates=script_args.verification_num_candidates,
        long_thought_base_penalty=script_args.long_thought_base_penalty,
        long_thought_base_penalty_start=script_args.long_thought_base_penalty_start,
        long_thought_base_penalty_full=script_args.long_thought_base_penalty_full,
        long_thought_base_penalty_weight=script_args.long_thought_base_penalty_weight,
        adaptive_completion_length=script_args.adaptive_completion_length,
        adaptive_completion_target=script_args.adaptive_completion_target,
        adaptive_completion_window_steps=script_args.adaptive_completion_window_steps,
        adaptive_completion_length_increment=script_args.adaptive_completion_length_increment,
        adaptive_max_completion_length=script_args.adaptive_max_completion_length,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
