# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os
import random
import re
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.utils import (
    is_flash_attn_2_available,
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)

from trl.data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset
from trl.extras.profiling import profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    empty_cache,
    ensure_master_addr_port,
    pad,
)
from trl.experimental.gold.gold_config import GOLDConfig
from data_collator import SelfDistillationDataCollator


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


class EMAUpdateCallback(TrainerCallback):
    """Update EMA teacher weights after each optimizer step."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only update when the optimizer actually stepped (end of a gradient accumulation cycle)
        if self.trainer.use_ema_teacher and self.trainer.accelerator.sync_gradients:
            self.trainer._update_ema()


class GOLDVLLMSyncCallback(TrainerCallback):
    """Sync the model weights to vLLM after training steps when it's safe to do so."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync weights after training step when DeepSpeed is stable."""
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            # Check if this is a step where gradients are synchronized
            # This happens at the end of gradient accumulation cycles
            if (
                hasattr(self.trainer.accelerator, "sync_gradients")
                and self.trainer.accelerator.sync_gradients
            ):
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class BestCheckpointEvaluationCallback(TrainerCallback):
    """Evaluate saved checkpoints and update the frozen best-policy anchor."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.trainer._initialize_best_checkpoint_anchor(state.global_step)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.trainer._evaluate_and_maybe_promote_checkpoint(state.global_step)


class OPSDTrainer(SFTTrainer):
    _tag_names = ["trl", "opsd"]
    _name = "OPSD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        use_thinking_machines_loss: bool = False,
        fixed_teacher: bool = False,
        reason_first: bool = False,
        shuffled_worked_example: bool = False,
        shuffled_worked_example_subject: str = "mathematics",
        target_only_teacher: bool = False,
        answer_only_teacher: bool = False,
        top_k_loss: int | None = None,
        jsd_token_clip: float | None = None,
        contrastive_teacher: bool = False,
        contrastive_alpha: float = 0.2,
        contrastive_good_weight: float = 1.0,
        reverse_teacher_generation: bool = False,
        reverse_teacher_weight: float = 1.0,
        reverse_teacher_cache_required: bool = False,
        reward_guided_distillation: bool = False,
        reward_guided_alpha: float = 1.0,
        reward_guided_advantage_epsilon: float = 1e-6,
        wrong_boxed_only_distillation: bool = False,
        mock_student_distillation: bool = False,
        change_to_wrong_distillation: bool = False,
        change_to_wrong_number_rate: float = 0.1,
        change_to_wrong_connector_rate: float = 0.1,
        change_to_wrong_include_no_boxed: bool = False,
        localized_error_recovery_distillation: bool = False,
        localized_error_recovery_tokens: int = 256,
        localized_error_recovery_min_prefix_tokens: int = 32,
        prefix_consistency_distillation: bool = False,
        prefix_consistency_fraction: float = 0.75,
        prefix_consistency_num_regenerations: int = 3,
        prefix_consistency_suffix_tokens: int = 256,
        prefix_consistency_include_no_boxed: bool = False,
        prefix_consistency_selection_mode: str = "all",
        prefix_consistency_normalize_by_eligible: bool = False,
        prefix_consistency_outcome_alpha: float = 0.0,
        wrong_answer_teacher_correction_distillation: bool = False,
        wrong_answer_branch_contrastive: bool = False,
        branch_contrastive_tokens: int = 100,
        branch_contrastive_beta: float = 1.0,
        branch_error_locator_max_tokens: int = 256,
        best_checkpoint_distillation: bool = False,
        best_checkpoint_eval_dataset: str = "aime24",
        best_checkpoint_eval_val_n: int = 12,
        best_checkpoint_eval_max_new_tokens: int = 32768,
        best_checkpoint_eval_seed: int = 12345,
        best_checkpoint_baseline_score: float | None = None,
        best_checkpoint_independent_verification: bool = False,
        verification_num_candidates: int = 1,
        long_thought_base_penalty: bool = False,
        long_thought_base_penalty_start: int = 2048,
        long_thought_base_penalty_full: int = 4096,
        long_thought_base_penalty_weight: float = 2.0,
        adaptive_completion_length: bool = False,
        adaptive_completion_target: float = 0.3,
        adaptive_completion_window_steps: int = 50,
        adaptive_completion_length_increment: int = 1024,
        adaptive_max_completion_length: int = 4096,
        use_ema_teacher: bool = False,
        ema_decay: float = 0.999,
        student_thinking: bool = False,
        teacher_thinking: bool = True,
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        # Custom data collator for self-distillation
        if data_collator is None:
            data_collator = SelfDistillationDataCollator(
                tokenizer=processing_class,
                max_length=args.max_length,
                reason_first=reason_first,
                contrastive_teacher=contrastive_teacher,
                reverse_teacher_generation=reverse_teacher_generation,
                reverse_teacher_cache_required=reverse_teacher_cache_required,
                mock_student_distillation=mock_student_distillation,
                shuffled_worked_example=shuffled_worked_example,
                shuffled_worked_example_subject=(
                    shuffled_worked_example_subject
                ),
                target_only_teacher=target_only_teacher,
                answer_only_teacher=answer_only_teacher,
                student_thinking=student_thinking,
                teacher_thinking=teacher_thinking,
            )

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.use_thinking_machines_loss = use_thinking_machines_loss
        self.fixed_teacher = fixed_teacher
        self.reason_first = reason_first
        self.shuffled_worked_example = shuffled_worked_example
        self.target_only_teacher = target_only_teacher
        self.answer_only_teacher = answer_only_teacher
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip
        self.contrastive_teacher = contrastive_teacher
        self.contrastive_alpha = contrastive_alpha
        self.contrastive_good_weight = contrastive_good_weight
        self.reverse_teacher_generation = reverse_teacher_generation
        self.reverse_teacher_weight = reverse_teacher_weight
        self.reward_guided_distillation = reward_guided_distillation
        self.reward_guided_alpha = reward_guided_alpha
        self.reward_guided_advantage_epsilon = reward_guided_advantage_epsilon
        self.wrong_boxed_only_distillation = wrong_boxed_only_distillation
        self.mock_student_distillation = mock_student_distillation
        self._mock_student_rollout_count = 0
        self.change_to_wrong_distillation = change_to_wrong_distillation
        self.change_to_wrong_number_rate = change_to_wrong_number_rate
        self.change_to_wrong_connector_rate = change_to_wrong_connector_rate
        self.change_to_wrong_include_no_boxed = change_to_wrong_include_no_boxed
        self.localized_error_recovery_distillation = localized_error_recovery_distillation
        self.localized_error_recovery_tokens = localized_error_recovery_tokens
        self.localized_error_recovery_min_prefix_tokens = (
            localized_error_recovery_min_prefix_tokens
        )
        self.prefix_consistency_distillation = prefix_consistency_distillation
        self.prefix_consistency_fraction = prefix_consistency_fraction
        self.prefix_consistency_num_regenerations = (
            prefix_consistency_num_regenerations
        )
        self.prefix_consistency_suffix_tokens = prefix_consistency_suffix_tokens
        self.prefix_consistency_include_no_boxed = (
            prefix_consistency_include_no_boxed
        )
        self.prefix_consistency_selection_mode = prefix_consistency_selection_mode
        self.prefix_consistency_normalize_by_eligible = (
            prefix_consistency_normalize_by_eligible
        )
        self.prefix_consistency_outcome_alpha = prefix_consistency_outcome_alpha
        self.wrong_answer_teacher_correction_distillation = (
            wrong_answer_teacher_correction_distillation
        )
        self.wrong_answer_branch_contrastive = wrong_answer_branch_contrastive
        self.branch_contrastive_tokens = branch_contrastive_tokens
        self.branch_contrastive_beta = branch_contrastive_beta
        self.branch_error_locator_max_tokens = branch_error_locator_max_tokens
        self.best_checkpoint_distillation = best_checkpoint_distillation
        self.best_checkpoint_eval_dataset = best_checkpoint_eval_dataset
        self.best_checkpoint_eval_val_n = best_checkpoint_eval_val_n
        self.best_checkpoint_eval_max_new_tokens = best_checkpoint_eval_max_new_tokens
        self.best_checkpoint_eval_seed = best_checkpoint_eval_seed
        self.best_checkpoint_baseline_score = best_checkpoint_baseline_score
        self.best_checkpoint_independent_verification = best_checkpoint_independent_verification
        self.verification_num_candidates = verification_num_candidates
        self._best_anchor_params = None
        self._best_checkpoint_score = None
        self._best_checkpoint_step = 0
        self.long_thought_base_penalty = long_thought_base_penalty
        self.long_thought_base_penalty_start = long_thought_base_penalty_start
        self.long_thought_base_penalty_full = long_thought_base_penalty_full
        self.long_thought_base_penalty_weight = long_thought_base_penalty_weight
        self.adaptive_completion_length = adaptive_completion_length
        self.adaptive_completion_target = adaptive_completion_target
        self.adaptive_completion_window_steps = adaptive_completion_window_steps
        self.adaptive_completion_length_increment = adaptive_completion_length_increment
        self.adaptive_max_completion_length = adaptive_max_completion_length
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self._adaptive_completion_history = deque(maxlen=max(1, int(adaptive_completion_window_steps)))
        self._adaptive_completion_optimizer_steps = 0
        self._adaptive_completion_last_check_step = 0
        self._adaptive_pending_boxed = 0
        self._adaptive_pending_total = 0
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self._ema_params = None  # lazily initialized on first optimizer step

        # Validate fixed_teacher option
        if self.fixed_teacher and peft_config is None:
            raise ValueError(
                "fixed_teacher=True requires a PEFT config (use_peft=True). "
                "The fixed teacher is implemented by disabling LoRA adapters during teacher forward passes."
            )

        if self.target_only_teacher and not self.fixed_teacher:
            raise ValueError(
                "target_only_teacher=True requires fixed_teacher=True. "
                "The reference-free control uses the initial base policy as its teacher."
            )

        if self.answer_only_teacher and not self.fixed_teacher:
            raise ValueError(
                "answer_only_teacher=True requires fixed_teacher=True. "
                "The gold-answer-only control uses the initial base policy as its teacher."
            )
        if self.target_only_teacher and self.answer_only_teacher:
            raise ValueError(
                "target_only_teacher=True and answer_only_teacher=True are mutually exclusive."
            )

        if self.use_ema_teacher and self.fixed_teacher:
            raise ValueError(
                "use_ema_teacher=True and fixed_teacher=True are mutually exclusive teacher strategies."
            )

        if self.contrastive_teacher and self.use_thinking_machines_loss:
            raise ValueError("contrastive_teacher=True is implemented for the full-vocab distillation loss.")

        if self.reverse_teacher_generation and self.use_thinking_machines_loss:
            raise ValueError("reverse_teacher_generation=True is implemented for the full-vocab distillation loss.")
        if self.reverse_teacher_generation and not self.fixed_teacher:
            raise ValueError("reverse_teacher_generation=True requires fixed_teacher=True.")

        if self.reward_guided_distillation:
            if self.use_thinking_machines_loss:
                raise ValueError("reward_guided_distillation=True is implemented for the full-vocab loss.")
            if peft_config is None:
                raise ValueError(
                    "reward_guided_distillation=True requires PEFT/LoRA so the base model can be evaluated "
                    "by disabling adapters."
                )
            if self.contrastive_teacher or self.reverse_teacher_generation or self.long_thought_base_penalty:
                raise ValueError(
                    "reward_guided_distillation=True replaces the standard OPSD target selection and is mutually "
                    "exclusive with contrastive_teacher, reverse_teacher_generation, and long_thought_base_penalty."
                )
            if self.reward_guided_alpha < 0:
                raise ValueError("reward_guided_alpha must be non-negative.")
            if self.reward_guided_advantage_epsilon <= 0:
                raise ValueError("reward_guided_advantage_epsilon must be positive.")

        if self.wrong_boxed_only_distillation:
            if self.use_thinking_machines_loss:
                raise ValueError("wrong_boxed_only_distillation=True requires the full-vocabulary loss.")
            if (
                self.contrastive_teacher
                or self.reverse_teacher_generation
                or self.reward_guided_distillation
                or self.best_checkpoint_distillation
            ):
                raise ValueError(
                    "wrong_boxed_only_distillation=True is mutually exclusive with contrastive_teacher, "
                    "reverse_teacher_generation, reward_guided_distillation, and best_checkpoint_distillation."
                )

        if self.mock_student_distillation:
            if self.use_thinking_machines_loss:
                raise ValueError("mock_student_distillation=True requires the full-vocabulary loss.")
            if not self.fixed_teacher or peft_config is None:
                raise ValueError(
                    "mock_student_distillation=True requires fixed_teacher=True and PEFT/LoRA."
                )
            if (
                self.wrong_boxed_only_distillation
                or self.wrong_answer_teacher_correction_distillation
                or self.wrong_answer_branch_contrastive
                or self.contrastive_teacher
                or self.reverse_teacher_generation
                or self.reward_guided_distillation
                or self.best_checkpoint_distillation
                or self.long_thought_base_penalty
                or self.reason_first
            ):
                raise ValueError(
                    "mock_student_distillation=True is mutually exclusive with reason_first and the other "
                    "specialized distillation modes."
                )

        if self.change_to_wrong_distillation:
            if self.use_thinking_machines_loss:
                raise ValueError("change_to_wrong_distillation=True requires the full-vocabulary loss.")
            if not self.fixed_teacher or peft_config is None:
                raise ValueError(
                    "change_to_wrong_distillation=True requires fixed_teacher=True and PEFT/LoRA."
                )
            if not 0.0 <= self.change_to_wrong_number_rate <= 1.0:
                raise ValueError("change_to_wrong_number_rate must be in [0, 1].")
            if not 0.0 <= self.change_to_wrong_connector_rate <= 1.0:
                raise ValueError("change_to_wrong_connector_rate must be in [0, 1].")
            if (
                self.wrong_boxed_only_distillation
                or self.mock_student_distillation
                or self.wrong_answer_teacher_correction_distillation
                or self.wrong_answer_branch_contrastive
                or self.contrastive_teacher
                or self.reverse_teacher_generation
                or self.reward_guided_distillation
                or self.best_checkpoint_distillation
                or self.long_thought_base_penalty
                or self.reason_first
            ):
                raise ValueError(
                    "change_to_wrong_distillation=True is mutually exclusive with reason_first and the other "
                    "specialized distillation modes."
                )

        if self.localized_error_recovery_distillation:
            if self.use_thinking_machines_loss:
                raise ValueError(
                    "localized_error_recovery_distillation=True requires the full-vocabulary loss."
                )
            if int(self.args.per_device_train_batch_size) != 1:
                raise ValueError(
                    "localized_error_recovery_distillation=True currently requires "
                    "per_device_train_batch_size=1."
                )
            if args.use_vllm and args.vllm_enable_sleep_mode:
                raise ValueError(
                    "localized_error_recovery_distillation=True currently does not support "
                    "vllm_enable_sleep_mode."
                )
            if args.use_vllm and (
                args.vllm_mode != "colocate" or args.vllm_tensor_parallel_size != 1
            ):
                raise ValueError(
                    "localized_error_recovery_distillation=True currently requires colocated "
                    "vLLM with vllm_tensor_parallel_size=1."
                )
            if not self.fixed_teacher or peft_config is None:
                raise ValueError(
                    "localized_error_recovery_distillation=True requires fixed_teacher=True and PEFT/LoRA."
                )
            if self.localized_error_recovery_tokens <= 0:
                raise ValueError("localized_error_recovery_tokens must be positive.")
            if self.localized_error_recovery_tokens >= int(args.max_completion_length):
                raise ValueError(
                    "localized_error_recovery_tokens must be smaller than max_completion_length so the "
                    "corrupted prefix and recovery continuation share the original completion budget."
                )
            if self.localized_error_recovery_min_prefix_tokens < 0:
                raise ValueError(
                    "localized_error_recovery_min_prefix_tokens must be non-negative."
                )
            if (
                self.wrong_boxed_only_distillation
                or self.mock_student_distillation
                or self.change_to_wrong_distillation
                or self.wrong_answer_teacher_correction_distillation
                or self.wrong_answer_branch_contrastive
                or self.contrastive_teacher
                or self.reverse_teacher_generation
                or self.reward_guided_distillation
                or self.best_checkpoint_distillation
                or self.long_thought_base_penalty
                or self.reason_first
            ):
                raise ValueError(
                    "localized_error_recovery_distillation=True is mutually exclusive with reason_first and "
                    "the other specialized distillation modes."
                )

        if self.prefix_consistency_distillation:
            allowed_selection_modes = {
                "all",
                "wrong_boxed_all",
                "wrong_boxed_mixed_all",
            }
            if self.prefix_consistency_selection_mode not in allowed_selection_modes:
                raise ValueError(
                    "prefix_consistency_selection_mode must be one of "
                    f"{sorted(allowed_selection_modes)}; got "
                    f"{self.prefix_consistency_selection_mode!r}."
                )
            if self.use_thinking_machines_loss:
                raise ValueError(
                    "prefix_consistency_distillation=True requires the full-vocabulary loss."
                )
            if not self.fixed_teacher or peft_config is None:
                raise ValueError(
                    "prefix_consistency_distillation=True requires fixed_teacher=True and PEFT/LoRA."
                )
            if not args.use_vllm:
                raise ValueError(
                    "prefix_consistency_distillation=True currently requires colocated vLLM."
                )
            if args.vllm_enable_sleep_mode:
                raise ValueError(
                    "prefix_consistency_distillation=True currently does not support "
                    "vllm_enable_sleep_mode."
                )
            if args.vllm_mode != "colocate" or args.vllm_tensor_parallel_size != 1:
                raise ValueError(
                    "prefix_consistency_distillation=True currently requires colocated "
                    "vLLM with vllm_tensor_parallel_size=1."
                )
            if not 0.0 < self.prefix_consistency_fraction < 1.0:
                raise ValueError("prefix_consistency_fraction must be strictly between 0 and 1.")
            if self.prefix_consistency_num_regenerations <= 0:
                raise ValueError("prefix_consistency_num_regenerations must be positive.")
            if (
                self.prefix_consistency_selection_mode
                == "wrong_boxed_mixed_all"
                and self.prefix_consistency_num_regenerations < 2
            ):
                raise ValueError(
                    "wrong_boxed_mixed_all requires at least two regenerations."
                )
            if self.prefix_consistency_suffix_tokens <= 0:
                raise ValueError("prefix_consistency_suffix_tokens must be positive.")
            if self.prefix_consistency_outcome_alpha < 0.0:
                raise ValueError("prefix_consistency_outcome_alpha must be non-negative.")
            if self.prefix_consistency_selection_mode in {
                "wrong_boxed_all",
                "wrong_boxed_mixed_all",
            }:
                if self.prefix_consistency_include_no_boxed:
                    raise ValueError(
                        f"{self.prefix_consistency_selection_mode} requires "
                        "prefix_consistency_include_no_boxed=False."
                    )
                if self.prefix_consistency_outcome_alpha != 0.0:
                    raise ValueError(
                        f"{self.prefix_consistency_selection_mode} is an OPSD-only ablation and requires "
                        "prefix_consistency_outcome_alpha=0."
                    )
            elif self.prefix_consistency_normalize_by_eligible:
                raise ValueError(
                    "prefix_consistency_normalize_by_eligible=True currently requires "
                    "a wrong-boxed conditional selection mode."
                )
            max_completion_tokens = int(args.max_completion_length)
            max_natural_prefix_tokens = math.ceil(
                self.prefix_consistency_fraction * max_completion_tokens
            )
            if (
                max_natural_prefix_tokens + self.prefix_consistency_suffix_tokens
                > max_completion_tokens
            ):
                raise ValueError(
                    "The natural prefix plus regenerated suffix exceeds max_completion_length: "
                    f"ceil({self.prefix_consistency_fraction} * {max_completion_tokens}) + "
                    f"{self.prefix_consistency_suffix_tokens} > {max_completion_tokens}."
                )
            if (
                self.wrong_boxed_only_distillation
                or self.mock_student_distillation
                or self.change_to_wrong_distillation
                or self.localized_error_recovery_distillation
                or self.wrong_answer_teacher_correction_distillation
                or self.wrong_answer_branch_contrastive
                or self.contrastive_teacher
                or self.reverse_teacher_generation
                or self.reward_guided_distillation
                or self.best_checkpoint_distillation
                or self.long_thought_base_penalty
                or self.adaptive_completion_length
                or self.reason_first
            ):
                raise ValueError(
                    "prefix_consistency_distillation=True is mutually exclusive with reason_first, "
                    "adaptive completion length, and the other specialized distillation modes."
                )

        if self.wrong_answer_teacher_correction_distillation:
            if self.use_thinking_machines_loss:
                raise ValueError(
                    "wrong_answer_teacher_correction_distillation=True requires the full-vocabulary loss."
                )
            if not self.fixed_teacher or peft_config is None:
                raise ValueError(
                    "wrong_answer_teacher_correction_distillation=True requires fixed_teacher=True and PEFT/LoRA."
                )
            if (
                self.wrong_boxed_only_distillation
                or self.contrastive_teacher
                or self.reverse_teacher_generation
                or self.reward_guided_distillation
                or self.best_checkpoint_distillation
                or self.reason_first
            ):
                raise ValueError(
                    "wrong_answer_teacher_correction_distillation=True is mutually exclusive with reason_first, "
                    "wrong_boxed_only_distillation, contrastive_teacher, reverse_teacher_generation, "
                    "reward_guided_distillation, and best_checkpoint_distillation."
                )

        if self.wrong_answer_branch_contrastive:
            if self.use_thinking_machines_loss:
                raise ValueError(
                    "wrong_answer_branch_contrastive=True uses its own pairwise loss and is not compatible "
                    "with use_thinking_machines_loss."
                )
            if not self.fixed_teacher or peft_config is None:
                raise ValueError(
                    "wrong_answer_branch_contrastive=True requires fixed_teacher=True and PEFT/LoRA."
                )
            if self.branch_contrastive_tokens <= 0:
                raise ValueError("branch_contrastive_tokens must be positive.")
            if self.branch_contrastive_beta <= 0:
                raise ValueError("branch_contrastive_beta must be positive.")
            if self.branch_error_locator_max_tokens <= 0:
                raise ValueError("branch_error_locator_max_tokens must be positive.")
            if (
                self.wrong_boxed_only_distillation
                or self.wrong_answer_teacher_correction_distillation
                or self.contrastive_teacher
                or self.reverse_teacher_generation
                or self.reward_guided_distillation
                or self.best_checkpoint_distillation
                or self.long_thought_base_penalty
                or self.reason_first
            ):
                raise ValueError(
                    "wrong_answer_branch_contrastive=True is mutually exclusive with reason_first, "
                    "wrong_boxed_only_distillation, wrong_answer_teacher_correction_distillation, "
                    "contrastive_teacher, reverse_teacher_generation, reward_guided_distillation, "
                    "best_checkpoint_distillation, and long_thought_base_penalty."
                )

        if self.best_checkpoint_independent_verification and not self.best_checkpoint_distillation:
            raise ValueError(
                "best_checkpoint_independent_verification=True requires best_checkpoint_distillation=True."
            )

        if self.best_checkpoint_distillation:
            if self.use_thinking_machines_loss:
                raise ValueError("best_checkpoint_distillation=True requires the full-vocabulary loss.")
            if peft_config is None or not self.fixed_teacher:
                raise ValueError(
                    "best_checkpoint_distillation=True requires PEFT/LoRA and fixed_teacher=True."
                )
            if self.contrastive_teacher or self.reverse_teacher_generation or self.reward_guided_distillation:
                raise ValueError(
                    "best_checkpoint_distillation=True is mutually exclusive with contrastive_teacher, "
                    "reverse_teacher_generation, and reward_guided_distillation."
                )
            if self.long_thought_base_penalty:
                raise ValueError(
                    "best_checkpoint_distillation=True is mutually exclusive with long_thought_base_penalty."
                )
            if (
                not self.best_checkpoint_independent_verification
                and args.per_device_train_batch_size != 1
            ):
                raise ValueError(
                    "best_checkpoint_distillation with 4096-token full-vocabulary KL requires "
                    "per_device_train_batch_size=1. Use gradient accumulation for the rollout group."
                )
            if not args.use_vllm or args.vllm_mode != "colocate":
                raise ValueError(
                    "best_checkpoint_distillation checkpoint evaluation requires colocated vLLM."
                )
            if args.vllm_tensor_parallel_size != 1:
                raise ValueError(
                    "best_checkpoint_distillation checkpoint evaluation currently requires "
                    "vllm_tensor_parallel_size=1."
                )
            if self.best_checkpoint_eval_dataset != "aime24":
                raise ValueError("best_checkpoint_eval_dataset currently supports only 'aime24'.")
            if self.best_checkpoint_eval_val_n <= 0:
                raise ValueError("best_checkpoint_eval_val_n must be positive.")
            if self.best_checkpoint_eval_max_new_tokens <= 0:
                raise ValueError("best_checkpoint_eval_max_new_tokens must be positive.")
            if self.best_checkpoint_eval_max_new_tokens >= args.max_length:
                raise ValueError(
                    "best_checkpoint_eval_max_new_tokens must be smaller than max_length so checkpoint "
                    "evaluation has room for the prompt. Training and evaluation completion limits may differ."
                )
            if self.best_checkpoint_independent_verification:
                if self.verification_num_candidates <= 0:
                    raise ValueError("verification_num_candidates must be positive.")
                if self.reason_first:
                    raise ValueError(
                        "best_checkpoint_independent_verification is not compatible with reason_first."
                    )

        if self.long_thought_base_penalty:
            if self.use_thinking_machines_loss:
                raise ValueError("long_thought_base_penalty=True is implemented for the full-vocab loss.")
            if peft_config is None:
                raise ValueError(
                    "long_thought_base_penalty=True requires PEFT/LoRA so the base model can be evaluated "
                    "by disabling adapters."
                )
            if self.long_thought_base_penalty_start < 0:
                raise ValueError("long_thought_base_penalty_start must be non-negative.")
            if self.long_thought_base_penalty_full <= self.long_thought_base_penalty_start:
                raise ValueError(
                    "long_thought_base_penalty_full must be greater than long_thought_base_penalty_start."
                )
            if self.long_thought_base_penalty_weight < 0:
                raise ValueError("long_thought_base_penalty_weight must be non-negative.")

        if self.adaptive_completion_length:
            if not (0.0 <= self.adaptive_completion_target <= 1.0):
                raise ValueError("adaptive_completion_target must be in [0, 1].")
            if self.adaptive_completion_window_steps <= 0:
                raise ValueError("adaptive_completion_window_steps must be positive.")
            if self.adaptive_completion_length_increment <= 0:
                raise ValueError("adaptive_completion_length_increment must be positive.")
            if self.adaptive_max_completion_length < args.max_completion_length:
                raise ValueError(
                    "adaptive_max_completion_length must be greater than or equal to max_completion_length."
                )

        if self.use_ema_teacher:
            self.add_callback(EMAUpdateCallback(self))
            print(f"\n{'='*80}")
            print("EMA TEACHER MODE ENABLED")
            print(f"EMA decay: {self.ema_decay}")
            print("Teacher is an exponential moving average of the student weights.")
            print("EMA parameters are initialized on the first optimizer step.")
            print(f"{'='*80}\n")

        if self.fixed_teacher:
            print(f"\n{'='*80}")
            print("FIXED TEACHER MODE ENABLED")
            print("Teacher will use the initial policy (base model without LoRA adapters)")
            print("Student will update with LoRA adapters")
            print(f"{'='*80}\n")

        if self.target_only_teacher:
            print(f"\n{'='*80}")
            print("TARGET-ONLY TEACHER MODE ENABLED")
            print("Teacher user input = target problem + the same solve instruction as the student.")
            print("No reference solution or unrelated worked example is included in the teacher prompt.")
            print(
                "Student/teacher chat templates still use their configured thinking modes: "
                f"student_thinking={self.student_thinking}, "
                f"teacher_thinking={self.teacher_thinking}."
            )
            print(f"{'='*80}\n")

        if self.answer_only_teacher:
            print(f"\n{'='*80}")
            print("GOLD-ANSWER-ONLY TEACHER MODE ENABLED")
            print(
                "Teacher user input = target problem + explicit final Answer; "
                "the reference reasoning is excluded."
            )
            print("Student user input remains target problem only.")
            print(
                "Student/teacher chat templates use their configured thinking modes: "
                f"student_thinking={self.student_thinking}, "
                f"teacher_thinking={self.teacher_thinking}."
            )
            print(f"{'='*80}\n")

        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASON FIRST MODE ENABLED")
            print("Teacher will first reason about the privileged solution, then evaluate student's response")
            print(f"{'='*80}\n")

        if self.contrastive_teacher:
            print(f"\n{'='*80}")
            print("CONTRASTIVE TEACHER MODE ENABLED")
            print(
                "Loss = good_weight * good_teacher_loss "
                "- alpha * subtle_mistake_bad_teacher_loss"
            )
            print(f"Contrastive good weight: {self.contrastive_good_weight}")
            print(f"Contrastive alpha: {self.contrastive_alpha}")
            print(f"{'='*80}\n")

        if self.reverse_teacher_generation:
            print(f"\n{'='*80}")
            print("REVERSE TEACHER GENERATION MODE ENABLED")
            print("Loss = normal_good_teacher_loss - reverse_teacher_weight * teacher_generated_loss")
            print("The reverse teacher completion is generated by the fixed/base teacher from the reference solution.")
            print(f"Reverse teacher weight: {self.reverse_teacher_weight}")
            print(f"{'='*80}\n")

        if self.reward_guided_distillation:
            print(f"\n{'='*80}")
            print("REWARD-GUIDED DISTILLATION MODE ENABLED")
            print("Rewards: correct boxed answer = +1, otherwise = -1.")
            print(
                "Group advantages are computed over the gathered rollout group. "
                "Positive advantage moves toward the privileged teacher; negative advantage moves toward base."
            )
            print(f"Reward-guided alpha: {self.reward_guided_alpha}")
            print(f"{'='*80}\n")

        if self.wrong_boxed_only_distillation:
            print(f"\n{'='*80}")
            print("WRONG-BOXED-ONLY DISTILLATION MODE ENABLED")
            print("Loss = normal OPSD teacher loss only for boxed but incorrect student rollouts.")
            print("Correct boxed and no-boxed rollouts are skipped with zero sample weight.")
            print(f"{'='*80}\n")

        if self.mock_student_distillation:
            print(f"\n{'='*80}")
            print("MOCK-STUDENT DISTILLATION MODE ENABLED")
            print("On-policy rollouts alternate between the normal student prompt and a subtle-error prompt.")
            print(
                "Only boxed incorrect rollouts are distilled. The generated tokens are always scored under "
                "the normal student prompt against the fixed privileged teacher."
            )
            print(f"{'='*80}\n")

        if self.change_to_wrong_distillation:
            print(f"\n{'='*80}")
            print("CHANGE-TO-WRONG DISTILLATION MODE ENABLED")
            if self.change_to_wrong_include_no_boxed:
                print(
                    "No-boxed rollouts are changed by replacing numbers and discourse connectives, then "
                    "distilled with normal fixed-teacher OPSD."
                )
            else:
                print("No-boxed rollouts are skipped.")
            print("Boxed incorrect rollouts are distilled unchanged.")
            print(
                "Boxed correct rollouts are changed into incorrect trajectories by replacing numbers and "
                "discourse connectives, then distilled with normal fixed-teacher OPSD."
            )
            print(f"Number replacement rate: {self.change_to_wrong_number_rate}")
            print(f"Connector replacement rate: {self.change_to_wrong_connector_rate}")
            print(f"Include modified no-boxed rollouts: {self.change_to_wrong_include_no_boxed}")
            print(f"{'='*80}\n")

        if self.localized_error_recovery_distillation:
            print(f"\n{'='*80}")
            print("LOCALIZED ERROR-RECOVERY DISTILLATION MODE ENABLED")
            print("Only verified-correct boxed initial student rollouts are eligible.")
            print(
                "Exactly one reasoning number is replaced, the original suffix is discarded, and the student "
                "regenerates from the corrupted prefix."
            )
            print(
                "Loss = fixed privileged-teacher OPSD only on the regenerated continuation; the problem prompt "
                "and corrupted prefix are context-only."
            )
            print(f"Recovery continuation tokens: {self.localized_error_recovery_tokens}")
            print(
                "Minimum corrupted-prefix tokens: "
                f"{self.localized_error_recovery_min_prefix_tokens}"
            )
            print(f"{'='*80}\n")

        if self.prefix_consistency_distillation:
            print(f"\n{'='*80}")
            print("PREFIX-CONSISTENCY DISTILLATION MODE ENABLED")
            if self.prefix_consistency_selection_mode == "wrong_boxed_all":
                print("Eligible initial rollouts: wrong-boxed only.")
                print(
                    "K suffixes are sampled and all K receive equal-weight privileged-teacher OPSD, "
                    "regardless of whether each suffix is correct, wrong, or unboxed."
                )
            elif (
                self.prefix_consistency_selection_mode
                == "wrong_boxed_mixed_all"
            ):
                print("Regenerated initial rollouts: wrong-boxed only.")
                print(
                    "A group is selected only when its K suffixes contain both a correct and a "
                    "non-correct outcome (0 < gold_count < K)."
                )
                print(
                    "For every selected mixed-outcome group, all K suffixes receive equal-weight "
                    "privileged-teacher OPSD."
                )
            elif self.prefix_consistency_include_no_boxed:
                print(
                    "Eligible initial rollouts: every trajectory with a valid reference answer, including "
                    "no-boxed and malformed-boxed outputs."
                )
            else:
                print("Eligible initial rollouts: correct-boxed and wrong-boxed; no-boxed is skipped.")
            print(
                "The initial completion is truncated at an unchanged natural token prefix, then the student "
                "samples multiple independent suffixes from that same prefix."
            )
            if self.prefix_consistency_outcome_alpha > 0.0:
                print(
                    "Loss = equal-weight privileged-teacher OPSD plus a centered binary outcome-advantage "
                    "policy loss over regenerated suffixes only."
                )
                print(
                    "Homogeneous all-correct or all-wrong groups have zero outcome advantage and retain "
                    "OPSD supervision."
                )
            else:
                print(
                    "Loss = the equal-weight mean of privileged-teacher OPSD over regenerated suffixes only; "
                    "the problem prompt and natural prefix are context-only."
                )
            print(f"Natural prefix fraction: {self.prefix_consistency_fraction}")
            print(
                "Regenerations per prefix: "
                f"{self.prefix_consistency_num_regenerations}"
            )
            print(f"Maximum tokens per suffix: {self.prefix_consistency_suffix_tokens}")
            print(
                "Include initial trajectories without a valid boxed answer: "
                f"{self.prefix_consistency_include_no_boxed}"
            )
            print(
                "Regeneration selection mode: "
                f"{self.prefix_consistency_selection_mode}"
            )
            print(
                "Normalize loss by globally eligible prefixes: "
                f"{self.prefix_consistency_normalize_by_eligible}"
            )
            print(f"Outcome-advantage alpha: {self.prefix_consistency_outcome_alpha}")
            if self.prefix_consistency_outcome_alpha == 0.0:
                print("Prefix-consistency scores are logged for analysis and do not weight the loss.")
            print(
                "A boxed marker already present inside the natural prefix is logged as leakage but is not "
                "filtered, matching the token-truncation definition."
            )
            print(f"{'='*80}\n")

        if self.wrong_answer_teacher_correction_distillation:
            print(f"\n{'='*80}")
            print("WRONG-ANSWER TEACHER-CORRECTION DISTILLATION MODE ENABLED")
            print("First rollout: generate normally and select only boxed but incorrect answers.")
            print(
                "For each selected answer, the fixed privileged teacher repeats the candidate up to its first "
                "error and continues with a corrected solution."
            )
            print(
                "Loss = token-level distillation on the teacher-generated corrected trajectory; "
                "correct boxed and no-boxed initial rollouts are skipped."
            )
            print(f"{'='*80}\n")

        if self.wrong_answer_branch_contrastive:
            print(f"\n{'='*80}")
            print("WRONG-ANSWER BRANCH CONTRASTIVE MODE ENABLED")
            print("Only boxed but incorrect initial student rollouts are eligible.")
            print(
                "A fixed teacher first quotes the first incorrect excerpt, then generates a corrected "
                "continuation from the exact student prefix before that excerpt."
            )
            print(
                "Loss = softplus(-beta * (mean_logp(correct continuation) - "
                "mean_logp(wrong continuation))). No OPSD/KL loss is added."
            )
            print(f"Continuation tokens: {self.branch_contrastive_tokens}")
            print(f"Contrastive beta: {self.branch_contrastive_beta}")
            print(f"Error locator max tokens: {self.branch_error_locator_max_tokens}")
            print(f"{'='*80}\n")

        if self.best_checkpoint_distillation:
            effective_rollout_group = (
                self.accelerator.num_processes
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
            )
            print(f"\n{'='*80}")
            print("BEST-CHECKPOINT DISTILLATION MODE ENABLED")
            print(f"Effective optimizer-step rollout group: {effective_rollout_group}")
            if self.best_checkpoint_independent_verification:
                print(
                    f"Independent verification candidates per problem: {self.verification_num_candidates}."
                )
                print("Correct candidate: Forward KL(student, privileged teacher) on the original prompt.")
                print("Incorrect candidate: generate an independent verification rollout.")
                print(
                    "Correct verification: Forward KL(student, privileged teacher) on its verification prompt."
                )
                print("Incorrect or unboxed verification: skip the loss. No reverse-best term is used.")
            else:
                print("Advantages: correct boxed = +1, wrong boxed = -1, no boxed = +0.25")
                print("Positive advantages use Forward KL(student, privileged teacher).")
                print("Negative advantages use Forward KL(student, frozen best policy).")
            print(
                f"Checkpoint selection: {self.best_checkpoint_eval_dataset} "
                f"Avg@{self.best_checkpoint_eval_val_n}, "
                f"max_new_tokens={self.best_checkpoint_eval_max_new_tokens}, "
                f"seed={self.best_checkpoint_eval_seed}"
            )
            print(f"{'='*80}\n")

        if self.long_thought_base_penalty:
            print(f"\n{'='*80}")
            print("LONG-THOUGHT BASE PENALTY MODE ENABLED")
            print(
                "Adds C(length) * JSD(student, base) on the student prompt, "
                "where base is the model with LoRA adapters disabled."
            )
            print(f"C(length) = 0 up to {self.long_thought_base_penalty_start} generated tokens")
            print(
                f"C(length) grows linearly to {self.long_thought_base_penalty_weight} "
                f"at {self.long_thought_base_penalty_full} generated tokens"
            )
            print(f"{'='*80}\n")

        if self.adaptive_completion_length:
            print(f"\n{'='*80}")
            print("ADAPTIVE COMPLETION LENGTH MODE ENABLED")
            print(
                f"If boxed rate over {self.adaptive_completion_window_steps} optimizer steps falls below "
                f"{self.adaptive_completion_target:.3f}, max_new_tokens increases by "
                f"{self.adaptive_completion_length_increment}."
            )
            print(f"Initial max_new_tokens: {args.max_completion_length}")
            print(f"Maximum max_new_tokens: {self.adaptive_max_completion_length}")
            print(f"{'='*80}\n")

        # Track per-step loss statistics for on/off-policy batches (used in logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        self.use_transformers_paged = args.use_transformers_paged or False

        # Track generation outputs for saving
        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5  # Save every 5 steps

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Generation config for reasoning phase (when reason_first=True)
        max_reasoning_length = getattr(args, "max_reasoning_length", 4096)
        self.reasoning_generation_config = GenerationConfig(
            max_new_tokens=max_reasoning_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.reasoning_generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                # Make sure tensor_parallel_size divides world size evenly
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                )

                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

            self.add_callback(GOLDVLLMSyncCallback(self))

        if self.best_checkpoint_distillation:
            self.add_callback(BestCheckpointEvaluationCallback(self))

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "problem",
            "solution",
        ]
        if self.answer_only_teacher:
            required_columns.append("Answer")
        if self.shuffled_worked_example:
            required_columns.extend(
                [
                    "teacher_example_problem",
                    "teacher_example_solution",
                    "teacher_example_source_index",
                ]
            )
        if self.reverse_teacher_generation:
            required_columns.append("reverse_teacher_completion_token_ids")
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
        token_clip=None,
        sample_weights=None,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1. beta=0 gives forward KL,
                KL(teacher || student); beta=1 gives reverse KL, KL(student || teacher).
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')
            top_k:
                If set, restricts the loss to only the top-k tokens of the teacher distribution. Both student and
                teacher distributions are renormalized over these k tokens before computing JSD. This reduces memory
                and focuses distillation on the teacher's most probable tokens. (default: None = full vocabulary)
            token_clip:
                if set, clips per-token divergence values to this maximum before reduction. Prevents style tokens from dominating the gradient signal over math tokens.
            sample_weights:
                Optional tensor of shape (batch_size,) that weights each sample's token-level divergence before
                reduction. The denominator remains the number of valid tokens, so the loss scale is proportional
                to the average sample weight.

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            # Apply temperature scaling to logits before computing probabilities
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature

            if top_k is not None and top_k > 0:
                # Restrict to top-k tokens of the teacher distribution and renormalize.
                # Shape: [batch, seq_len, top_k]
                _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
                student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
                teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

            # Compute log probabilities for student and probabilities for teacher
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Per-token clipping: cap each token's divergence value
        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)

        if sample_weights is not None:
            sample_weights = sample_weights.to(device=jsd.device, dtype=jsd.dtype)
            token_jsd = jsd.sum(dim=-1)
            if labels is not None:
                mask = labels != -100
                token_jsd = token_jsd * mask.to(token_jsd.dtype)
                weighted = token_jsd * sample_weights[:, None]
                return weighted.sum() / mask.sum().clamp_min(1)
            return (token_jsd * sample_weights[:, None]).sum() / token_jsd.numel()

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return (
                jsd.sum() / mask.sum().clamp_min(1)
                if labels is not None
                else jsd.sum() / jsd.size(0)
            )
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def _update_ema(self):
        """Update EMA parameters after an optimizer step.

        On the very first call this lazily initializes the EMA state as an exact copy of the
        current (trainable) model parameters, then returns without applying a decay step.
        Subsequent calls apply: ema = decay * ema + (1 - decay) * student.

        Only trainable parameters are tracked (i.e. LoRA adapter weights for PEFT models,
        or all parameters for full fine-tuning).

        ZeRO-3 note: with ZeRO-3 each rank only holds a shard of every parameter.
        We use `deepspeed.zero.GatheredParameters` (read-only, modifier_rank=None) so that
        every rank sees the full parameter tensor when snapshotting / updating the EMA.
        The EMA tensors are therefore full-sized copies, which is also required by
        `_ema_teacher_context` when it swaps the gathered student weights with EMA values.
        """
        decay = self.ema_decay
        unwrapped = self.accelerator.unwrap_model(self.model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            trainable = [(name, param) for name, param in unwrapped.named_parameters() if param.requires_grad]
            params_list = [p for _, p in trainable]

            # modifier_rank=None → read-only gather; original partitions are restored on exit.
            with deepspeed.zero.GatheredParameters(params_list):
                if self._ema_params is None:
                    self._ema_params = {name: param.data.clone().detach() for name, param in trainable}
                    n_tensors = len(self._ema_params)
                    n_params = sum(p.numel() for p in self._ema_params.values())
                    print(
                        f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                        f"(decay={decay})"
                    )
                    return  # first call = initialization only, no decay update

                for name, param in trainable:
                    if name not in self._ema_params:
                        continue
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    ema.mul_(decay).add_(param.data, alpha=1.0 - decay)
        else:
            if self._ema_params is None:
                # Lazy init: snapshot the current weights as the initial EMA state.
                self._ema_params = {
                    name: param.data.clone().detach()
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                }
                n_tensors = len(self._ema_params)
                n_params = sum(p.numel() for p in self._ema_params.values())
                print(
                    f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                    f"(decay={decay})"
                )
                return  # first call = initialization only, no decay update

            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                # Move EMA buffer to the same device as the live param (handles multi-GPU setups)
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                ema.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @contextmanager
    def _ema_teacher_context(self, model):
        """Context manager that temporarily loads EMA weights for the teacher forward pass.

        Swaps `param.data` of every tracked (trainable) parameter with its EMA counterpart,
        runs the body (teacher forward), then restores the student weights unconditionally.
        Safe to use inside `torch.no_grad()`.  If EMA has not been initialized yet (step 0),
        this is a no-op and the current student weights are used instead.

        ZeRO-3 note: direct `param.data` assignment bypasses ZeRO-3's shard lifecycle and
        corrupts its internal state, causing size-mismatch errors during gradient-checkpoint
        recomputation.  When ZeRO-3 is active we therefore wrap the swap inside
        `deepspeed.zero.GatheredParameters` so the parameters are fully materialised on every
        rank before we touch them, and ZeRO-3 re-partitions cleanly when the context exits.
        """
        if self._ema_params is None:
            yield  # EMA not yet initialized; fall back to current weights
            return

        unwrapped = self.accelerator.unwrap_model(model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            name_to_param = {
                name: param
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and name in self._ema_params
            }
            params_list = list(name_to_param.values())

            # modifier_rank=0 causes ZeRO-3 to re-partition from rank-0's param.data on exit,
            # which will be the restored student weights.
            with deepspeed.zero.GatheredParameters(params_list, modifier_rank=0):
                saved = {}
                for name, param in name_to_param.items():
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    saved[name] = param.data.clone()
                    param.data.copy_(ema)
                try:
                    yield
                finally:
                    for name, param in name_to_param.items():
                        if name in saved:
                            param.data.copy_(saved[name])
        else:
            saved = {}
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                saved[name] = param.data
                param.data = ema
            try:
                yield
            finally:
                for name, param in unwrapped.named_parameters():
                    if name in saved:
                        param.data = saved[name]

    def _snapshot_current_policy_as_best(self):
        """Freeze the current LoRA weights as the best-policy anchor on every rank."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        self._best_anchor_params = {
            name: param.data.clone().detach()
            for name, param in unwrapped.named_parameters()
            if param.requires_grad
        }

    @contextmanager
    def _best_anchor_context(self, model):
        """Temporarily evaluate the student prompt under the frozen best policy."""
        unwrapped = self.accelerator.unwrap_model(model)
        if self._best_anchor_params is None:
            with unwrapped.disable_adapter():
                yield
            return

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        name_to_param = {
            name: param
            for name, param in unwrapped.named_parameters()
            if param.requires_grad and name in self._best_anchor_params
        }

        if zero_stage_3:
            import deepspeed

            with deepspeed.zero.GatheredParameters(list(name_to_param.values()), modifier_rank=0):
                saved = {name: param.data.clone() for name, param in name_to_param.items()}
                for name, param in name_to_param.items():
                    param.data.copy_(self._best_anchor_params[name].to(param.data.device))
                try:
                    yield
                finally:
                    for name, param in name_to_param.items():
                        param.data.copy_(saved[name])
        else:
            saved = {}
            for name, param in name_to_param.items():
                anchor = self._best_anchor_params[name]
                if anchor.device != param.data.device:
                    anchor = anchor.to(param.data.device)
                    self._best_anchor_params[name] = anchor
                saved[name] = param.data
                param.data = anchor
            try:
                yield
            finally:
                for name, param in name_to_param.items():
                    param.data = saved[name]

    @staticmethod
    def _extract_boxed_answer(text: str):
        idx = text.rfind("\\boxed")
        if idx < 0:
            return None

        i = idx
        num_left_braces = 0
        right_brace_idx = None
        while i < len(text):
            if text[i] == "{":
                num_left_braces += 1
            elif text[i] == "}":
                num_left_braces -= 1
                if num_left_braces == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            return None

        boxed_str = text[idx : right_brace_idx + 1]
        if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
            return boxed_str[7:-1].strip()
        return None

    @staticmethod
    def _normalize_answer_for_reward(answer: str | None) -> str:
        if answer is None:
            return ""
        return re.sub(r"\s+", "", answer.replace("$", "")).lower().strip()

    def _grade_boxed_answer_for_reward(self, predicted: str | None, ground_truth: str | None) -> bool:
        if predicted is None or ground_truth is None:
            return False

        if self._normalize_answer_for_reward(predicted) == self._normalize_answer_for_reward(ground_truth):
            return True

        try:
            from math_verify import parse, verify

            pred_latex = predicted if "$" in predicted else f"${predicted}$"
            gt_latex = ground_truth if "$" in ground_truth else f"${ground_truth}$"
            pred_parsed = parse(pred_latex, fallback_mode="no_fallback")
            gt_parsed = parse(gt_latex, fallback_mode="no_fallback")
            return bool(verify(gt_parsed, pred_parsed, timeout_seconds=2))
        except Exception:
            return False

    @staticmethod
    def _last_boxed_content_span(text: str):
        boxed_index = text.rfind("\\boxed")
        if boxed_index < 0:
            return None
        opening = text.find("{", boxed_index)
        if opening < 0:
            return None
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return opening + 1, index
        return None

    @staticmethod
    def _change_numeric_literal(value: str) -> str:
        plain = value.replace(",", "")
        delta = random.choice([-10, -2, -1, 1, 2, 10])
        try:
            if "." in plain:
                decimals = len(plain.rsplit(".", 1)[1])
                changed = f"{float(plain) + delta:.{decimals}f}"
            else:
                changed = str(int(plain) + delta)
                if "," in value:
                    changed = f"{int(changed):,}"
            if value.startswith("+") and not changed.startswith("-"):
                changed = "+" + changed.lstrip("+")
            return changed
        except ValueError:
            return value

    @staticmethod
    def _replace_random_number_occurrences(text: str, rate: float):
        if rate <= 0:
            return text, 0
        pattern = re.compile(
            r"(?<![A-Za-z\\])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return text, 0
        count = min(len(matches), max(1, int(round(rate * len(matches)))))
        selected = random.sample(matches, count)
        replacements = [
            (match.start(), match.end(), OPSDTrainer._change_numeric_literal(match.group(0)))
            for match in selected
        ]
        for start, end, replacement in sorted(replacements, reverse=True):
            text = text[:start] + replacement + text[end:]
        return text, len(replacements)

    @staticmethod
    def _replace_random_connector_occurrences(text: str, rate: float):
        if rate <= 0:
            return text, 0
        connectors = (
            "however",
            "therefore",
            "thus",
            "hence",
            "consequently",
            "moreover",
            "furthermore",
            "nevertheless",
            "instead",
            "although",
            "because",
            "since",
            "then",
            "finally",
            "conversely",
            "similarly",
        )
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(word) for word in connectors) + r")\b",
            flags=re.IGNORECASE,
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return text, 0
        count = min(len(matches), max(1, int(round(rate * len(matches)))))
        selected = random.sample(matches, count)
        replacements = []
        for match in selected:
            original = match.group(0)
            choices = [word for word in connectors if word.lower() != original.lower()]
            replacement = random.choice(choices)
            if original.isupper():
                replacement = replacement.upper()
            elif original[:1].isupper():
                replacement = replacement.capitalize()
            replacements.append((match.start(), match.end(), replacement))
        for start, end, replacement in sorted(replacements, reverse=True):
            text = text[:start] + replacement + text[end:]
        return text, len(replacements)

    def _force_incorrect_last_boxed_answer(self, text: str, ground_truth: str):
        current_answer = self._extract_boxed_answer(text)
        if current_answer is None:
            return text, False, None
        if not self._grade_boxed_answer_for_reward(current_answer, ground_truth):
            return text, False, current_answer

        candidates = []
        stripped = current_answer.strip()
        choice_match = re.fullmatch(r"(?:\\text\w*\{)?\(?([A-E])\)?(?:\})?", stripped)
        if choice_match:
            current_choice = choice_match.group(1)
            candidates.extend(choice for choice in "ABCDE" if choice != current_choice)
        if stripped.lower() in {"yes", "no", "true", "false"}:
            candidates.append(
                {"yes": "No", "no": "Yes", "true": "False", "false": "True"}[
                    stripped.lower()
                ]
            )

        number_match = re.search(
            r"(?<![A-Za-z\\])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?",
            stripped,
        )
        if number_match:
            changed_number = self._change_numeric_literal(number_match.group(0))
            candidates.append(
                stripped[: number_match.start()]
                + changed_number
                + stripped[number_match.end() :]
            )
        candidates.extend(["0", "1", "-1", "2", "42", "\\text{None}"])

        span = self._last_boxed_content_span(text)
        if span is None:
            return text, False, current_answer
        start, end = span
        for candidate in candidates:
            if candidate == current_answer:
                continue
            changed_text = text[:start] + candidate + text[end:]
            changed_answer = self._extract_boxed_answer(changed_text)
            if not self._grade_boxed_answer_for_reward(changed_answer, ground_truth):
                return changed_text, True, changed_answer
        return text, False, current_answer

    def _perturb_change_to_wrong_completion(
        self,
        completion: str,
        ground_truth: str,
        force_incorrect_boxed_answer: bool,
    ):
        changed, number_replacements = self._replace_random_number_occurrences(
            completion, self.change_to_wrong_number_rate
        )
        changed, connector_replacements = self._replace_random_connector_occurrences(
            changed, self.change_to_wrong_connector_rate
        )
        if force_incorrect_boxed_answer:
            changed, boxed_answer_forced, target_answer = (
                self._force_incorrect_last_boxed_answer(changed, ground_truth)
            )
        else:
            boxed_answer_forced = False
            target_answer = self._extract_boxed_answer(changed)
        return {
            "text": changed,
            "number_replacements": number_replacements,
            "connector_replacements": connector_replacements,
            "boxed_answer_forced": boxed_answer_forced,
            "target_answer": target_answer,
        }

    def _select_localized_numeric_corruption(self, completion: str):
        """Replace one pre-answer number and cut the trajectory immediately after it."""
        boxed_index = completion.rfind("\\boxed")
        reasoning_text = completion[:boxed_index] if boxed_index >= 0 else completion
        number_pattern = re.compile(
            r"(?<![A-Za-z\\])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
        )
        matches = list(number_pattern.finditer(reasoning_text))
        if not matches:
            return None, "no_reasoning_number"

        max_prefix_tokens = (
            int(self.generation_config.max_new_tokens)
            - int(self.localized_error_recovery_tokens)
        )
        candidates = []
        for match in matches:
            original_number = match.group(0)
            replacement_number = self._change_numeric_literal(original_number)
            if replacement_number == original_number:
                continue
            clean_prefix = completion[: match.end()]
            corrupted_prefix = completion[: match.start()] + replacement_number
            prefix_ids = self.processing_class.encode(
                corrupted_prefix,
                add_special_tokens=False,
            )
            prefix_token_count = len(prefix_ids)
            if prefix_token_count < int(self.localized_error_recovery_min_prefix_tokens):
                continue
            if prefix_token_count > max_prefix_tokens:
                continue
            candidates.append(
                {
                    "clean_prefix": clean_prefix,
                    "corrupted_prefix": corrupted_prefix,
                    "original_number": original_number,
                    "replacement_number": replacement_number,
                    "number_char_start": match.start(),
                    "number_char_end": match.end(),
                    "prefix_token_count": prefix_token_count,
                }
            )

        if not candidates:
            return None, "no_number_within_prefix_budget"
        return random.choice(candidates), "selected"

    def _tokenize_localized_recovery_prompts(self, prompt_texts, recovery_tokens):
        previous_padding_side = self.processing_class.padding_side
        try:
            self.processing_class.padding_side = "right"
            encoded = self.processing_class(
                prompt_texts,
                padding="longest",
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
        finally:
            self.processing_class.padding_side = previous_padding_side

        prompt_width = int(encoded["input_ids"].shape[1])
        if prompt_width + int(recovery_tokens) > int(self.args.max_length):
            raise ValueError(
                "A localized error-recovery sequence exceeds max_length: "
                f"prompt_width={prompt_width}, recovery_tokens={recovery_tokens}, "
                f"max_length={self.args.max_length}. Increase MAX_LENGTH."
            )
        return {
            "input_ids": encoded["input_ids"].to(self.accelerator.device),
            "attention_mask": encoded["attention_mask"].to(self.accelerator.device),
        }

    def _generate_localized_student_recoveries(self, model, prompt_texts):
        """Generate student continuations from a selected set of corrupted prefixes."""
        if not prompt_texts:
            return None, [], []

        recovery_tokens = int(self.localized_error_recovery_tokens)
        recovery_config = copy.deepcopy(self.generation_config)
        recovery_config.max_new_tokens = recovery_tokens
        encoded = self._tokenize_localized_recovery_prompts(
            prompt_texts,
            recovery_tokens,
        )
        generation_inputs = {
            "localized_recovery_prompts": encoded["input_ids"],
            "localized_recovery_prompt_attention_mask": encoded["attention_mask"],
        }

        if self.use_vllm:
            if self.vllm_mode != "colocate" or self.vllm_tensor_parallel_size != 1:
                raise RuntimeError(
                    "Localized error recovery currently requires colocated vLLM with "
                    "vllm_tensor_parallel_size=1."
                )
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                generation_inputs,
                recovery_config,
                self.processing_class.pad_token_id,
                prompt_key="localized_recovery_prompts",
                attention_mask_key="localized_recovery_prompt_attention_mask",
            )
            generated_ids, _, _, _, completion_texts = result
            completion_ids = generated_ids[:, -recovery_tokens:]
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                generated_ids, _, _ = self.generate_on_policy_outputs(
                    unwrapped_model,
                    generation_inputs,
                    recovery_config,
                    self.processing_class.pad_token_id,
                    prompt_key="localized_recovery_prompts",
                    attention_mask_key="localized_recovery_prompt_attention_mask",
                )
            prompt_width = encoded["input_ids"].shape[1]
            completion_ids = generated_ids[:, prompt_width:]
            completion_texts = self.processing_class.batch_decode(
                completion_ids,
                skip_special_tokens=False,
            )

        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id or 0
        if completion_ids.shape[1] < recovery_tokens:
            completion_ids = torch.cat(
                [
                    completion_ids,
                    torch.full(
                        (completion_ids.shape[0], recovery_tokens - completion_ids.shape[1]),
                        pad_token_id,
                        dtype=completion_ids.dtype,
                        device=completion_ids.device,
                    ),
                ],
                dim=1,
            )
        elif completion_ids.shape[1] > recovery_tokens:
            completion_ids = completion_ids[:, :recovery_tokens]

        token_counts = (
            (completion_ids != pad_token_id).sum(dim=1).detach().cpu().tolist()
        )
        return completion_ids, completion_texts, token_counts

    def _prepare_localized_error_recovery_batch(
        self,
        model,
        inputs,
        initial_completion_texts,
    ):
        """Build context-only corrupted prefixes followed by student recovery tokens."""
        device = inputs["student_prompts"].device
        batch_size = len(initial_completion_texts)
        solutions = inputs.get("reward_solutions", [])
        if len(solutions) != batch_size:
            raise RuntimeError(
                "Localized error recovery requires one reference solution per rollout."
            )

        base_student_prompt_texts = [
            self._clean_decoded_prompt(
                self.processing_class.decode(prompt_ids, skip_special_tokens=False)
            )
            for prompt_ids in inputs["student_prompts"]
        ]
        base_teacher_prompt_texts = [
            self._clean_decoded_prompt(
                self.processing_class.decode(prompt_ids, skip_special_tokens=False)
            )
            for prompt_ids in inputs["teacher_prompts"]
        ]

        predicted_answers = []
        ground_truth_answers = []
        outcomes = []
        corruption_records = [None] * batch_size
        selected_indices = []
        recovery_student_prompt_texts = []

        for index, (completion, solution) in enumerate(
            zip(initial_completion_texts, solutions)
        ):
            predicted_answer = self._extract_boxed_answer(completion or "")
            ground_truth_answer = self._extract_boxed_answer(solution or "")
            predicted_answers.append(predicted_answer)
            ground_truth_answers.append(ground_truth_answer)
            if predicted_answer is None:
                outcomes.append("no_boxed_skipped")
                continue
            if not self._grade_boxed_answer_for_reward(
                predicted_answer,
                ground_truth_answer,
            ):
                outcomes.append("wrong_boxed_skipped")
                continue

            corruption, selection_outcome = self._select_localized_numeric_corruption(
                completion or ""
            )
            if corruption is None:
                outcomes.append(selection_outcome)
                continue
            corruption_records[index] = corruption
            outcomes.append("correct_boxed_selected")
            selected_indices.append(index)
            recovery_student_prompt_texts.append(
                base_student_prompt_texts[index] + corruption["corrupted_prefix"]
            )

        selected_completion_ids, selected_completion_texts, selected_token_counts = (
            self._generate_localized_student_recoveries(
                model,
                recovery_student_prompt_texts,
            )
        )

        recovery_tokens = int(self.localized_error_recovery_tokens)
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id or 0
        recovery_ids = torch.full(
            (batch_size, recovery_tokens),
            pad_token_id,
            dtype=inputs["student_prompts"].dtype,
            device=device,
        )
        recovery_completion_texts = [None] * batch_size
        recovery_token_counts = [0] * batch_size
        recovery_answers = [None] * batch_size
        recovery_correct = [None] * batch_size
        for selected_row, index in enumerate(selected_indices):
            recovery_ids[index] = selected_completion_ids[selected_row].to(
                device=device,
                dtype=recovery_ids.dtype,
            )
            recovery_text = selected_completion_texts[selected_row]
            recovery_completion_texts[index] = recovery_text
            recovery_token_counts[index] = int(selected_token_counts[selected_row])
            recovery_answer = self._extract_boxed_answer(recovery_text or "")
            recovery_answers[index] = recovery_answer
            recovery_correct[index] = self._grade_boxed_answer_for_reward(
                recovery_answer,
                ground_truth_answers[index],
            )

        loss_student_prompt_texts = list(base_student_prompt_texts)
        loss_teacher_prompt_texts = list(base_teacher_prompt_texts)
        for index in selected_indices:
            corrupted_prefix = corruption_records[index]["corrupted_prefix"]
            loss_student_prompt_texts[index] += corrupted_prefix
            loss_teacher_prompt_texts[index] += corrupted_prefix

        student_encoded = self._tokenize_localized_recovery_prompts(
            loss_student_prompt_texts,
            recovery_tokens,
        )
        teacher_encoded = self._tokenize_localized_recovery_prompts(
            loss_teacher_prompt_texts,
            recovery_tokens,
        )
        student_prompt_lengths = student_encoded["attention_mask"].sum(dim=1)
        teacher_prompt_lengths = teacher_encoded["attention_mask"].sum(dim=1)

        inputs["student_prompts"] = student_encoded["input_ids"]
        inputs["student_prompt_attention_mask"] = student_encoded["attention_mask"]
        inputs["student_prompt_length"] = int(student_encoded["input_ids"].shape[1])
        inputs["student_prompt_lengths_per_example"] = student_prompt_lengths
        inputs["teacher_prompts"] = teacher_encoded["input_ids"]
        inputs["teacher_prompt_attention_mask"] = teacher_encoded["attention_mask"]
        inputs["teacher_prompt_length"] = int(teacher_encoded["input_ids"].shape[1])
        inputs["teacher_prompt_lengths_per_example"] = teacher_prompt_lengths

        student_generated_ids = torch.cat(
            [inputs["student_prompts"], recovery_ids],
            dim=1,
        )
        student_attention_mask = torch.ones_like(student_generated_ids)
        student_attention_mask[student_generated_ids == pad_token_id] = 0

        mode = "train" if self.model.training else "eval"
        selected_count = len(selected_indices)
        self._metrics[mode]["localized_recovery_selected_count"].append(
            float(selected_count)
        )
        self._metrics[mode]["localized_recovery_selected_rate"].append(
            float(selected_count / max(1, batch_size))
        )
        if selected_count:
            self._metrics[mode]["localized_recovery_avg_prefix_tokens"].append(
                float(
                    sum(
                        corruption_records[index]["prefix_token_count"]
                        for index in selected_indices
                    )
                    / selected_count
                )
            )
            self._metrics[mode]["localized_recovery_avg_continuation_tokens"].append(
                float(
                    sum(recovery_token_counts[index] for index in selected_indices)
                    / selected_count
                )
            )
            self._metrics[mode]["localized_recovery_boxed_rate"].append(
                float(
                    sum(recovery_answers[index] is not None for index in selected_indices)
                    / selected_count
                )
            )
            self._metrics[mode]["localized_recovery_correct_rate"].append(
                float(
                    sum(bool(recovery_correct[index]) for index in selected_indices)
                    / selected_count
                )
            )

        return (
            student_generated_ids,
            student_attention_mask,
            recovery_ids,
            {
                "predicted_answers": predicted_answers,
                "ground_truth_answers": ground_truth_answers,
                "outcomes": outcomes,
                "corruptions": corruption_records,
                "recovery_completion_texts": recovery_completion_texts,
                "recovery_token_counts": recovery_token_counts,
                "recovery_answers": recovery_answers,
                "recovery_correct": recovery_correct,
            },
        )

    def _sample_vllm_completion_ids(
        self,
        prompts: list[str] | list[list[int]],
        generation_config: GenerationConfig,
        num_samples: int,
        prompts_are_token_ids: bool = False,
    ) -> list[list[list[int]]]:
        """Sample one or more completions for each local prompt from colocated vLLM."""
        if self.vllm_mode != "colocate" or self.vllm_tensor_parallel_size != 1:
            raise RuntimeError(
                "Multi-sample local generation requires colocated vLLM with tensor_parallel_size=1."
            )

        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        guided_decoding = (
            GuidedDecodingParams(backend="outlines", regex=self.vllm_guided_decoding_regex)
            if self.vllm_guided_decoding_regex
            else None
        )
        sampling_params = SamplingParams(
            n=num_samples,
            repetition_penalty=getattr(self.args, "repetition_penalty", 1.0),
            temperature=generation_config.temperature,
            top_p=getattr(self.args, "top_p", 1.0),
            top_k=top_k,
            min_p=getattr(self.args, "min_p", 0.0),
            max_tokens=generation_config.max_new_tokens,
            presence_penalty=getattr(self.args, "presence_penalty", 0.0),
            guided_decoding=guided_decoding,
        )
        vllm_prompts = (
            [
                {"prompt_token_ids": [int(token_id) for token_id in prompt_ids]}
                for prompt_ids in prompts
            ]
            if prompts_are_token_ids
            else prompts
        )
        outputs = self.vllm_engine.generate(
            vllm_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return [
            [list(completion.token_ids) for completion in request_output.outputs]
            for request_output in outputs
        ]

    def _build_student_batch_from_completion_ids(
        self,
        inputs,
        completion_ids,
        pad_token_id,
        max_completion_length=None,
    ):
        device = self.accelerator.device
        max_completion_length = int(
            self.generation_config.max_new_tokens
            if max_completion_length is None
            else max_completion_length
        )
        completion = torch.tensor(completion_ids[:max_completion_length], dtype=torch.long, device=device)
        if completion.numel() < max_completion_length:
            completion = torch.cat(
                [
                    completion,
                    torch.full(
                        (max_completion_length - completion.numel(),),
                        pad_token_id,
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )
        completion = completion.unsqueeze(0)
        generated_ids = torch.cat([inputs["student_prompts"], completion], dim=1)
        attention_mask = torch.ones_like(generated_ids)
        attention_mask[generated_ids == pad_token_id] = 0
        labels = generated_ids.clone()
        labels[generated_ids == pad_token_id] = -100
        return generated_ids, attention_mask, labels

    def _set_single_loss_prompt(
        self,
        inputs,
        prompt_key,
        prompt_text,
        max_completion_length=None,
    ):
        """Replace a one-example loss prompt while preserving the full completion budget."""
        max_completion_length = int(
            self.generation_config.max_new_tokens
            if max_completion_length is None
            else max_completion_length
        )
        max_prompt_length = int(self.args.max_length) - max_completion_length
        if max_prompt_length <= 0:
            raise ValueError(
                "max_length must be larger than max_completion_length for independent verification."
            )

        encoded = self.processing_class(
            prompt_text,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        prompt_ids = encoded["input_ids"].to(self.accelerator.device)
        attention_mask = encoded["attention_mask"].to(self.accelerator.device)
        prompt_length = int(prompt_ids.shape[1])
        if prompt_length > max_prompt_length:
            raise ValueError(
                f"The {prompt_key} verification prompt has {prompt_length} tokens, but only "
                f"{max_prompt_length} prompt tokens fit within max_length={self.args.max_length} "
                f"and max_completion_length={max_completion_length}. Increase max_length."
            )

        inputs[f"{prompt_key}_prompts"] = prompt_ids
        inputs[f"{prompt_key}_prompt_attention_mask"] = attention_mask
        inputs[f"{prompt_key}_prompt_length"] = prompt_length
        inputs[f"{prompt_key}_prompt_lengths_per_example"] = torch.tensor(
            [prompt_length], dtype=torch.long, device=self.accelerator.device
        )

    def _set_single_loss_prompt_ids(
        self,
        inputs,
        prompt_key,
        prompt_ids,
        max_completion_length,
    ):
        """Install one already-tokenized prompt without a decode/re-tokenize round trip."""
        max_completion_length = int(max_completion_length)
        max_prompt_length = int(self.args.max_length) - max_completion_length
        if max_prompt_length <= 0:
            raise ValueError(
                "max_length must be larger than the prefix-consistency suffix length."
            )

        prompt_ids = [int(token_id) for token_id in prompt_ids]
        prompt_length = len(prompt_ids)
        if prompt_length == 0:
            raise ValueError(f"The {prompt_key} prefix-consistency prompt is empty.")
        if prompt_length > max_prompt_length:
            raise ValueError(
                f"The {prompt_key} prefix-consistency prompt has {prompt_length} tokens, but only "
                f"{max_prompt_length} prompt tokens fit within max_length={self.args.max_length} "
                f"and suffix length={max_completion_length}. Increase max_length."
            )

        prompt_tensor = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.accelerator.device,
        )
        attention_mask = torch.ones_like(prompt_tensor)
        inputs[f"{prompt_key}_prompts"] = prompt_tensor
        inputs[f"{prompt_key}_prompt_attention_mask"] = attention_mask
        inputs[f"{prompt_key}_prompt_length"] = prompt_length
        inputs[f"{prompt_key}_prompt_lengths_per_example"] = torch.tensor(
            [prompt_length],
            dtype=torch.long,
            device=self.accelerator.device,
        )

    def _clean_decoded_prompt(self, prompt_text):
        if self.processing_class.pad_token:
            return prompt_text.replace(self.processing_class.pad_token, "")
        return prompt_text

    @staticmethod
    def _prefix_consistency_answer_status(text, answer):
        if answer is not None and str(answer).strip():
            return "boxed"
        if "\\boxed" in (text or ""):
            return "malformed_boxed"
        return "no_boxed"

    def _prefix_consistency_prefix_length(self, completion_length):
        """Return the paper-defined ceil(tau * |completion|) natural-prefix length."""
        completion_length = int(completion_length)
        if completion_length <= 0:
            return 0
        return max(
            1,
            math.ceil(self.prefix_consistency_fraction * completion_length),
        )

    @staticmethod
    def _prefix_consistency_outcome_advantages(gold_flags):
        """Center binary rewards within one prefix's K regenerated suffixes."""
        if not gold_flags:
            return []
        rewards = [float(bool(flag)) for flag in gold_flags]
        mean_reward = sum(rewards) / len(rewards)
        return [reward - mean_reward for reward in rewards]

    @staticmethod
    def _prefix_consistency_outcome_policy_loss(
        student_logits,
        labels,
        advantage,
        temperature,
    ):
        """Return -A times suffix mean log-probability and the detached-free mean NLL."""
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive for the outcome policy loss.")
        token_nll = F.cross_entropy(
            (student_logits / temperature).transpose(1, 2),
            labels,
            reduction="none",
            ignore_index=-100,
        )
        token_mask = labels != -100
        sequence_nll = (
            token_nll * token_mask.to(token_nll.dtype)
        ).sum() / token_mask.sum().clamp_min(1)
        advantage_tensor = torch.as_tensor(
            advantage,
            dtype=sequence_nll.dtype,
            device=sequence_nll.device,
        ).detach()
        return advantage_tensor * sequence_nll, sequence_nll

    def _cluster_prefix_consistency_answers(self, source_answers, denominator):
        """Group parseable answers by the same equivalence check used for reward grading."""
        groups = []
        for source, answer in source_answers:
            if answer is None or not str(answer).strip():
                continue
            matched_group = None
            for group in groups:
                if self._grade_boxed_answer_for_reward(
                    answer,
                    group["representative"],
                ):
                    matched_group = group
                    break
            if matched_group is None:
                matched_group = {
                    "representative": answer,
                    "representative_normalized": self._normalize_answer_for_reward(answer),
                    "sources": [],
                    "count": 0,
                }
                groups.append(matched_group)
            matched_group["sources"].append(source)
            matched_group["count"] += 1

        for group in groups:
            group["consistency"] = group["count"] / max(1, denominator)
        return groups

    def _prefix_consistency_uses_wrong_boxed_only(self):
        return getattr(
            self,
            "prefix_consistency_selection_mode",
            "all",
        ) in {
            "wrong_boxed_all",
            "wrong_boxed_mixed_all",
        }

    def _prefix_consistency_requires_mixed_outcomes(self):
        return (
            getattr(self, "prefix_consistency_selection_mode", "all")
            == "wrong_boxed_mixed_all"
        )

    def _prefix_consistency_loss_scale(
        self,
        local_eligible_count,
        local_batch_size,
    ):
        """Return the per-branch batch mean scale and global eligible-prefix count."""
        local_batch_size = int(local_batch_size)
        if local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive.")
        local_eligible_count = int(local_eligible_count)
        if not 0 <= local_eligible_count <= local_batch_size:
            raise ValueError(
                "local_eligible_count must be between zero and local_batch_size."
            )
        eligible_count = torch.tensor(
            [local_eligible_count],
            dtype=torch.float32,
            device=self.accelerator.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(eligible_count, op=dist.ReduceOp.SUM)
        global_eligible_count = int(eligible_count.item())
        if getattr(self, "prefix_consistency_normalize_by_eligible", False):
            # DDP averages rank gradients by world_size. Multiplying each active
            # branch by world_size/(global_eligible*K) therefore yields the exact
            # global conditional mean over eligible prefixes for this distributed
            # microbatch. If no rank has an eligible prefix, every slot is zero.
            world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )
            scale = (
                world_size
                / (
                    global_eligible_count
                    * self.prefix_consistency_num_regenerations
                )
                if global_eligible_count > 0
                else 0.0
            )
        else:
            # Fixed full-batch normalization preserves the legacy objective and
            # treats filtered prefixes as zero-weight examples.
            scale = 1.0 / (
                local_batch_size * self.prefix_consistency_num_regenerations
            )
        return float(scale), global_eligible_count

    def _build_prefix_consistency_loss_inputs(
        self,
        base_inputs,
        loss_spec,
        pad_token_id,
    ):
        """Build one suffix-only OPSD branch while keeping K branches memory-sequential."""
        branch_inputs = dict(base_inputs)
        suffix_tokens = int(self.prefix_consistency_suffix_tokens)
        # Ineligible ranks still execute K backward passes so DDP/DeepSpeed remain
        # synchronized, but one masked token is enough for their differentiable zero.
        branch_completion_tokens = (
            suffix_tokens if float(loss_spec["loss_scale"]) > 0.0 else 1
        )
        self._set_single_loss_prompt_ids(
            branch_inputs,
            "student",
            loss_spec["student_prompt_ids"],
            max_completion_length=branch_completion_tokens,
        )
        self._set_single_loss_prompt_ids(
            branch_inputs,
            "teacher",
            loss_spec["teacher_prompt_ids"],
            max_completion_length=branch_completion_tokens,
        )

        generated_ids, generated_attention_mask, _ = (
            self._build_student_batch_from_completion_ids(
                branch_inputs,
                loss_spec["completion_ids"],
                pad_token_id,
                max_completion_length=branch_completion_tokens,
            )
        )
        student_prompt_len = branch_inputs["student_prompt_length"]
        generation_ids = generated_ids[:, student_prompt_len:]
        teacher_full_ids = torch.cat(
            [branch_inputs["teacher_prompts"], generation_ids],
            dim=1,
        )
        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        teacher_attention_mask[teacher_full_ids == pad_token_id] = 0

        labels = generated_ids.clone()
        labels[:, :student_prompt_len] = -100
        labels[generated_ids == pad_token_id] = -100
        if float(loss_spec["loss_scale"]) == 0.0:
            labels[:, student_prompt_len:] = -100

        branch_inputs["student_input_ids"] = generated_ids
        branch_inputs["student_attention_mask"] = generated_attention_mask
        branch_inputs["teacher_input_ids"] = teacher_full_ids
        branch_inputs["teacher_attention_mask"] = teacher_attention_mask
        branch_inputs["labels"] = labels
        branch_inputs["prefix_consistency_loss_scale"] = torch.tensor(
            [loss_spec["loss_scale"]],
            dtype=torch.float32,
            device=self.accelerator.device,
        )
        branch_inputs["prefix_consistency_outcome_advantage"] = torch.tensor(
            [loss_spec["outcome_advantage"]],
            dtype=torch.float32,
            device=self.accelerator.device,
        )
        return branch_inputs

    def _generate_prefix_consistency_rollouts_vllm(self, inputs):
        """Generate one initial trace and K suffixes for every local example."""
        batch_size = int(inputs["student_prompts"].shape[0])
        if batch_size <= 0:
            raise RuntimeError("Prefix-consistency received an empty local batch.")

        # Every rank executes exactly B*K backward calls, including masked calls
        # for skipped samples. Fail together if local batch sizes differ.
        if dist.is_available() and dist.is_initialized():
            minimum_batch_size = torch.tensor(
                [batch_size], dtype=torch.int64, device=self.accelerator.device
            )
            maximum_batch_size = minimum_batch_size.clone()
            dist.all_reduce(minimum_batch_size, op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum_batch_size, op=dist.ReduceOp.MAX)
            if int(minimum_batch_size.item()) != int(maximum_batch_size.item()):
                raise RuntimeError(
                    "Prefix-consistency requires the same local batch size on every rank; "
                    f"observed min={int(minimum_batch_size.item())}, "
                    f"max={int(maximum_batch_size.item())}."
                )

        problems = list(inputs.get("reward_problems", []))
        reference_solutions = list(inputs.get("reward_solutions", []))
        if len(problems) != batch_size or len(reference_solutions) != batch_size:
            raise RuntimeError(
                "Prefix-consistency distillation requires one problem and reference solution per batch item."
            )

        student_prompt_lengths = [
            int(value)
            for value in inputs["student_prompt_lengths_per_example"]
            .detach()
            .cpu()
            .tolist()
        ]
        teacher_prompt_lengths = [
            int(value)
            for value in inputs["teacher_prompt_lengths_per_example"]
            .detach()
            .cpu()
            .tolist()
        ]
        if (
            len(student_prompt_lengths) != batch_size
            or len(teacher_prompt_lengths) != batch_size
        ):
            raise RuntimeError(
                "Prefix-consistency prompt-length metadata does not match the local batch size."
            )

        original_student_prompt_ids_batch = []
        original_teacher_prompt_ids_batch = []
        original_student_prompts = []
        ground_truth_answers = []
        ground_truth_valid_flags = []
        for sample_index in range(batch_size):
            student_prompt_length = student_prompt_lengths[sample_index]
            teacher_prompt_length = teacher_prompt_lengths[sample_index]
            if not 0 < student_prompt_length <= inputs["student_prompts"].shape[1]:
                raise RuntimeError(
                    f"Invalid student prompt length for sample {sample_index}: "
                    f"{student_prompt_length}."
                )
            if not 0 < teacher_prompt_length <= inputs["teacher_prompts"].shape[1]:
                raise RuntimeError(
                    f"Invalid teacher prompt length for sample {sample_index}: "
                    f"{teacher_prompt_length}."
                )

            student_prompt_ids = (
                inputs["student_prompts"][sample_index, :student_prompt_length]
                .detach()
                .cpu()
                .tolist()
            )
            teacher_prompt_ids = (
                inputs["teacher_prompts"][sample_index, :teacher_prompt_length]
                .detach()
                .cpu()
                .tolist()
            )
            original_student_prompt_ids_batch.append(student_prompt_ids)
            original_teacher_prompt_ids_batch.append(teacher_prompt_ids)
            original_student_prompts.append(
                self._clean_decoded_prompt(
                    self.processing_class.decode(
                        student_prompt_ids,
                        skip_special_tokens=False,
                    )
                )
            )
            ground_truth_answer = self._extract_boxed_answer(
                reference_solutions[sample_index]
            )
            ground_truth_answers.append(ground_truth_answer)
            ground_truth_valid_flags.append(
                bool(
                    ground_truth_answer is not None
                    and self._normalize_answer_for_reward(ground_truth_answer)
                )
            )

        invalid_reference_count = torch.tensor(
            [sum(not flag for flag in ground_truth_valid_flags)],
            dtype=torch.int64,
            device=self.accelerator.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(invalid_reference_count, op=dist.ReduceOp.SUM)
        global_invalid_reference_count = int(invalid_reference_count.item())

        initial_groups = self._sample_vllm_completion_ids(
            original_student_prompt_ids_batch,
            self.generation_config,
            1,
            prompts_are_token_ids=True,
        )
        if len(initial_groups) != batch_size or any(
            len(group) != 1 for group in initial_groups
        ):
            raise RuntimeError(
                "Expected exactly one initial prefix-consistency rollout per local example."
            )

        wrong_boxed_only = self._prefix_consistency_uses_wrong_boxed_only()
        sample_states = []
        eligible_prompt_ids = []
        eligible_sample_indices = []
        for sample_index in range(batch_size):
            initial_ids = list(initial_groups[sample_index][0])
            initial_completion = self.processing_class.decode(
                initial_ids,
                skip_special_tokens=False,
            )
            initial_answer = self._extract_boxed_answer(initial_completion)
            initial_parse_status = self._prefix_consistency_answer_status(
                initial_completion,
                initial_answer,
            )
            ground_truth_answer = ground_truth_answers[sample_index]
            ground_truth_is_valid = ground_truth_valid_flags[sample_index]
            initial_correct = bool(
                ground_truth_is_valid
                and self._grade_boxed_answer_for_reward(
                    initial_answer,
                    ground_truth_answer,
                )
            )
            if not ground_truth_is_valid:
                initial_status = "invalid_reference"
            elif initial_parse_status != "boxed":
                initial_status = initial_parse_status
            elif initial_correct:
                initial_status = "correct_boxed"
            else:
                initial_status = "wrong_boxed"

            # The wrong-boxed-only ablation admits only initially wrong boxed
            # trajectories. Legacy mode keeps correct+wrong boxed trajectories
            # and its optional no-box extension.
            if wrong_boxed_only:
                eligible = bool(
                    ground_truth_is_valid
                    and initial_status == "wrong_boxed"
                )
            else:
                eligible = bool(
                    ground_truth_is_valid
                    and (
                        initial_parse_status == "boxed"
                        or self.prefix_consistency_include_no_boxed
                    )
                )
            prefix_ids = []
            prefix_text = None
            prefix_token_count = 0
            prefix_contains_box = False
            skip_reason = None
            if eligible:
                prefix_token_count = self._prefix_consistency_prefix_length(
                    len(initial_ids)
                )
                prefix_ids = list(initial_ids[:prefix_token_count])
                prefix_text = self.processing_class.decode(
                    prefix_ids,
                    skip_special_tokens=False,
                )
                prefix_contains_box = "\\boxed" in prefix_text
            if not eligible and skip_reason is None:
                skip_reason = (
                    "invalid_reference"
                    if not ground_truth_is_valid
                    else initial_status
                )

            loss_student_prompt_ids = list(
                original_student_prompt_ids_batch[sample_index]
            )
            loss_teacher_prompt_ids = list(
                original_teacher_prompt_ids_batch[sample_index]
            )
            if eligible:
                loss_student_prompt_ids.extend(prefix_ids)
                loss_teacher_prompt_ids.extend(prefix_ids)
                eligible_sample_indices.append(sample_index)
                eligible_prompt_ids.append(loss_student_prompt_ids)

            sample_states.append(
                {
                    "sample_index": sample_index,
                    "initial_ids": initial_ids,
                    "initial_completion": initial_completion,
                    "initial_answer": initial_answer,
                    "initial_parse_status": initial_parse_status,
                    "initial_correct": initial_correct,
                    "initial_status": initial_status,
                    "ground_truth_answer": ground_truth_answer,
                    "ground_truth_is_valid": ground_truth_is_valid,
                    "eligible": eligible,
                    "prefix_ids": prefix_ids,
                    "prefix_text": prefix_text,
                    "prefix_token_count": prefix_token_count,
                    "prefix_contains_box": prefix_contains_box,
                    "skip_reason": skip_reason,
                    "loss_student_prompt_ids": loss_student_prompt_ids,
                    "loss_teacher_prompt_ids": loss_teacher_prompt_ids,
                    "suffix_ids_group": [],
                }
            )

        if eligible_prompt_ids:
            suffix_config = copy.deepcopy(self.generation_config)
            suffix_config.max_new_tokens = int(
                self.prefix_consistency_suffix_tokens
            )
            suffix_groups = self._sample_vllm_completion_ids(
                eligible_prompt_ids,
                suffix_config,
                self.prefix_consistency_num_regenerations,
                prompts_are_token_ids=True,
            )
            if len(suffix_groups) != len(eligible_prompt_ids) or any(
                len(group) != self.prefix_consistency_num_regenerations
                for group in suffix_groups
            ):
                raise RuntimeError(
                    "vLLM did not return K prefix-conditioned regenerations "
                    "for every eligible local example."
                )
            for sample_index, suffix_ids_group in zip(
                eligible_sample_indices,
                suffix_groups,
            ):
                sample_states[sample_index]["suffix_ids_group"] = [
                    list(suffix_ids) for suffix_ids in suffix_ids_group
                ]

        # Outcome-conditioned selection must be decided before the distributed
        # loss denominator is computed.  Decode and grade every regenerated
        # candidate once, retain the records for logging below, and distinguish
        # initial eligibility (the suffixes were sampled) from loss selection
        # (the group contributes gradients).
        k = self.prefix_consistency_num_regenerations
        mixed_outcomes_required = (
            self._prefix_consistency_requires_mixed_outcomes()
        )
        for state in sample_states:
            regeneration_records = []
            regeneration_answers = []
            regeneration_gold_flags = []
            regeneration_original_match_flags = []
            if state["eligible"]:
                for branch_index, suffix_ids in enumerate(
                    state["suffix_ids_group"]
                ):
                    suffix_text = self.processing_class.decode(
                        suffix_ids,
                        skip_special_tokens=False,
                    )
                    full_completion = self.processing_class.decode(
                        state["prefix_ids"] + list(suffix_ids),
                        skip_special_tokens=False,
                    )
                    regenerated_answer = self._extract_boxed_answer(
                        full_completion
                    )
                    regenerated_status = (
                        self._prefix_consistency_answer_status(
                            full_completion,
                            regenerated_answer,
                        )
                    )
                    is_gold = self._grade_boxed_answer_for_reward(
                        regenerated_answer,
                        state["ground_truth_answer"],
                    )
                    matches_original = (
                        self._grade_boxed_answer_for_reward(
                            regenerated_answer,
                            state["initial_answer"],
                        )
                        if state["initial_parse_status"] == "boxed"
                        else None
                    )
                    regeneration_answers.append(regenerated_answer)
                    regeneration_gold_flags.append(bool(is_gold))
                    if matches_original is not None:
                        regeneration_original_match_flags.append(
                            bool(matches_original)
                        )
                    regeneration_records.append(
                        {
                            "branch_index": branch_index,
                            "suffix_completion": suffix_text,
                            "suffix_token_ids": [
                                int(token_id) for token_id in suffix_ids
                            ],
                            "full_regenerated_completion": full_completion,
                            "answer": regenerated_answer,
                            "answer_normalized": (
                                self._normalize_answer_for_reward(
                                    regenerated_answer
                                )
                            ),
                            "status": regenerated_status,
                            "is_gold": bool(is_gold),
                            "matches_original": matches_original,
                            "token_count": len(suffix_ids),
                            "finish_reason_inferred": (
                                "length"
                                if len(suffix_ids)
                                >= self.prefix_consistency_suffix_tokens
                                else "stop"
                            ),
                        }
                    )

            regeneration_gold_count = (
                sum(regeneration_gold_flags)
                if state["eligible"]
                else None
            )
            mixed_outcome = bool(
                state["eligible"]
                and 0 < regeneration_gold_count < k
            )
            selected_for_loss = bool(
                state["eligible"]
                and (
                    not mixed_outcomes_required
                    or mixed_outcome
                )
            )
            if (
                state["eligible"]
                and mixed_outcomes_required
                and not selected_for_loss
            ):
                state["skip_reason"] = (
                    "all_regenerations_non_gold"
                    if regeneration_gold_count == 0
                    else "all_regenerations_gold"
                )

            state["regeneration_records"] = regeneration_records
            state["regeneration_answers"] = regeneration_answers
            state["regeneration_gold_flags"] = regeneration_gold_flags
            state["regeneration_original_match_flags"] = (
                regeneration_original_match_flags
            )
            state["regeneration_gold_count"] = regeneration_gold_count
            state["mixed_outcome"] = mixed_outcome
            state["selected_for_loss"] = selected_for_loss

        local_selected_count = sum(
            int(state["selected_for_loss"])
            for state in sample_states
        )
        branch_loss_scale, global_eligible_count = (
            self._prefix_consistency_loss_scale(
                local_selected_count,
                batch_size,
            )
        )
        loss_specs = []
        records = []
        mode = "train" if self.model.training else "eval"

        for state in sample_states:
            sample_index = state["sample_index"]
            eligible = state["eligible"]
            initial_ids = state["initial_ids"]
            initial_completion = state["initial_completion"]
            initial_answer = state["initial_answer"]
            initial_parse_status = state["initial_parse_status"]
            initial_correct = state["initial_correct"]
            initial_status = state["initial_status"]
            ground_truth_answer = state["ground_truth_answer"]
            ground_truth_is_valid = state["ground_truth_is_valid"]
            prefix_ids = state["prefix_ids"]
            prefix_text = state["prefix_text"]
            prefix_token_count = state["prefix_token_count"]
            prefix_contains_box = state["prefix_contains_box"]
            loss_student_prompt_ids = state["loss_student_prompt_ids"]
            loss_teacher_prompt_ids = state["loss_teacher_prompt_ids"]
            suffix_ids_group = state["suffix_ids_group"]
            selected_for_loss = state["selected_for_loss"]
            loss_scale = (
                branch_loss_scale if selected_for_loss else 0.0
            )

            regeneration_records = state["regeneration_records"]
            regeneration_answers = state["regeneration_answers"]
            regeneration_gold_flags = state["regeneration_gold_flags"]
            regeneration_original_match_flags = state[
                "regeneration_original_match_flags"
            ]
            if eligible:
                # Records were decoded and graded in the pre-pass so the
                # mixed-outcome selection and its global denominator use the
                # exact same outcomes that are logged here.
                regeneration_records = [
                    dict(record) for record in regeneration_records
                ]
            else:
                regeneration_records = [
                    {
                        "branch_index": branch_index,
                        "suffix_completion": None,
                        "suffix_token_ids": [],
                        "full_regenerated_completion": None,
                        "answer": None,
                        "answer_normalized": "",
                        "status": "not_generated",
                        "is_gold": None,
                        "matches_original": None,
                        "token_count": 0,
                        "finish_reason_inferred": None,
                    }
                    for branch_index in range(k)
                ]

            outcome_advantages = (
                self._prefix_consistency_outcome_advantages(
                    regeneration_gold_flags
                )
                if eligible
                else [0.0] * k
            )
            outcome_rewards = (
                [float(flag) for flag in regeneration_gold_flags]
                if eligible
                else [None] * k
            )
            for branch_index, regeneration_record in enumerate(
                regeneration_records
            ):
                regeneration_record["outcome_reward"] = outcome_rewards[
                    branch_index
                ]
                regeneration_record["outcome_advantage"] = outcome_advantages[
                    branch_index
                ]

            for branch_index in range(k):
                completion_ids = (
                    suffix_ids_group[branch_index] if eligible else []
                )
                loss_specs.append(
                    {
                        "sample_index": sample_index,
                        "branch_index": branch_index,
                        "student_prompt_ids": loss_student_prompt_ids,
                        "teacher_prompt_ids": loss_teacher_prompt_ids,
                        "completion_ids": completion_ids,
                        "loss_scale": loss_scale,
                        "outcome_advantage": outcome_advantages[branch_index],
                    }
                )

            reproduction_count = (
                sum(regeneration_original_match_flags)
                if eligible and initial_parse_status == "boxed"
                else None
            )
            regeneration_gold_count = (
                sum(regeneration_gold_flags) if eligible else None
            )
            regeneration_parseable_count = (
                sum(
                    self._prefix_consistency_answer_status("", answer)
                    == "boxed"
                    for answer in regeneration_answers
                )
                if eligible
                else None
            )
            original_consistency = (
                (1 + reproduction_count) / (k + 1)
                if eligible and reproduction_count is not None
                else None
            )
            original_reproduction_rate = (
                reproduction_count / k
                if eligible and reproduction_count is not None
                else None
            )
            regeneration_gold_value = (
                regeneration_gold_count / k if eligible else None
            )
            group_gold_count = (
                int(initial_correct) + regeneration_gold_count
                if eligible
                else None
            )
            group_gold_value = (
                group_gold_count / (k + 1) if eligible else None
            )
            source_answers = [("original", initial_answer)] + [
                (f"regen_{index}", answer)
                for index, answer in enumerate(regeneration_answers)
            ]
            candidate_groups = (
                self._cluster_prefix_consistency_answers(
                    source_answers,
                    k + 1,
                )
                if eligible
                else []
            )
            outcome_active = bool(
                eligible and 0 < regeneration_gold_count < k
            )
            mean_abs_outcome_advantage = (
                sum(abs(value) for value in outcome_advantages) / k
                if eligible
                else 0.0
            )

            record = {
                "step": self.state.global_step,
                "rollout_step": self.state.global_step + 1,
                "component": "prefix_consistency_distillation",
                "microbatch_sample_index": sample_index,
                "local_microbatch_size": batch_size,
                "problem": problems[sample_index],
                "prompt": original_student_prompts[sample_index],
                "student_prompt_token_count": len(
                    original_student_prompt_ids_batch[sample_index]
                ),
                "teacher_prompt_token_count": len(
                    original_teacher_prompt_ids_batch[sample_index]
                ),
                "completion": initial_completion,
                "original_completion": initial_completion,
                "original_completion_token_count": len(initial_ids),
                "original_answer": initial_answer,
                "original_answer_normalized": (
                    self._normalize_answer_for_reward(initial_answer)
                ),
                "original_status": initial_status,
                "original_is_gold": (
                    bool(initial_correct) if ground_truth_is_valid else None
                ),
                "ground_truth_answer": ground_truth_answer,
                "reference_answer_is_valid": bool(ground_truth_is_valid),
                "global_invalid_references_in_microbatch": (
                    global_invalid_reference_count
                ),
                "prefix_fraction": self.prefix_consistency_fraction,
                "prefix_token_count": prefix_token_count,
                "prefix_token_ids": [int(token_id) for token_id in prefix_ids],
                "prefix_text": prefix_text,
                "conditioned_student_prompt_token_count": len(
                    loss_student_prompt_ids
                ),
                "conditioned_teacher_prompt_token_count": len(
                    loss_teacher_prompt_ids
                ),
                "prefix_contains_boxed_marker": prefix_contains_box,
                "num_regenerations": k,
                "suffix_max_tokens": self.prefix_consistency_suffix_tokens,
                "regenerations": regeneration_records,
                "candidate_groups": candidate_groups,
                "original_reproduction_count": reproduction_count,
                "original_reproduction_rate": original_reproduction_rate,
                "original_consistency": original_consistency,
                "regeneration_gold_count": regeneration_gold_count,
                "regeneration_gold_value": regeneration_gold_value,
                "group_gold_count": group_gold_count,
                "group_gold_value": group_gold_value,
                "regeneration_parseable_count": regeneration_parseable_count,
                "regeneration_boxed_rate": (
                    regeneration_parseable_count / k if eligible else None
                ),
                "include_no_boxed_initial": (
                    self.prefix_consistency_include_no_boxed
                ),
                "selection_mode": self.prefix_consistency_selection_mode,
                "normalize_by_eligible": (
                    self.prefix_consistency_normalize_by_eligible
                ),
                "outcome_alpha": self.prefix_consistency_outcome_alpha,
                "outcome_rewards": outcome_rewards,
                "outcome_advantages": outcome_advantages,
                "outcome_active": outcome_active,
                "mean_abs_outcome_advantage": mean_abs_outcome_advantage,
                "used_for_outcome_loss": bool(
                    outcome_active
                    and self.prefix_consistency_outcome_alpha > 0.0
                ),
                "initially_eligible_for_regeneration": bool(eligible),
                "mixed_outcome_selected": bool(
                    state["mixed_outcome"]
                    and mixed_outcomes_required
                ),
                "used_for_loss": bool(selected_for_loss),
                "loss_scale_per_branch": loss_scale,
                "global_eligible_prefixes_in_microbatch": (
                    global_eligible_count
                ),
                "global_selected_prefixes_in_microbatch": (
                    global_eligible_count
                ),
                "skip_reason": state["skip_reason"],
            }
            records.append(record)

            count_metrics = {
                "prefix_consistency_initial_count": 1,
                "prefix_consistency_initial_boxed_count": int(
                    initial_parse_status == "boxed"
                ),
                "prefix_consistency_initial_correct_count": int(
                    initial_status == "correct_boxed"
                ),
                "prefix_consistency_initial_wrong_count": int(
                    initial_status == "wrong_boxed"
                ),
                "prefix_consistency_initial_no_boxed_count": int(
                    initial_parse_status == "no_boxed"
                ),
                "prefix_consistency_initial_malformed_boxed_count": int(
                    initial_parse_status == "malformed_boxed"
                ),
                "prefix_consistency_invalid_reference_count": int(
                    not ground_truth_is_valid
                ),
                "prefix_consistency_prefix_box_leak_count": int(
                    prefix_contains_box
                ),
                "prefix_consistency_initial_eligible_count": int(eligible),
                "prefix_consistency_selected_count": int(
                    selected_for_loss
                ),
                "prefix_consistency_mixed_selected_count": int(
                    mixed_outcomes_required and selected_for_loss
                ),
                "prefix_consistency_outcome_active_group_count": int(
                    outcome_active
                ),
                "prefix_consistency_regeneration_count": k if eligible else 0,
                "prefix_consistency_regeneration_boxed_count": (
                    regeneration_parseable_count if eligible else 0
                ),
                "prefix_consistency_regeneration_gold_count": (
                    regeneration_gold_count if eligible else 0
                ),
                "prefix_consistency_correct_regeneration_count": (
                    k if eligible and initial_status == "correct_boxed" else 0
                ),
                "prefix_consistency_correct_reproduction_count": (
                    reproduction_count
                    if eligible and initial_status == "correct_boxed"
                    else 0
                ),
                "prefix_consistency_correct_gold_count": (
                    regeneration_gold_count
                    if eligible and initial_status == "correct_boxed"
                    else 0
                ),
                "prefix_consistency_wrong_regeneration_count": (
                    k if eligible and initial_status == "wrong_boxed" else 0
                ),
                "prefix_consistency_wrong_reproduction_count": (
                    reproduction_count
                    if eligible and initial_status == "wrong_boxed"
                    else 0
                ),
                "prefix_consistency_wrong_gold_count": (
                    regeneration_gold_count
                    if eligible and initial_status == "wrong_boxed"
                    else 0
                ),
                "prefix_consistency_unboxed_initial_regeneration_count": (
                    k if eligible and initial_parse_status != "boxed" else 0
                ),
                "prefix_consistency_unboxed_initial_gold_count": (
                    regeneration_gold_count
                    if eligible and initial_parse_status != "boxed"
                    else 0
                ),
            }
            for key, value in count_metrics.items():
                self._metrics[mode][key].append(float(value))
            self._metrics[mode][
                "prefix_consistency_global_eligible_count"
            ].append(float(global_eligible_count))
            self._metrics[mode][
                "prefix_consistency_global_selected_count"
            ].append(float(global_eligible_count))
            self._metrics[mode][
                "prefix_consistency_loss_scale_per_branch"
            ].append(float(loss_scale))
            if eligible:
                self._metrics[mode][
                    "prefix_consistency_initial_tokens"
                ].append(float(len(initial_ids)))
                self._metrics[mode][
                    "prefix_consistency_prefix_tokens"
                ].append(float(prefix_token_count))
                self._metrics[mode][
                    "prefix_consistency_suffix_tokens"
                ].append(
                    float(
                        sum(
                            item["token_count"]
                            for item in regeneration_records
                        )
                        / k
                    )
                )
                self._metrics[mode][
                    "prefix_consistency_regeneration_gold_value"
                ].append(float(regeneration_gold_value))
                self._metrics[mode][
                    "prefix_consistency_group_gold_value"
                ].append(float(group_gold_value))
                self._metrics[mode][
                    "prefix_consistency_mean_abs_outcome_advantage"
                ].append(float(mean_abs_outcome_advantage))
                if initial_parse_status == "boxed":
                    self._metrics[mode][
                        "prefix_consistency_original_consistency"
                    ].append(float(original_consistency))
                    self._metrics[mode][
                        "prefix_consistency_original_reproduction"
                    ].append(float(original_reproduction_rate))
                    status_prefix = "correct" if initial_correct else "wrong"
                    self._metrics[mode][
                        f"prefix_consistency_{status_prefix}_original_consistency"
                    ].append(float(original_consistency))
                    self._metrics[mode][
                        f"prefix_consistency_{status_prefix}_regeneration_gold_value"
                    ].append(float(regeneration_gold_value))
                self._metrics[mode][
                    "prefix_consistency_mixed_gold_outcome"
                ].append(float(outcome_active))
                for gold_count in range(k + 1):
                    self._metrics[mode][
                        f"prefix_consistency_gold_count_{gold_count}_rate"
                    ].append(float(regeneration_gold_count == gold_count))

        return loss_specs, records

    def _build_verification_prompt_pair(self, problem, reference_solution, candidate_solution):
        verification_instruction = (
            "Verify the candidate using a different method. If you find any error, correct it. "
            "Then give the complete corrected solution and put the final answer within \\boxed{}."
        )
        student_message = (
            f"Problem: {problem}\n\n"
            "A candidate solution is shown below.\n"
            "=== Candidate Solution Begin ===\n"
            f"{candidate_solution}\n"
            "=== Candidate Solution End ===\n\n"
            f"{verification_instruction}"
        )
        teacher_message = (
            f"Problem: {problem}\n\n"
            "Here is a reference solution to this problem:\n"
            "=== Reference Solution Begin ===\n"
            f"{reference_solution}\n"
            "=== Reference Solution End ===\n\n"
            "A candidate solution is shown below.\n"
            "=== Candidate Solution Begin ===\n"
            f"{candidate_solution}\n"
            "=== Candidate Solution End ===\n\n"
            f"{verification_instruction}"
        )
        student_prompt = self.processing_class.apply_chat_template(
            [{"role": "user", "content": student_message}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.student_thinking,
        )
        teacher_prompt = self.processing_class.apply_chat_template(
            [{"role": "user", "content": teacher_message}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.teacher_thinking,
        )
        return student_prompt, teacher_prompt

    def _build_wrong_answer_correction_teacher_prompt(
        self, problem, reference_solution, candidate_solution
    ):
        correction_instruction = (
            "Repeat the student's response up to the point where it first goes wrong, then continue from there "
            "with a corrected solution. Please reason step by step, and put your final answer within \\boxed{}."
        )
        teacher_message = (
            f"Problem: {problem}\n\n"
            "Here is a reference solution to this problem:\n"
            "=== Reference Solution Begin ===\n"
            f"{reference_solution}\n"
            "=== Reference Solution End ===\n\n"
            "Here is the student's incorrect response:\n"
            "=== Incorrect Student Response Begin ===\n"
            f"{candidate_solution}\n"
            "=== Incorrect Student Response End ===\n\n"
            f"{correction_instruction}"
        )
        return self.processing_class.apply_chat_template(
            [{"role": "user", "content": teacher_message}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.teacher_thinking,
        )

    def _build_branch_error_locator_prompt(self, problem, reference_solution, candidate_solution):
        locator_message = (
            f"Problem: {problem}\n\n"
            "Here is a correct reference solution:\n"
            "=== Reference Solution Begin ===\n"
            f"{reference_solution}\n"
            "=== Reference Solution End ===\n\n"
            "Here is an incorrect student response:\n"
            "=== Incorrect Student Response Begin ===\n"
            f"{candidate_solution}\n"
            "=== Incorrect Student Response End ===\n\n"
            "Find the first mathematical error in the student response. Quote a short, exact, contiguous "
            "substring from the student response that begins at that first incorrect step. Do not correct the "
            "solution and do not paraphrase the quote. Output only the quote between these tags:\n"
            "<error_start_quote>exact verbatim quote</error_start_quote>"
        )
        return self.processing_class.apply_chat_template(
            [{"role": "user", "content": locator_message}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _build_branch_corrected_continuation_prompt(
        self,
        problem,
        reference_solution,
        correct_student_prefix,
        first_wrong_quote,
    ):
        continuation_message = (
            f"Problem: {problem}\n\n"
            "Here is a correct reference solution:\n"
            "=== Reference Solution Begin ===\n"
            f"{reference_solution}\n"
            "=== Reference Solution End ===\n\n"
            "The student's response is mathematically valid up to the following prefix, which ends immediately "
            "before the first incorrect step:\n"
            "=== Valid Student Prefix Begin ===\n"
            f"{correct_student_prefix}\n"
            "=== Valid Student Prefix End ===\n\n"
            "The first incorrect step originally began with this exact excerpt:\n"
            "=== First Incorrect Excerpt Begin ===\n"
            f"{first_wrong_quote}\n"
            "=== First Incorrect Excerpt End ===\n\n"
            "Continue directly from the valid student prefix with corrected reasoning. Do not repeat or restart "
            "the prefix. Please reason step by step, and put your final answer within \\boxed{}."
        )
        return self.processing_class.apply_chat_template(
            [{"role": "user", "content": continuation_message}],
            tokenize=False,
            add_generation_prompt=True,
            # The generated tokens will be spliced after an existing student prefix. Avoid a new <think> block.
            enable_thinking=False,
        )

    @staticmethod
    def _extract_branch_error_quote(locator_text):
        if not locator_text:
            return None
        match = re.search(
            r"<error_start_quote>\s*(.*?)\s*</error_start_quote>",
            locator_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            quote = match.group(1).strip()
        else:
            match = re.search(
                r"ERROR_START_QUOTE\s*:\s*(.+)", locator_text, flags=re.IGNORECASE
            )
            if not match:
                return None
            quote = match.group(1).strip()
        quote = quote.strip("`\"'").strip()
        if not quote or quote.lower() in {"none", "n/a", "no error"}:
            return None
        return quote

    @staticmethod
    def _match_branch_quote(candidate_text, quote):
        direct_index = candidate_text.find(quote)
        if direct_index >= 0:
            return direct_index, "exact"

        quote_parts = re.split(r"\s+", quote.strip())
        if not quote_parts:
            return None, None
        flexible_pattern = r"\s+".join(re.escape(part) for part in quote_parts if part)
        if not flexible_pattern:
            return None, None
        match = re.search(flexible_pattern, candidate_text, flags=re.DOTALL)
        return (match.start(), "whitespace_normalized") if match else (None, None)

    @staticmethod
    def _find_branch_quote_start(candidate_text, quote):
        """Backward-compatible convenience wrapper returning only the match position."""
        return OPSDTrainer._match_branch_quote(candidate_text, quote)[0]

    def _split_candidate_at_branch(self, candidate_text, error_char_start):
        encoded = self.processing_class(
            candidate_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        branch_token_index = None
        for index, (_, end) in enumerate(offsets):
            if end > error_char_start:
                branch_token_index = index
                break
        if branch_token_index is None or branch_token_index >= len(token_ids):
            return None

        prefix_ids = token_ids[:branch_token_index]
        wrong_ids = token_ids[
            branch_token_index : branch_token_index + self.branch_contrastive_tokens
        ]
        if not wrong_ids:
            return None
        prefix_text = self.processing_class.decode(prefix_ids, skip_special_tokens=False)
        return prefix_ids, wrong_ids, prefix_text, branch_token_index

    def _generate_fixed_teacher_correction_ids(
        self,
        model,
        teacher_prompt_texts,
        generation_config,
    ):
        """Generate corrected trajectories with the fixed/base teacher."""
        if not teacher_prompt_texts:
            return [], [], []

        max_completion_length = int(generation_config.max_new_tokens)
        max_prompt_length = int(self.args.max_length) - max_completion_length
        if max_prompt_length <= 0:
            raise ValueError(
                "max_length must be larger than max_completion_length for teacher correction generation."
            )

        previous_padding_side = self.processing_class.padding_side
        try:
            self.processing_class.padding_side = "left"
            encoded = self.processing_class(
                teacher_prompt_texts,
                padding="longest",
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
        finally:
            self.processing_class.padding_side = previous_padding_side

        prompt_width = int(encoded["input_ids"].shape[1])
        if prompt_width > max_prompt_length:
            raise ValueError(
                "A wrong-answer teacher-correction prompt has "
                f"{prompt_width} tokens, but only {max_prompt_length} prompt tokens fit within "
                f"max_length={self.args.max_length} and max_completion_length={max_completion_length}. "
                "Increase MAX_LENGTH."
            )

        generation_inputs = {
            "correction_teacher_prompts": encoded["input_ids"].to(self.accelerator.device),
            "correction_teacher_prompt_attention_mask": encoded["attention_mask"].to(
                self.accelerator.device
            ),
        }
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            adapter_context = (
                self.accelerator.unwrap_model(model).disable_adapter()
                if self.fixed_teacher and is_peft_model(model)
                else nullcontext()
            )
            with torch.no_grad(), adapter_context:
                generated_ids, _, _ = self.generate_on_policy_outputs(
                    unwrapped_model,
                    generation_inputs,
                    generation_config,
                    self.processing_class.pad_token_id,
                    prompt_key="correction_teacher_prompts",
                    attention_mask_key="correction_teacher_prompt_attention_mask",
                )

        padded_completion_ids = generated_ids[:, prompt_width:]
        pad_token_id = self.processing_class.pad_token_id
        completion_ids = []
        token_counts = []
        for row in padded_completion_ids:
            row_ids = row.detach().cpu().tolist()
            if pad_token_id is not None:
                while row_ids and row_ids[-1] == pad_token_id:
                    row_ids.pop()
            completion_ids.append(row_ids)
            token_counts.append(len(row_ids))
        completion_texts = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=False
        )
        return completion_ids, completion_texts, token_counts

    def _prepare_wrong_answer_teacher_correction_batch(
        self,
        model,
        inputs,
        initial_generation_ids,
        initial_completion_texts,
    ):
        """Replace wrong boxed rollouts with fixed-teacher corrected trajectories for the loss."""
        device = initial_generation_ids.device
        (
            weights,
            predicted_answers,
            ground_truth_answers,
            outcomes,
        ) = self._compute_wrong_boxed_only_weights(
            initial_completion_texts,
            inputs.get("reward_solutions", []),
            device,
            metric_prefix="wrong_answer_teacher_correction",
        )
        wrong_indices = [index for index, outcome in enumerate(outcomes) if outcome == "wrong_boxed"]
        batch_size = len(initial_completion_texts)
        if len(inputs.get("reward_problems", [])) != batch_size or len(
            inputs.get("reward_solutions", [])
        ) != batch_size:
            raise RuntimeError(
                "Wrong-answer teacher correction requires one problem and reference solution per rollout."
            )

        max_completion_length = int(self.generation_config.max_new_tokens)
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        if initial_generation_ids.shape[1] < max_completion_length:
            missing = max_completion_length - initial_generation_ids.shape[1]
            initial_generation_ids = torch.cat(
                [
                    initial_generation_ids,
                    torch.full(
                        (initial_generation_ids.shape[0], missing),
                        pad_token_id,
                        dtype=initial_generation_ids.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
        elif initial_generation_ids.shape[1] > max_completion_length:
            initial_generation_ids = initial_generation_ids[:, :max_completion_length]

        original_teacher_prompt_texts = [
            self._clean_decoded_prompt(
                self.processing_class.decode(prompt_ids, skip_special_tokens=False)
            )
            for prompt_ids in inputs["teacher_prompts"]
        ]
        loss_teacher_prompt_texts = list(original_teacher_prompt_texts)
        target_completion_texts = list(initial_completion_texts)
        target_generation_ids = initial_generation_ids.clone()
        correction_prompt_texts = [None] * len(initial_completion_texts)
        correction_completion_texts = [None] * len(initial_completion_texts)
        correction_token_counts = [0] * len(initial_completion_texts)

        if wrong_indices:
            prompts_to_generate = []
            for index in wrong_indices:
                candidate_solution = self.processing_class.decode(
                    initial_generation_ids[index], skip_special_tokens=True
                )
                correction_prompt = self._build_wrong_answer_correction_teacher_prompt(
                    inputs["reward_problems"][index],
                    inputs["reward_solutions"][index],
                    candidate_solution,
                )
                prompts_to_generate.append(correction_prompt)
                correction_prompt_texts[index] = correction_prompt

            correction_ids, generated_corrections, token_counts = (
                self._generate_fixed_teacher_correction_ids(
                    model, prompts_to_generate, self.generation_config
                )
            )
            for index, token_ids, correction_text, token_count in zip(
                wrong_indices, correction_ids, generated_corrections, token_counts
            ):
                token_ids = token_ids[:max_completion_length]
                padded_ids = token_ids + [pad_token_id] * (max_completion_length - len(token_ids))
                target_generation_ids[index] = torch.tensor(
                    padded_ids, dtype=target_generation_ids.dtype, device=device
                )
                loss_teacher_prompt_texts[index] = correction_prompt_texts[index]
                target_completion_texts[index] = correction_text
                correction_completion_texts[index] = correction_text
                correction_token_counts[index] = token_count

        max_prompt_length = int(self.args.max_length) - max_completion_length
        previous_padding_side = self.processing_class.padding_side
        try:
            self.processing_class.padding_side = "right"
            teacher_encoded = self.processing_class(
                loss_teacher_prompt_texts,
                padding="longest",
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
        finally:
            self.processing_class.padding_side = previous_padding_side
        teacher_prompt_length = int(teacher_encoded["input_ids"].shape[1])
        if teacher_prompt_length > max_prompt_length:
            raise ValueError(
                "A teacher-correction loss prompt has "
                f"{teacher_prompt_length} tokens, but only {max_prompt_length} prompt tokens fit. "
                "Increase MAX_LENGTH."
            )

        teacher_prompts = teacher_encoded["input_ids"].to(device)
        teacher_prompt_attention_mask = teacher_encoded["attention_mask"].to(device)
        teacher_prompt_lengths = teacher_prompt_attention_mask.sum(dim=1)
        student_generated_ids = torch.cat(
            [inputs["student_prompts"], target_generation_ids], dim=1
        )
        student_attention_mask = torch.ones_like(student_generated_ids)
        if self.processing_class.pad_token_id is not None:
            student_attention_mask[
                student_generated_ids == self.processing_class.pad_token_id
            ] = 0

        inputs["teacher_prompts"] = teacher_prompts
        inputs["teacher_prompt_attention_mask"] = teacher_prompt_attention_mask
        inputs["teacher_prompt_length"] = teacher_prompt_length
        inputs["teacher_prompt_lengths_per_example"] = teacher_prompt_lengths
        inputs["wrong_boxed_only_weights"] = weights

        mode = "train" if self.model.training else "eval"
        selected_count = len(wrong_indices)
        total_count = max(1, len(initial_completion_texts))
        self._metrics[mode]["wrong_answer_teacher_correction_selected_count"].append(
            float(selected_count)
        )
        self._metrics[mode]["wrong_answer_teacher_correction_selected_rate"].append(
            float(selected_count / total_count)
        )
        if selected_count:
            selected_lengths = [correction_token_counts[index] for index in wrong_indices]
            selected_texts = [correction_completion_texts[index] for index in wrong_indices]
            self._metrics[mode]["wrong_answer_teacher_correction_avg_tokens"].append(
                float(sum(selected_lengths) / selected_count)
            )
            self._metrics[mode]["wrong_answer_teacher_correction_boxed_rate"].append(
                float(
                    sum(self._extract_boxed_answer(text or "") is not None for text in selected_texts)
                    / selected_count
                )
            )

        metadata = {
            "weights": weights,
            "predicted_answers": predicted_answers,
            "ground_truth_answers": ground_truth_answers,
            "outcomes": outcomes,
            "correction_prompt_texts": correction_prompt_texts,
            "correction_completion_texts": correction_completion_texts,
            "correction_token_counts": correction_token_counts,
            "target_completion_texts": target_completion_texts,
        }
        return student_generated_ids, student_attention_mask, target_generation_ids, metadata

    def _prepare_change_to_wrong_batch(
        self,
        inputs,
        initial_generation_ids,
        initial_completion_texts,
    ):
        """Perturb no-box/correct trajectories and retain boxed incorrect ones."""
        device = initial_generation_ids.device
        batch_size = len(initial_completion_texts)
        solutions = inputs.get("reward_solutions", [])
        if len(solutions) != batch_size:
            raise RuntimeError(
                "Change-to-wrong distillation requires one reference solution per rollout."
            )

        max_completion_length = int(self.generation_config.max_new_tokens)
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        target_generation_ids = initial_generation_ids.clone()
        target_completion_texts = list(initial_completion_texts)
        predicted_answers = []
        ground_truth_answers = []
        target_answers = []
        outcomes = []
        weights = []
        number_replacements = []
        connector_replacements = []
        boxed_answer_forced = []

        for index, (completion, solution) in enumerate(
            zip(initial_completion_texts, solutions)
        ):
            predicted_answer = self._extract_boxed_answer(completion or "")
            ground_truth_answer = self._extract_boxed_answer(solution or "")
            predicted_answers.append(predicted_answer)
            ground_truth_answers.append(ground_truth_answer)

            if predicted_answer is None:
                if not self.change_to_wrong_include_no_boxed:
                    outcomes.append("no_boxed_skipped")
                    weights.append(0.0)
                    target_answers.append(None)
                    number_replacements.append(0)
                    connector_replacements.append(0)
                    boxed_answer_forced.append(False)
                    continue

                corruption = self._perturb_change_to_wrong_completion(
                    completion,
                    ground_truth_answer,
                    force_incorrect_boxed_answer=False,
                )
                encoded = self.processing_class(
                    corruption["text"],
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"][:max_completion_length]
                padded = encoded + [pad_token_id] * (max_completion_length - len(encoded))
                decoded_target = self.processing_class.decode(
                    encoded, skip_special_tokens=False
                )
                target_generation_ids[index] = torch.tensor(
                    padded,
                    dtype=target_generation_ids.dtype,
                    device=device,
                )
                target_completion_texts[index] = decoded_target
                outcomes.append("no_boxed_changed")
                weights.append(1.0)
                target_answers.append(self._extract_boxed_answer(decoded_target))
                number_replacements.append(corruption["number_replacements"])
                connector_replacements.append(corruption["connector_replacements"])
                boxed_answer_forced.append(False)
                continue

            original_is_correct = self._grade_boxed_answer_for_reward(
                predicted_answer, ground_truth_answer
            )
            if not original_is_correct:
                outcomes.append("wrong_boxed_unchanged")
                weights.append(1.0)
                target_answers.append(predicted_answer)
                number_replacements.append(0)
                connector_replacements.append(0)
                boxed_answer_forced.append(False)
                continue

            corruption = self._perturb_change_to_wrong_completion(
                completion,
                ground_truth_answer,
                force_incorrect_boxed_answer=True,
            )
            encoded = self.processing_class(
                corruption["text"],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"][:max_completion_length]
            padded = encoded + [pad_token_id] * (max_completion_length - len(encoded))
            decoded_target = self.processing_class.decode(
                encoded, skip_special_tokens=False
            )
            decoded_target_answer = self._extract_boxed_answer(decoded_target)
            corruption_succeeded = (
                decoded_target_answer is not None
                and not self._grade_boxed_answer_for_reward(
                    decoded_target_answer, ground_truth_answer
                )
            )

            number_replacements.append(corruption["number_replacements"])
            connector_replacements.append(corruption["connector_replacements"])
            boxed_answer_forced.append(corruption["boxed_answer_forced"])
            target_answers.append(decoded_target_answer)
            if corruption_succeeded:
                target_generation_ids[index] = torch.tensor(
                    padded,
                    dtype=target_generation_ids.dtype,
                    device=device,
                )
                target_completion_texts[index] = decoded_target
                outcomes.append("correct_boxed_changed_to_wrong")
                weights.append(1.0)
            else:
                outcomes.append("correct_boxed_corruption_failed")
                weights.append(0.0)

        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        student_generated_ids = torch.cat(
            [inputs["student_prompts"], target_generation_ids], dim=1
        )
        student_attention_mask = torch.ones_like(student_generated_ids)
        if self.processing_class.pad_token_id is not None:
            student_attention_mask[
                student_generated_ids == self.processing_class.pad_token_id
            ] = 0

        mode = "train" if self.model.training else "eval"
        total = max(1, batch_size)
        changed_outcomes = {
            "correct_boxed_changed_to_wrong",
            "no_boxed_changed",
        }
        changed_count = sum(outcome in changed_outcomes for outcome in outcomes)
        original_correct_count = sum(
            outcome
            in {"correct_boxed_changed_to_wrong", "correct_boxed_corruption_failed"}
            for outcome in outcomes
        )
        original_wrong_count = sum(
            outcome == "wrong_boxed_unchanged"
            for outcome in outcomes
        )
        no_boxed_count = sum(
            outcome in {"no_boxed_skipped", "no_boxed_changed"}
            for outcome in outcomes
        )
        self._metrics[mode]["change_to_wrong_original_correct_rate"].append(
            original_correct_count / total
        )
        self._metrics[mode]["change_to_wrong_original_wrong_rate"].append(
            original_wrong_count / total
        )
        self._metrics[mode]["change_to_wrong_no_boxed_rate"].append(
            no_boxed_count / total
        )
        self._metrics[mode]["change_to_wrong_used_rate"].append(
            float(weight_tensor.mean().item())
        )
        eligible_corruption_count = original_correct_count + (
            no_boxed_count if self.change_to_wrong_include_no_boxed else 0
        )
        self._metrics[mode]["change_to_wrong_corruption_success_count"].append(
            float(changed_count)
        )
        self._metrics[mode]["change_to_wrong_corruption_eligible_count"].append(
            float(eligible_corruption_count)
        )
        self._metrics[mode]["change_to_wrong_modified_no_boxed_rate"].append(
            sum(outcome == "no_boxed_changed" for outcome in outcomes)
            / total
        )
        if changed_count:
            changed_indices = [
                index
                for index, outcome in enumerate(outcomes)
                if outcome in changed_outcomes
            ]
            self._metrics[mode]["change_to_wrong_avg_number_replacements"].append(
                sum(number_replacements[index] for index in changed_indices)
                / changed_count
            )
            self._metrics[mode]["change_to_wrong_avg_connector_replacements"].append(
                sum(connector_replacements[index] for index in changed_indices)
                / changed_count
            )
            self._metrics[mode]["change_to_wrong_forced_boxed_answer_rate"].append(
                sum(boxed_answer_forced[index] for index in changed_indices)
                / changed_count
            )

        metadata = {
            "weights": weight_tensor,
            "predicted_answers": predicted_answers,
            "ground_truth_answers": ground_truth_answers,
            "target_answers": target_answers,
            "outcomes": outcomes,
            "target_completion_texts": target_completion_texts,
            "number_replacements": number_replacements,
            "connector_replacements": connector_replacements,
            "boxed_answer_forced": boxed_answer_forced,
        }
        return (
            student_generated_ids,
            student_attention_mask,
            target_generation_ids,
            metadata,
        )

    def _prepare_wrong_answer_branch_contrastive_batch(
        self,
        model,
        inputs,
        initial_completion_texts,
    ):
        """Locate the first error and build correct/wrong continuation pairs from one exact prefix."""
        device = inputs["student_prompts"].device
        batch_size = len(initial_completion_texts)
        (
            _,
            predicted_answers,
            ground_truth_answers,
            outcomes,
        ) = self._compute_wrong_boxed_only_weights(
            initial_completion_texts,
            inputs.get("reward_solutions", []),
            device,
            metric_prefix="wrong_answer_branch_contrastive_initial",
        )
        if len(inputs.get("reward_problems", [])) != batch_size or len(
            inputs.get("reward_solutions", [])
        ) != batch_size:
            raise RuntimeError(
                "Wrong-answer branch contrastive mode requires one problem and reference solution per rollout."
            )

        wrong_indices = [index for index, outcome in enumerate(outcomes) if outcome == "wrong_boxed"]
        locator_prompts = [None] * batch_size
        locator_outputs = [None] * batch_size
        error_quotes = [None] * batch_size
        error_quote_parsed = [False] * batch_size
        error_quote_found = [False] * batch_size
        error_quote_match_modes = [None] * batch_size
        error_char_starts = [None] * batch_size
        branch_token_indices = [None] * batch_size
        prefix_ids_by_index = [[] for _ in range(batch_size)]
        prefix_texts = [None] * batch_size
        wrong_ids_by_index = [[] for _ in range(batch_size)]
        wrong_continuation_texts = [None] * batch_size
        correction_prompts = [None] * batch_size
        correction_ids_by_index = [[] for _ in range(batch_size)]
        correction_texts = [None] * batch_size
        skip_reasons = [outcome for outcome in outcomes]

        if wrong_indices:
            prompts = []
            for index in wrong_indices:
                prompt = self._build_branch_error_locator_prompt(
                    inputs["reward_problems"][index],
                    inputs["reward_solutions"][index],
                    initial_completion_texts[index],
                )
                prompts.append(prompt)
                locator_prompts[index] = prompt

            locator_config = copy.deepcopy(self.generation_config)
            locator_config.max_new_tokens = int(self.branch_error_locator_max_tokens)
            locator_config.do_sample = False
            locator_config.temperature = None
            locator_config.top_p = None
            locator_config.top_k = None
            _, generated_locator_texts, _ = self._generate_fixed_teacher_correction_ids(
                model, prompts, locator_config
            )

            for index, locator_text in zip(wrong_indices, generated_locator_texts):
                locator_outputs[index] = locator_text
                quote = self._extract_branch_error_quote(locator_text)
                if quote is None:
                    skip_reasons[index] = "locator_parse_failed"
                    continue
                error_quotes[index] = quote
                error_quote_parsed[index] = True
                error_start, match_mode = self._match_branch_quote(
                    initial_completion_texts[index], quote
                )
                if error_start is None:
                    skip_reasons[index] = "locator_quote_not_found"
                    continue
                error_quote_found[index] = True
                error_quote_match_modes[index] = match_mode
                error_char_starts[index] = error_start
                split = self._split_candidate_at_branch(
                    initial_completion_texts[index], error_start
                )
                if split is None:
                    skip_reasons[index] = "branch_split_failed"
                    continue
                prefix_ids, wrong_ids, prefix_text, branch_token_index = split
                prefix_ids_by_index[index] = prefix_ids
                prefix_texts[index] = prefix_text
                wrong_ids_by_index[index] = wrong_ids
                wrong_continuation_texts[index] = self.processing_class.decode(
                    wrong_ids, skip_special_tokens=False
                )
                branch_token_indices[index] = branch_token_index
                correction_prompts[index] = self._build_branch_corrected_continuation_prompt(
                    inputs["reward_problems"][index],
                    inputs["reward_solutions"][index],
                    prefix_text,
                    quote,
                )
                skip_reasons[index] = "pending_correction"

        correction_indices = [
            index for index, prompt in enumerate(correction_prompts) if prompt is not None
        ]
        if correction_indices:
            correction_config = copy.deepcopy(self.generation_config)
            correction_config.max_new_tokens = int(self.branch_contrastive_tokens)
            correction_ids, generated_correction_texts, _ = (
                self._generate_fixed_teacher_correction_ids(
                    model,
                    [correction_prompts[index] for index in correction_indices],
                    correction_config,
                )
            )
            for index, token_ids, correction_text in zip(
                correction_indices, correction_ids, generated_correction_texts
            ):
                wrong_text = wrong_continuation_texts[index] or ""
                leading_whitespace = re.match(r"\s*", wrong_text).group(0)
                if not leading_whitespace:
                    leading_whitespace = " "
                correction_body = correction_text.lstrip()
                combined_text = prefix_texts[index] + leading_whitespace + correction_body
                combined_ids = self.processing_class.encode(
                    combined_text, add_special_tokens=False
                )
                prefix_ids = prefix_ids_by_index[index]
                if combined_ids[: len(prefix_ids)] == prefix_ids:
                    token_ids = combined_ids[len(prefix_ids) :]
                else:
                    # This fallback remains well-formed even if decode/encode does not reproduce
                    # the exact prefix tokenization for an unusual special-token boundary.
                    token_ids = self.processing_class.encode(
                        leading_whitespace + correction_body,
                        add_special_tokens=False,
                    )
                token_ids = token_ids[: self.branch_contrastive_tokens]
                if not token_ids:
                    skip_reasons[index] = "empty_teacher_correction"
                    continue
                correction_ids_by_index[index] = token_ids
                correction_texts[index] = self.processing_class.decode(
                    token_ids, skip_special_tokens=False
                )
                skip_reasons[index] = "used_for_loss"

        sample_weights = torch.tensor(
            [float(reason == "used_for_loss") for reason in skip_reasons],
            dtype=torch.float32,
            device=device,
        )
        fallback_token_id = self.processing_class.eos_token_id
        if fallback_token_id is None:
            fallback_token_id = self.processing_class.pad_token_id or 0

        good_sequences = []
        bad_sequences = []
        good_labels = []
        bad_labels = []
        for index in range(batch_size):
            prompt_mask = inputs["student_prompt_attention_mask"][index].bool()
            prompt_ids = inputs["student_prompts"][index][prompt_mask].detach().tolist()
            prefix_ids = prefix_ids_by_index[index]
            good_ids = correction_ids_by_index[index] or [fallback_token_id]
            bad_ids = wrong_ids_by_index[index] or [fallback_token_id]
            common_ids = prompt_ids + prefix_ids
            good_sequence = common_ids + good_ids
            bad_sequence = common_ids + bad_ids
            if max(len(good_sequence), len(bad_sequence)) > int(self.args.max_length):
                raise ValueError(
                    "A branch contrastive scoring sequence exceeds max_length. Increase MAX_LENGTH."
                )
            good_sequences.append(good_sequence)
            bad_sequences.append(bad_sequence)
            good_labels.append([-100] * len(common_ids) + good_ids)
            bad_labels.append([-100] * len(common_ids) + bad_ids)

        all_sequences = good_sequences + bad_sequences
        all_labels = good_labels + bad_labels
        max_sequence_length = max(len(sequence) for sequence in all_sequences)
        input_ids = torch.full(
            (2 * batch_size, max_sequence_length),
            self.processing_class.pad_token_id or 0,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for row, (sequence, row_labels) in enumerate(zip(all_sequences, all_labels)):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(row_labels, dtype=torch.long, device=device)

        inputs["branch_contrastive_input_ids"] = input_ids
        inputs["branch_contrastive_attention_mask"] = attention_mask
        inputs["branch_contrastive_labels"] = labels
        inputs["branch_contrastive_sample_weights"] = sample_weights

        mode = "train" if self.model.training else "eval"
        selected_count = int(sample_weights.sum().item())
        local_quote_stats = {
            "wrong": len(wrong_indices),
            "parsed": sum(error_quote_parsed[index] for index in wrong_indices),
            "found": sum(error_quote_found[index] for index in wrong_indices),
            "exact": sum(
                error_quote_match_modes[index] == "exact" for index in wrong_indices
            ),
            "whitespace_normalized": sum(
                error_quote_match_modes[index] == "whitespace_normalized"
                for index in wrong_indices
            ),
        }
        gathered_quote_stats = gather_object([local_quote_stats])
        global_quote_stats = {
            key: sum(int(stats[key]) for stats in gathered_quote_stats)
            for key in local_quote_stats
        }
        global_wrong_count = global_quote_stats["wrong"]
        quote_denominator = max(1, global_wrong_count)
        self._metrics[mode]["branch_contrastive_wrong_count"].append(
            float(global_wrong_count)
        )
        self._metrics[mode]["branch_contrastive_error_quote_parsed_count"].append(
            float(global_quote_stats["parsed"])
        )
        self._metrics[mode]["branch_contrastive_error_quote_parsed_rate"].append(
            float(global_quote_stats["parsed"] / quote_denominator)
        )
        self._metrics[mode]["branch_contrastive_error_quote_found_count"].append(
            float(global_quote_stats["found"])
        )
        self._metrics[mode]["branch_contrastive_error_quote_found_rate"].append(
            float(global_quote_stats["found"] / quote_denominator)
        )
        self._metrics[mode]["branch_contrastive_error_quote_exact_match_rate"].append(
            float(global_quote_stats["exact"] / quote_denominator)
        )
        self._metrics[mode][
            "branch_contrastive_error_quote_whitespace_match_rate"
        ].append(float(global_quote_stats["whitespace_normalized"] / quote_denominator))
        # Retain the old metric name for existing analysis scripts.
        self._metrics[mode]["branch_contrastive_locator_success_rate"].append(
            float(global_quote_stats["found"] / quote_denominator)
        )
        self._metrics[mode]["branch_contrastive_selected_count"].append(float(selected_count))
        self._metrics[mode]["branch_contrastive_selected_rate"].append(
            float(selected_count / max(1, batch_size))
        )

        return {
            "predicted_answers": predicted_answers,
            "ground_truth_answers": ground_truth_answers,
            "outcomes": outcomes,
            "sample_weights": sample_weights,
            "locator_prompts": locator_prompts,
            "locator_outputs": locator_outputs,
            "error_quotes": error_quotes,
            "error_quote_parsed": error_quote_parsed,
            "error_quote_found": error_quote_found,
            "error_quote_match_modes": error_quote_match_modes,
            "error_char_starts": error_char_starts,
            "branch_token_indices": branch_token_indices,
            "prefix_texts": prefix_texts,
            "wrong_continuation_texts": wrong_continuation_texts,
            "correction_prompts": correction_prompts,
            "correction_texts": correction_texts,
            "skip_reasons": skip_reasons,
        }

    def _build_independent_loss_inputs(self, base_inputs, loss_spec, pad_token_id):
        branch_inputs = dict(base_inputs)
        self._set_single_loss_prompt(branch_inputs, "student", loss_spec["student_prompt"])
        self._set_single_loss_prompt(branch_inputs, "teacher", loss_spec["teacher_prompt"])

        generated_ids, generated_attention_mask, _ = self._build_student_batch_from_completion_ids(
            branch_inputs, loss_spec["completion_ids"], pad_token_id
        )
        student_prompt_len = branch_inputs["student_prompt_length"]
        generation_ids = generated_ids[:, student_prompt_len:]
        teacher_full_ids = torch.cat([branch_inputs["teacher_prompts"], generation_ids], dim=1)
        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        teacher_attention_mask[teacher_full_ids == pad_token_id] = 0

        labels = generated_ids.clone()
        labels[:, :student_prompt_len] = -100
        labels[generated_ids == pad_token_id] = -100

        branch_inputs["student_input_ids"] = generated_ids
        branch_inputs["student_attention_mask"] = generated_attention_mask
        branch_inputs["teacher_input_ids"] = teacher_full_ids
        branch_inputs["teacher_attention_mask"] = teacher_attention_mask
        branch_inputs["labels"] = labels
        branch_inputs["best_checkpoint_update_mode"] = torch.tensor(
            [loss_spec["update_mode"]], dtype=torch.long, device=self.accelerator.device
        )
        branch_inputs["independent_loss_scale"] = torch.tensor(
            [loss_spec["loss_scale"]], dtype=torch.float32, device=self.accelerator.device
        )
        return branch_inputs

    def _generate_independent_verification_rollouts_vllm(self, inputs, generation_config):
        """Grade each candidate independently and verify each incorrect candidate separately."""
        batch_size = int(inputs["student_prompts"].shape[0])
        if batch_size <= 0:
            raise RuntimeError("Independent verification received an empty training batch.")

        problems = list(inputs.get("reward_problems", []))
        reference_solutions = list(inputs.get("reward_solutions", []))
        if len(problems) != batch_size or len(reference_solutions) != batch_size:
            raise RuntimeError(
                "Independent verification requires one problem and reference solution per batch item: "
                f"batch_size={batch_size}, problems={len(problems)}, "
                f"solutions={len(reference_solutions)}."
            )

        original_student_prompts = [
            self._clean_decoded_prompt(
                self.processing_class.decode(prompt_ids, skip_special_tokens=False)
            )
            for prompt_ids in inputs["student_prompts"]
        ]
        original_teacher_prompts = [
            self._clean_decoded_prompt(
                self.processing_class.decode(prompt_ids, skip_special_tokens=False)
            )
            for prompt_ids in inputs["teacher_prompts"]
        ]
        ground_truths = [
            self._extract_boxed_answer(reference_solution)
            for reference_solution in reference_solutions
        ]

        candidate_groups = self._sample_vllm_completion_ids(
            original_student_prompts,
            generation_config,
            self.verification_num_candidates,
        )
        if len(candidate_groups) != batch_size:
            raise RuntimeError(
                f"Expected candidate groups for {batch_size} problems, got {len(candidate_groups)}."
            )

        total_candidates = batch_size * self.verification_num_candidates
        loss_scale = 1.0 / total_candidates
        loss_specs = []
        records = []
        pending_verifications = []
        candidate_correct_count = 0

        for problem_index, candidate_ids in enumerate(candidate_groups):
            if len(candidate_ids) != self.verification_num_candidates:
                raise RuntimeError(
                    f"Expected {self.verification_num_candidates} independent candidates for problem "
                    f"{problem_index}, got {len(candidate_ids)}."
                )

            original_student_prompt = original_student_prompts[problem_index]
            original_teacher_prompt = original_teacher_prompts[problem_index]
            problem = problems[problem_index]
            reference_solution = reference_solutions[problem_index]
            ground_truth = ground_truths[problem_index]

            for candidate_index, token_ids in enumerate(candidate_ids):
                candidate_text = self.processing_class.decode(token_ids, skip_special_tokens=False)
                candidate_solution = self.processing_class.decode(token_ids, skip_special_tokens=True)
                candidate_answer = self._extract_boxed_answer(candidate_text)
                candidate_correct = self._grade_boxed_answer_for_reward(candidate_answer, ground_truth)
                candidate_correct_count += int(candidate_correct)
                record_index = len(records)

                record = {
                    "step": self.state.global_step,
                    "rollout_step": self.state.global_step + 1,
                    "problem_index": problem_index,
                    "candidate_index": candidate_index,
                    "candidate_completion": candidate_text,
                    "candidate_predicted_answer": candidate_answer,
                    "candidate_correct": candidate_correct,
                    "ground_truth_answer": ground_truth,
                    "verification_used": not candidate_correct,
                    "verification_completion": None,
                    "verification_predicted_answer": None,
                    "verification_correct": None,
                    "loss_context": "original_problem" if candidate_correct else "verification",
                    "used_for_loss": candidate_correct,
                    "update_mode": 1 if candidate_correct else 0,
                    "outcome": "candidate_correct" if candidate_correct else "pending_verification",
                    "prompt": original_student_prompt,
                    "completion": candidate_text,
                }
                records.append(record)

                if candidate_correct:
                    loss_specs.append(
                        {
                            "student_prompt": original_student_prompt,
                            "teacher_prompt": original_teacher_prompt,
                            "completion_ids": token_ids,
                            "update_mode": 1,
                            "loss_scale": loss_scale,
                        }
                    )
                    continue

                (
                    student_verification_prompt,
                    teacher_verification_prompt,
                ) = self._build_verification_prompt_pair(
                    problem,
                    reference_solution,
                    candidate_solution,
                )
                pending_verifications.append(
                    (
                        record_index,
                        ground_truth,
                        student_verification_prompt,
                        teacher_verification_prompt,
                    )
                )

        if pending_verifications:
            verification_prompts = [item[2] for item in pending_verifications]
            verification_groups = self._sample_vllm_completion_ids(
                verification_prompts, generation_config, 1
            )
            if len(verification_groups) != len(pending_verifications):
                raise RuntimeError(
                    f"Expected {len(pending_verifications)} verification outputs, got {len(verification_groups)}."
                )

            for pending, output_group in zip(pending_verifications, verification_groups):
                (
                    record_index,
                    ground_truth,
                    student_verification_prompt,
                    teacher_verification_prompt,
                ) = pending
                if len(output_group) != 1:
                    raise RuntimeError("Each independent candidate must receive exactly one verification rollout.")
                verification_ids = output_group[0]
                verification_text = self.processing_class.decode(
                    verification_ids, skip_special_tokens=False
                )
                verification_answer = self._extract_boxed_answer(verification_text)
                verification_correct = self._grade_boxed_answer_for_reward(
                    verification_answer, ground_truth
                )
                update_mode = 2 if verification_correct else 0
                record = records[record_index]
                record.update(
                    {
                        "verification_completion": verification_text,
                        "verification_predicted_answer": verification_answer,
                        "verification_correct": verification_correct,
                        "used_for_loss": verification_correct,
                        "update_mode": update_mode,
                        "outcome": (
                            "verification_correct"
                            if verification_correct
                            else "verification_incorrect_skipped"
                        ),
                        "prompt": student_verification_prompt,
                        "completion": verification_text,
                    }
                )
                loss_specs.append(
                    {
                        "student_prompt": student_verification_prompt,
                        "teacher_prompt": teacher_verification_prompt,
                        "completion_ids": verification_ids,
                        # Keep a zero-loss branch for rejected verifications so every rank
                        # executes the same number of backwards at synchronization boundaries.
                        "update_mode": update_mode,
                        "loss_scale": loss_scale,
                    }
                )

        if len(loss_specs) != total_candidates:
            raise RuntimeError(
                "Independent verification must produce one loss branch per candidate: "
                f"expected {total_candidates}, got {len(loss_specs)}."
            )

        if self.vllm_enable_sleep_mode:
            self.vllm_engine.sleep(level=2)

        mode = "train" if self.model.training else "eval"
        verification_count = len(pending_verifications)
        verification_correct_count = sum(
            int(record["verification_correct"] is True) for record in records
        )
        self._metrics[mode]["independent_candidate_correct_count"].append(
            float(candidate_correct_count)
        )
        self._metrics[mode]["independent_candidate_count"].append(float(total_candidates))
        self._metrics[mode]["independent_verification_count"].append(float(verification_count))
        self._metrics[mode]["independent_verification_correct_count"].append(
            float(verification_correct_count)
        )
        update_count = sum(int(record["used_for_loss"]) for record in records)
        self._metrics[mode]["independent_update_count"].append(float(update_count))

        return loss_specs, records

    def _compute_best_checkpoint_advantages(self, completion_texts, solution_texts, device):
        advantages = []
        predicted_answers = []
        ground_truth_answers = []
        outcomes = []

        for completion, solution in zip(completion_texts, solution_texts):
            predicted = self._extract_boxed_answer(completion or "")
            ground_truth = self._extract_boxed_answer(solution or "")
            if predicted is None:
                advantage = 0.25
                outcome = "no_boxed"
            elif self._grade_boxed_answer_for_reward(predicted, ground_truth):
                advantage = 1.0
                outcome = "correct_boxed"
            else:
                advantage = -1.0
                outcome = "wrong_boxed"

            advantages.append(advantage)
            predicted_answers.append(predicted)
            ground_truth_answers.append(ground_truth)
            outcomes.append(outcome)

        advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        mode = "train" if self.model.training else "eval"
        if advantage_tensor.numel():
            self._metrics[mode]["best_checkpoint_advantage_mean"].append(
                advantage_tensor.mean().item()
            )
            self._metrics[mode]["best_checkpoint_correct_boxed_rate"].append(
                float(sum(outcome == "correct_boxed" for outcome in outcomes) / len(outcomes))
            )
            self._metrics[mode]["best_checkpoint_wrong_boxed_rate"].append(
                float(sum(outcome == "wrong_boxed" for outcome in outcomes) / len(outcomes))
            )
            self._metrics[mode]["best_checkpoint_no_boxed_rate"].append(
                float(sum(outcome == "no_boxed" for outcome in outcomes) / len(outcomes))
            )

        return advantage_tensor, predicted_answers, ground_truth_answers, outcomes

    def _compute_reward_guided_advantages(self, completion_texts, solution_texts, device, problem_texts=None):
        local_rewards = []
        local_predicted_answers = []
        local_ground_truth_answers = []
        local_problem_texts = list(problem_texts or [""] * len(completion_texts))
        for completion, solution in zip(completion_texts, solution_texts):
            predicted_answer = self._extract_boxed_answer(completion or "")
            ground_truth_answer = self._extract_boxed_answer(solution or "")
            reward = 1.0 if self._grade_boxed_answer_for_reward(predicted_answer, ground_truth_answer) else -1.0
            local_rewards.append(reward)
            local_predicted_answers.append(predicted_answer)
            local_ground_truth_answers.append(ground_truth_answer)

        local_records = [
            {"problem": problem or "", "reward": reward}
            for problem, reward in zip(local_problem_texts, local_rewards)
        ]
        gathered_records = gather_object(local_records)
        gathered_rewards = [float(record["reward"]) for record in gathered_records]
        reward_tensor = torch.tensor(gathered_rewards, dtype=torch.float32, device=device)
        global_reward_mean = reward_tensor.mean() if reward_tensor.numel() > 0 else torch.tensor(0.0, device=device)
        global_reward_std = (
            reward_tensor.std(unbiased=False) if reward_tensor.numel() > 1 else torch.tensor(0.0, device=device)
        )

        grouped_rewards = defaultdict(list)
        for record in gathered_records:
            grouped_rewards[record["problem"]].append(float(record["reward"]))
        use_problem_groups = any(len(rewards) > 1 for rewards in grouped_rewards.values())
        group_stats = {}
        for problem, rewards in grouped_rewards.items():
            group_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            group_stats[problem] = (
                group_tensor.mean(),
                group_tensor.std(unbiased=False) if group_tensor.numel() > 1 else torch.tensor(0.0, device=device),
            )

        local_advantages_list = []
        for problem, reward in zip(local_problem_texts, local_rewards):
            local_reward = torch.tensor(float(reward), dtype=torch.float32, device=device)
            if use_problem_groups:
                reward_mean, reward_std = group_stats.get(problem, (global_reward_mean, global_reward_std))
            else:
                reward_mean, reward_std = global_reward_mean, global_reward_std
            if reward_std.item() <= self.reward_guided_advantage_epsilon:
                local_advantages_list.append(torch.tensor(0.0, dtype=torch.float32, device=device))
            else:
                local_advantages_list.append(
                    (local_reward - reward_mean) / (reward_std + self.reward_guided_advantage_epsilon)
                )

        local_reward_tensor = torch.tensor(local_rewards, dtype=torch.float32, device=device)
        local_advantages = (
            torch.stack(local_advantages_list)
            if local_advantages_list
            else torch.empty(0, dtype=torch.float32, device=device)
        )

        mode = "train" if self.model.training else "eval"
        positive_rate = (local_advantages > 0).to(torch.float32).mean().item() if local_advantages.numel() else 0.0
        negative_rate = (local_advantages < 0).to(torch.float32).mean().item() if local_advantages.numel() else 0.0
        zero_rate = (local_advantages == 0).to(torch.float32).mean().item() if local_advantages.numel() else 0.0
        self._metrics[mode]["reward_guided_reward_mean"].append(local_reward_tensor.mean().item())
        self._metrics[mode]["reward_guided_global_reward_mean"].append(global_reward_mean.detach().item())
        self._metrics[mode]["reward_guided_global_reward_std"].append(global_reward_std.detach().item())
        self._metrics[mode]["reward_guided_advantage_mean"].append(local_advantages.mean().item())
        self._metrics[mode]["reward_guided_positive_advantage_rate"].append(positive_rate)
        self._metrics[mode]["reward_guided_negative_advantage_rate"].append(negative_rate)
        self._metrics[mode]["reward_guided_zero_advantage_rate"].append(zero_rate)
        self._metrics[mode]["reward_guided_problem_grouped"].append(float(use_problem_groups))

        return (
            local_reward_tensor,
            local_advantages,
            local_predicted_answers,
            local_ground_truth_answers,
        )

    def _compute_wrong_boxed_only_weights(
        self,
        completion_texts,
        solution_texts,
        device,
        metric_prefix="wrong_boxed_only",
    ):
        weights = []
        predicted_answers = []
        ground_truth_answers = []
        outcomes = []
        for completion, solution in zip(completion_texts, solution_texts):
            predicted_answer = self._extract_boxed_answer(completion or "")
            ground_truth_answer = self._extract_boxed_answer(solution or "")
            predicted_answers.append(predicted_answer)
            ground_truth_answers.append(ground_truth_answer)

            if predicted_answer is None:
                outcome = "no_boxed"
                weight = 0.0
            elif self._grade_boxed_answer_for_reward(predicted_answer, ground_truth_answer):
                outcome = "correct_boxed"
                weight = 0.0
            else:
                outcome = "wrong_boxed"
                weight = 1.0
            outcomes.append(outcome)
            weights.append(weight)

        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        mode = "train" if self.model.training else "eval"
        if outcomes:
            total = float(len(outcomes))
            self._metrics[mode][f"{metric_prefix}_wrong_rate"].append(
                sum(outcome == "wrong_boxed" for outcome in outcomes) / total
            )
            self._metrics[mode][f"{metric_prefix}_correct_rate"].append(
                sum(outcome == "correct_boxed" for outcome in outcomes) / total
            )
            self._metrics[mode][f"{metric_prefix}_no_boxed_rate"].append(
                sum(outcome == "no_boxed" for outcome in outcomes) / total
            )
            self._metrics[mode][f"{metric_prefix}_mean_sample_weight"].append(
                weight_tensor.detach().mean().item()
            )

        return weight_tensor, predicted_answers, ground_truth_answers, outcomes

    def _record_adaptive_completion_batch(self, gathered_completion_texts):
        if not self.adaptive_completion_length:
            return

        total = len(gathered_completion_texts)
        if total == 0:
            return

        boxed = sum(
            1
            for completion in gathered_completion_texts
            if self._extract_boxed_answer(completion or "") is not None
        )
        self._adaptive_pending_boxed += boxed
        self._adaptive_pending_total += total

    def _maybe_update_adaptive_completion_length(self):
        if not self.adaptive_completion_length or not self.accelerator.sync_gradients:
            return

        if self._adaptive_pending_total <= 0:
            return

        boxed = int(self._adaptive_pending_boxed)
        total = int(self._adaptive_pending_total)
        step_rate = boxed / max(1, total)
        self._adaptive_completion_history.append((boxed, total))
        self._adaptive_completion_optimizer_steps += 1
        self._adaptive_pending_boxed = 0
        self._adaptive_pending_total = 0

        current_length = int(self.generation_config.max_new_tokens)
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["adaptive_boxed_step_rate"].append(float(step_rate))
        self._metrics[mode]["adaptive_current_max_completion_length"].append(float(current_length))

        steps_since_check = (
            self._adaptive_completion_optimizer_steps - self._adaptive_completion_last_check_step
        )
        if (
            len(self._adaptive_completion_history) < self.adaptive_completion_window_steps
            or steps_since_check < self.adaptive_completion_window_steps
        ):
            return

        window_boxed = sum(item[0] for item in self._adaptive_completion_history)
        window_total = sum(item[1] for item in self._adaptive_completion_history)
        window_rate = window_boxed / max(1, window_total)
        self._metrics[mode]["adaptive_boxed_window_rate"].append(float(window_rate))
        self._metrics[mode]["adaptive_boxed_window_samples"].append(float(window_total))

        next_length = current_length
        if window_rate < self.adaptive_completion_target and current_length < self.adaptive_max_completion_length:
            next_length = min(
                self.adaptive_max_completion_length,
                current_length + self.adaptive_completion_length_increment,
            )
            self.generation_config.max_new_tokens = next_length
            if self.accelerator.is_main_process:
                print(f"\n{'='*80}")
                print("ADAPTIVE COMPLETION LENGTH UPDATE")
                print(
                    f"Optimizer step window ending at {self._adaptive_completion_optimizer_steps}: "
                    f"boxed rate {window_rate:.3f} < target {self.adaptive_completion_target:.3f}."
                )
                print(f"max_new_tokens: {current_length} -> {next_length}")
                print(f"{'='*80}\n")
        elif self.accelerator.is_main_process:
            print(f"\n{'='*80}")
            print("ADAPTIVE COMPLETION LENGTH CHECK")
            print(
                f"Optimizer step window ending at {self._adaptive_completion_optimizer_steps}: "
                f"boxed rate {window_rate:.3f}, current max_new_tokens {current_length}."
            )
            print(f"{'='*80}\n")

        self._metrics[mode]["adaptive_next_max_completion_length"].append(float(next_length))
        self._adaptive_completion_last_check_step = self._adaptive_completion_optimizer_steps

    def _compute_branch_contrastive_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["branch_contrastive_input_ids"]
        attention_mask = inputs["branch_contrastive_attention_mask"]
        labels = inputs["branch_contrastive_labels"]
        sample_weights = inputs["branch_contrastive_sample_weights"].to(
            device=input_ids.device, dtype=torch.float32
        )
        batch_size = int(sample_weights.numel())
        if input_ids.shape[0] != 2 * batch_size:
            raise RuntimeError(
                "Branch contrastive inputs must contain all correct continuations followed by all wrong continuations."
            )

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        token_mask = shifted_labels != -100
        token_nll = F.cross_entropy(
            shifted_logits.transpose(1, 2),
            shifted_labels,
            reduction="none",
            ignore_index=-100,
        )
        sequence_log_probs = (-(token_nll) * token_mask.to(token_nll.dtype)).sum(dim=1)
        sequence_token_counts = token_mask.sum(dim=1).clamp_min(1)
        mean_log_probs = sequence_log_probs / sequence_token_counts
        good_mean_log_probs = mean_log_probs[:batch_size]
        bad_mean_log_probs = mean_log_probs[batch_size:]
        margins = good_mean_log_probs - bad_mean_log_probs
        per_sample_losses = F.softplus(-float(self.branch_contrastive_beta) * margins)
        loss = (
            per_sample_losses * sample_weights.to(per_sample_losses.dtype)
        ).sum() / max(1, batch_size)

        mode = "train" if self.model.training else "eval"
        selected = sample_weights > 0
        if selected.any():
            self._metrics[mode]["branch_contrastive_good_mean_logp"].append(
                good_mean_log_probs[selected].detach().mean().item()
            )
            self._metrics[mode]["branch_contrastive_bad_mean_logp"].append(
                bad_mean_log_probs[selected].detach().mean().item()
            )
            self._metrics[mode]["branch_contrastive_margin"].append(
                margins[selected].detach().mean().item()
            )
            self._metrics[mode]["branch_contrastive_pair_loss"].append(
                per_sample_losses[selected].detach().mean().item()
            )
        self._metrics[mode]["branch_contrastive_loss"].append(loss.detach().item())
        self._metrics[mode]["branch_contrastive_beta"].append(
            float(self.branch_contrastive_beta)
        )

        if return_outputs:
            class MinimalOutput:
                pass

            minimal_output = MinimalOutput()
            minimal_output.loss = loss
            del outputs, shifted_logits
            return loss, minimal_output
        del outputs, shifted_logits
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the self-distillation loss with memory-efficient log-prob extraction.

        Memory optimization: Extract only needed log-probs immediately and free large tensors.
        """
        if self.wrong_answer_branch_contrastive:
            return self._compute_branch_contrastive_loss(model, inputs, return_outputs)

        # Get batch-level prompt lengths
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]
        reverse_student_logits_for_loss = None
        prefix_consistency_loss_scale_input = None
        prefix_consistency_outcome_advantage_input = None
        if self.prefix_consistency_distillation:
            prefix_consistency_loss_scale_input = inputs.get(
                "prefix_consistency_loss_scale"
            )
            prefix_consistency_outcome_advantage_input = inputs.get(
                "prefix_consistency_outcome_advantage"
            )
            if (
                prefix_consistency_loss_scale_input is None
                or prefix_consistency_loss_scale_input.numel() != 1
            ):
                raise RuntimeError(
                    "Prefix-consistency loss requires exactly one prefix_consistency_loss_scale."
                )
            if (
                prefix_consistency_outcome_advantage_input is None
                or prefix_consistency_outcome_advantage_input.numel() != 1
            ):
                raise RuntimeError(
                    "Prefix-consistency loss requires exactly one "
                    "prefix_consistency_outcome_advantage."
                )

        # === STUDENT FORWARD - Extract log-probs immediately ===
        if self.reverse_teacher_generation and not self.use_thinking_machines_loss:
            reverse_student_prompt_len = inputs["reverse_student_prompt_length"]
            if reverse_student_prompt_len != student_prompt_len:
                raise ValueError(
                    "reverse_teacher_generation expects student and reverse-student prompt lengths to match."
                )

            normal_student_seq_len = inputs["student_input_ids"].shape[1]
            reverse_student_seq_len = inputs["reverse_student_input_ids"].shape[1]
            combined_seq_len = max(normal_student_seq_len, reverse_student_seq_len)
            pad_token_id = self.processing_class.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0

            def pad_sequence_dim(tensor, target_len, value):
                if tensor.shape[1] >= target_len:
                    return tensor
                pad_width = target_len - tensor.shape[1]
                padding = torch.full(
                    (tensor.shape[0], pad_width),
                    value,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                return torch.cat([tensor, padding], dim=1)

            normal_student_ids = pad_sequence_dim(inputs["student_input_ids"], combined_seq_len, pad_token_id)
            reverse_student_ids = pad_sequence_dim(
                inputs["reverse_student_input_ids"], combined_seq_len, pad_token_id
            )
            normal_student_mask = pad_sequence_dim(inputs["student_attention_mask"], combined_seq_len, 0)
            reverse_student_mask = pad_sequence_dim(
                inputs["reverse_student_attention_mask"], combined_seq_len, 0
            )

            combined_student_ids = torch.cat([normal_student_ids, reverse_student_ids], dim=0)
            combined_student_mask = torch.cat([normal_student_mask, reverse_student_mask], dim=0)

            outputs_student = model(
                input_ids=combined_student_ids,
                attention_mask=combined_student_mask,
            )
            combined_student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
            normal_batch_size = inputs["student_input_ids"].shape[0]
            normal_generation_len = normal_student_seq_len - student_prompt_len
            reverse_generation_len = reverse_student_seq_len - reverse_student_prompt_len
            student_logits = combined_student_logits[:normal_batch_size, :normal_generation_len, :]
            reverse_student_logits_for_loss = combined_student_logits[
                normal_batch_size:, :reverse_generation_len, :
            ]
            del (
                combined_student_ids,
                combined_student_mask,
                combined_student_logits,
                normal_student_ids,
                reverse_student_ids,
                normal_student_mask,
                reverse_student_mask,
            )
        else:
            outputs_student = model(
                input_ids=inputs["student_input_ids"],
                attention_mask=inputs["student_attention_mask"],
            )

            # Extract only what we need and convert to log-probs immediately
            student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]

        if self.use_thinking_machines_loss:
            # For reverse KL, we only need log-probs of sampled tokens
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            student_log_probs_sampled = torch.gather(
                student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            del student_logits, student_log_probs  # Free immediately!
        else:
            # For JSD, keep logits (temperature will be applied in generalized_jsd_loss)
            student_logits_for_loss = student_logits
            del student_logits

        # Free the full outputs (but keep reference for return_outputs if needed)
        if return_outputs:
            # Create a minimal output object to return (just the loss, no logits)
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student
        empty_cache()

        prefix_consistency_loss_scale = None
        prefix_consistency_outcome_advantage = None
        if self.prefix_consistency_distillation:
            prefix_consistency_loss_scale = (
                prefix_consistency_loss_scale_input.to(
                    device=student_logits_for_loss.device,
                    dtype=student_logits_for_loss.dtype,
                )
                .reshape(())
            )
            prefix_consistency_outcome_advantage = (
                prefix_consistency_outcome_advantage_input.to(
                    device=student_logits_for_loss.device,
                    dtype=student_logits_for_loss.dtype,
                )
                .reshape(())
                .detach()
            )

        if self.best_checkpoint_independent_verification:
            update_modes = inputs.get("best_checkpoint_update_mode")
            if update_modes is None or update_modes.numel() != 1:
                raise RuntimeError(
                    "Independent verification requires exactly one best_checkpoint_update_mode."
                )
            if int(update_modes.item()) == 0:
                loss = student_logits_for_loss.sum() * 0.0
                mode = "train" if self.model.training else "eval"
                self._metrics[mode]["best_checkpoint_teacher_target_loss"].append(0.0)
                self._metrics[mode]["best_checkpoint_best_target_loss"].append(0.0)
                self._metrics[mode]["best_checkpoint_total_loss"].append(0.0)
                self._metrics[mode]["independent_update_mode"].append(0.0)
                self._metrics[mode]["best_checkpoint_anchor_step"].append(
                    float(self._best_checkpoint_step)
                )
                if self._best_checkpoint_score is not None:
                    self._metrics[mode]["best_checkpoint_anchor_score"].append(
                        float(self._best_checkpoint_score)
                    )
                del student_logits_for_loss
                empty_cache()
                if return_outputs:
                    minimal_output.loss = loss
                    return loss, minimal_output
                return loss

        # === TEACHER FORWARD - Extract log-probs immediately ===
        # Choose teacher context based on mode:
        #   use_ema_teacher  → swap in EMA weights temporarily
        #   fixed_teacher    → disable LoRA adapters (base model = initial policy)
        #   default (dynamic)→ no-op, use current student weights
        def teacher_forward_context():
            if self.use_ema_teacher:
                return self._ema_teacher_context(model)
            if self.fixed_teacher and is_peft_model(model):
                return self.accelerator.unwrap_model(model).disable_adapter()
            return nullcontext()

        with torch.no_grad(), teacher_forward_context():
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )

            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]

            if self.use_thinking_machines_loss:
                teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                teacher_log_probs_sampled = torch.gather(
                    teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                del teacher_logits, teacher_log_probs  # Free immediately!
            else:
                teacher_logits_for_loss = teacher_logits
                del teacher_logits

            del outputs_teacher
            empty_cache()

        # === COMPUTE LOSS with only small tensors ===
        if self.use_thinking_machines_loss:
            # Thinking Machines uses RL-style policy gradient:
            # Advantage = log π_teacher(x) - log π_student(x)
            # Loss = -E[Advantage * log π_student(x)]
            #
            # CRITICAL: advantage must be detached to prevent gradients flowing through it.
            # We want: ∇θ L = -E[A(x) * ∇θ log π_student(x)]
            # NOT: ∇θ L = -E[(T(x) - S(x)) * ∇θ S(x)] where both terms differentiate

            advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()

            # Apply masking before computing loss
            if shifted_labels is not None:
                mask = shifted_labels != -100
                advantage = advantage[mask]
                student_log_probs_sampled_masked = student_log_probs_sampled[mask]
            else:
                student_log_probs_sampled_masked = student_log_probs_sampled

            # Policy gradient loss: -advantage * log π_student
            # Negative because we minimize loss (gradient descent), but want to maximize reward
            loss = -(advantage * student_log_probs_sampled_masked).mean()

            del (
                student_log_probs_sampled,
                teacher_log_probs_sampled,
                advantage,
                student_log_probs_sampled_masked,
            )
        else:
            if self.best_checkpoint_distillation:
                mode = "train" if self.model.training else "eval"
                advantages = None
                update_mode = None
                if self.best_checkpoint_independent_verification:
                    if "best_checkpoint_update_mode" not in inputs:
                        raise RuntimeError(
                            "Independent verification expected best_checkpoint_update_mode in inputs."
                        )
                    update_modes = inputs["best_checkpoint_update_mode"]
                    if update_modes.numel() != 1:
                        raise RuntimeError(
                            "Independent verification requires a loss microbatch of exactly one sample."
                        )
                    update_mode = int(update_modes.item())
                    if update_mode not in (0, 1, 2):
                        raise RuntimeError(f"Unknown independent verification update mode: {update_mode}")
                    loss_scale = inputs.get("independent_loss_scale")
                    if loss_scale is None or loss_scale.numel() != 1:
                        raise RuntimeError(
                            "Independent verification requires exactly one independent_loss_scale."
                        )
                    sample_weight = loss_scale.to(
                        dtype=student_logits_for_loss.dtype,
                        device=student_logits_for_loss.device,
                    )
                else:
                    if "best_checkpoint_advantages" not in inputs:
                        raise RuntimeError(
                            "best_checkpoint_distillation=True expected best_checkpoint_advantages in inputs."
                        )
                    advantages = inputs["best_checkpoint_advantages"].to(
                        device=student_logits_for_loss.device,
                        dtype=student_logits_for_loss.dtype,
                    )
                    if advantages.numel() != 1:
                        raise RuntimeError(
                            "best_checkpoint_distillation requires a loss microbatch of exactly one sample."
                        )
                    advantage = float(advantages.item())
                    sample_weight = torch.tensor(
                        [abs(advantage)],
                        dtype=student_logits_for_loss.dtype,
                        device=student_logits_for_loss.device,
                    )

                if self.best_checkpoint_independent_verification and update_mode == 0:
                    del teacher_logits_for_loss
                    loss = student_logits_for_loss.sum() * 0.0
                    teacher_target_loss_value = 0.0
                    best_target_loss_value = 0.0
                elif self.best_checkpoint_independent_verification:
                    teacher_target_loss = self.generalized_jsd_loss(
                        student_logits=student_logits_for_loss,
                        teacher_logits=teacher_logits_for_loss,
                        labels=shifted_labels,
                        beta=0,
                        temperature=self.temperature,
                        top_k=self.top_k_loss,
                        token_clip=self.jsd_token_clip,
                        sample_weights=sample_weight,
                    )
                    loss = teacher_target_loss
                    best_target_loss_value = 0.0
                    teacher_target_loss_value = teacher_target_loss.detach().item()
                    del teacher_logits_for_loss, teacher_target_loss
                    empty_cache()
                elif advantage > 0:
                    teacher_target_loss = self.generalized_jsd_loss(
                        student_logits=student_logits_for_loss,
                        teacher_logits=teacher_logits_for_loss,
                        labels=shifted_labels,
                        beta=0,
                        temperature=self.temperature,
                        top_k=self.top_k_loss,
                        token_clip=self.jsd_token_clip,
                        sample_weights=sample_weight,
                    )
                    loss = teacher_target_loss
                    best_target_loss_value = 0.0
                    teacher_target_loss_value = teacher_target_loss.detach().item()
                    del teacher_logits_for_loss, teacher_target_loss
                else:
                    # The privileged teacher is not needed for a negative sample. Free its
                    # full-vocabulary logits before running the frozen best policy.
                    del teacher_logits_for_loss
                    empty_cache()
                    with torch.no_grad(), self._best_anchor_context(model):
                        outputs_best_student = model(
                            input_ids=inputs["student_input_ids"],
                            attention_mask=inputs["student_attention_mask"],
                        )
                        best_student_logits_for_loss = outputs_best_student.logits[
                            :, student_prompt_len - 1 : -1, :
                        ]
                        del outputs_best_student
                        empty_cache()

                    best_target_loss = self.generalized_jsd_loss(
                        student_logits=student_logits_for_loss,
                        teacher_logits=best_student_logits_for_loss,
                        labels=shifted_labels,
                        beta=0,
                        temperature=self.temperature,
                        top_k=self.top_k_loss,
                        token_clip=self.jsd_token_clip,
                        sample_weights=sample_weight,
                    )
                    loss = best_target_loss
                    teacher_target_loss_value = 0.0
                    best_target_loss_value = best_target_loss.detach().item()
                    del best_student_logits_for_loss, best_target_loss

                self._metrics[mode]["best_checkpoint_teacher_target_loss"].append(
                    teacher_target_loss_value
                )
                self._metrics[mode]["best_checkpoint_best_target_loss"].append(
                    best_target_loss_value
                )
                self._metrics[mode]["best_checkpoint_total_loss"].append(loss.detach().item())
                if self.best_checkpoint_independent_verification:
                    self._metrics[mode]["independent_update_mode"].append(float(update_mode))
                    self._metrics[mode]["independent_loss_scale"].append(float(sample_weight.item()))
                else:
                    self._metrics[mode]["best_checkpoint_abs_advantage"].append(abs(advantage))
                self._metrics[mode]["best_checkpoint_anchor_step"].append(
                    float(self._best_checkpoint_step)
                )
                if self._best_checkpoint_score is not None:
                    self._metrics[mode]["best_checkpoint_anchor_score"].append(
                        float(self._best_checkpoint_score)
                    )
                if advantages is not None:
                    del advantages
                del sample_weight

            elif self.reward_guided_distillation:
                if "reward_guided_advantages" not in inputs:
                    raise RuntimeError(
                        "reward_guided_distillation=True expected reward_guided_advantages in inputs."
                    )
                advantages = inputs["reward_guided_advantages"].to(
                    device=student_logits_for_loss.device,
                    dtype=student_logits_for_loss.dtype,
                )
                alpha = float(self.reward_guided_alpha)
                positive_weights = advantages.clamp(min=0.0) * alpha
                negative_weights = (-advantages).clamp(min=0.0) * alpha

                reward_teacher_loss = self.generalized_jsd_loss(
                    student_logits=student_logits_for_loss,
                    teacher_logits=teacher_logits_for_loss,
                    labels=shifted_labels,
                    beta=0.5,
                    temperature=self.temperature,
                    top_k=self.top_k_loss,
                    token_clip=self.jsd_token_clip,
                    sample_weights=positive_weights,
                )
                del teacher_logits_for_loss

                with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
                    outputs_base_student = model(
                        input_ids=inputs["student_input_ids"],
                        attention_mask=inputs["student_attention_mask"],
                    )
                    base_student_logits_for_loss = outputs_base_student.logits[
                        :, student_prompt_len - 1 : -1, :
                    ]
                    del outputs_base_student
                    empty_cache()

                reward_base_loss = self.generalized_jsd_loss(
                    student_logits=student_logits_for_loss,
                    teacher_logits=base_student_logits_for_loss,
                    labels=shifted_labels,
                    beta=0.5,
                    temperature=self.temperature,
                    top_k=self.top_k_loss,
                    token_clip=self.jsd_token_clip,
                    sample_weights=negative_weights,
                )
                loss = reward_teacher_loss + reward_base_loss

                mode = "train" if self.model.training else "eval"
                self._metrics[mode]["reward_guided_teacher_loss"].append(
                    reward_teacher_loss.detach().item()
                )
                self._metrics[mode]["reward_guided_base_loss"].append(
                    reward_base_loss.detach().item()
                )
                self._metrics[mode]["reward_guided_total_loss"].append(loss.detach().item())
                self._metrics[mode]["reward_guided_positive_weight_mean"].append(
                    positive_weights.detach().mean().item()
                )
                self._metrics[mode]["reward_guided_negative_weight_mean"].append(
                    negative_weights.detach().mean().item()
                )
                self._metrics[mode]["reward_guided_alpha"].append(alpha)

                del (
                    base_student_logits_for_loss,
                    reward_teacher_loss,
                    reward_base_loss,
                    positive_weights,
                    negative_weights,
                    advantages,
                )
            else:
                wrong_boxed_only_weights_for_loss = None
                if (
                    self.wrong_boxed_only_distillation
                    or self.mock_student_distillation
                    or self.change_to_wrong_distillation
                    or self.wrong_answer_teacher_correction_distillation
                ):
                    if "wrong_boxed_only_weights" not in inputs:
                        raise RuntimeError(
                            "Wrong-answer filtered distillation expected wrong_boxed_only_weights in inputs."
                        )
                    wrong_boxed_only_weights_for_loss = inputs["wrong_boxed_only_weights"].to(
                        device=student_logits_for_loss.device,
                        dtype=student_logits_for_loss.dtype,
                    )

                # Temperature is applied inside generalized_jsd_loss
                good_loss = self.generalized_jsd_loss(
                    student_logits=student_logits_for_loss,
                    teacher_logits=teacher_logits_for_loss,
                    labels=shifted_labels,
                    beta=self.beta,
                    temperature=self.temperature,  # Let the function handle temperature
                    top_k=self.top_k_loss,
                    token_clip=self.jsd_token_clip,
                    sample_weights=wrong_boxed_only_weights_for_loss,
                )
                if self.prefix_consistency_distillation:
                    outcome_advantage_value = float(
                        prefix_consistency_outcome_advantage.item()
                    )
                    outcome_sequence_nll = good_loss.new_zeros(())
                    outcome_unscaled_loss = good_loss.new_zeros(())
                    outcome_weighted_loss = good_loss.new_zeros(())
                    combined_unscaled_loss = good_loss
                    if (
                        self.prefix_consistency_outcome_alpha > 0.0
                        and outcome_advantage_value != 0.0
                    ):
                        (
                            outcome_unscaled_loss,
                            outcome_sequence_nll,
                        ) = self._prefix_consistency_outcome_policy_loss(
                            student_logits_for_loss,
                            shifted_labels,
                            prefix_consistency_outcome_advantage,
                            self.temperature,
                        )
                        outcome_weighted_loss = (
                            float(self.prefix_consistency_outcome_alpha)
                            * outcome_unscaled_loss
                        )
                        combined_unscaled_loss = (
                            good_loss + outcome_weighted_loss
                        )
                    loss = combined_unscaled_loss * prefix_consistency_loss_scale
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode][
                        "prefix_consistency_branch_unscaled_loss"
                    ].append(good_loss.detach().item())
                    self._metrics[mode][
                        "prefix_consistency_outcome_advantage"
                    ].append(outcome_advantage_value)
                    if (
                        self.prefix_consistency_outcome_alpha > 0.0
                        and outcome_advantage_value != 0.0
                    ):
                        self._metrics[mode][
                            "prefix_consistency_active_outcome_sequence_nll"
                        ].append(outcome_sequence_nll.detach().item())
                        active_opsd_abs = abs(good_loss.detach().item())
                        active_outcome_abs = abs(
                            outcome_weighted_loss.detach().item()
                        )
                        self._metrics[mode][
                            "prefix_consistency_active_branch_opsd_abs_loss"
                        ].append(active_opsd_abs)
                        self._metrics[mode][
                            "prefix_consistency_active_outcome_abs_loss"
                        ].append(active_outcome_abs)
                        self._metrics[mode][
                            "prefix_consistency_active_outcome_to_opsd_ratio"
                        ].append(
                            active_outcome_abs / max(active_opsd_abs, 1e-12)
                        )
                    self._metrics[mode][
                        "prefix_consistency_outcome_unscaled_loss"
                    ].append(outcome_unscaled_loss.detach().item())
                    self._metrics[mode][
                        "prefix_consistency_outcome_weighted_loss"
                    ].append(outcome_weighted_loss.detach().item())
                    self._metrics[mode][
                        "prefix_consistency_combined_unscaled_loss"
                    ].append(combined_unscaled_loss.detach().item())
                    self._metrics[mode][
                        "prefix_consistency_branch_weighted_loss"
                    ].append(loss.detach().item())
                else:
                    loss = good_loss
                if self.wrong_boxed_only_distillation:
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode]["wrong_boxed_only_loss"].append(good_loss.detach().item())
                if self.mock_student_distillation:
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode]["mock_student_loss"].append(good_loss.detach().item())
                if self.change_to_wrong_distillation:
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode]["change_to_wrong_loss"].append(good_loss.detach().item())
                if self.localized_error_recovery_distillation:
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode]["localized_recovery_loss"].append(
                        good_loss.detach().item()
                    )
                if self.wrong_answer_teacher_correction_distillation:
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode]["wrong_answer_teacher_correction_loss"].append(
                        good_loss.detach().item()
                    )
                del teacher_logits_for_loss
                if wrong_boxed_only_weights_for_loss is not None:
                    del wrong_boxed_only_weights_for_loss

            if self.contrastive_teacher:
                bad_teacher_prompt_len = inputs["bad_teacher_prompt_length"]

                with torch.no_grad(), teacher_forward_context():
                    outputs_bad_teacher = model(
                        input_ids=inputs["bad_teacher_input_ids"],
                        attention_mask=inputs["bad_teacher_attention_mask"],
                    )
                    bad_teacher_logits_for_loss = outputs_bad_teacher.logits[
                        :, bad_teacher_prompt_len - 1 : -1, :
                    ]
                    del outputs_bad_teacher
                    empty_cache()

                bad_loss = self.generalized_jsd_loss(
                    student_logits=student_logits_for_loss,
                    teacher_logits=bad_teacher_logits_for_loss,
                    labels=shifted_labels,
                    beta=self.beta,
                    temperature=self.temperature,
                    top_k=self.top_k_loss,
                    token_clip=self.jsd_token_clip,
                )
                good_weight = float(self.contrastive_good_weight)
                loss = good_weight * good_loss - self.contrastive_alpha * bad_loss

                mode = "train" if self.model.training else "eval"
                self._metrics[mode]["contrastive_good_loss"].append(good_loss.detach().item())
                self._metrics[mode]["contrastive_good_weighted_loss"].append(
                    (good_weight * good_loss).detach().item()
                )
                self._metrics[mode]["contrastive_bad_loss"].append(bad_loss.detach().item())
                self._metrics[mode]["contrastive_bad_weighted_loss"].append(
                    (-self.contrastive_alpha * bad_loss).detach().item()
                )
                self._metrics[mode]["contrastive_total_loss"].append(loss.detach().item())
                self._metrics[mode]["contrastive_good_weight"].append(good_weight)
                self._metrics[mode]["contrastive_alpha"].append(float(self.contrastive_alpha))

                del bad_teacher_logits_for_loss, bad_loss

            if self.reverse_teacher_generation:
                reverse_student_prompt_len = inputs["reverse_student_prompt_length"]
                reverse_teacher_prompt_len = inputs["reverse_teacher_prompt_length"]
                reverse_shifted_labels = inputs["reverse_labels"][:, reverse_student_prompt_len:]
                if reverse_student_logits_for_loss is None:
                    raise RuntimeError(
                        "reverse_teacher_generation expected precomputed reverse student logits."
                    )

                with torch.no_grad(), teacher_forward_context():
                    outputs_reverse_teacher = model(
                        input_ids=inputs["reverse_teacher_input_ids"],
                        attention_mask=inputs["reverse_teacher_attention_mask"],
                    )
                    reverse_teacher_logits_for_loss = outputs_reverse_teacher.logits[
                        :, reverse_teacher_prompt_len - 1 : -1, :
                    ]
                    del outputs_reverse_teacher
                    empty_cache()

                reverse_seq_len = min(
                    reverse_student_logits_for_loss.shape[1],
                    reverse_teacher_logits_for_loss.shape[1],
                    reverse_shifted_labels.shape[1],
                )
                reverse_teacher_loss = self.generalized_jsd_loss(
                    student_logits=reverse_student_logits_for_loss[:, :reverse_seq_len, :],
                    teacher_logits=reverse_teacher_logits_for_loss[:, :reverse_seq_len, :],
                    labels=reverse_shifted_labels[:, :reverse_seq_len],
                    beta=self.beta,
                    temperature=self.temperature,
                    top_k=self.top_k_loss,
                    token_clip=self.jsd_token_clip,
                )
                reverse_weight = float(self.reverse_teacher_weight)
                reverse_weighted_loss = -reverse_weight * reverse_teacher_loss
                loss = loss + reverse_weighted_loss

                mode = "train" if self.model.training else "eval"
                self._metrics[mode]["reverse_teacher_generated_loss"].append(
                    reverse_teacher_loss.detach().item()
                )
                self._metrics[mode]["reverse_teacher_generated_weighted_loss"].append(
                    reverse_weighted_loss.detach().item()
                )
                self._metrics[mode]["reverse_teacher_weight"].append(reverse_weight)
                self._metrics[mode]["combined_total_loss"].append(loss.detach().item())

                del (
                    reverse_teacher_logits_for_loss,
                    reverse_teacher_loss,
                    reverse_weighted_loss,
                )
                reverse_student_logits_for_loss = None

            if self.long_thought_base_penalty:
                completion_token_counts = (shifted_labels != -100).sum(dim=1).to(student_logits_for_loss.dtype)
                ramp_denominator = float(
                    self.long_thought_base_penalty_full - self.long_thought_base_penalty_start
                )
                base_penalty_weights = (
                    (completion_token_counts - float(self.long_thought_base_penalty_start))
                    / ramp_denominator
                ).clamp(min=0.0, max=1.0)
                base_penalty_weights = base_penalty_weights * float(self.long_thought_base_penalty_weight)

                with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
                    outputs_base_student = model(
                        input_ids=inputs["student_input_ids"],
                        attention_mask=inputs["student_attention_mask"],
                    )
                    base_student_logits_for_loss = outputs_base_student.logits[
                        :, student_prompt_len - 1 : -1, :
                    ]
                    del outputs_base_student
                    empty_cache()

                base_penalty_loss = self.generalized_jsd_loss(
                    student_logits=student_logits_for_loss,
                    teacher_logits=base_student_logits_for_loss,
                    labels=shifted_labels,
                    beta=0.5,
                    temperature=self.temperature,
                    top_k=self.top_k_loss,
                    token_clip=self.jsd_token_clip,
                    sample_weights=base_penalty_weights,
                )
                loss = loss + base_penalty_loss

                mode = "train" if self.model.training else "eval"
                self._metrics[mode]["long_thought_base_penalty_loss"].append(
                    base_penalty_loss.detach().item()
                )
                self._metrics[mode]["long_thought_base_penalty_mean_weight"].append(
                    base_penalty_weights.detach().mean().item()
                )
                self._metrics[mode]["long_thought_base_penalty_max_weight"].append(
                    base_penalty_weights.detach().max().item()
                )
                self._metrics[mode]["long_thought_base_penalty_active_rate"].append(
                    (base_penalty_weights.detach() > 0).to(torch.float32).mean().item()
                )
                self._metrics[mode]["long_thought_base_penalty_avg_tokens"].append(
                    completion_token_counts.detach().mean().item()
                )
                self._metrics[mode]["combined_total_loss"].append(loss.detach().item())

                del base_student_logits_for_loss, base_penalty_loss, base_penalty_weights

            if self.reward_guided_distillation or self.best_checkpoint_distillation:
                del student_logits_for_loss
            else:
                del student_logits_for_loss, good_loss

        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return (loss, minimal_output)
        else:
            return loss

    def generate_teacher_reasoning(
        self, model, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning about the solution."""
        if self.use_vllm:
            # Use vLLM for fast reasoning generation
            return self._generate_teacher_reasoning_vllm(teacher_reasoning_prompts)
        else:
            # Use transformers generation (slower)
            with torch.no_grad():
                # Temporarily enable KV cache
                original_use_cache = model.config.use_cache
                original_gen_use_cache = self.reasoning_generation_config.use_cache

                model.config.use_cache = True
                self.reasoning_generation_config.use_cache = True

                # If fixed_teacher=True, disable LoRA adapters
                adapter_context = (
                    self.accelerator.unwrap_model(model).disable_adapter()
                    if self.fixed_teacher and is_peft_model(model)
                    else nullcontext()
                )

                try:
                    with adapter_context:
                        reasoning_outputs = model.generate(
                            input_ids=teacher_reasoning_prompts,
                            attention_mask=teacher_reasoning_attention_mask,
                            generation_config=self.reasoning_generation_config,
                            return_dict_in_generate=True,
                            use_cache=True,
                        )
                        reasoning_ids = reasoning_outputs.sequences
                finally:
                    model.config.use_cache = original_use_cache
                    self.reasoning_generation_config.use_cache = original_gen_use_cache

                return reasoning_ids

    def generate_on_policy_outputs(
        self,
        model,
        inputs,
        generation_config,
        pad_token_id=None,
        prompt_key="student_prompts",
        attention_mask_key="student_prompt_attention_mask",
    ):
        """Generate outputs from the prompts stored in inputs[prompt_key]."""
        import time

        start_time = time.time()

        # Temporarily enable KV cache for generation if it was disabled for training
        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache

        model.config.use_cache = True
        generation_config.use_cache = True

        print(f"\n{'='*80}")
        print(f"GENERATION DEBUG INFO:")
        print(f"  Model dtype: {model.dtype}")
        print(f"  Model config use_cache: {model.config.use_cache}")
        print(f"  Attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")
        print(f"  Generation config use_cache: {generation_config.use_cache}")
        print(f"  Prompt key: {prompt_key}")
        print(f"  Batch size: {inputs[prompt_key].shape[0]}")
        print(f"  Prompt length: {inputs[prompt_key].shape[1]}")
        print(f"  Max new tokens: {generation_config.max_new_tokens}")
        print(f"{'='*80}\n")

        # Generate output with respect to the student prompt only
        try:
            generated_outputs = model.generate(
                input_ids=inputs[prompt_key],
                attention_mask=inputs.get(attention_mask_key, None),
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            # Get the generated token IDs
            generated_tokens = generated_outputs.sequences
        finally:
            # Restore original settings
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        elapsed_time = time.time() - start_time
        num_prompts = generated_tokens.shape[0]
        total_completion_tokens = generated_tokens.shape[1] - inputs[prompt_key].shape[1]
        num_tokens = total_completion_tokens * num_prompts
        avg_completion_length = total_completion_tokens
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {num_tokens}, avg length: {avg_completion_length}, speed: {tokens_per_sec:.1f} tok/s"
        )

        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(
        self,
        inputs,
        generation_config,
        pad_token_id=None,
        prompt_key="student_prompts",
        attention_mask_key="student_prompt_attention_mask",
    ):
        """Generate on-policy outputs from the selected prompts using vLLM."""
        import time

        device = self.accelerator.device

        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs[prompt_key],
            skip_special_tokens=False,
        )
        # Remove padding token text if it appears, as vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [
                p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm
            ]

        # Also decode prompts WITH special tokens for logging
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs[prompt_key],
            skip_special_tokens=False,
        )

        # system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        # target_system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        # prompts_text = [p.replace(target_system_prompt, system_prompt) for p in prompts_text]
        # Add system prompt to prompts

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None.
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        # top_p, repetition_penalty, min_p, presence_penalty are not directly in generation_config, get from trainer args
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0
        presence_penalty = self.args.presence_penalty if hasattr(self.args, "presence_penalty") else 0.0

        # Start timing for vLLM generation
        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,  # In GKD, we generate 1 completion per prompt from student
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=self.vllm_guided_decoding_regex
                )
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                presence_penalty=presence_penalty,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(
                    gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group
                )
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # Calculate and print vLLM generation statistics
        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        avg_completion_length = total_completion_tokens / num_prompts if num_prompts > 0 else 0
        tokens_per_sec = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"vLLM generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {total_completion_tokens}, avg length: {avg_completion_length:.1f}, speed: {tokens_per_sec:.1f} tok/s"
        )

        # We need to combine prompt and completion for new_input_ids
        # Tokenize prompts again to get prompt_ids on the correct device and format
        # Use prompts_text_for_vllm (without special tokens) for tokenization since vLLM expects clean text
        # Ensure add_special_tokens=False as vLLM typically handles prompts as raw text
        # Calculate max_length for prompts, ensuring it's positive
        prompt_max_length = (
            max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        )
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_completion_length length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype
                        ),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)

        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Extract completion texts from the generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, prompts_text_with_special, completion_texts

    def _generate_teacher_reasoning_vllm(
        self, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning using vLLM."""
        import time

        device = self.accelerator.device

        # Decode prompts for vLLM
        prompts_text = self.processing_class.batch_decode(
            teacher_reasoning_prompts,
            skip_special_tokens=True,
        )
        if self.processing_class.pad_token:
            prompts_text = [p.replace(self.processing_class.pad_token, "") for p in prompts_text]

        max_reasoning_length = self.reasoning_generation_config.max_new_tokens
        temperature = self.reasoning_generation_config.temperature
        top_k = (
            self.reasoning_generation_config.top_k
            if self.reasoning_generation_config.top_k and self.reasoning_generation_config.top_k > 0
            else -1
        )
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0

        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_reasoning_length,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            completion_ids = completion_ids[process_slice]

        elif self.vllm_mode == "colocate":
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_reasoning_length,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)

        elapsed_time = time.time() - start_time
        total_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        print(
            f"vLLM teacher reasoning generation done - elapsed: {elapsed_time:.2f}s, prompts: {num_prompts}, tokens: {total_tokens}, speed: {total_tokens/elapsed_time:.1f} tok/s"
        )

        # Combine prompt + completion
        prompt_tokenized = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        padded_completions = pad(
            completion_ids_tensors, padding_value=self.processing_class.pad_token_id, padding_side="right"
        )

        reasoning_ids = torch.cat([prompt_ids, padded_completions], dim=1)

        return reasoning_ids

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            # recurse into the child
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = (
                            self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        )
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Synchronize student model weights to vLLM engine."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                # use memory-efficient post-order traversal for FSDP
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # For DeepSpeed ZeRO-3, gather each parameter individually like GRPO trainer
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    def _initialize_best_checkpoint_anchor(self, initial_global_step: int):
        """Initialize best=base, or restore a saved best anchor when resuming."""
        import json
        from pathlib import Path

        best_dir = Path(self.args.output_dir) / "best"
        metadata_path = best_dir / "metadata.json"
        anchor_path = best_dir / "anchor_state.pt"

        if initial_global_step > 0 and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._best_checkpoint_score = float(metadata["score"])
            self._best_checkpoint_step = int(metadata.get("step", 0))
            if self._best_checkpoint_step > 0:
                if not anchor_path.exists():
                    raise FileNotFoundError(
                        f"Missing best anchor state while resuming: {anchor_path}"
                    )
                saved_state = torch.load(anchor_path, map_location="cpu", weights_only=True)
                unwrapped = self.accelerator.unwrap_model(self.model)
                trainable = {
                    name: param
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                }
                missing = sorted(set(trainable) - set(saved_state))
                if missing:
                    raise RuntimeError(
                        f"Best anchor state is missing {len(missing)} trainable tensors; "
                        f"first missing key: {missing[0]}"
                    )
                self._best_anchor_params = {
                    name: saved_state[name].to(param.device).detach()
                    for name, param in trainable.items()
                }
            print(
                f"Restored best policy from step {self._best_checkpoint_step} "
                f"with score {self._best_checkpoint_score:.4f}."
            )
            return

        self._best_anchor_params = None
        self._best_checkpoint_step = 0
        if self.best_checkpoint_baseline_score is None:
            evaluation = self._evaluate_current_vllm_policy("base", step=0)
            self._best_checkpoint_score = float(evaluation["average_at_n_pct"])
        else:
            self._best_checkpoint_score = float(self.best_checkpoint_baseline_score)
            evaluation = {
                "label": "base",
                "step": 0,
                "dataset": self.best_checkpoint_eval_dataset,
                "val_n": self.best_checkpoint_eval_val_n,
                "max_new_tokens": self.best_checkpoint_eval_max_new_tokens,
                "average_at_n_pct": self._best_checkpoint_score,
                "score_source": "provided_baseline",
            }

        self._persist_best_checkpoint(step=0, evaluation=evaluation, promoted=True)
        print(f"Initial best policy is base with score {self._best_checkpoint_score:.4f}.")

    def _evaluate_and_maybe_promote_checkpoint(self, step: int):
        """Evaluate a newly saved checkpoint and promote it only on strict improvement."""
        if step <= 0:
            return

        if self._last_vllm_sync_step != step:
            self._move_model_to_vllm()
            self._last_vllm_sync_step = step

        evaluation = self._evaluate_current_vllm_policy(f"checkpoint-{step}", step=step)
        score = float(evaluation["average_at_n_pct"])
        promoted = self._best_checkpoint_score is None or score > self._best_checkpoint_score
        if promoted:
            self._snapshot_current_policy_as_best()
            self._best_checkpoint_score = score
            self._best_checkpoint_step = step

        self._persist_best_checkpoint(step=step, evaluation=evaluation, promoted=promoted)
        decision = "PROMOTED" if promoted else "kept existing best"
        print(
            f"Checkpoint {step} evaluation: {score:.4f}; {decision}. "
            f"Best step={self._best_checkpoint_step}, score={self._best_checkpoint_score:.4f}."
        )

    def _evaluate_current_vllm_policy(self, label: str, step: int):
        """Evaluate the currently loaded vLLM policy using deterministic AIME24 sampling."""
        from datasets import load_dataset

        if not hasattr(self, "_best_checkpoint_eval_examples"):
            dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
            self._best_checkpoint_eval_examples = [
                {
                    "index": index,
                    "problem": example["problem"],
                    "ground_truth": str(example["answer"]),
                    "problem_id": example.get("id", index),
                }
                for index, example in enumerate(dataset)
            ]

        examples = self._best_checkpoint_eval_examples
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        local_examples = examples[rank::world_size]
        local_prompts = []
        for example in local_examples:
            user_message = (
                f"{example['problem']}\n\nPlease reason step by step, and put your final answer "
                "within \\boxed{}."
            )
            local_prompts.append(
                self.processing_class.apply_chat_template(
                    [{"role": "user", "content": user_message}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )

        if local_prompts:
            longest_prompt = max(
                len(self.processing_class.encode(prompt, add_special_tokens=False))
                for prompt in local_prompts
            )
            if longest_prompt + self.best_checkpoint_eval_max_new_tokens > self.args.max_length:
                raise ValueError(
                    "Checkpoint evaluation prompt plus max_new_tokens exceeds max_length: "
                    f"{longest_prompt} + {self.best_checkpoint_eval_max_new_tokens} > "
                    f"{self.args.max_length}."
                )

        self._wake_vllm_if_needed()
        sampling_params = SamplingParams(
            n=self.best_checkpoint_eval_val_n,
            temperature=1.0,
            top_p=0.95,
            top_k=-1,
            min_p=0.0,
            max_tokens=self.best_checkpoint_eval_max_new_tokens,
            presence_penalty=0.0,
            seed=self.best_checkpoint_eval_seed,
        )
        outputs = self.vllm_engine.generate(
            local_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        local_records = []
        for example, output in zip(local_examples, outputs):
            generations = []
            for candidate in output.outputs:
                text = candidate.text
                predicted = self._extract_boxed_answer(text)
                correct = self._grade_boxed_answer_for_reward(
                    predicted, example["ground_truth"]
                )
                generations.append(
                    {
                        "predicted_answer": predicted,
                        "correct": bool(correct),
                        "formatted": predicted is not None,
                        "token_count": len(candidate.token_ids),
                        "full_generation": text,
                    }
                )
            local_records.append({**example, "generations": generations})

        records = sorted(gather_object(local_records), key=lambda record: record["index"])
        total = sum(len(record["generations"]) for record in records)
        correct = sum(
            generation["correct"]
            for record in records
            for generation in record["generations"]
        )
        formatted = sum(
            generation["formatted"]
            for record in records
            for generation in record["generations"]
        )
        average_at_n_pct = 100.0 * correct / max(1, total)
        format_rate = 100.0 * formatted / max(1, total)
        return {
            "label": label,
            "step": step,
            "dataset": self.best_checkpoint_eval_dataset,
            "val_n": self.best_checkpoint_eval_val_n,
            "max_new_tokens": self.best_checkpoint_eval_max_new_tokens,
            "seed": self.best_checkpoint_eval_seed,
            "num_problems": len(records),
            "total_generations": total,
            "correct_generations": int(correct),
            "average_at_n_pct": average_at_n_pct,
            "formatted_generations": int(formatted),
            "format_rate": format_rate,
            "results": records,
        }

    def _persist_best_checkpoint(self, step: int, evaluation: dict, promoted: bool):
        """Persist every evaluation and keep a pruning-proof copy of the best LoRA adapter."""
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        import json
        import shutil
        from pathlib import Path

        output_dir = Path(self.args.output_dir)
        evaluations_dir = output_dir / "checkpoint_evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        evaluation_path = evaluations_dir / f"{evaluation['label']}.json"
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if not promoted:
            self.accelerator.wait_for_everyone()
            return

        best_dir = output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        for stale_name in ("adapter_model.safetensors", "adapter_model.bin", "anchor_state.pt"):
            stale_path = best_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

        if step > 0:
            checkpoint_dir = output_dir / f"checkpoint-{step}"
            for filename in (
                "adapter_model.safetensors",
                "adapter_model.bin",
                "adapter_config.json",
                "README.md",
                "chat_template.jinja",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
            ):
                source = checkpoint_dir / filename
                if source.exists():
                    shutil.copy2(source, best_dir / filename)
            cpu_anchor = {
                name: tensor.detach().cpu()
                for name, tensor in self._best_anchor_params.items()
            }
            torch.save(cpu_anchor, best_dir / "anchor_state.pt")

        metadata = {
            "step": self._best_checkpoint_step,
            "score": self._best_checkpoint_score,
            "metric": f"{self.best_checkpoint_eval_dataset}_average_at_{self.best_checkpoint_eval_val_n}",
            "max_new_tokens": self.best_checkpoint_eval_max_new_tokens,
            "seed": self.best_checkpoint_eval_seed,
            "source": "base" if self._best_checkpoint_step == 0 else f"checkpoint-{self._best_checkpoint_step}",
        }
        (best_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.accelerator.wait_for_everyone()

    def _save_generation_outputs(self, step: int):
        """Save generation outputs to disk."""
        import json
        from pathlib import Path

        local_records = [
            {**record, "process_index": self.accelerator.process_index}
            for record in self._generation_outputs_buffer
        ]
        gathered_records = gather_object(local_records)
        self._generation_outputs_buffer.clear()

        if not self.accelerator.is_main_process or not gathered_records:
            return

        # Create generations directory in output_dir
        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        output_file = generations_dir / f"generations_step_{step}.json"

        generations = []
        for record in gathered_records:
            record_with_save_step = dict(record)
            record_with_save_step["step"] = step
            generations.append(record_with_save_step)

        output_data = {
            "step": step,
            "num_samples": len(generations),
            "generations": generations,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Saved {len(generations)} generation outputs to:")
        print(f"  {output_file}")
        print(f"{'='*80}\n")

    def _training_step_prefix_consistency(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Generate K suffixes and backpropagate mean OPSD plus optional outcome loss sequentially."""
        self._wake_vllm_if_needed()
        loss_specs, records = self._generate_prefix_consistency_rollouts_vllm(
            inputs
        )
        local_batch_size = int(inputs["student_prompts"].shape[0])
        expected_branches = (
            local_batch_size * int(self.prefix_consistency_num_regenerations)
        )
        if len(loss_specs) != expected_branches:
            raise RuntimeError(
                "Prefix-consistency distillation must create exactly "
                f"B*K={expected_branches} loss branches on every rank."
            )

        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        total_loss = None
        outer_sync_gradients = self.accelerator.sync_gradients
        try:
            for branch_index, loss_spec in enumerate(loss_specs):
                is_last_branch = branch_index == expected_branches - 1
                branch_sync_gradients = outer_sync_gradients and is_last_branch
                self.accelerator.gradient_state._set_sync_gradients(
                    branch_sync_gradients
                )
                branch_sync_context = (
                    self.accelerator.no_sync(model)
                    if outer_sync_gradients
                    and not is_last_branch
                    and self.accelerator.distributed_type
                    != DistributedType.DEEPSPEED
                    else nullcontext()
                )
                with branch_sync_context:
                    branch_inputs = self._build_prefix_consistency_loss_inputs(
                        inputs,
                        loss_spec,
                        pad_token_id,
                    )
                    branch_loss = super().training_step(
                        model,
                        branch_inputs,
                        num_items_in_batch,
                    )
                total_loss = (
                    branch_loss
                    if total_loss is None
                    else total_loss + branch_loss
                )
                del branch_inputs
                empty_cache()
        finally:
            self.accelerator.gradient_state._set_sync_gradients(
                outer_sync_gradients
            )

        prompt_texts = [record["prompt"] for record in records]
        completion_texts = [record["completion"] for record in records]
        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(
            gather_object(completion_texts)
        )
        self._generation_outputs_buffer.extend(records)

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["prefix_consistency_training_step_loss"].append(
            float(total_loss.detach())
        )

        completed_optimizer_step = self.state.global_step + 1
        if (
            completed_optimizer_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(completed_optimizer_step)

        loss_scalar = float(total_loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        self._on_policy_loss_total += loss_scalar
        self._on_policy_step_equiv += 1.0 / ga
        return total_loss

    def _training_step_independent_verification(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Train on independently graded candidates without batching full-vocabulary loss tensors."""
        self._wake_vllm_if_needed()
        loss_specs, records = self._generate_independent_verification_rollouts_vllm(
            inputs, self.generation_config
        )
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        total_loss = None
        outer_sync_gradients = self.accelerator.sync_gradients
        try:
            for branch_index, loss_spec in enumerate(loss_specs):
                is_last_branch = branch_index == len(loss_specs) - 1
                branch_sync_gradients = outer_sync_gradients and is_last_branch
                self.accelerator.gradient_state._set_sync_gradients(branch_sync_gradients)

                # DeepSpeed steps inside accelerator.backward() whenever sync_gradients is true.
                # Keep intermediate trajectory backwards inside the same Trainer microbatch from
                # advancing the optimizer; for DDP, also avoid redundant gradient all-reduces.
                branch_sync_context = (
                    self.accelerator.no_sync(model)
                    if outer_sync_gradients
                    and not is_last_branch
                    and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                    else nullcontext()
                )
                with branch_sync_context:
                    branch_inputs = self._build_independent_loss_inputs(
                        inputs, loss_spec, pad_token_id
                    )
                    branch_loss = super().training_step(
                        model, branch_inputs, num_items_in_batch
                    )
                total_loss = branch_loss if total_loss is None else total_loss + branch_loss
                del branch_inputs
                empty_cache()
        finally:
            self.accelerator.gradient_state._set_sync_gradients(outer_sync_gradients)

        prompt_texts = [record["prompt"] for record in records]
        completion_texts = [record["completion"] for record in records]
        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))

        gathered_candidates = gather_object(
            [record["candidate_completion"] for record in records]
        )
        self._record_adaptive_completion_batch(gathered_candidates)
        self._generation_outputs_buffer.extend(records)

        self._maybe_update_adaptive_completion_length()
        completed_optimizer_step = self.state.global_step + 1
        if (
            completed_optimizer_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(completed_optimizer_step)

        loss_scalar = float(total_loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        self._on_policy_loss_total += loss_scalar
        self._on_policy_step_equiv += 1.0 / ga
        return total_loss

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step with self-distillation.

        If reason_first=True:
        1. Generate teacher's reasoning about the solution
        2. Append reasoning to teacher prompt
        3. Generate completions from student prompts
        4. Compute JSD loss

        Otherwise:
        1. Generate completions from student prompts
        2. Construct full sequences for both student and teacher with the generation
        3. Compute JSD loss on the generation tokens
        """
        if self.prefix_consistency_distillation:
            return self._training_step_prefix_consistency(
                model,
                inputs,
                num_items_in_batch,
            )

        if self.best_checkpoint_independent_verification:
            return self._training_step_independent_verification(
                model, inputs, num_items_in_batch
            )

        on_policy = True

        # === REASONING PHASE (if enabled) ===
        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASONING PHASE: Teacher analyzing solution...")
            print(f"{'='*80}\n")

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                # Generate teacher's reasoning
                teacher_reasoning_ids = self.generate_teacher_reasoning(
                    unwrapped_model,
                    inputs["teacher_reasoning_prompts"],
                    inputs.get("teacher_reasoning_attention_mask"),
                )

                # Decode reasoning
                reasoning_prompt_len = inputs["teacher_reasoning_prompt_length"]
                reasoning_completions = teacher_reasoning_ids[:, reasoning_prompt_len:]
                reasoning_texts = self.processing_class.batch_decode(
                    reasoning_completions, skip_special_tokens=True
                )

                # Occasionally print reasoning
                if random.random() < 0.01:
                    print(f"\n{'='*80}")
                    print(f"TEACHER REASONING SAMPLE (Step {self.state.global_step}):")
                    print(f"{'='*80}")
                    sample_idx = random.randint(0, len(reasoning_texts) - 1)
                    print(f"\n{'='*80}")
                    # Decode the prompt from token IDs to text
                    sample_prompt = self.processing_class.decode(
                        inputs["teacher_reasoning_prompts"][sample_idx], skip_special_tokens=False
                    )
                    print(f"PROMPT:\n{sample_prompt}")
                    print(f"\nReasoning:\n{reasoning_texts[sample_idx]}")
                    print(f"{'='*80}\n")

                # Update teacher prompts with reasoning
                # Construct: [teacher_reasoning_prompt][reasoning][transition_to_teaching]
                teacher_prompts_with_reasoning = torch.cat(
                    [
                        inputs["teacher_reasoning_prompts"],
                        reasoning_completions,
                        inputs["teacher_transition_tokens"],
                    ],
                    dim=1,
                )

                # Update inputs with new teacher prompts
                inputs["teacher_prompts"] = teacher_prompts_with_reasoning
                teacher_attention_mask = torch.ones_like(teacher_prompts_with_reasoning)
                if self.processing_class.pad_token_id is not None:
                    teacher_attention_mask[
                        teacher_prompts_with_reasoning == self.processing_class.pad_token_id
                    ] = 0
                inputs["teacher_prompt_attention_mask"] = teacher_attention_mask
                inputs["teacher_prompt_length"] = teacher_prompts_with_reasoning.shape[1]

        # === GENERATION PHASE ===
        use_mock_student_rollout = False
        generation_prompt_key = "student_prompts"
        generation_attention_mask_key = "student_prompt_attention_mask"
        if self.mock_student_distillation:
            use_mock_student_rollout = self._mock_student_rollout_count % 2 == 1
            self._mock_student_rollout_count += 1
            if use_mock_student_rollout:
                generation_prompt_key = "mock_student_prompts"
                generation_attention_mask_key = "mock_student_prompt_attention_mask"
                if generation_prompt_key not in inputs:
                    raise RuntimeError(
                        "Mock-student distillation expected mock_student_prompts from the data collator."
                    )

        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs,
                self.generation_config,
                self.processing_class.pad_token_id,
                prompt_key=generation_prompt_key,
                attention_mask_key=generation_attention_mask_key,
            )
            rollout_generated_ids, rollout_attention_mask, _, prompt_texts, completion_texts = result
            rollout_prompt_width = (
                rollout_generated_ids.shape[1] - int(self.generation_config.max_new_tokens)
            )
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model,
                    inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                    prompt_key=generation_prompt_key,
                    attention_mask_key=generation_attention_mask_key,
                )
                rollout_generated_ids, rollout_attention_mask, _ = result
                rollout_prompt_width = inputs[generation_prompt_key].shape[1]
                # Decode for logging
                prompt_texts = self.processing_class.batch_decode(
                    inputs[generation_prompt_key], skip_special_tokens=False
                )
                completion_ids = rollout_generated_ids[:, rollout_prompt_width:]
                completion_texts = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=False
                )

        # Get batch-level student prompt length
        student_prompt_len = inputs["student_prompt_length"]

        # The rollout may come from the mock prompt, but OPSD always scores its tokens under
        # the normal student prompt. This is the defining separation between data generation
        # and the student distribution updated by the loss.
        generation_ids = rollout_generated_ids[:, rollout_prompt_width:]
        if use_mock_student_rollout:
            generated_ids = torch.cat([inputs["student_prompts"], generation_ids], dim=1)
            generated_attention_mask = torch.ones_like(generated_ids)
            if self.processing_class.pad_token_id is not None:
                generated_attention_mask[
                    generated_ids == self.processing_class.pad_token_id
                ] = 0
        else:
            generated_ids = rollout_generated_ids
            generated_attention_mask = rollout_attention_mask

        if self.mock_student_distillation:
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["mock_student_rollout_rate"].append(
                float(use_mock_student_rollout)
            )

        teacher_correction_metadata = None
        if self.wrong_answer_teacher_correction_distillation:
            (
                generated_ids,
                generated_attention_mask,
                generation_ids,
                teacher_correction_metadata,
            ) = self._prepare_wrong_answer_teacher_correction_batch(
                model,
                inputs,
                generation_ids,
                completion_texts,
            )

        change_to_wrong_metadata = None
        if self.change_to_wrong_distillation:
            (
                generated_ids,
                generated_attention_mask,
                generation_ids,
                change_to_wrong_metadata,
            ) = self._prepare_change_to_wrong_batch(
                inputs,
                generation_ids,
                completion_texts,
            )
            inputs["wrong_boxed_only_weights"] = change_to_wrong_metadata["weights"]

        localized_recovery_metadata = None
        if self.localized_error_recovery_distillation:
            (
                generated_ids,
                generated_attention_mask,
                generation_ids,
                localized_recovery_metadata,
            ) = self._prepare_localized_error_recovery_batch(
                model,
                inputs,
                completion_texts,
            )
            student_prompt_len = inputs["student_prompt_length"]

        branch_contrastive_metadata = None
        if self.wrong_answer_branch_contrastive:
            branch_contrastive_metadata = (
                self._prepare_wrong_answer_branch_contrastive_batch(
                    model,
                    inputs,
                    completion_texts,
                )
            )

        # Construct student full sequence: [student_prompt][generation]
        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask

        # Construct teacher full sequence: [teacher_prompt][generation]
        teacher_prompts = inputs["teacher_prompts"]
        teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)

        # Create attention mask for teacher
        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0

        inputs["teacher_input_ids"] = teacher_full_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask

        if self.contrastive_teacher:
            bad_teacher_prompts = inputs["bad_teacher_prompts"]
            bad_teacher_full_ids = torch.cat([bad_teacher_prompts, generation_ids], dim=1)

            bad_teacher_attention_mask = torch.ones_like(bad_teacher_full_ids)
            if self.processing_class.pad_token_id is not None:
                bad_teacher_attention_mask[
                    bad_teacher_full_ids == self.processing_class.pad_token_id
                ] = 0

            inputs["bad_teacher_input_ids"] = bad_teacher_full_ids
            inputs["bad_teacher_attention_mask"] = bad_teacher_attention_mask
            inputs["bad_teacher_prompt_length"] = bad_teacher_prompts.shape[1]

        # Create labels for generation tokens
        # Mask prompt tokens (use per-example lengths for accurate masking)
        labels = generated_ids.clone()
        for i in range(labels.shape[0]):
            actual_prompt_len = inputs["student_prompt_lengths_per_example"][i].item()
            labels[i, :actual_prompt_len] = -100  # Mask actual prompt

        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        inputs["labels"] = labels

        reward_guided_rewards = None
        reward_guided_advantages = None
        reward_guided_predicted_answers = []
        reward_guided_ground_truth_answers = []
        if teacher_correction_metadata is not None:
            wrong_boxed_only_weights = teacher_correction_metadata["weights"]
            wrong_boxed_only_predicted_answers = teacher_correction_metadata[
                "predicted_answers"
            ]
            wrong_boxed_only_ground_truth_answers = teacher_correction_metadata[
                "ground_truth_answers"
            ]
            wrong_boxed_only_outcomes = teacher_correction_metadata["outcomes"]
        elif change_to_wrong_metadata is not None:
            wrong_boxed_only_weights = change_to_wrong_metadata["weights"]
            wrong_boxed_only_predicted_answers = change_to_wrong_metadata[
                "predicted_answers"
            ]
            wrong_boxed_only_ground_truth_answers = change_to_wrong_metadata[
                "ground_truth_answers"
            ]
            wrong_boxed_only_outcomes = change_to_wrong_metadata["outcomes"]
        else:
            wrong_boxed_only_weights = None
            wrong_boxed_only_predicted_answers = []
            wrong_boxed_only_ground_truth_answers = []
            wrong_boxed_only_outcomes = []
        if self.reward_guided_distillation:
            (
                reward_guided_rewards,
                reward_guided_advantages,
                reward_guided_predicted_answers,
                reward_guided_ground_truth_answers,
            ) = self._compute_reward_guided_advantages(
                completion_texts,
                inputs.get("reward_solutions", []),
                generated_ids.device,
                problem_texts=inputs.get("reward_problems", []),
            )
            inputs["reward_guided_rewards"] = reward_guided_rewards
            inputs["reward_guided_advantages"] = reward_guided_advantages

        if self.wrong_boxed_only_distillation or self.mock_student_distillation:
            (
                wrong_boxed_only_weights,
                wrong_boxed_only_predicted_answers,
                wrong_boxed_only_ground_truth_answers,
                wrong_boxed_only_outcomes,
            ) = self._compute_wrong_boxed_only_weights(
                completion_texts,
                inputs.get("reward_solutions", []),
                generated_ids.device,
                metric_prefix=(
                    "mock_student_mock_rollout"
                    if use_mock_student_rollout
                    else (
                        "mock_student_normal_rollout"
                        if self.mock_student_distillation
                        else "wrong_boxed_only"
                    )
                ),
            )
            inputs["wrong_boxed_only_weights"] = wrong_boxed_only_weights

        best_checkpoint_advantages = None
        best_checkpoint_predicted_answers = []
        best_checkpoint_ground_truth_answers = []
        best_checkpoint_outcomes = []
        if self.best_checkpoint_distillation:
            (
                best_checkpoint_advantages,
                best_checkpoint_predicted_answers,
                best_checkpoint_ground_truth_answers,
                best_checkpoint_outcomes,
            ) = self._compute_best_checkpoint_advantages(
                completion_texts,
                inputs.get("reward_solutions", []),
                generated_ids.device,
            )
            inputs["best_checkpoint_advantages"] = best_checkpoint_advantages

        reverse_teacher_prompt_texts = []
        reverse_teacher_completion_texts = []
        reverse_teacher_token_counts = []
        if self.reverse_teacher_generation:
            reverse_teacher_prompt_len = inputs["reverse_teacher_prompt_length"]
            reverse_teacher_prompt_texts = self.processing_class.batch_decode(
                inputs["reverse_teacher_prompts"], skip_special_tokens=False
            )

            if "reverse_teacher_cached_completion_ids" in inputs:
                reverse_generation_ids = inputs["reverse_teacher_cached_completion_ids"]
                max_new_tokens = int(self.generation_config.max_new_tokens)
                if reverse_generation_ids.shape[1] > max_new_tokens:
                    reverse_generation_ids = reverse_generation_ids[:, :max_new_tokens]
                reverse_teacher_generated_ids = torch.cat(
                    [inputs["reverse_teacher_prompts"], reverse_generation_ids], dim=1
                )
                reverse_teacher_attention_mask = torch.ones_like(reverse_teacher_generated_ids)
                if self.processing_class.pad_token_id is not None:
                    reverse_teacher_attention_mask[
                        reverse_teacher_generated_ids == self.processing_class.pad_token_id
                    ] = 0
                reverse_teacher_completion_texts = self.processing_class.batch_decode(
                    reverse_generation_ids, skip_special_tokens=False
                )
            else:
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    adapter_context = (
                        self.accelerator.unwrap_model(model).disable_adapter()
                        if self.fixed_teacher and is_peft_model(model)
                        else nullcontext()
                    )
                    with adapter_context:
                        reverse_result = self.generate_on_policy_outputs(
                            unwrapped_model,
                            inputs,
                            self.generation_config,
                            self.processing_class.pad_token_id,
                            prompt_key="reverse_teacher_prompts",
                            attention_mask_key="reverse_teacher_prompt_attention_mask",
                        )
                        reverse_teacher_generated_ids, reverse_teacher_attention_mask, _ = reverse_result

                reverse_generation_ids = reverse_teacher_generated_ids[:, reverse_teacher_prompt_len:]
                reverse_teacher_completion_texts = self.processing_class.batch_decode(
                    reverse_generation_ids, skip_special_tokens=False
                )

            reverse_student_generated_ids = torch.cat([inputs["student_prompts"], reverse_generation_ids], dim=1)
            reverse_student_attention_mask = torch.ones_like(reverse_student_generated_ids)
            reverse_teacher_attention_mask = torch.ones_like(reverse_teacher_generated_ids)
            if self.processing_class.pad_token_id is not None:
                reverse_student_attention_mask[
                    reverse_student_generated_ids == self.processing_class.pad_token_id
                ] = 0
                reverse_teacher_attention_mask[
                    reverse_teacher_generated_ids == self.processing_class.pad_token_id
                ] = 0

            reverse_labels = reverse_student_generated_ids.clone()
            for i in range(reverse_labels.shape[0]):
                actual_prompt_len = inputs["student_prompt_lengths_per_example"][i].item()
                reverse_labels[i, :actual_prompt_len] = -100
            if self.processing_class.pad_token_id is not None:
                reverse_labels[reverse_labels == self.processing_class.pad_token_id] = -100

            inputs["reverse_student_input_ids"] = reverse_student_generated_ids
            inputs["reverse_student_attention_mask"] = reverse_student_attention_mask
            inputs["reverse_teacher_input_ids"] = reverse_teacher_generated_ids
            inputs["reverse_teacher_attention_mask"] = reverse_teacher_attention_mask
            inputs["reverse_labels"] = reverse_labels
            inputs["reverse_student_prompt_length"] = inputs["student_prompt_length"]
            inputs["reverse_teacher_prompt_length"] = reverse_teacher_prompt_len

            if self.processing_class.pad_token_id is not None:
                reverse_teacher_token_counts = (
                    reverse_generation_ids != self.processing_class.pad_token_id
                ).sum(dim=1).detach().cpu().tolist()
            else:
                reverse_teacher_token_counts = [reverse_generation_ids.shape[1]] * reverse_generation_ids.shape[0]

            if reverse_teacher_completion_texts:
                boxed_rate = sum(
                    self._extract_boxed_answer(text or "") is not None
                    for text in reverse_teacher_completion_texts
                ) / len(reverse_teacher_completion_texts)
                self._metrics["train"]["reverse_teacher_generated_boxed_rate"].append(float(boxed_rate))
                self._metrics["train"]["reverse_teacher_generated_avg_tokens"].append(
                    float(sum(reverse_teacher_token_counts) / len(reverse_teacher_token_counts))
                )

        # Log prompt and completion texts
        gathered_prompt_texts = gather_object(prompt_texts)
        gathered_completion_texts = gather_object(completion_texts)
        self._textual_logs["prompt"].extend(gathered_prompt_texts)
        self._textual_logs["completion"].extend(gathered_completion_texts)
        if self.reverse_teacher_generation:
            self._textual_logs["prompt"].extend(gather_object(reverse_teacher_prompt_texts))
            self._textual_logs["completion"].extend(gather_object(reverse_teacher_completion_texts))
        self._record_adaptive_completion_batch(gathered_completion_texts)

        # Collect generation outputs for saving
        normal_student_prompt_texts = None
        if self.mock_student_distillation:
            normal_student_prompt_texts = self.processing_class.batch_decode(
                inputs["student_prompts"], skip_special_tokens=False
            )
        for idx, (prompt, completion) in enumerate(zip(prompt_texts, completion_texts)):
            output_record = {
                "step": self.state.global_step,
                "rollout_step": self.state.global_step + 1,
                "prompt": prompt,
                "completion": completion,
            }
            if self.mock_student_distillation:
                output_record.update(
                    {
                        "component": "mock_student_distillation",
                        "rollout_source": (
                            "mock_student" if use_mock_student_rollout else "student"
                        ),
                        "loss_student_prompt": normal_student_prompt_texts[idx],
                    }
                )
            if self.reward_guided_distillation:
                reward_value = (
                    reward_guided_rewards[idx].detach().item()
                    if reward_guided_rewards is not None and idx < reward_guided_rewards.numel()
                    else None
                )
                advantage_value = (
                    reward_guided_advantages[idx].detach().item()
                    if reward_guided_advantages is not None and idx < reward_guided_advantages.numel()
                    else None
                )
                output_record.update(
                    {
                        "reward": reward_value,
                        "advantage": advantage_value,
                        "predicted_answer": (
                            reward_guided_predicted_answers[idx]
                            if idx < len(reward_guided_predicted_answers)
                            else None
                        ),
                        "ground_truth_answer": (
                            reward_guided_ground_truth_answers[idx]
                            if idx < len(reward_guided_ground_truth_answers)
                            else None
                        ),
                    }
                )
            if (
                self.wrong_boxed_only_distillation
                or self.mock_student_distillation
                or self.change_to_wrong_distillation
                or self.wrong_answer_teacher_correction_distillation
            ):
                weight_value = (
                    wrong_boxed_only_weights[idx].detach().item()
                    if wrong_boxed_only_weights is not None and idx < wrong_boxed_only_weights.numel()
                    else None
                )
                output_record.update(
                    {
                        "wrong_boxed_only_weight": weight_value,
                        "wrong_boxed_only_outcome": (
                            wrong_boxed_only_outcomes[idx]
                            if idx < len(wrong_boxed_only_outcomes)
                            else None
                        ),
                        "predicted_answer": (
                            wrong_boxed_only_predicted_answers[idx]
                            if idx < len(wrong_boxed_only_predicted_answers)
                            else None
                        ),
                        "ground_truth_answer": (
                            wrong_boxed_only_ground_truth_answers[idx]
                            if idx < len(wrong_boxed_only_ground_truth_answers)
                            else None
                        ),
                        "used_for_loss": bool(weight_value),
                    }
                )
            if self.change_to_wrong_distillation:
                output_record.update(
                    {
                        "component": "change_to_wrong_distillation",
                        "change_to_wrong_outcome": change_to_wrong_metadata["outcomes"][idx],
                        "target_completion": change_to_wrong_metadata[
                            "target_completion_texts"
                        ][idx],
                        "target_answer": change_to_wrong_metadata["target_answers"][idx],
                        "number_replacements": change_to_wrong_metadata[
                            "number_replacements"
                        ][idx],
                        "connector_replacements": change_to_wrong_metadata[
                            "connector_replacements"
                        ][idx],
                        "boxed_answer_forced": change_to_wrong_metadata[
                            "boxed_answer_forced"
                        ][idx],
                    }
                )
            if self.localized_error_recovery_distillation:
                corruption = localized_recovery_metadata["corruptions"][idx]
                output_record.update(
                    {
                        "component": "localized_error_recovery_distillation",
                        "localized_recovery_outcome": localized_recovery_metadata[
                            "outcomes"
                        ][idx],
                        "used_for_loss": corruption is not None,
                        "predicted_answer": localized_recovery_metadata[
                            "predicted_answers"
                        ][idx],
                        "ground_truth_answer": localized_recovery_metadata[
                            "ground_truth_answers"
                        ][idx],
                        "clean_prefix": (
                            corruption["clean_prefix"] if corruption is not None else None
                        ),
                        "corrupted_prefix": (
                            corruption["corrupted_prefix"]
                            if corruption is not None
                            else None
                        ),
                        "original_number": (
                            corruption["original_number"] if corruption is not None else None
                        ),
                        "replacement_number": (
                            corruption["replacement_number"]
                            if corruption is not None
                            else None
                        ),
                        "corrupted_prefix_token_count": (
                            corruption["prefix_token_count"]
                            if corruption is not None
                            else 0
                        ),
                        "recovery_completion": localized_recovery_metadata[
                            "recovery_completion_texts"
                        ][idx],
                        "recovery_token_count": localized_recovery_metadata[
                            "recovery_token_counts"
                        ][idx],
                        "recovery_answer": localized_recovery_metadata[
                            "recovery_answers"
                        ][idx],
                        "recovery_correct": localized_recovery_metadata[
                            "recovery_correct"
                        ][idx],
                    }
                )
            if self.wrong_answer_teacher_correction_distillation:
                correction_text = teacher_correction_metadata[
                    "correction_completion_texts"
                ][idx]
                output_record.update(
                    {
                        "component": "wrong_answer_teacher_correction",
                        "used_for_loss": bool(weight_value),
                        "teacher_correction_prompt": teacher_correction_metadata[
                            "correction_prompt_texts"
                        ][idx],
                        "teacher_correction_completion": correction_text,
                        "teacher_correction_token_count": teacher_correction_metadata[
                            "correction_token_counts"
                        ][idx],
                        "teacher_correction_has_boxed": (
                            self._extract_boxed_answer(correction_text or "") is not None
                            if correction_text is not None
                            else None
                        ),
                    }
                )
            if self.wrong_answer_branch_contrastive:
                branch_weight = float(
                    branch_contrastive_metadata["sample_weights"][idx].detach().item()
                )
                output_record.update(
                    {
                        "component": "wrong_answer_branch_contrastive",
                        "used_for_loss": bool(branch_weight),
                        "branch_contrastive_weight": branch_weight,
                        "outcome": branch_contrastive_metadata["outcomes"][idx],
                        "predicted_answer": branch_contrastive_metadata[
                            "predicted_answers"
                        ][idx],
                        "ground_truth_answer": branch_contrastive_metadata[
                            "ground_truth_answers"
                        ][idx],
                        "error_locator_prompt": branch_contrastive_metadata[
                            "locator_prompts"
                        ][idx],
                        "error_locator_output": branch_contrastive_metadata[
                            "locator_outputs"
                        ][idx],
                        "error_start_quote": branch_contrastive_metadata[
                            "error_quotes"
                        ][idx],
                        "error_quote_parsed": branch_contrastive_metadata[
                            "error_quote_parsed"
                        ][idx],
                        "error_quote_found_in_student": branch_contrastive_metadata[
                            "error_quote_found"
                        ][idx],
                        "error_quote_match_mode": branch_contrastive_metadata[
                            "error_quote_match_modes"
                        ][idx],
                        "error_char_start": branch_contrastive_metadata[
                            "error_char_starts"
                        ][idx],
                        "branch_token_index": branch_contrastive_metadata[
                            "branch_token_indices"
                        ][idx],
                        "valid_student_prefix": branch_contrastive_metadata[
                            "prefix_texts"
                        ][idx],
                        "wrong_continuation": branch_contrastive_metadata[
                            "wrong_continuation_texts"
                        ][idx],
                        "teacher_correction_prompt": branch_contrastive_metadata[
                            "correction_prompts"
                        ][idx],
                        "correct_continuation": branch_contrastive_metadata[
                            "correction_texts"
                        ][idx],
                        "branch_skip_reason": branch_contrastive_metadata[
                            "skip_reasons"
                        ][idx],
                    }
                )
            if self.best_checkpoint_distillation:
                advantage_value = (
                    best_checkpoint_advantages[idx].detach().item()
                    if best_checkpoint_advantages is not None
                    and idx < best_checkpoint_advantages.numel()
                    else None
                )
                output_record["advantage"] = advantage_value
                output_record.update(
                    {
                        "outcome": (
                            best_checkpoint_outcomes[idx]
                            if idx < len(best_checkpoint_outcomes)
                            else None
                        ),
                        "predicted_answer": (
                            best_checkpoint_predicted_answers[idx]
                            if idx < len(best_checkpoint_predicted_answers)
                            else None
                        ),
                        "ground_truth_answer": (
                            best_checkpoint_ground_truth_answers[idx]
                            if idx < len(best_checkpoint_ground_truth_answers)
                            else None
                        ),
                        "best_checkpoint_step": self._best_checkpoint_step,
                        "best_checkpoint_score": self._best_checkpoint_score,
                    }
                )
            if self.adaptive_completion_length:
                output_record["max_completion_length"] = int(self.generation_config.max_new_tokens)
            self._generation_outputs_buffer.append(output_record)
        if self.reverse_teacher_generation:
            for prompt, completion, token_count in zip(
                reverse_teacher_prompt_texts,
                reverse_teacher_completion_texts,
                reverse_teacher_token_counts,
            ):
                output_record = {
                    "step": self.state.global_step,
                    "rollout_step": self.state.global_step + 1,
                    "component": "reverse_teacher_generation",
                    "prompt": prompt,
                    "completion": completion,
                    "completion_token_count": int(token_count),
                    "has_boxed": self._extract_boxed_answer(completion or "") is not None,
                }
                if self.adaptive_completion_length:
                    output_record["max_completion_length"] = int(self.generation_config.max_new_tokens)
                self._generation_outputs_buffer.append(output_record)

        # Occasionally print student's generation with 1% probability
        if random.random() < 0.01:
            print(f"\n{'='*80}")
            print(f"STUDENT GENERATION SAMPLE (Step {self.state.global_step}):")
            print(f"{'='*80}")
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            print(f"\nPrompt:\n{prompt_texts[sample_idx]}")
            print(f"\nCompletion:\n{completion_texts[sample_idx]}")
            print(f"{'='*80}\n")

        loss = super().training_step(model, inputs, num_items_in_batch)
        self._maybe_update_adaptive_completion_length()

        # Save generation outputs every N steps
        completed_optimizer_step = self.state.global_step + 1
        if (
            completed_optimizer_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(completed_optimizer_step)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        local_metric_items = [
            (key, float(value))
            for key, values in self._metrics[mode].items()
            for value in values
        ]
        if (
            getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
            and dist.is_available()
            and dist.is_initialized()
        ):
            metric_items = gather_object(local_metric_items)
        else:
            metric_items = local_metric_items

        metric_values = defaultdict(list)
        for key, value in metric_items:
            metric_values[key].append(value)
        metrics = {
            key: sum(values) / len(values)
            for key, values in metric_values.items()
            if values
        }

        derived_rate_counts = {
            "independent_candidate_correct_rate": (
                "independent_candidate_correct_count",
                "independent_candidate_count",
            ),
            "independent_verification_rate": (
                "independent_verification_count",
                "independent_candidate_count",
            ),
            "independent_verification_correct_rate": (
                "independent_verification_correct_count",
                "independent_verification_count",
            ),
            "independent_update_rate": (
                "independent_update_count",
                "independent_candidate_count",
            ),
            "change_to_wrong_corruption_success_rate": (
                "change_to_wrong_corruption_success_count",
                "change_to_wrong_corruption_eligible_count",
            ),
            "prefix_consistency_initial_boxed_rate": (
                "prefix_consistency_initial_boxed_count",
                "prefix_consistency_initial_count",
            ),
            "prefix_consistency_initial_correct_rate": (
                "prefix_consistency_initial_correct_count",
                "prefix_consistency_initial_count",
            ),
            "prefix_consistency_initial_wrong_rate": (
                "prefix_consistency_initial_wrong_count",
                "prefix_consistency_initial_count",
            ),
            "prefix_consistency_initial_no_boxed_rate": (
                "prefix_consistency_initial_no_boxed_count",
                "prefix_consistency_initial_count",
            ),
            "prefix_consistency_initial_malformed_boxed_rate": (
                "prefix_consistency_initial_malformed_boxed_count",
                "prefix_consistency_initial_count",
            ),
            "prefix_consistency_invalid_reference_rate": (
                "prefix_consistency_invalid_reference_count",
                "prefix_consistency_initial_count",
            ),
            "prefix_consistency_prefix_box_leak_rate": (
                "prefix_consistency_prefix_box_leak_count",
                "prefix_consistency_initial_eligible_count",
            ),
            "prefix_consistency_initial_eligible_rate": (
                "prefix_consistency_initial_eligible_count",
                "prefix_consistency_initial_count",
            ),
            "prefix_consistency_selected_rate": (
                "prefix_consistency_selected_count",
                "prefix_consistency_initial_count",
            ),
            "prefix_consistency_mixed_selection_rate": (
                "prefix_consistency_mixed_selected_count",
                "prefix_consistency_initial_eligible_count",
            ),
            "prefix_consistency_regeneration_boxed_rate": (
                "prefix_consistency_regeneration_boxed_count",
                "prefix_consistency_regeneration_count",
            ),
            "prefix_consistency_regeneration_gold_value": (
                "prefix_consistency_regeneration_gold_count",
                "prefix_consistency_regeneration_count",
            ),
            "prefix_consistency_reproduction_correct": (
                "prefix_consistency_correct_reproduction_count",
                "prefix_consistency_correct_regeneration_count",
            ),
            "prefix_consistency_reproduction_wrong": (
                "prefix_consistency_wrong_reproduction_count",
                "prefix_consistency_wrong_regeneration_count",
            ),
            "prefix_consistency_correct_regeneration_gold_value": (
                "prefix_consistency_correct_gold_count",
                "prefix_consistency_correct_regeneration_count",
            ),
            "prefix_consistency_wrong_recovery_rate": (
                "prefix_consistency_wrong_gold_count",
                "prefix_consistency_wrong_regeneration_count",
            ),
            "prefix_consistency_unboxed_initial_recovery_rate": (
                "prefix_consistency_unboxed_initial_gold_count",
                "prefix_consistency_unboxed_initial_regeneration_count",
            ),
            "prefix_consistency_outcome_active_group_rate": (
                "prefix_consistency_outcome_active_group_count",
                "prefix_consistency_initial_eligible_count",
            ),
        }
        for rate_key, (numerator_key, denominator_key) in derived_rate_counts.items():
            if numerator_key not in metric_values or denominator_key not in metric_values:
                continue
            numerator = sum(metric_values[numerator_key])
            denominator = sum(metric_values[denominator_key])
            metrics[rate_key] = numerator / denominator if denominator > 0 else 0.0
            metrics.pop(numerator_key, None)
            metrics.pop(denominator_key, None)

        if (
            "prefix_consistency_reproduction_correct" in metrics
            and "prefix_consistency_reproduction_wrong" in metrics
            and sum(
                metric_values.get(
                    "prefix_consistency_correct_regeneration_count", []
                )
            )
            > 0
            and sum(
                metric_values.get(
                    "prefix_consistency_wrong_regeneration_count", []
                )
            )
            > 0
        ):
            metrics["prefix_consistency_discrimination_gap"] = (
                metrics["prefix_consistency_reproduction_correct"]
                - metrics["prefix_consistency_reproduction_wrong"]
            )

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            # Track on/off-policy loss statistics
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                ],
                dtype=torch.float64,
                device=device,
            )

            # Sum across processes so we mirror Trainer's distributed reduction
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
            ) = vec.tolist()

            # Compute category averages over the *same window* as Trainer's logs
            # (avoid div-by-zero if, e.g., no on-policy steps in the window)
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            # Reset window accumulators after logging (just like Trainer resets its window)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        # SFTTrainer.log() also reads self._metrics. We already gathered and reduced those
        # values above, so clear the buffer before delegating to avoid re-inserting local
        # per-rank values (and internal count metrics) into the final logs.
        self._metrics[mode].clear()
        super().log(logs, start_time)

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                if self.num_completions_to_print and len(df) > 0:
                    df = df.sample(n=self.num_completions_to_print, random_state=42)
                wandb.log({"completions": wandb.Table(dataframe=df)})
