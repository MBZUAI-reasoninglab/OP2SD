#!/usr/bin/env python3
"""Aggregate repeated decoding-seed evaluations without changing metric definitions."""

import argparse
import json
import math
import statistics
from pathlib import Path


def corrected_mc_se_pct(correct_counts, samples_per_problem):
    """Unbiased plug-in Monte Carlo SE for a mean over fixed problems."""
    if samples_per_problem <= 1:
        raise ValueError("corrected SE requires at least two generations per problem")
    num_problems = len(correct_counts)
    if num_problems == 0:
        raise ValueError("cannot aggregate an empty problem set")
    variance_sum = 0.0
    for count in correct_counts:
        p_hat = count / samples_per_problem
        variance_sum += p_hat * (1.0 - p_hat)
    return 100.0 / num_problems * math.sqrt(
        variance_sum / (samples_per_problem - 1)
    )


def clipped_interval(center, half_width):
    return [max(0.0, center - half_width), min(100.0, center + half_width)]


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_seed_result(path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    required = [
        "dataset",
        "val_n",
        "num_problems",
        "results",
        "seed",
        "checkpoint_loaded",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path}: missing summary fields: {', '.join(missing)}")
    if data["seed"] is None:
        raise ValueError(f"{path}: seed metadata is null")
    if not isinstance(data["checkpoint_loaded"], bool):
        raise ValueError(f"{path}: checkpoint_loaded must be a boolean")
    checkpoint_requested = data.get("checkpoint_dir") is not None
    if data["checkpoint_loaded"] != checkpoint_requested:
        raise ValueError(
            f"{path}: checkpoint request/load metadata is inconsistent"
        )

    val_n = int(data["val_n"])
    results = data["results"]
    if len(results) != int(data["num_problems"]):
        raise ValueError(
            f"{path}: results={len(results)} but num_problems={data['num_problems']}"
        )

    identities = []
    correct_counts = []
    pass_flags = []
    majority_flags = []
    formatted_count = 0

    for index, result in enumerate(results):
        generations = result.get("generations", [])
        if len(generations) != val_n:
            raise ValueError(
                f"{path}: problem {index} has {len(generations)} generations, expected {val_n}"
            )
        correct_count = sum(bool(item.get("correct")) for item in generations)
        stored_correct_count = int(result.get("num_correct", -1))
        if correct_count != stored_correct_count:
            raise ValueError(
                f"{path}: problem {index} correct count {correct_count} != stored {stored_correct_count}"
            )
        pass_flag = bool(result.get("pass_at_n"))
        if pass_flag != (correct_count > 0):
            raise ValueError(f"{path}: problem {index} has inconsistent pass_at_n")

        identities.append(
            (
                result.get("problem_id", index),
                result.get("problem"),
                result.get("ground_truth"),
            )
        )
        correct_counts.append(correct_count)
        pass_flags.append(pass_flag)
        majority_flags.append(bool(result.get("majority_vote_correct")))
        formatted_count += sum(bool(item.get("formatted")) for item in generations)

    num_problems = len(results)
    total_generations = num_problems * val_n
    metrics = {
        "seed": int(data["seed"]),
        "avg_at_n_pct": 100.0 * sum(correct_counts) / total_generations,
        "corrected_mc_se_pct": corrected_mc_se_pct(correct_counts, val_n),
        "pass_at_n_pct": 100.0 * sum(pass_flags) / num_problems,
        "majority_at_n_pct": 100.0 * sum(majority_flags) / num_problems,
        "format_rate_pct": 100.0 * formatted_count / total_generations,
        "correct_counts": correct_counts,
    }
    metrics["two_sigma_interval_pct"] = clipped_interval(
        metrics["avg_at_n_pct"], 2.0 * metrics["corrected_mc_se_pct"]
    )
    metrics["normal_95_interval_pct"] = clipped_interval(
        metrics["avg_at_n_pct"], 1.96 * metrics["corrected_mc_se_pct"]
    )

    comparable_config = {
        key: data.get(key)
        for key in [
            "dataset",
            "dataset_path",
            "dataset_sha256",
            "base_model",
            "checkpoint_dir",
            "checkpoint_loaded",
            "enable_thinking",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "max_new_tokens",
            "max_model_len",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "val_n",
            "num_problems",
        ]
    }
    return data, comparable_config, identities, metrics


def format_interval(interval):
    return f"[{interval[0]:.2f}, {interval[1]:.2f}]"


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate seed-specific evaluate_math.py JSON files."
    )
    parser.add_argument("result_files", nargs="+", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    loaded = [load_seed_result(path) for path in args.result_files]
    first_config = loaded[0][1]
    first_identities = loaded[0][2]
    for path, (_, config, identities, _) in zip(args.result_files[1:], loaded[1:]):
        if config != first_config:
            raise ValueError(f"{path}: evaluation config differs from the first seed")
        if identities != first_identities:
            raise ValueError(f"{path}: problem IDs/order/ground truths differ from the first seed")

    seed_metrics = [item[3] for item in loaded]
    seeds = [item["seed"] for item in seed_metrics]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"decoding seeds must be distinct, got {seeds}")

    val_n = int(first_config["val_n"])
    num_problems = int(first_config["num_problems"])
    samples_per_problem = val_n * len(seed_metrics)
    pooled_counts = [
        sum(metric["correct_counts"][problem_index] for metric in seed_metrics)
        for problem_index in range(num_problems)
    ]

    avg_values = [metric["avg_at_n_pct"] for metric in seed_metrics]
    pass_values = [metric["pass_at_n_pct"] for metric in seed_metrics]
    majority_values = [metric["majority_at_n_pct"] for metric in seed_metrics]
    format_values = [metric["format_rate_pct"] for metric in seed_metrics]

    mean_avg = statistics.mean(avg_values)
    pooled_avg = 100.0 * sum(pooled_counts) / (num_problems * samples_per_problem)
    if not math.isclose(mean_avg, pooled_avg, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("mean seed Avg@N does not equal pooled correctness rate")

    pooled_se = corrected_mc_se_pct(pooled_counts, samples_per_problem)
    aggregate = {
        "metric": f"Mean Avg@{val_n} over {len(seeds)} decoding seeds",
        "mean_avg_at_n_pct": mean_avg,
        "pooled_correctness_pct": pooled_avg,
        "corrected_mc_se_pct": pooled_se,
        "two_sigma_interval_pct": clipped_interval(mean_avg, 2.0 * pooled_se),
        "normal_95_interval_pct": clipped_interval(mean_avg, 1.96 * pooled_se),
        "empirical_seed_sd_pct": sample_sd(avg_values),
        "empirical_seed_sem_pct": sample_sd(avg_values) / math.sqrt(len(avg_values)),
        "mean_pass_at_n_pct": statistics.mean(pass_values),
        "pass_at_n_seed_sd_pct": sample_sd(pass_values),
        "mean_majority_at_n_pct": statistics.mean(majority_values),
        "majority_at_n_seed_sd_pct": sample_sd(majority_values),
        "mean_format_rate_pct": statistics.mean(format_values),
        "format_rate_seed_sd_pct": sample_sd(format_values),
        "samples_per_problem_pooled": samples_per_problem,
        "total_generations": num_problems * samples_per_problem,
        "boundary_problem_count": sum(
            count in (0, samples_per_problem) for count in pooled_counts
        ),
    }

    output = {
        "label": args.label,
        "seeds": seeds,
        "config": first_config,
        "per_seed": [
            {key: value for key, value in metric.items() if key != "correct_counts"}
            for metric in seed_metrics
        ],
        "aggregate": aggregate,
        "notes": [
            "The corrected Monte Carlo SE treats the benchmark problems as fixed and measures decoding randomness only.",
            f"Pass@N and Majority@N are computed within each seed and then averaged; they are not converted to pooled Pass@{samples_per_problem} or Majority@{samples_per_problem} metrics.",
            "The 2-sigma interval is a clipped normal-approximation interval using plus/minus 2 corrected SE.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    markdown_path = args.output.with_suffix(".md")
    lines = [
        f"# {len(seeds)}-seed evaluation: {args.label}",
        "",
        f"Seeds: `{', '.join(map(str, seeds))}`",
        "",
        f"Problems: {num_problems}; generations per problem per seed: {val_n}; pooled: {samples_per_problem}",
        "",
        f"| Seed | Avg@{val_n} | Corrected SE | 2-sigma interval | Pass@{val_n} | Majority@{val_n} | Format |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in seed_metrics:
        lines.append(
            f"| {metric['seed']} | {metric['avg_at_n_pct']:.2f}% | "
            f"{metric['corrected_mc_se_pct']:.2f} pp | "
            f"{format_interval(metric['two_sigma_interval_pct'])} | "
            f"{metric['pass_at_n_pct']:.2f}% | {metric['majority_at_n_pct']:.2f}% | "
            f"{metric['format_rate_pct']:.2f}% |"
        )
    lines.extend(
        [
            f"| **{len(seeds)}-seed mean** | **{mean_avg:.2f}%** | **{pooled_se:.2f} pp** | "
            f"**{format_interval(aggregate['two_sigma_interval_pct'])}** | "
            f"**{aggregate['mean_pass_at_n_pct']:.2f}%** | "
            f"**{aggregate['mean_majority_at_n_pct']:.2f}%** | "
            f"**{aggregate['mean_format_rate_pct']:.2f}%** |",
            "",
            f"Empirical seed SD of Avg@{val_n}: **{aggregate['empirical_seed_sd_pct']:.2f} percentage points**.",
            "",
            "The corrected SE and 2-sigma interval measure decoding randomness on this fixed problem set; they do not include training-seed or problem-set uncertainty.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"{args.label}: mean Avg@{val_n}={mean_avg:.2f}%")
    print(f"Corrected pooled SE={pooled_se:.2f} percentage points")
    print(f"2-sigma interval={format_interval(aggregate['two_sigma_interval_pct'])}")
    print(f"Empirical seed SD={aggregate['empirical_seed_sd_pct']:.2f} percentage points")
    print(f"Saved JSON: {args.output}")
    print(f"Saved Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
