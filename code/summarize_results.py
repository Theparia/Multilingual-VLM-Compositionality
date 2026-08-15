from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_config import BENCHMARK_NAMES, BENCHMARK_SUBSETS


LANGUAGE_ORDER = {"en": 0, "de": 1, "es": 2, "fa": 3}

# Where translate.py wrote the translated data and its collapse reports.
DATA_ROOT = Path("data")

# Per-sample field holding the per-record outcome for each reported metric.
METRIC_FIELDS = {"Accuracy": "correct", "ITT": "ITT", "TOT": "TOT"}


def metric_from_filename(benchmark: str, path: Path) -> str:
    """Infer the reported evaluation metric from a benchmark and result filename."""
    if benchmark == "sc":
        return "Accuracy"
    if path.name.startswith("ITT_"):
        return "ITT"
    if path.name.startswith("TOT_"):
        return "TOT"
    raise ValueError(f"Unrecognized SugarCrepe++ aggregate file: {path}")


def read_excluded_ids(benchmark: str) -> dict[str, set[str]]:
    """Collect, per subset, the sample ids flagged as collapsed in any language."""
    excluded: dict[str, set[str]] = {subset: set() for subset in BENCHMARK_SUBSETS[benchmark]}
    benchmark_data_root = DATA_ROOT / benchmark
    if not benchmark_data_root.is_dir():
        raise FileNotFoundError(
            f"Translation reports not found: {benchmark_data_root}. "
            "Run translate.py first, or point DATA_ROOT at the translated data."
        )

    for language_dir in sorted(path for path in benchmark_data_root.iterdir() if path.is_dir()):
        for subset in BENCHMARK_SUBSETS[benchmark]:
            issues_path = language_dir / f"{subset}_issues.json"
            if not issues_path.is_file():
                continue
            with issues_path.open(encoding="utf-8") as handle:
                issues = json.load(handle)
            # One entry per colliding field pair, so a record can appear more
            # than once; the set collapses those back to one exclusion.
            excluded[subset].update(str(issue["sample_id"]) for issue in issues)
    return excluded


def read_per_sample(language_dir: Path) -> dict[str, dict[str, dict[str, object]]]:
    """Load the single per-sample result file stored in a language directory."""
    per_sample_paths = sorted(language_dir.glob("*_per_sample.json"))
    if len(per_sample_paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_per_sample.json in {language_dir}, "
            f"found {len(per_sample_paths)}"
        )
    with per_sample_paths[0].open(encoding="utf-8") as handle:
        result = json.load(handle)
    subsets = result.get("subsets")
    if not isinstance(subsets, dict):
        raise ValueError(f"No per-sample 'subsets' object in {per_sample_paths[0]}")
    return subsets


def score_subsets(
    per_sample: dict[str, dict[str, dict[str, object]]],
    benchmark: str,
    metric: str,
    excluded: dict[str, set[str]],
    source: Path,
) -> tuple[list[float], int]:
    """Compute subset accuracies after applying the shared exclusion set.
    Return the subset scores and the total number of retained examples."""
    field = METRIC_FIELDS[metric]
    values: list[float] = []
    kept_total = 0

    for subset in BENCHMARK_SUBSETS[benchmark]:
        samples = per_sample.get(subset)
        if not isinstance(samples, dict):
            raise ValueError(f"Missing per-sample records for {subset} in {source}")

        kept = [
            sample
            for sample_id, sample in samples.items()
            if str(sample_id) not in excluded[subset]
        ]
        if not kept:
            raise ValueError(f"Every {subset} record was excluded in {source}")
        for sample in kept:
            if not isinstance(sample.get(field), bool):
                raise ValueError(f"Missing boolean '{field}' for a {subset} record in {source}")

        values.append(sum(bool(sample[field]) for sample in kept) / len(kept))
        kept_total += len(kept)

    return values, kept_total


def read_rows(output_root: Path, benchmark: str) -> list[dict[str, object]]:
    """Collect scored result rows for every model, language, and metric."""
    benchmark_root = output_root / benchmark
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Benchmark output directory not found: {benchmark_root}")

    excluded = read_excluded_ids(benchmark)
    rows: list[dict[str, object]] = []
    for model_dir in sorted(path for path in benchmark_root.iterdir() if path.is_dir()):
        language_dirs = sorted(
            (path for path in model_dir.iterdir() if path.is_dir()),
            key=lambda path: (LANGUAGE_ORDER.get(path.name, 100), path.name),
        )
        for language_dir in language_dirs:
            per_sample = read_per_sample(language_dir)
            for result_path in sorted(language_dir.glob("*.json")):
                if result_path.name.endswith("_per_sample.json"):
                    continue
                metric = metric_from_filename(benchmark, result_path)
                subset_values, kept_total = score_subsets(
                    per_sample, benchmark, metric, excluded, language_dir
                )

                rows.append(
                    {
                        "model": model_dir.name,
                        "language": language_dir.name,
                        "metric": metric,
                        "values": subset_values,
                        "kept": kept_total,
                    }
                )
    return rows


def print_table(
    title: str,
    subsets: tuple[str, ...],
    rows: list[dict[str, object]],
    decimals: int,
) -> None:
    """Print a formatted result table with subset and macro-average scores."""
    headers = ["Model", "Lang", *subsets, "Average", "N"]
    body = []
    for row in rows:
        values = row["values"]
        assert isinstance(values, list)
        displayed_values = [round(100 * value, decimals) for value in values]
        body.append(
            [
                str(row["model"]),
                str(row["language"]),
                *(f"{value:.{decimals}f}" for value in displayed_values),
                f"{sum(displayed_values) / len(displayed_values):.{decimals}f}",
                str(row["kept"]),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in body))
        for index in range(len(headers))
    ]

    print(f"\n=== {title} (accuracy %) ===")
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in body:
        formatted = [
            value.ljust(widths[index]) if index < 2 else value.rjust(widths[index])
            for index, value in enumerate(row)
        ]
        print("  ".join(formatted))


def main() -> None:
    """Parse command-line options and print the requested benchmark tables."""
    parser = argparse.ArgumentParser(
        description="Print SugarCrepe and SugarCrepe++ multilingual result tables."
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--benchmark", choices=("all", "sc", "scpp"), default="all")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()

    benchmarks = ("sc", "scpp") if args.benchmark == "all" else (args.benchmark,)
    for benchmark in benchmarks:
        excluded = read_excluded_ids(benchmark)
        dropped = sum(len(ids) for ids in excluded.values())
        rows = read_rows(args.output_root, benchmark)
        metrics = ("Accuracy",) if benchmark == "sc" else ("ITT", "TOT")
        for metric in metrics:
            metric_rows = [row for row in rows if row["metric"] == metric]
            title = BENCHMARK_NAMES[benchmark]
            if benchmark == "scpp":
                title += f" — {metric}"
            title += f", {dropped} collapsed-translation records excluded in all languages"
            print_table(title, BENCHMARK_SUBSETS[benchmark], metric_rows, args.decimals)


if __name__ == "__main__":
    main()
