"""CLIP-ready SBERT prompt selection for RSU traffic-scene ensembles.

This script keeps the lexicon-prototype SBERT method:

lexicon terms -> SBERT -> class prototype
generated prompt -> SBERT -> cosine similarity to all prototypes

It then improves the final ensemble candidates by filtering weak prompts,
deduplicating near-identical meanings, and selecting across semantic clusters.
No CLIP model or image is used here; CLIP evaluation happens after this step.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


CATEGORY_ORDER = ["accident", "normal_flow", "congestion", "invalid_scene"]
CATEGORY_HEADINGS = {
    "accident": "[ACCIDENT]",
    "normal_flow": "[NORMAL FLOW]",
    "congestion": "[CONGESTION]",
    "invalid_scene": "[INVALID SCENE]",
}
PROMPT_HEADING_TO_CATEGORY = {
    "ACCIDENT": "accident",
    "NORMAL FLOW": "normal_flow",
    "SMOOTH FLOW": "normal_flow",
    "CONGESTION": "congestion",
    "INVALID SCENE": "invalid_scene",
}
LEXICON_CLASS_TO_CATEGORY = {
    "ACCIDENT": "accident",
    "NORMAL FLOW": "normal_flow",
    "CONGESTION": "congestion",
    "INVALID SCENE": "invalid_scene",
}

PROMPT_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s*)?\[([A-Za-z ]+)\]")
LEXICON_CLASS_RE = re.compile(r"^##\s+Class\s+\d+:\s+(.+?)\s*$", re.IGNORECASE)
MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class PromptRecord:
    category: str
    prompt: str
    original_line_number: int


@dataclass(frozen=True)
class EncodedPrompt:
    record: PromptRecord
    embedding: np.ndarray
    own_score: float
    rival_score: float
    margin: float
    closest_rival_class: str
    predicted_category: str
    cluster_id: int = -1
    passed_filter: bool = False


def parse_prompt_file(input_path: Path) -> tuple[dict[str, list[PromptRecord]], int]:
    """Parse generated prompts from Markdown category sections."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input prompt file does not exist: {input_path}")

    prompts_by_category = {category: [] for category in CATEGORY_ORDER}
    seen_categories: set[str] = set()
    current_category: str | None = None
    empty_lines = 0

    lines = input_path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            empty_lines += 1
            continue

        heading_match = PROMPT_HEADING_RE.match(stripped)
        if heading_match:
            heading = heading_match.group(1).upper().strip()
            if heading in PROMPT_HEADING_TO_CATEGORY:
                current_category = PROMPT_HEADING_TO_CATEGORY[heading]
                seen_categories.add(current_category)
                continue

        if current_category is None:
            continue

        prompt = stripped[2:].strip() if stripped.startswith("- ") else stripped
        if prompt:
            prompts_by_category[current_category].append(
                PromptRecord(current_category, prompt, line_number)
            )

    missing = [category for category in CATEGORY_ORDER if category not in seen_categories]
    if missing:
        formatted = ", ".join(CATEGORY_HEADINGS[category] for category in missing)
        raise ValueError(f"Required prompt category heading missing: {formatted}")

    return prompts_by_category, empty_lines


def parse_lexicon_file(lexicon_path: Path) -> dict[str, list[str]]:
    """Parse comma-separated seed terms from each lexicon class section."""

    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon file does not exist: {lexicon_path}")

    lexicon_terms = {category: [] for category in CATEGORY_ORDER}
    current_category: str | None = None

    lines = lexicon_path.read_text(encoding="utf-8").splitlines()
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        class_match = LEXICON_CLASS_RE.match(stripped)
        if class_match:
            class_name = class_match.group(1).upper().strip()
            current_category = LEXICON_CLASS_TO_CATEGORY.get(class_name)
            continue

        if current_category is None or MARKDOWN_HEADING_RE.match(stripped):
            continue
        if "," not in stripped:
            continue

        for term in stripped.split(","):
            cleaned = term.strip()
            if cleaned:
                lexicon_terms[current_category].append(cleaned)

    missing = [category for category in CATEGORY_ORDER if not lexicon_terms[category]]
    if missing:
        raise ValueError(f"Lexicon has no parsed seed terms for: {', '.join(missing)}")

    return lexicon_terms


def unique_prompt_records(records: list[PromptRecord]) -> list[PromptRecord]:
    """Remove exact duplicate prompt text while preserving the first occurrence."""

    seen: set[str] = set()
    unique: list[PromptRecord] = []
    for record in records:
        if record.prompt not in seen:
            seen.add(record.prompt)
            unique.append(record)
    return unique


def unique_texts(texts: list[str]) -> list[str]:
    """Remove exact duplicate strings while preserving order."""

    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    label: str,
    show_progress_bar: bool,
) -> np.ndarray:
    """Encode text into normalized SBERT vectors and validate the matrix."""

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Model returned {embeddings.shape[0]} embeddings for {len(texts)} texts in {label}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Embeddings contain NaN or infinity for {label}.")
    return embeddings


def normalize_vector(vector: np.ndarray, label: str) -> np.ndarray:
    """Normalize one vector so dot product equals cosine similarity."""

    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Cannot normalize zero or invalid vector for {label}.")
    return vector / norm


def build_lexicon_prototypes(
    model: SentenceTransformer,
    lexicon_terms: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Encode lexicon terms and average them into normalized class prototypes."""

    prototypes: dict[str, np.ndarray] = {}
    term_counts: dict[str, int] = {}
    for category in CATEGORY_ORDER:
        terms = unique_texts(lexicon_terms[category])
        term_counts[category] = len(terms)
        embeddings = encode_texts(
            model=model,
            texts=terms,
            label=f"{category} lexicon",
            show_progress_bar=False,
        )
        prototypes[category] = normalize_vector(
            embeddings.mean(axis=0),
            f"{category} lexicon prototype",
        )
    return prototypes, term_counts


def score_prompt_embeddings(
    records_by_category: dict[str, list[PromptRecord]],
    embeddings_by_category: dict[str, np.ndarray],
    prototypes: dict[str, np.ndarray],
    min_own_score: float,
    min_margin: float,
) -> dict[str, list[EncodedPrompt]]:
    """Calculate own score, rival score, and margin for every prompt."""

    scored: dict[str, list[EncodedPrompt]] = {category: [] for category in CATEGORY_ORDER}
    for category in CATEGORY_ORDER:
        for record, embedding in zip(
            records_by_category[category],
            embeddings_by_category[category],
            strict=True,
        ):
            similarities = {
                prototype_category: float(embedding @ prototypes[prototype_category])
                for prototype_category in CATEGORY_ORDER
            }
            predicted_category = max(similarities, key=similarities.get)
            rival_category = max(
                (candidate for candidate in CATEGORY_ORDER if candidate != category),
                key=lambda candidate: similarities[candidate],
            )
            own_score = similarities[category]
            rival_score = similarities[rival_category]
            margin = own_score - rival_score
            scored[category].append(
                EncodedPrompt(
                    record=record,
                    embedding=embedding,
                    own_score=own_score,
                    rival_score=rival_score,
                    margin=margin,
                    closest_rival_class=rival_category,
                    predicted_category=predicted_category,
                    passed_filter=own_score >= min_own_score and margin >= min_margin,
                )
            )
    return scored


def assign_clusters(
    prompts: list[EncodedPrompt],
    n_clusters: int,
    seed: int,
) -> list[EncodedPrompt]:
    """Cluster prompt embeddings inside one class for semantic diversity."""

    if not prompts:
        return []
    cluster_count = min(n_clusters, len(prompts))
    matrix = np.vstack([prompt.embedding for prompt in prompts])
    if cluster_count == 1:
        labels = np.zeros(len(prompts), dtype=np.int32)
    else:
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=seed,
            n_init=20,
        ).fit_predict(matrix)

    assigned = [
        EncodedPrompt(
            record=prompt.record,
            embedding=prompt.embedding,
            own_score=prompt.own_score,
            rival_score=prompt.rival_score,
            margin=prompt.margin,
            closest_rival_class=prompt.closest_rival_class,
            predicted_category=prompt.predicted_category,
            cluster_id=int(label),
            passed_filter=prompt.passed_filter,
        )
        for prompt, label in zip(prompts, labels, strict=True)
    ]
    represented = {prompt.cluster_id for prompt in assigned}
    if len(represented) != cluster_count:
        raise ValueError("KMeans produced an empty cluster.")
    return assigned


def rank_key(prompt: EncodedPrompt) -> tuple[float, float, int]:
    """Rank by distinctiveness first, relevance second, source order third."""

    return (-prompt.margin, -prompt.own_score, prompt.record.original_line_number)


def is_not_duplicate(
    candidate: EncodedPrompt,
    selected: list[EncodedPrompt],
    threshold: float,
) -> bool:
    """Reject a candidate if it is too close to an already selected prompt."""

    if not selected:
        return True
    selected_matrix = np.vstack([prompt.embedding for prompt in selected])
    max_similarity = float(np.max(selected_matrix @ candidate.embedding))
    return max_similarity < threshold


def select_for_target_count(
    scored_prompts: list[EncodedPrompt],
    target_count: int,
    min_count: int,
    duplicate_threshold: float,
    relaxed_duplicate_threshold: float,
) -> list[EncodedPrompt]:
    """Select diverse prompts using filter, rank, deduplication, and clusters."""

    if target_count <= 0:
        raise ValueError("Target prompt count must be positive.")
    if min_count > target_count:
        raise ValueError("Minimum prompt count cannot exceed target prompt count.")

    filtered = [prompt for prompt in scored_prompts if prompt.passed_filter]
    candidate_pool = filtered if len(filtered) >= min_count else scored_prompts
    ranked = sorted(candidate_pool, key=rank_key)

    selected: list[EncodedPrompt] = []
    grouped: dict[int, list[EncodedPrompt]] = defaultdict(list)
    for prompt in ranked:
        grouped[prompt.cluster_id].append(prompt)

    # Stage A: take the strongest non-duplicate prompt from each semantic cluster.
    for cluster_id in sorted(grouped):
        for candidate in grouped[cluster_id]:
            if candidate in selected:
                continue
            if is_not_duplicate(candidate, selected, duplicate_threshold):
                selected.append(candidate)
                break
        if len(selected) >= target_count:
            return selected[:target_count]

    # Stage B: fill from the global ranking while keeping the strict threshold.
    for candidate in ranked:
        if candidate in selected:
            continue
        if is_not_duplicate(candidate, selected, duplicate_threshold):
            selected.append(candidate)
        if len(selected) >= target_count:
            return selected[:target_count]

    # Stage C: relax deduplication before accepting near-duplicates.
    threshold = duplicate_threshold + 0.03
    while len(selected) < target_count and threshold <= relaxed_duplicate_threshold + 1e-9:
        for candidate in ranked:
            if candidate in selected:
                continue
            if is_not_duplicate(candidate, selected, threshold):
                selected.append(candidate)
            if len(selected) >= target_count:
                return selected[:target_count]
        threshold += 0.03

    # Stage D: final fallback; this is reported as a weaker selection condition.
    for candidate in ranked:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= target_count:
            return selected[:target_count]

    if len(selected) < min_count:
        raise ValueError(
            f"Only selected {len(selected)} prompts; required at least {min_count}."
        )
    return selected


def selected_rows_for_count(
    scored_by_category: dict[str, list[EncodedPrompt]],
    target_count: int,
    min_count: int,
    duplicate_threshold: float,
    relaxed_duplicate_threshold: float,
) -> list[dict[str, Any]]:
    """Select one target prompt count for every class."""

    selected_rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        selected = select_for_target_count(
            scored_prompts=scored_by_category[category],
            target_count=target_count,
            min_count=min_count,
            duplicate_threshold=duplicate_threshold,
            relaxed_duplicate_threshold=relaxed_duplicate_threshold,
        )
        for rank, prompt in enumerate(selected, start=1):
            selected_rows.append(
                {
                    "category": category,
                    "prompt": prompt.record.prompt,
                    "own_score": prompt.own_score,
                    "rival_score": prompt.rival_score,
                    "margin": prompt.margin,
                    "closest_rival_class": prompt.closest_rival_class,
                    "predicted_category": prompt.predicted_category,
                    "cluster_id": prompt.cluster_id,
                    "passed_filter": prompt.passed_filter,
                    "selection_rank": rank,
                    "original_line_number": prompt.record.original_line_number,
                }
            )
    return selected_rows


def format_markdown(selected_rows: list[dict[str, Any]]) -> str:
    """Format prompts in the four-section Markdown style CLIP expects."""

    lines: list[str] = []
    for category in CATEGORY_ORDER:
        lines.append(CATEGORY_HEADINGS[category])
        rows = sorted(
            (row for row in selected_rows if row["category"] == category),
            key=lambda row: int(row["selection_rank"]),
        )
        for row in rows:
            lines.append(str(row["prompt"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_all_prompts_by_cluster_markdown(all_score_rows: list[dict[str, Any]]) -> str:
    """Group every scored prompt by class and semantic cluster for inspection."""

    lines = [
        "# All S-BERT Prompts by Semantic Cluster",
        "",
        "This file is for prompt-selection review. It lists every scored generated prompt under its assigned KMeans semantic cluster.",
        "",
        "Sorting inside each cluster uses the selection priority: margin first, then own score.",
        "",
    ]
    for category in CATEGORY_ORDER:
        category_rows = [row for row in all_score_rows if row["category"] == category]
        cluster_ids = sorted({int(row["cluster_id"]) for row in category_rows})
        lines.extend([f"## {CATEGORY_HEADINGS[category]}", ""])
        for cluster_id in cluster_ids:
            cluster_rows = sorted(
                (row for row in category_rows if int(row["cluster_id"]) == cluster_id),
                key=lambda row: (
                    -float(row["margin"]),
                    -float(row["own_score"]),
                    int(row["original_line_number"]),
                ),
            )
            lines.extend(
                [
                    f"### Cluster {cluster_id} ({len(cluster_rows)} prompts)",
                    "",
                    "| Rank | Prompt | Own Score | Rival Score | Margin | Rival Class | Selected In |",
                    "|---:|---|---:|---:|---:|---|---|",
                ]
            )
            for rank, row in enumerate(cluster_rows, start=1):
                selected_counts = str(row["selected_counts"]) if row["selected_counts"] else "-"
                prompt_text = str(row["prompt"]).replace("|", "\\|")
                lines.append(
                    f"| {rank} | {prompt_text} | "
                    f"{float(row['own_score']):.4f} | {float(row['rival_score']):.4f} | "
                    f"{float(row['margin']):.4f} | {row['closest_rival_class']} | {selected_counts} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write dictionaries to CSV with a stable column order."""

    try:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise OSError(f"Could not write CSV output file: {path}") from exc


def write_outputs(
    output_dir: Path,
    run_name: str,
    selected_by_count: dict[int, list[dict[str, Any]]],
    all_score_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    report_text: str,
) -> None:
    """Export all selection reports without touching old output folders."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create output directory: {output_dir}") from exc

    score_fields = [
        "category",
        "prompt",
        "own_score",
        "rival_score",
        "margin",
        "closest_rival_class",
        "predicted_category",
        "cluster_id",
        "passed_filter",
        "selected_counts",
        "original_line_number",
    ]
    selected_fields = [
        "category",
        "prompt",
        "own_score",
        "rival_score",
        "margin",
        "closest_rival_class",
        "predicted_category",
        "cluster_id",
        "passed_filter",
        "selection_rank",
        "original_line_number",
    ]
    summary_fields = [
        "target_count",
        "category",
        "generated",
        "unique_prompts",
        "passed_filter",
        "selected",
        "mean_selected_own_score",
        "mean_selected_margin",
        "clusters_represented",
    ]

    write_csv(
        output_dir / f"sbert_{run_name}_clip_ready_all_prompt_scores.csv",
        all_score_rows,
        score_fields,
    )
    write_csv(
        output_dir / f"sbert_{run_name}_clip_ready_selection_summary.csv",
        summary_rows,
        summary_fields,
    )

    for target_count, selected_rows in selected_by_count.items():
        count_dir = output_dir / f"k{target_count}"
        count_dir.mkdir(parents=True, exist_ok=True)
        base = count_dir / f"sbert_{run_name}_clip_ready_candidates_{target_count}"
        base.with_suffix(".md").write_text(format_markdown(selected_rows), encoding="utf-8")
        write_csv(base.with_suffix(".csv"), selected_rows, selected_fields)

    (output_dir / f"sbert_{run_name}_clip_ready_all_prompts_by_cluster.md").write_text(
        format_all_prompts_by_cluster_markdown(all_score_rows),
        encoding="utf-8",
    )
    (output_dir / f"sbert_{run_name}_clip_ready_report.md").write_text(
        report_text,
        encoding="utf-8",
    )


def build_all_score_rows(
    scored_by_category: dict[str, list[EncodedPrompt]],
    selected_by_count: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Create one row per prompt and mark which target sets selected it."""

    selected_lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    for target_count, rows in selected_by_count.items():
        for row in rows:
            selected_lookup[(row["category"], row["prompt"])].append(target_count)

    rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        for prompt in sorted(scored_by_category[category], key=rank_key):
            selected_counts = selected_lookup.get((category, prompt.record.prompt), [])
            rows.append(
                {
                    "category": category,
                    "prompt": prompt.record.prompt,
                    "own_score": prompt.own_score,
                    "rival_score": prompt.rival_score,
                    "margin": prompt.margin,
                    "closest_rival_class": prompt.closest_rival_class,
                    "predicted_category": prompt.predicted_category,
                    "cluster_id": prompt.cluster_id,
                    "passed_filter": prompt.passed_filter,
                    "selected_counts": "|".join(str(count) for count in selected_counts),
                    "original_line_number": prompt.record.original_line_number,
                }
            )
    return rows


def build_summary_rows(
    records_by_category: dict[str, list[PromptRecord]],
    scored_by_category: dict[str, list[EncodedPrompt]],
    selected_by_count: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Summarize prompt quality and diversity by class and target count."""

    rows: list[dict[str, Any]] = []
    for target_count, selected_rows in selected_by_count.items():
        for category in CATEGORY_ORDER:
            category_selected = [row for row in selected_rows if row["category"] == category]
            own_scores = [float(row["own_score"]) for row in category_selected]
            margins = [float(row["margin"]) for row in category_selected]
            clusters = {int(row["cluster_id"]) for row in category_selected}
            rows.append(
                {
                    "target_count": target_count,
                    "category": category,
                    "generated": len(records_by_category[category]),
                    "unique_prompts": len(scored_by_category[category]),
                    "passed_filter": sum(
                        1 for prompt in scored_by_category[category] if prompt.passed_filter
                    ),
                    "selected": len(category_selected),
                    "mean_selected_own_score": float(np.mean(own_scores)) if own_scores else 0.0,
                    "mean_selected_margin": float(np.mean(margins)) if margins else 0.0,
                    "clusters_represented": len(clusters),
                }
            )
    return rows


def build_report(
    input_path: Path,
    lexicon_path: Path,
    model_name: str,
    run_name: str,
    empty_lines: int,
    lexicon_term_counts: dict[str, int],
    records_by_category: dict[str, list[PromptRecord]],
    scored_by_category: dict[str, list[EncodedPrompt]],
    selected_by_count: dict[int, list[dict[str, Any]]],
    min_own_score: float,
    min_margin: float,
    duplicate_threshold: float,
    diversity_clusters: int,
) -> str:
    """Create a readable Markdown report for the improved SBERT selector."""

    lines = [
        f"# S-BERT CLIP-ready prompt-selection report ({run_name})",
        "",
        "## Method",
        "",
        "This run uses Sentence-BERT for text-prompt selection only. It does not use CLIP, images, or model training.",
        "",
        "Selection objective:",
        "",
        "```text",
        "Selected prompt = relevant + distinctive + diverse",
        "```",
        "",
        "- Relevant: high cosine similarity to the intended class lexicon prototype.",
        "- Distinctive: positive margin against the strongest rival class prototype.",
        "- Diverse: semantic deduplication plus KMeans cluster coverage.",
        "",
        "## Inputs",
        "",
        f"- prompt file: `{input_path}`",
        f"- lexicon file: `{lexicon_path}`",
        f"- SBERT model: `{model_name}`",
        f"- empty lines in prompt file: {empty_lines}",
        "",
        "## Parameters",
        "",
        f"- minimum own score: {min_own_score}",
        f"- minimum margin: {min_margin}",
        f"- duplicate threshold: {duplicate_threshold}",
        f"- diversity clusters per class: {diversity_clusters}",
        "",
        "## Lexicon Terms",
        "",
    ]
    for category in CATEGORY_ORDER:
        lines.append(f"- {category}: {lexicon_term_counts[category]} terms")

    lines.extend(["", "## Category Summary", ""])
    for category in CATEGORY_ORDER:
        scored = scored_by_category[category]
        passed = [prompt for prompt in scored if prompt.passed_filter]
        margins = np.array([prompt.margin for prompt in scored], dtype=np.float32)
        own_scores = np.array([prompt.own_score for prompt in scored], dtype=np.float32)
        lines.append(
            f"- {category}: generated={len(records_by_category[category])}, "
            f"unique={len(scored)}, passed_filter={len(passed)}, "
            f"mean_own={own_scores.mean():.4f}, mean_margin={margins.mean():.4f}"
        )

    lines.extend(["", "## Selected Sets", ""])
    for target_count, selected_rows in selected_by_count.items():
        lines.append(f"### Target {target_count} per class")
        for category in CATEGORY_ORDER:
            category_selected = [row for row in selected_rows if row["category"] == category]
            own_scores = [float(row["own_score"]) for row in category_selected]
            margins = [float(row["margin"]) for row in category_selected]
            clusters = {int(row["cluster_id"]) for row in category_selected}
            lines.append(
                f"- {category}: selected={len(category_selected)}, "
                f"mean_own={np.mean(own_scores):.4f}, "
                f"mean_margin={np.mean(margins):.4f}, "
                f"clusters={len(clusters)}"
            )
        lines.append("")

    lines.extend(
        [
            "## How to Use in CLIP",
            "",
            "Use each exported Markdown file as `--prompt-file` in the CLIP evaluator, then compare CLIP accuracy and macro F1 against the 4-prompt baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_target_counts(value: str) -> list[int]:
    """Parse comma-separated target counts such as 10,15,20,30."""

    counts = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("Target counts must be positive integers.")
    return counts


def sanitize_run_name(run_name: str) -> str:
    """Validate the label inserted into output filenames."""

    cleaned = run_name.strip()
    if not cleaned or not RUN_NAME_RE.fullmatch(cleaned):
        raise ValueError("--run-name must contain only letters, numbers, and underscores.")
    return cleaned


def parse_args() -> argparse.Namespace:
    """Define the command-line interface."""

    parser = argparse.ArgumentParser(
        description="Select CLIP-ready ensemble prompts with SBERT scoring and diversity."
    )
    parser.add_argument("--input", type=Path, default=Path("sbert/docs/2000prompts_opus_4_8.md"))
    parser.add_argument("--lexicon", type=Path, default=Path("sbert/docs/lexicon.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("sbert/output/cluster_equals_k/opus_4_8"))
    parser.add_argument("--run-name", default="opus_4_8")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--target-counts", type=parse_target_counts, default=[30])
    parser.add_argument("--min-count", type=int, default=15)
    parser.add_argument("--min-own-score", type=float, default=0.45)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--relaxed-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--diversity-clusters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the improved SBERT prompt-selection pipeline."""

    args = parse_args()
    run_name = sanitize_run_name(args.run_name)
    if args.diversity_clusters <= 0:
        raise ValueError("--diversity-clusters must be positive.")
    if args.min_count <= 0:
        raise ValueError("--min-count must be positive.")

    print("S-BERT CLIP-ready prompt selection")
    print(f"Input: {args.input}")
    print(f"Lexicon: {args.lexicon}")
    print(f"Targets: {args.target_counts}")

    records_by_category, empty_lines = parse_prompt_file(args.input)
    unique_records_by_category = {
        category: unique_prompt_records(records_by_category[category])
        for category in CATEGORY_ORDER
    }
    lexicon_terms = parse_lexicon_file(args.lexicon)

    model = SentenceTransformer(args.model_name)
    prototypes, lexicon_term_counts = build_lexicon_prototypes(model, lexicon_terms)

    embeddings_by_category: dict[str, np.ndarray] = {}
    for category in CATEGORY_ORDER:
        texts = [record.prompt for record in unique_records_by_category[category]]
        embeddings_by_category[category] = encode_texts(
            model=model,
            texts=texts,
            label=f"{category} generated prompts",
            show_progress_bar=True,
        )
        print(f"{category}: {len(texts)} unique prompts, embedding shape {embeddings_by_category[category].shape}")

    scored_by_category = score_prompt_embeddings(
        records_by_category=unique_records_by_category,
        embeddings_by_category=embeddings_by_category,
        prototypes=prototypes,
        min_own_score=args.min_own_score,
        min_margin=args.min_margin,
    )

    clustered_by_category: dict[str, list[EncodedPrompt]] = {}
    for category in CATEGORY_ORDER:
        clustered_by_category[category] = assign_clusters(
            prompts=scored_by_category[category],
            n_clusters=args.diversity_clusters,
            seed=args.seed,
        )

    selected_by_count: dict[int, list[dict[str, Any]]] = {}
    for target_count in args.target_counts:
        selected_by_count[target_count] = selected_rows_for_count(
            scored_by_category=clustered_by_category,
            target_count=target_count,
            min_count=min(args.min_count, target_count),
            duplicate_threshold=args.duplicate_threshold,
            relaxed_duplicate_threshold=args.relaxed_duplicate_threshold,
        )

    all_score_rows = build_all_score_rows(clustered_by_category, selected_by_count)
    summary_rows = build_summary_rows(
        unique_records_by_category,
        clustered_by_category,
        selected_by_count,
    )
    report_text = build_report(
        input_path=args.input,
        lexicon_path=args.lexicon,
        model_name=args.model_name,
        run_name=run_name,
        empty_lines=empty_lines,
        lexicon_term_counts=lexicon_term_counts,
        records_by_category=unique_records_by_category,
        scored_by_category=clustered_by_category,
        selected_by_count=selected_by_count,
        min_own_score=args.min_own_score,
        min_margin=args.min_margin,
        duplicate_threshold=args.duplicate_threshold,
        diversity_clusters=args.diversity_clusters,
    )
    write_outputs(
        output_dir=args.output_dir,
        run_name=run_name,
        selected_by_count=selected_by_count,
        all_score_rows=all_score_rows,
        summary_rows=summary_rows,
        report_text=report_text,
    )

    print("Summary")
    for row in summary_rows:
        if int(row["target_count"]) == max(args.target_counts):
            print(
                f"{row['category']}: generated={row['generated']}, "
                f"passed_filter={row['passed_filter']}, selected={row['selected']}, "
                f"mean_margin={float(row['mean_selected_margin']):.4f}"
            )
    print(f"Reports written to: {args.output_dir}")


if __name__ == "__main__":
    main()
