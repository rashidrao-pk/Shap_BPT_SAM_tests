#!/usr/bin/env python
"""Compare multiple full-image-set XAI experiment result folders.

The script loads saved AUC/faithfulness CSV files from folders such as
``xai_results_E1_1_2`` or ``xai_results_E3_1`` and creates aggregate comparison
tables, plots, and an HTML page.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


# METHOD_ORDER = ["BPT", "sam", "coverage", "compact", "filled", "refined", "refined_filled"]
METHOD_ORDER = ["BPT", "sam", "greedy_selection", "compact_selection", "filled", "boosted", "boosted_filled", "sam_kernelshap"]


METRIC_DIRECTIONS = {
    "auc_ins": "higher",
    "auc_del": "lower",
    "drop_at_10": "higher",
    "drop_at_20": "higher",
    "drop_at_30": "higher",
    "insert_at_10": "higher",
    "insert_at_20": "higher",
    "insert_at_30": "higher",
    "sufficiency_at_10": "higher",
    "sufficiency_at_20": "higher",
    "sufficiency_gap_at_10": "lower",
    "sufficiency_gap_at_20": "lower",
    "comprehensiveness_at_10": "higher",
    "comprehensiveness_at_20": "higher",
    "sensitivity_n_corr": "higher",
    "stability_top10_jaccard": "higher",
    "total_time_mean": "lower",
}

DEFAULT_METRICS = [
    "auc_ins",
    "auc_del",
    "sensitivity_n_corr",
    # "comprehensiveness_at_20",
    # "sufficiency_gap_at_20",
    "total_time_mean",
]

EXP_MODEL_NAMES = {
    "E1": "YOLOv11",
    "E2": "ResNet-50",
    "E3": "ViT-B/16",
    "E4": "DETR",
}

METHOD_REGION_COLUMNS = {
    "BPT": ["n_bpt_regions", "n_regions_bpt"],
    "sam": ["n_sam_masks", "n_masks"],
    "coverage": ["n_coverage_masks"],
    "compact": ["n_compact_masks", "n_unique_compact", "n_coverage_masks"],
    "filled": ["n_filled_masks", "n_unique_filler"],
    "refined": ["n_refined_instances", "n_refined_masks"],
    "refined_filled": ["n_refined_filled", "n_unique_refined_filled", "n_unique_filler"],
}


def latest_file(folder: Path, pattern: str, exclude=()) -> Path | None:
    exclude = set(exclude)
    matches = [path for path in folder.glob(pattern) if path.name not in exclude]
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def infer_exp_no(results_dir: Path, run_info: dict | None = None) -> str:
    if run_info and run_info.get("exp_no"):
        return str(run_info["exp_no"])
    match = re.search(r"(E\d(?:_[A-Za-z0-9]+)+)", results_dir.name)
    return match.group(1) if match else results_dir.name


def infer_model_name(exp_no: str, run_info: dict | None = None) -> str:
    if run_info:
        if run_info.get("model_group") == "yolov11" or str(run_info.get("model", "")).lower().startswith("yolo"):
            return "YOLOv11"
        if run_info.get("model_type") == "resnet50" or "resnet" in str(run_info.get("model", "")).lower():
            return "ResNet-50"
        if run_info.get("model_type") == "vit_b_16" or "vit" in str(run_info.get("model", "")).lower():
            return "ViT-B/16"
        if run_info.get("model_group") == "detr" or "detr" in str(run_info.get("model", "")).lower():
            return "DETR"
    prefix = exp_no.split("_", 1)[0]
    return EXP_MODEL_NAMES.get(prefix, prefix)


def infer_exp_type(exp_no: str) -> str:
    match = re.match(r"(E\d+)", str(exp_no))
    return match.group(1) if match else str(exp_no)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def resolve_auc_csv(results_dir: Path) -> Path:
    final_csv = latest_file(results_dir, "auc_results_all_images_*.csv", exclude={"auc_results_all_images_current_run.csv"})
    if final_csv is not None:
        return final_csv
    current_csv = results_dir / "auc_results_all_images_current_run.csv"
    if current_csv.exists():
        return current_csv
    raise FileNotFoundError(f"No AUC CSV found in {results_dir}")


def normalize_image_id(value) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(12)


def load_total_time(results_dir: Path) -> pd.DataFrame:
    total_path = latest_file(results_dir, "total_time_by_method_*.csv")
    if total_path is None:
        return pd.DataFrame(columns=["method", "total_time_mean", "total_time_sum"])
    df = pd.read_csv(total_path)
    keep = [col for col in ["method", "total_time_mean", "total_time_sum", "partition_time_mean", "explanation_time_mean"] if col in df.columns]
    return df[keep].copy()


def load_partition_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "partition_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "image_id" not in df.columns:
        return pd.DataFrame()
    return df


def load_region_counts(results_dir: Path, exp_no: str, model_name: str) -> pd.DataFrame:
    partition_df = load_partition_summary(results_dir)
    if partition_df.empty:
        return pd.DataFrame()
    partition_df = partition_df.copy()
    partition_df["image_id_str"] = partition_df["image_id"].apply(normalize_image_id)
    rows = []
    for _, row in partition_df.iterrows():
        for method, candidates in METHOD_REGION_COLUMNS.items():
            region_count = np.nan
            region_source = ""
            for column in candidates:
                if column in partition_df.columns and pd.notna(row.get(column)):
                    region_count = row[column]
                    region_source = column
                    break
            if pd.isna(region_count):
                continue
            rows.append(
                {
                    "experiment": exp_no,
                    "exp_type": infer_exp_type(exp_no),
                    "model": model_name,
                    "image_id_str": row["image_id_str"],
                    "method": method,
                    "n_regions": float(region_count),
                    "region_source": region_source,
                    "results_dir": str(results_dir),
                }
            )
    return pd.DataFrame(rows)


def load_experiment(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    results_dir = results_dir.expanduser().resolve()
    run_info = read_json(results_dir / "run_info.json") or {}
    exp_no = infer_exp_no(results_dir, run_info)
    model_name = infer_model_name(exp_no, run_info)
    auc_csv = resolve_auc_csv(results_dir)
    auc_df = pd.read_csv(auc_csv)
    auc_df["experiment"] = exp_no
    auc_df["exp_type"] = infer_exp_type(exp_no)
    auc_df["model"] = model_name
    auc_df["results_dir"] = str(results_dir)
    auc_df["auc_csv"] = str(auc_csv)
    if "image_id" in auc_df.columns:
        auc_df["image_id_str"] = auc_df["image_id"].apply(normalize_image_id)
    metadata = {
        "experiment": exp_no,
        "exp_type": infer_exp_type(exp_no),
        "model": model_name,
        "results_dir": str(results_dir),
        "auc_csv": str(auc_csv),
        "run_info": run_info,
    }
    return auc_df, metadata


def summarize_by_experiment_method(combined: pd.DataFrame, results_dirs: list[Path]) -> pd.DataFrame:
    metric_cols = [metric for metric in METRIC_DIRECTIONS if metric in combined.columns and metric != "total_time_mean"]
    agg_spec = {
        "images": ("image_no", "nunique") if "image_no" in combined.columns else ("image_id", "nunique"),
    }
    for metric in metric_cols:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
        agg_spec[f"{metric}_median"] = (metric, "median")

    summary = (
        combined.groupby(["experiment", "exp_type", "model", "method"], dropna=False, observed=True)
        .agg(**agg_spec)
        .reset_index()
    )

    time_frames = []
    for results_dir in results_dirs:
        run_info = read_json(results_dir / "run_info.json") or {}
        exp_no = infer_exp_no(results_dir, run_info)
        exp_type = infer_exp_type(exp_no)
        model_name = infer_model_name(exp_no, run_info)
        time_df = load_total_time(results_dir)
        if time_df.empty:
            continue
        time_df["experiment"] = exp_no
        time_df["exp_type"] = exp_type
        time_df["model"] = model_name
        time_frames.append(time_df)
    if time_frames:
        time_all = pd.concat(time_frames, ignore_index=True)
        summary = summary.merge(time_all, on=["experiment", "exp_type", "model", "method"], how="left")
    return summary


def normalize_metric(series: pd.Series, direction: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    min_value = values.min(skipna=True)
    max_value = values.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or np.isclose(max_value, min_value):
        return pd.Series(np.nan, index=series.index)
    score = (values - min_value) / (max_value - min_value)
    if direction == "lower":
        score = 1.0 - score
    return score


def add_ranking_columns(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    ranked = summary.copy()
    score_cols = []
    for metric in metrics:
        source_col = metric if metric == "total_time_mean" else f"{metric}_mean"
        if source_col not in ranked.columns:
            continue
        score_col = f"{metric}_score"
        ranked[score_col] = normalize_metric(ranked[source_col], METRIC_DIRECTIONS[metric])
        score_cols.append(score_col)
    if score_cols:
        ranked["composite_score"] = ranked[score_cols].mean(axis=1, skipna=True)
        ranked["composite_rank"] = ranked["composite_score"].rank(ascending=False, method="min")
    else:
        ranked["composite_score"] = np.nan
        ranked["composite_rank"] = np.nan
    return ranked.sort_values(["composite_rank", "experiment", "method"]).reset_index(drop=True)


def raw_metric_list(combined: pd.DataFrame, metrics: list[str]) -> list[str]:
    return [metric for metric in metrics if metric in combined.columns and metric != "total_time_mean"]


def summarize_raw_by_method(combined: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    metric_cols = raw_metric_list(combined, metrics)
    if not metric_cols:
        return pd.DataFrame()
    agg_spec = {
        "images": ("image_no", "nunique") if "image_no" in combined.columns else ("image_id", "nunique"),
        "rows": ("method", "size"),
    }
    for metric in metric_cols:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
        agg_spec[f"{metric}_median"] = (metric, "median")

    by = [*group_cols, "method"] if group_cols else ["method"]
    return combined.groupby(by, dropna=False, observed=True).agg(**agg_spec).reset_index()


def best_method_for_metric(view: pd.DataFrame, metric: str) -> pd.Series | None:
    col = metric if metric == "total_time_mean" else f"{metric}_mean"
    if col not in view.columns:
        return None
    values = view.dropna(subset=[col])
    if values.empty:
        return None
    ascending = METRIC_DIRECTIONS.get(metric, "higher") == "lower"
    return values.sort_values(col, ascending=ascending).iloc[0]


def build_winner_table(named_summaries: list[tuple[str, list[str], pd.DataFrame]], metrics: list[str]) -> pd.DataFrame:
    rows = []
    metric_cols = [metric for metric in metrics if metric in METRIC_DIRECTIONS]
    for scope_name, group_cols, summary in named_summaries:
        if summary.empty:
            continue
        grouped = [((), summary)] if not group_cols else summary.groupby(group_cols, dropna=False, observed=True)
        for group_key, view in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(group_cols, group_key))
            for metric in metric_cols:
                best = best_method_for_metric(view, metric)
                if best is None:
                    continue
                rows.append(
                    {
                        "scope": scope_name,
                        **group_values,
                        "metric": metric,
                        "best_direction": METRIC_DIRECTIONS[metric],
                        "winning_method": best["method"],
                        "winning_mean": best[metric if metric == "total_time_mean" else f"{metric}_mean"],
                        "images": best.get("images", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def build_overall_vote_table(winner_table: pd.DataFrame) -> pd.DataFrame:
    if winner_table.empty:
        return pd.DataFrame()
    votes = (
        winner_table.groupby(["scope", "winning_method"], dropna=False)
        .size()
        .reset_index(name="metric_wins")
        .sort_values(["scope", "metric_wins", "winning_method"], ascending=[True, False, True])
    )
    totals = votes.groupby("scope")["metric_wins"].transform("sum")
    votes["win_percent"] = (100.0 * votes["metric_wins"] / totals).round(2)
    return votes


def attach_region_counts(combined: pd.DataFrame, region_counts: pd.DataFrame) -> pd.DataFrame:
    if region_counts.empty:
        return pd.DataFrame()
    auc = combined.copy()
    if "image_id_str" not in auc.columns:
        image_col = "image_id" if "image_id" in auc.columns else "image_no"
        auc["image_id_str"] = auc[image_col].apply(normalize_image_id)
    region_counts = region_counts.copy()
    region_counts["method"] = region_counts["method"].astype(str)
    auc["method"] = auc["method"].astype(str)
    merge_cols = ["experiment", "image_id_str", "method"]
    keep = [*merge_cols, "n_regions", "region_source"]
    return auc.merge(region_counts[keep], on=merge_cols, how="inner")


def summarize_region_metric_relationship(region_metric_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if region_metric_df.empty:
        return pd.DataFrame()
    rows = []
    available = [metric for metric in ["auc_ins", "auc_del", *metrics] if metric in region_metric_df.columns]
    available = list(dict.fromkeys(available))
    for (experiment, model, method), group in region_metric_df.groupby(["experiment", "model", "method"], dropna=False):
        for metric in available:
            view = group[["n_regions", metric]].dropna()
            if len(view) < 2:
                continue
            if view["n_regions"].nunique() < 2 or view[metric].nunique() < 2:
                corr = np.nan
            else:
                corr = view["n_regions"].corr(view[metric], method="spearman")
            rows.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "method": method,
                    "metric": metric,
                    "n": len(view),
                    "mean_regions": view["n_regions"].mean(),
                    "mean_metric": view[metric].mean(),
                    "spearman_regions_vs_metric": corr,
                }
            )
    return pd.DataFrame(rows)


def paired_key_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    if "experiment" in df.columns:
        cols.append("experiment")
    if "image_id_str" in df.columns:
        cols.append("image_id_str")
    elif "image_id" in df.columns:
        cols.append("image_id")
    elif "image_no" in df.columns:
        cols.append("image_no")
    return cols


def wilcoxon_pvalue(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if len(diff) < 2 or wilcoxon is None:
        return np.nan
    try:
        return float(wilcoxon(diff, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def significance_for_scope(combined: pd.DataFrame, group_cols: list[str], metrics: list[str], scope_name: str) -> pd.DataFrame:
    metric_cols = raw_metric_list(combined, metrics)
    if not metric_cols:
        return pd.DataFrame()
    df = combined.copy()
    if "image_id_str" not in df.columns:
        image_col = "image_id" if "image_id" in df.columns else "image_no"
        df["image_id_str"] = df[image_col].apply(normalize_image_id)
    df["method"] = df["method"].astype(str)

    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False, observed=True)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(group_cols, group_key))
        key_cols = paired_key_columns(group)
        for metric in metric_cols:
            means = group.groupby("method", observed=True)[metric].mean().dropna()
            if len(means) < 2:
                continue
            best_method = means.idxmin() if METRIC_DIRECTIONS[metric] == "lower" else means.idxmax()
            pivot = group.pivot_table(index=key_cols, columns="method", values=metric, aggfunc="mean")
            if best_method not in pivot.columns:
                continue
            for method in means.index:
                if method == best_method or method not in pivot.columns:
                    continue
                pair = pivot[[best_method, method]].dropna()
                if len(pair) < 2:
                    continue
                if METRIC_DIRECTIONS[metric] == "lower":
                    diff = pair[method].to_numpy(dtype=float) - pair[best_method].to_numpy(dtype=float)
                else:
                    diff = pair[best_method].to_numpy(dtype=float) - pair[method].to_numpy(dtype=float)
                rows.append(
                    {
                        "scope": scope_name,
                        **group_values,
                        "metric": metric,
                        "direction": METRIC_DIRECTIONS[metric],
                        "best_method": best_method,
                        "compared_method": method,
                        "n_pairs": len(pair),
                        "best_mean": pair[best_method].mean(),
                        "compared_mean": pair[method].mean(),
                        "mean_improvement_in_better_direction": float(np.nanmean(diff)),
                        "median_improvement_in_better_direction": float(np.nanmedian(diff)),
                        "win_rate": float(np.mean(diff > 0)),
                        "wilcoxon_p": wilcoxon_pvalue(diff),
                    }
                )
    return pd.DataFrame(rows)


def build_significance_table(combined: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    tables = [
        significance_for_scope(combined, ["experiment"], metrics, "per_experiment"),
        significance_for_scope(combined, ["exp_type", "model"], metrics, "per_experiment_type"),
        significance_for_scope(combined, [], metrics, "overall"),
    ]
    tables = [table for table in tables if not table.empty]
    if not tables:
        return pd.DataFrame()
    result = pd.concat(tables, ignore_index=True)
    if "wilcoxon_p" in result.columns:
        result["significant_p_lt_0_05"] = result["wilcoxon_p"] < 0.05
    return result


def plot_metric_bars(ranked: pd.DataFrame, metrics: list[str], output_dir: Path) -> Path:
    available = []
    for metric in metrics:
        col = metric if metric == "total_time_mean" else f"{metric}_mean"
        if col in ranked.columns:
            available.append((metric, col))
    n_cols = 2
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5 * n_cols, 4.2 * n_rows), squeeze=False)

    for ax, (metric, col) in zip(axes.flat, available):
        view = ranked[["experiment", "method", col]].dropna().copy()
        view["label"] = view["experiment"].astype(str) + " / " + view["method"].astype(str)
        view = view.sort_values(col, ascending=METRIC_DIRECTIONS[metric] == "lower")
        ax.barh(view["label"], view[col], color="#4f83cc", alpha=0.85)
        ax.invert_yaxis()
        ax.set_title(f"{metric} ({METRIC_DIRECTIONS[metric]} is better)")
        ax.grid(axis="x", alpha=0.25)

    for ax in axes.flat[len(available):]:
        ax.axis("off")
    plt.tight_layout()
    out_path = output_dir / "overall_metric_bars.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_composite_heatmap(ranked: pd.DataFrame, metrics: list[str], output_dir: Path) -> Path:
    score_cols = [f"{metric}_score" for metric in metrics if f"{metric}_score" in ranked.columns]
    labels = (ranked["experiment"].astype(str) + " / " + ranked["method"].astype(str)).tolist()
    data = ranked[score_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(score_cols) * 1.2), max(6, len(labels) * 0.28)))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(len(score_cols)))
    ax.set_xticklabels([col.replace("_score", "") for col in score_cols], rotation=35, ha="right")
    ax.set_title("Normalized metric scores (1 is best)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    out_path = output_dir / "overall_normalized_score_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_quality_time_scatter(ranked: pd.DataFrame, output_dir: Path) -> Path | None:
    if "total_time_mean" not in ranked.columns or "composite_score" not in ranked.columns:
        return None
    view = ranked.dropna(subset=["total_time_mean", "composite_score"]).copy()
    if view.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    for experiment, group in view.groupby("experiment"):
        ax.scatter(group["total_time_mean"], group["composite_score"], s=70, label=experiment, alpha=0.85)
        for _, row in group.iterrows():
            ax.annotate(str(row["method"]), (row["total_time_mean"], row["composite_score"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Mean total time per image (seconds, lower better)")
    ax.set_ylabel("Composite normalized score (higher better)")
    ax.set_title("Quality-time tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(title="Experiment", fontsize=8)
    plt.tight_layout()
    out_path = output_dir / "quality_time_tradeoff.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_methods_boxplots(combined: pd.DataFrame, metrics: list[str], output_dir: Path) -> Path | None:
    available = raw_metric_list(combined, metrics)
    if not available:
        return None

    n_cols = 2
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5 * n_cols, 4.5 * n_rows), squeeze=False)
    methods = [method for method in METHOD_ORDER if method in set(combined["method"].astype(str))]

    for ax, metric in zip(axes.flat, available):
        values = [
            pd.to_numeric(combined.loc[combined["method"].astype(str) == method, metric], errors="coerce").dropna().to_numpy()
            for method in methods
        ]
        nonempty = [(method, vals) for method, vals in zip(methods, values) if len(vals)]
        if not nonempty:
            ax.axis("off")
            continue
        ascending = METRIC_DIRECTIONS[metric] == "lower"
        nonempty = sorted(nonempty, key=lambda item: np.mean(item[1]), reverse=not ascending)
        labels, data = zip(*nonempty)
        try:
            box = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        except TypeError:
            box = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        means = np.array([np.mean(vals) for vals in data], dtype=float)
        best_idx = 0
        worst_idx = len(labels) - 1
        for i, patch in enumerate(box["boxes"]):
            if i == best_idx:
                patch.set_facecolor("#22c55e")
                patch.set_alpha(0.70)
            elif i == worst_idx:
                patch.set_facecolor("#ef4444")
                patch.set_alpha(0.55)
            else:
                patch.set_facecolor("#93c5fd")
                patch.set_alpha(0.45)
        ax.set_title(f"{metric} across all experiments ({METRIC_DIRECTIONS[metric]} is better)")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        best_label = labels[best_idx]
        worst_label = labels[worst_idx]
        ax.text(
            0.02,
            0.98,
            f"Best mean: {best_label}\nWorst mean: {worst_label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#e5e7eb", alpha=0.85),
        )

    for ax in axes.flat[len(available):]:
        ax.axis("off")
    plt.tight_layout()
    out_path = output_dir / "all_methods_metric_boxplots.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_regions_vs_auc(region_metric_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if region_metric_df.empty or not {"n_regions", "auc_ins", "auc_del"}.issubset(region_metric_df.columns):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    plot_specs = [
        ("auc_ins", "Number of regions vs AUC+ (higher better)"),
        ("auc_del", "Number of regions vs AUC- (lower better)"),
    ]
    methods = [method for method in METHOD_ORDER if method in set(region_metric_df["method"].astype(str))]
    cmap = plt.get_cmap("tab10")
    for ax, (metric, title) in zip(axes.flat, plot_specs):
        for i, method in enumerate(methods):
            view = region_metric_df[region_metric_df["method"].astype(str) == method][["n_regions", metric]].dropna()
            if view.empty:
                continue
            ax.scatter(view["n_regions"], view[metric], s=28, alpha=0.55, label=method, color=cmap(i % cmap.N))
        ax.set_xlabel("Number of regions / masks")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(title="Method", fontsize=8)
    plt.tight_layout()
    out_path = output_dir / "regions_vs_auc_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def dataframe_to_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    return view.to_html(index=False, escape=True, border=0, classes="data-table")


def rel_path(path: Path | None, output_dir: Path) -> str:
    if path is None:
        return ""
    try:
        return html.escape(path.relative_to(output_dir).as_posix())
    except ValueError:
        return html.escape(str(path))


def build_html_report(
    ranked: pd.DataFrame,
    combined: pd.DataFrame,
    metadata: list[dict],
    winner_table: pd.DataFrame,
    vote_table: pd.DataFrame,
    experiment_type_summary: pd.DataFrame,
    overall_method_summary: pd.DataFrame,
    region_relationship: pd.DataFrame,
    significance_table: pd.DataFrame,
    image_paths: list[Path],
    output_dir: Path,
) -> Path:
    image_html = "\n".join(
        f"<figure><figcaption>{html.escape(path.stem)}</figcaption><img src='{rel_path(path, output_dir)}' alt='{html.escape(path.stem)}'></figure>"
        for path in image_paths
        if path is not None
    )
    metadata_df = pd.DataFrame([{k: v for k, v in item.items() if k != "run_info"} for item in metadata])
    sig_overall = significance_table[significance_table["scope"].astype(str) == "overall"].copy() if "scope" in significance_table.columns else pd.DataFrame()
    sig_by_model = significance_table[significance_table["scope"].astype(str) == "per_experiment_type"].copy() if "scope" in significance_table.columns else pd.DataFrame()
    sig_by_experiment = significance_table[significance_table["scope"].astype(str) == "per_experiment"].copy() if "scope" in significance_table.columns else pd.DataFrame()
    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Experiment Comparison</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #f8fafc; color: #0f172a; }}
header {{ padding: 22px 28px; background: #111827; color: white; }}
main {{ padding: 22px 28px; }}
section {{ margin: 0 0 24px; padding: 16px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; }}
.table-wrap {{ overflow: auto; max-height: 560px; border: 1px solid #e5e7eb; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; white-space: nowrap; }}
th {{ position: sticky; top: 0; background: #eef2ff; cursor: pointer; }}
figure {{ margin: 0 0 20px; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }}
figcaption {{ padding: 8px 10px; font-weight: 700; background: #f1f5f9; }}
img {{ display: block; width: 100%; height: auto; }}
details.explain {{ margin: 10px 0 12px; padding: 10px 12px; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; }}
details.explain summary {{ cursor: pointer; font-weight: 700; color: #1f2937; }}
details.explain p {{ margin: 8px 0; line-height: 1.45; }}
details.explain ul {{ margin: 8px 0 0 20px; padding: 0; line-height: 1.45; }}
code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
</style>
<script>
function sortTable(th) {{
  const table = th.closest('table');
  const tbody = table.tBodies[0];
  const index = Array.from(th.parentNode.children).indexOf(th);
  const asc = th.dataset.asc !== 'true';
  Array.from(table.querySelectorAll('th')).forEach(cell => cell.dataset.asc = '');
  th.dataset.asc = asc ? 'true' : 'false';
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {{
    const av = a.cells[index].innerText.trim();
    const bv = b.cells[index].innerText.trim();
    const an = Number(av);
    const bn = Number(bv);
    if (!Number.isNaN(an) && !Number.isNaN(bn)) {{
      return asc ? an - bn : bn - an;
    }}
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(row => tbody.appendChild(row));
}}
window.addEventListener('DOMContentLoaded', () => {{
  document.querySelectorAll('th').forEach(th => {{
    th.title = 'Click to sort';
    th.addEventListener('click', () => sortTable(th));
  }});
}});
</script>
</head>
<body>
<header>
  <h1>Experiment Comparison</h1>
  <p>{len(metadata)} experiments, {combined['image_no'].nunique() if 'image_no' in combined.columns else 'unknown'} unique image ids, {len(combined)} metric rows.</p>
</header>
<main>
  <section>
    <h2>Experiments</h2>
    <details class="explain">
      <summary>How to read this section</summary>
      <p>This table lists the experiment folders that were loaded, their inferred experiment id, model family, and source CSV. Use it first to verify that the comparison includes the intended runs, for example YOLO, ResNet-50, ViT-B/16, and DETR.</p>
      <p>The experiment id is parsed from folder names such as <code>xai_results_E1_1_4</code>. The first part usually identifies the model family: <code>E1</code> YOLO, <code>E2</code> ResNet-50, <code>E3</code> ViT-B/16, <code>E4</code> DETR.</p>
    </details>
    <div class="table-wrap">{dataframe_to_html(metadata_df)}</div>
  </section>
  <section>
    <h2>Overall Ranking</h2>
    <p>Composite score is the mean of normalized selected metrics. For lower-is-better metrics, the normalized score is inverted.</p>
    <details class="explain">
      <summary>How the ranking is computed</summary>
      <p>Each selected metric is min-max normalized across method/experiment rows. Metrics where lower is better, such as <code>auc_del</code>, <code>sufficiency_gap</code>, and <code>total_time_mean</code>, are inverted after normalization so that 1 always means best.</p>
      <p>The <code>composite_score</code> is the mean of those normalized metric scores. This is useful as a high-level summary, but the paper conclusion should also report the individual metrics because different metrics test different faithfulness properties.</p>
      <ul>
        <li><code>auc_ins</code> / AUC+ higher is better: important pixels should recover confidence quickly when inserted.</li>
        <li><code>auc_del</code> / AUC- lower is better: removing important pixels should reduce confidence quickly.</li>
        <li><code>total_time_mean</code> lower is better: it measures efficiency rather than explanation quality.</li>
      </ul>
    </details>
    <div class="table-wrap">{dataframe_to_html(ranked, max_rows=100)}</div>
  </section>
  <section>
    <h2>Winning Methods</h2>
    <p>This table reports the method with the best mean value for each metric and scope.</p>
    <details class="explain">
      <summary>How to use winner rows</summary>
      <p>This is the cleanest table for writing conclusions. For each scope and metric it reports the method with the best mean value according to the metric direction. Scopes are <code>per_experiment</code>, <code>per_experiment_type</code>, and <code>overall</code>.</p>
      <p>Use <code>per_experiment_type</code> to say which method wins for each model family. Use <code>overall</code> to support a global conclusion across all loaded experiments.</p>
    </details>
    <div class="table-wrap">{dataframe_to_html(winner_table, max_rows=200)}</div>
  </section>
  <section>
    <h2>Win Counts</h2>
    <p>Counts how often each method wins across the selected metrics.</p>
    <details class="explain">
      <summary>What win counts mean</summary>
      <p>This table counts how many metrics each method wins within a scope. It is a compact robustness check: a method that wins many metrics is more consistently strong than a method that wins only one metric.</p>
      <p>Do not treat win count as a statistical test. It is a descriptive summary. Pair it with the Wilcoxon table below for significance.</p>
    </details>
    <div class="table-wrap">{dataframe_to_html(vote_table, max_rows=100)}</div>
  </section>
  <section>
    <h2>Experiment-Type Means</h2>
    <details class="explain">
      <summary>How to read model-family means</summary>
      <p>This table averages metrics by experiment type and method, for example all <code>E1</code> rows for YOLO or all <code>E4</code> rows for DETR. It answers: which XAI method is best within each model family?</p>
      <p>This is useful because detection models and classification models can behave differently under insertion/deletion tests.</p>
    </details>
    <div class="table-wrap">{dataframe_to_html(experiment_type_summary, max_rows=120)}</div>
  </section>
  <section>
    <h2>Overall Method Means</h2>
    <details class="explain">
      <summary>How to read overall means</summary>
      <p>This table pools all loaded experiments and reports mean, standard deviation, and median for each method. It answers: if we ignore experiment type and look globally, which method has the best average behavior?</p>
      <p>Use this for the global headline result, but keep model-family results nearby so reviewers can see whether the global winner is consistent or driven by one model.</p>
    </details>
    <div class="table-wrap">{dataframe_to_html(overall_method_summary, max_rows=40)}</div>
  </section>
  <section>
    <h2>Regions Vs Metrics</h2>
    <p>Spearman correlation between number of regions/masks and each metric. Positive correlation means the metric increases as region count increases.</p>
    <details class="explain">
      <summary>Why region-count analysis matters</summary>
      <p>This checks whether better faithfulness is simply caused by having more or fewer regions. The table reports Spearman rank correlation between <code>n_regions</code> and each metric for each experiment/method.</p>
      <ul>
        <li>Positive correlation for <code>auc_ins</code>: more regions tend to improve insertion AUC.</li>
        <li>Negative correlation for <code>auc_del</code>: more regions tend to improve deletion AUC because lower AUC- is better.</li>
        <li>Near-zero correlation: method performance is less explained by region count alone.</li>
      </ul>
      <p>This is a complexity/fairness check for partition-based explanations.</p>
    </details>
    <div class="table-wrap">{dataframe_to_html(region_relationship, max_rows=120)}</div>
  </section>
  <section>
    <h2>Statistical Significance</h2>
    <p>Paired Wilcoxon tests compare the best-mean method against each other method using matched images. The improvement is positive when the best method is better according to the metric direction.</p>
    <details class="explain">
      <summary>How to interpret the statistical test</summary>
      <p>The Wilcoxon signed-rank test compares two methods on the same images. It is paired and non-parametric, so it is appropriate when metric distributions are not guaranteed to be Gaussian.</p>
      <p>The column <code>wilcoxon_p</code> is the p-value. <code>significant_p_lt_0_05</code> marks p &lt; 0.05. The columns <code>mean_improvement_in_better_direction</code> and <code>median_improvement_in_better_direction</code> are positive when the best method is better according to the metric direction.</p>
      <p>For paper text, prefer a statement like: "Method X improves AUC+ over Method Y with paired Wilcoxon p &lt; 0.05 on N matched images."</p>
    </details>
    <div class="table-wrap">{dataframe_to_html(significance_table, max_rows=200)}</div>
  </section>
  <section>
    <h2>Statistical Significance By Model</h2>
    <details class="explain" open>
      <summary>Why this section is important</summary>
      <p>This table repeats only the <code>per_experiment_type</code> Wilcoxon tests, so each model family is easy to inspect separately. For example, it lets you compare YOLO (<code>E1</code>), ResNet-50 (<code>E2</code>), ViT-B/16 (<code>E3</code>), and DETR (<code>E4</code>) without mixing them into the pooled overall result.</p>
      <p>Use this for statements such as: "For DETR, method X significantly improves AUC+ over method Y," or "The same winner remains significant across classification and detection models."</p>
    </details>
    <div class="table-wrap">{dataframe_to_html(sig_by_model, max_rows=None)}</div>
  </section>
  <section>
    <h2>Statistical Significance By Experiment</h2>
    <details class="explain">
      <summary>How this differs from model-level significance</summary>
      <p>This table uses the <code>per_experiment</code> scope, so each exact run folder is tested separately. This is useful when the same model has multiple backgrounds or budgets, for example <code>E1_1_1</code>, <code>E1_1_4</code>, or <code>E1_2_4</code>.</p>
    </details>
    <div class="table-wrap">{dataframe_to_html(sig_by_experiment, max_rows=None)}</div>
  </section>
  <section>
    <h2>Overall Statistical Significance</h2>
    <details class="explain">
      <summary>How to use the pooled test</summary>
      <p>This table uses all loaded experiments together. It is a high-level global test, but it should be interpreted alongside the model-level table above, because a pooled result can hide different behavior between detection and classification models.</p>
    </details>
    <div class="table-wrap">{dataframe_to_html(sig_overall, max_rows=None)}</div>
  </section>
  <section>
    <h2>Plots</h2>
    <details class="explain">
      <summary>How to read the plots</summary>
      <p>The plots provide visual checks for the tables above. The metric boxplots show method distributions across images and experiments; green marks the best mean and red marks the worst mean. The region scatter plot shows whether performance changes with partition complexity.</p>
      <p>Use plots for intuition and tables for exact values.</p>
    </details>
    {image_html}
  </section>
  <section>
    <h2>References</h2>
    <details class="explain" open>
      <summary>Methods and evaluation concepts used in this report</summary>
      <ul>
        <li>Shapley values: Shapley, L. S. (1953). "A value for n-person games." Contributions to the Theory of Games.</li>
        <li>Owen values / hierarchical coalitions: Owen, G. (1977). "Values of games with a priori unions." Mathematical Economics and Game Theory.</li>
        <li>Insertion/deletion faithfulness metrics: Petsiuk, V., Das, A., and Saenko, K. (2018). "RISE: Randomized Input Sampling for Explanation of Black-box Models." BMVC.</li>
        <li>Deletion/insertion style evaluation is also commonly used in saliency-map evaluation, for example Samek et al. (2017), "Evaluating the visualization of what a deep neural network has learned."</li>
        <li>Sufficiency and comprehensiveness: DeYoung et al. (2020). "ERASER: A Benchmark to Evaluate Rationalized NLP Models." ACL. The same idea is often adapted to vision by keeping or removing top-attributed evidence.</li>
        <li>Stability / sensitivity of explanations: Alvarez-Melis and Jaakkola (2018). "On the Robustness of Interpretability Methods." ICML Workshop.</li>
        <li>Spearman rank correlation: Spearman, C. (1904). "The proof and measurement of association between two things." American Journal of Psychology.</li>
        <li>Wilcoxon signed-rank test: Wilcoxon, F. (1945). "Individual comparisons by ranking methods." Biometrics Bulletin.</li>
        <li>Segment Anything masks: Kirillov et al. (2023). "Segment Anything." ICCV.</li>
        <li>DETR detection model: Carion et al. (2020). "End-to-End Object Detection with Transformers." ECCV.</li>
        <li>ViT classification model: Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words." ICLR.</li>
        <li>ResNet classification model: He et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.</li>
      </ul>
    </details>
  </section>
</main>
</body>
</html>
"""
    out_path = output_dir / "experiment_comparison.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def discover_results_dirs(results_root: Path, experiments: list[str] | None) -> list[Path]:
    if experiments:
        return [results_root / exp for exp in experiments]
    return sorted(path for path in results_root.glob("xai_results*") if path.is_dir())


def run(args):
    if args.results_dirs:
        results_dirs = [Path(path).expanduser().resolve() for path in args.results_dirs]
    else:
        results_root = Path(args.results_root).expanduser().resolve()
        results_dirs = discover_results_dirs(results_root, args.experiments)

    if not results_dirs:
        raise FileNotFoundError("No result directories were provided or discovered.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results_dirs[0].parent / "experiment_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, metadata, region_frames = [], [], []
    for results_dir in results_dirs:
        auc_df, meta = load_experiment(results_dir)
        frames.append(auc_df)
        metadata.append(meta)
        regions = load_region_counts(results_dir, meta["experiment"], meta["model"])
        if not regions.empty:
            region_frames.append(regions)
        print(f"Loaded {meta['experiment']}: {len(auc_df)} rows from {meta['auc_csv']}")

    combined = pd.concat(frames, ignore_index=True)
    if "method" in combined.columns:
        combined["method"] = pd.Categorical(combined["method"], categories=METHOD_ORDER, ordered=True)
        combined = combined.sort_values(["experiment", "method", "image_no" if "image_no" in combined.columns else "image_id"]).reset_index(drop=True)

    summary = summarize_by_experiment_method(combined, results_dirs)
    metrics = args.metrics or DEFAULT_METRICS
    ranked = add_ranking_columns(summary, metrics)
    experiment_type_summary = summarize_raw_by_method(combined, ["exp_type", "model"], metrics)
    overall_method_summary = summarize_raw_by_method(combined, [], metrics)
    winner_table = build_winner_table(
        [
            ("per_experiment", ["experiment"], summary),
            ("per_experiment_type", ["exp_type", "model"], experiment_type_summary),
            ("overall", [], overall_method_summary),
        ],
        metrics,
    )
    vote_table = build_overall_vote_table(winner_table)
    region_counts = pd.concat(region_frames, ignore_index=True) if region_frames else pd.DataFrame()
    region_metric_df = attach_region_counts(combined, region_counts)
    region_relationship = summarize_region_metric_relationship(region_metric_df, metrics)
    significance_table = build_significance_table(combined, metrics)

    combined_path = output_dir / "combined_auc_results.csv"
    summary_path = output_dir / "combined_method_summary.csv"
    ranked_path = output_dir / "combined_method_ranking.csv"
    experiment_type_summary_path = output_dir / "experiment_type_method_summary.csv"
    overall_method_summary_path = output_dir / "overall_method_summary.csv"
    winner_table_path = output_dir / "winner_table.csv"
    vote_table_path = output_dir / "winner_vote_counts.csv"
    region_counts_path = output_dir / "region_counts_by_image_method.csv"
    region_metric_path = output_dir / "region_counts_with_metrics.csv"
    region_relationship_path = output_dir / "regions_vs_metric_summary.csv"
    significance_path = output_dir / "statistical_significance_wilcoxon.csv"
    significance_by_model_path = output_dir / "statistical_significance_by_model.csv"
    significance_by_experiment_path = output_dir / "statistical_significance_by_experiment.csv"
    significance_overall_path = output_dir / "statistical_significance_overall.csv"
    combined.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    experiment_type_summary.to_csv(experiment_type_summary_path, index=False)
    overall_method_summary.to_csv(overall_method_summary_path, index=False)
    winner_table.to_csv(winner_table_path, index=False)
    vote_table.to_csv(vote_table_path, index=False)
    region_counts.to_csv(region_counts_path, index=False)
    region_metric_df.to_csv(region_metric_path, index=False)
    region_relationship.to_csv(region_relationship_path, index=False)
    significance_table.to_csv(significance_path, index=False)
    if "scope" in significance_table.columns:
        significance_table[significance_table["scope"].astype(str) == "per_experiment_type"].to_csv(significance_by_model_path, index=False)
        significance_table[significance_table["scope"].astype(str) == "per_experiment"].to_csv(significance_by_experiment_path, index=False)
        significance_table[significance_table["scope"].astype(str) == "overall"].to_csv(significance_overall_path, index=False)
    else:
        pd.DataFrame().to_csv(significance_by_model_path, index=False)
        pd.DataFrame().to_csv(significance_by_experiment_path, index=False)
        pd.DataFrame().to_csv(significance_overall_path, index=False)

    plot_paths = [
        plot_metric_bars(ranked, metrics, output_dir),
        plot_composite_heatmap(ranked, metrics, output_dir),
    ]
    all_methods_boxplot_path = plot_all_methods_boxplots(combined, metrics, output_dir)
    if all_methods_boxplot_path is not None:
        plot_paths.append(all_methods_boxplot_path)
    regions_vs_auc_path = plot_regions_vs_auc(region_metric_df, output_dir)
    if regions_vs_auc_path is not None:
        plot_paths.append(regions_vs_auc_path)
    quality_time_path = plot_quality_time_scatter(ranked, output_dir)
    if quality_time_path is not None:
        plot_paths.append(quality_time_path)

    report_path = build_html_report(
        ranked,
        combined,
        metadata,
        winner_table,
        vote_table,
        experiment_type_summary,
        overall_method_summary,
        region_relationship,
        significance_table,
        plot_paths,
        output_dir,
    )

    print(f"Saved combined rows: {combined_path}")
    print(f"Saved method summary: {summary_path}")
    print(f"Saved method ranking: {ranked_path}")
    print(f"Saved experiment-type summary: {experiment_type_summary_path}")
    print(f"Saved overall method summary: {overall_method_summary_path}")
    print(f"Saved winner table: {winner_table_path}")
    print(f"Saved winner vote counts: {vote_table_path}")
    print(f"Saved region counts: {region_counts_path}")
    print(f"Saved region-count metric table: {region_metric_path}")
    print(f"Saved region-count summary: {region_relationship_path}")
    print(f"Saved statistical significance table: {significance_path}")
    print(f"Saved statistical significance by model: {significance_by_model_path}")
    print(f"Saved statistical significance by experiment: {significance_by_experiment_path}")
    print(f"Saved overall statistical significance: {significance_overall_path}")
    print(f"Saved HTML report: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="examples/results", help="Folder containing xai_results_* directories.")
    parser.add_argument("--results-dirs", nargs="*", default=None, help="Explicit result directories to compare.")
    parser.add_argument("--experiments", nargs="*", default=None, help="Directory names under --results-root, e.g. xai_results_E1_1_2 xai_results_E3_1.")
    parser.add_argument("--output-dir", default=None, help="Where to save combined CSVs, plots, and HTML.")
    parser.add_argument("--metrics", nargs="*", default=None, choices=sorted(METRIC_DIRECTIONS), help="Metrics used for ranking and plots.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
