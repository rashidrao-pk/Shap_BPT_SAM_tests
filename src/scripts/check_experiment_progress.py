#!/usr/bin/env python
"""Check completion progress for all saved dataset XAI experiment folders.

The checker scans result folders such as ``xai_results_E1_1_4`` and reports:

- how many images have complete CSV rows for all expected methods
- how many images have complete per-image outputs
- which images are pending or partial
- which experiment folders are complete

It is intentionally lightweight and does not import torch, YOLO, or shap_bpt.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from config_paths import output_path, partition_path


DEFAULT_METHODS = ["BPT", "sam", "coverage", "compact", "filled", "refined", "refined_filled"]


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or pd.isna(seconds):
        return ""
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def normalize_image_id(value) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    text = Path(text).stem
    return text.zfill(12) if text.isdigit() else text


def latest_file(folder: Path, pattern: str, exclude=()) -> Path | None:
    matches = [path for path in folder.glob(pattern) if path.name not in set(exclude)]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else start.resolve()
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() or (path / "src").is_dir():
            return path
    return Path.cwd()


def load_config(project_root: Path, config_name_or_path: str | None) -> dict:
    if not config_name_or_path:
        return {}
    config_path = Path(config_name_or_path).expanduser()
    if not config_path.suffix:
        config_path = project_root / "src" / "configs" / f"{config_name_or_path}.yaml"
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_results_root(project_root: Path, args) -> Path:
    if args.results_root:
        path = Path(args.results_root).expanduser()
        return path if path.is_absolute() else project_root / path

    config = load_config(project_root, args.config)
    if config:
        return output_path(config, 'explanations', project_root)
    current = project_root / 'results/explanations'
    return current if current.is_dir() else project_root / 'outputs/results'


def resolve_masks_base_path(project_root: Path, args) -> Path | None:
    if args.masks_base_path:
        path = Path(args.masks_base_path).expanduser()
        return path if path.is_absolute() else project_root / path
    if not args.config:
        return None
    config = load_config(project_root, args.config)
    try:
        return partition_path(config, project_root)
    except (KeyError, ValueError):
        return None


def expected_image_ids_from_masks(masks_base_path: Path | None) -> set[str]:
    if masks_base_path is None or not masks_base_path.exists():
        return set()
    image_ids = set()
    for path in masks_base_path.glob('*.npy'):
        match = re.match(r'(.+)_(?:sam|compact|refined|boosted|filled|boosted_filled)$', path.stem)
        if match:
            image_ids.add(normalize_image_id(match.group(1)))
    return image_ids


def expected_image_ids_for_experiment(results_dir: Path, global_expected: set[str]) -> set[str]:
    selected = read_json(results_dir / 'run_info.json').get('image_ids')
    if selected:
        return {normalize_image_id(value) for value in selected}
    summary_path = results_dir / "partition_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if "image_id" in df.columns:
            return {normalize_image_id(value) for value in df["image_id"].dropna().unique()}
    return set(global_expected)


def resolve_auc_csv(results_dir: Path) -> Path | None:
    current = results_dir / "auc_results_all_images_current_run.csv"
    if current.exists():
        return current
    return latest_file(results_dir, "auc_results_all_images_*.csv", exclude={"auc_results_all_images_current_run.csv"})


def load_auc_table(results_dir: Path) -> tuple[Path | None, pd.DataFrame]:
    auc_csv = resolve_auc_csv(results_dir)
    if auc_csv is None:
        return None, pd.DataFrame()
    try:
        df = pd.read_csv(auc_csv)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        df = pd.DataFrame()  # A running writer may temporarily expose an empty CSV.
    return auc_csv, df


def load_auc_status(results_dir: Path, methods: list[str]) -> tuple[Path | None, dict[str, set[str]], set[str]]:
    auc_csv, df = load_auc_table(results_dir)
    if auc_csv is None:
        return None, {}, set()
    if df.empty:
        return auc_csv, {}, set()

    image_col = "image_id" if "image_id" in df.columns else "image_no"
    if image_col not in df.columns or "method" not in df.columns:
        return auc_csv, {}, set()

    rows_by_image: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        image_id = normalize_image_id(row[image_col])
        rows_by_image.setdefault(image_id, set()).add(str(row["method"]))

    required = set(methods)
    complete = {image_id for image_id, present in rows_by_image.items() if required.issubset(present)}
    return auc_csv, rows_by_image, complete


def load_partition_time_by_image(results_dir: Path) -> dict[str, float]:
    summary_path = results_dir / "partition_summary.csv"
    if not summary_path.exists():
        return {}
    df = pd.read_csv(summary_path)
    if df.empty or "image_id" not in df.columns:
        return {}
    time_col = "total_time_sec" if "total_time_sec" in df.columns else None
    if time_col is None:
        return {}
    times = {}
    for _, row in df.iterrows():
        image_id = normalize_image_id(row["image_id"])
        value = pd.to_numeric(row.get(time_col, 0.0), errors="coerce")
        times[image_id] = 0.0 if pd.isna(value) else float(value)
    return times


def estimate_time_progress(results_dir: Path, auc_csv: Path | None, completed_images: list[str], pending_images: int) -> dict:
    if auc_csv is None or not auc_csv.exists():
        return {
            # "time_taken_until_now_sec": 0.0,
            # "time_taken_until_now": "0s",
            "avg_time_per_completed_image_sec": 0.0,
            "avg_time_per_completed_image": "0s",
            "estimated_time_required_sec": 0.0,
            "estimated_time_required": "0s",
        }

    _, auc_df = load_auc_table(results_dir)
    if auc_df.empty:
        taken = 0.0
        completed_count = len(completed_images)
    else:
        image_col = "image_id" if "image_id" in auc_df.columns else "image_no"
        if image_col in auc_df.columns:
            auc_df["image_id_str"] = auc_df[image_col].apply(normalize_image_id)
        else:
            auc_df["image_id_str"] = ""
        for col in ["time_exp", "time_eval"]:
            if col not in auc_df.columns:
                auc_df[col] = 0.0
            auc_df[col] = pd.to_numeric(auc_df[col], errors="coerce").fillna(0.0)

        per_image = auc_df.groupby("image_id_str", dropna=False)[["time_exp", "time_eval"]].sum()
        per_image["auc_time_sec"] = per_image["time_exp"] + per_image["time_eval"]
        # Partition generation happened before XAI; do not add it to the XAI ETA.
        per_image["total_time_sec"] = per_image["auc_time_sec"]
        taken = float(per_image["total_time_sec"].sum())
        completed_set = set(completed_images)
        completed_per_image = per_image.loc[[idx for idx in per_image.index if idx in completed_set]]
        completed_count = len(completed_per_image) if not completed_per_image.empty else len(completed_images)
        if completed_count:
            avg = float(completed_per_image["total_time_sec"].mean()) if not completed_per_image.empty else taken / completed_count
        else:
            avg = 0.0
        required = avg * pending_images
        return {
            "time_taken_until_now_sec": round(taken, 3),
            "time_taken_until_now": format_duration(taken),
            "avg_time_per_completed_image_sec": round(avg, 3),
            "avg_time_per_completed_image": format_duration(avg),
            "estimated_time_required_sec": round(required, 3),
            "estimated_time_required": format_duration(required),
        }

    avg = taken / completed_count if completed_count else 0.0
    required = avg * pending_images
    return {
        "time_taken_until_now_sec": round(taken, 3),
        "time_taken_until_now": format_duration(taken),
        "avg_time_per_completed_image_sec": round(avg, 3),
        "avg_time_per_completed_image": format_duration(avg),
        "estimated_time_required_sec": round(required, 3),
        "estimated_time_required": format_duration(required),
    }


def image_dir_candidates(results_dir: Path, image_id: str) -> list[Path]:
    return ([results_dir / str(int(image_id)), results_dir / image_id] if image_id.isdigit()
            else [results_dir / image_id])


def existing_image_dir(results_dir: Path, image_id: str) -> Path | None:
    for path in image_dir_candidates(results_dir, image_id):
        if path.exists():
            return path
    return None


def completion_manifest(results_dir: Path, image_id: str):
    folder = existing_image_dir(results_dir, image_id)
    if folder is None:
        return None, None
    candidates = sorted(folder.glob('*_complete.json'))
    # YOLO's all-method marker takes precedence over an older ordinary marker.
    candidates.sort(key=lambda path: not path.name.endswith('_all_complete.json'))
    for path in candidates:
        stem = path.name.removesuffix('_all_complete.json') if path.name.endswith('_all_complete.json') else path.name.removesuffix('_complete.json')
        if normalize_image_id(stem) == normalize_image_id(image_id):
            return read_json(path), folder
    return None, folder


def output_status_for_image(results_dir: Path, image_id: str, methods: list[str], require_plots: bool) -> tuple[bool, list[str]]:
    folder = existing_image_dir(results_dir, image_id)
    if folder is None:
        return False, ["image folder"]

    manifest, _ = completion_manifest(results_dir, image_id)
    if manifest is not None:
        artifacts = manifest.get('artifacts', [])
        missing = [name for name in artifacts if not (folder / name).is_file()]
        if not artifacts:
            missing.append('completion artifact list')
        for name in artifacts:
            if name.endswith('_method_status.json'):
                status = read_json(folder / name)
                absent = set(methods) - set(status)
                missing.extend(f'{method} method status' for method in sorted(absent))
        signature = read_json(results_dir / 'run_info.json').get('signature')
        if signature and manifest.get('signature') != signature:
            missing.append('completion marker for current run settings')
        if require_plots:
            for pattern in ['*_xai_results.png', 'auc_results_*.png', 'faithfulness_curves_*.png']:
                if not any(folder.glob(pattern)):
                    missing.append(pattern)
        return not missing, missing

    image_no = int(image_id) if image_id.isdigit() else image_id
    missing = []
    if not (folder / f"{image_no}_predictions.json").exists() and not (folder / f"{image_id}_predictions.json").exists():
        missing.append("predictions.json")

    for method in methods:
        if not (folder / f"{method}_{image_no}.png").exists() and not (folder / f"{method}_{image_id}.png").exists():
            missing.append(f"{method} heatmap")

    if require_plots:
        expected_patterns = [
            "*_xai_results.png",
            "auc_results_*.png",
            "faithfulness_curves_*.png",
        ]
        for pattern in expected_patterns:
            if not any(folder.glob(pattern)):
                missing.append(pattern)

    return len(missing) == 0, missing


def discover_experiment_dirs(results_root: Path, experiments: list[str] | None) -> list[Path]:
    if experiments:
        dirs = []
        for exp in experiments:
            path = Path(exp).expanduser()
            if not path.is_absolute():
                path = results_root / exp
            dirs.append(path)
        return dirs
    dirs = {path.parent for path in results_root.rglob('run_info.json')}
    dirs.update(path for path in results_root.glob('xai_results*') if path.is_dir())
    return sorted(dirs)


def summarize_experiment(results_dir: Path, expected_ids: set[str], methods: list[str], require_plots: bool) -> tuple[dict, list[dict]]:
    run_info = read_json(results_dir / "run_info.json")
    experiment_name = run_info.get("exp_no", results_dir.name)
    auc_csv, csv_rows_by_image, csv_complete = load_auc_status(results_dir, methods)
    ids_seen = set(csv_rows_by_image)
    ids_seen.update(normalize_image_id(path.name) for path in results_dir.iterdir() if path.is_dir())
    image_ids = set(expected_ids) if expected_ids else ids_seen

    complete_images = []
    partial_images = 0
    partial_rows = []
    for image_id in sorted(image_ids):
        csv_methods = csv_rows_by_image.get(image_id, set())
        csv_ok = image_id in csv_complete
        output_ok, missing_outputs = output_status_for_image(results_dir, image_id, methods, require_plots)
        manifest, _ = completion_manifest(results_dir, image_id)
        complete = output_ok if manifest is not None else csv_ok and output_ok
        if complete:
            complete_images.append(image_id)
        else:
            if existing_image_dir(results_dir, image_id) is not None:
                partial_images += 1
            partial_rows.append(
                {
                    "experiment": experiment_name,
                    "folder": results_dir.name,
                    "image_id": image_id,
                    "csv_methods": len(csv_methods),
                    "missing_csv_methods": ",".join(method for method in methods if method not in csv_methods),
                    "missing_outputs": ",".join(missing_outputs),
                }
            )

    total = len(image_ids)
    completed = len(complete_images)
    pending = max(0, total - completed)
    time_progress = estimate_time_progress(results_dir, auc_csv, complete_images, pending)
    timings = {normalize_image_id(row['image_id']): float(row['total_time_sec'])
               for row in read_json(results_dir / 'timing_summary.json').get('images', [])}
    for image_id in complete_images:
        manifest, _ = completion_manifest(results_dir, image_id)
        if manifest:
            seconds = manifest.get('total_time_sec', manifest.get('timing', {}).get('total_time_sec'))
            if seconds is not None:
                timings[image_id] = float(seconds)
    measured = [timings[image_id] for image_id in complete_images if image_id in timings]
    if measured:
        average = sum(measured)/len(measured)
        time_progress.update(time_taken_until_now_sec=sum(measured),
                             time_taken_until_now=format_duration(sum(measured)),
                             avg_time_per_completed_image_sec=average,
                             avg_time_per_completed_image=format_duration(average),
                             estimated_time_required_sec=average*pending,
                             estimated_time_required=format_duration(average*pending))
    elif auc_csv is None or not csv_rows_by_image:
        time_progress.update(time_taken_until_now='n/a', avg_time_per_completed_image='n/a',
                             estimated_time_required='n/a')
    failed_ids = set()
    for filename in ['failures.json', 'full_image_set_failures.json']:
        path = results_dir / filename
        if path.is_file():
            try:
                failed_ids.update(normalize_image_id(row['image_id']) for row in json.loads(path.read_text()))
            except (ValueError, KeyError, TypeError):
                pass
    failed_ids &= image_ids - set(complete_images)
    runner = run_info.get('runner') or next((part.name for part in results_dir.parents
                                            if part.name.startswith(('OD_', 'OC_', 'COD_'))), experiment_name)
    partition = Path(run_info.get('masks_base_path', '')).name

    summary = {
        "experiment": runner,
        "partition": partition,
        "attribute": run_info.get("settings", {}).get("attribute") or (results_dir.parent.name if runner == "OC_CelebA" else ""),
        "folder": str(results_dir),
        "model": run_info.get("model", run_info.get("model_group", "")),
        "bg_type": run_info.get("background_replacement_values", run_info.get("bg_type", "")),
        "max_evals": run_info.get("max_evals", ""),
        "methods_expected": len(methods),
        "timed_images": len(measured),
        "expected_images": total,
        "csv_images_seen": len(csv_rows_by_image),
        "csv_images_complete": len(csv_complete),
        "completed_images": completed,
        "pending_images": pending,
        "partial_images": partial_images,
        "failed_images": len(failed_ids),
        "completion_percent": round(100.0 * completed / total, 2) if total else 0.0,
        "status": "complete" if total and pending == 0 else "partial",
        **time_progress,
        "auc_csv": str(auc_csv) if auc_csv else "",
    }
    return summary, partial_rows


def order_summary_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "experiment",
        "partition",
        "attribute",
        "model",
        "bg_type",
        "max_evals",
        "methods_expected",
        "expected_images",
        "csv_images_seen",
        "csv_images_complete",
        "completed_images",
        "pending_images",
        "partial_images",
        "failed_images",
        "completion_percent",
        "status",
        "time_taken_until_now",
        "estimated_time_required",
        "avg_time_per_completed_image",
        "time_taken_until_now_sec",
        "estimated_time_required_sec",
        "avg_time_per_completed_image_sec",
        "folder",
        "auc_csv",
    ]
    ordered_cols = [col for col in preferred if col in summary_df.columns] + [col for col in summary_df.columns if col not in preferred]
    return summary_df[ordered_cols]


def write_html_report(summary_df: pd.DataFrame, partial_df: pd.DataFrame, output_dir: Path) -> Path:
    summary_df = order_summary_columns(summary_df)
    summary_html = summary_df.to_html(index=False, escape=True, border=0, classes="data-table", table_id="summary-table")
    if partial_df.empty:
        partial_html = "<p>No pending images.</p>"
    else:
        partial_html = partial_df.to_html(index=False, escape=True, border=0, classes="data-table", table_id="pending-table")

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>XAI Experiment Progress</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #f8fafc; color: #0f172a; }}
header {{ padding: 22px 28px; background: #111827; color: white; }}
main {{ padding: 22px 28px; }}
section {{ margin: 0 0 24px; padding: 16px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; }}
.table-wrap {{ overflow: auto; max-height: 620px; border: 1px solid #e5e7eb; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; white-space: nowrap; }}
th {{ position: sticky; top: 0; background: #eef2ff; cursor: pointer; }}
details summary {{ cursor: pointer; font-weight: 700; }}
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
  <h1>XAI Experiment Progress</h1>
  <p>{len(summary_df)} experiment folders checked. Generated in {html.escape(str(output_dir))}.</p>
</header>
<main>
  <section>
    <h2>Summary</h2>
    <div class="table-wrap">{summary_html}</div>
  </section>
  <section>
    <details open>
      <summary>Pending / partial images ({len(partial_df)})</summary>
      <div class="table-wrap">{partial_html}</div>
    </details>
  </section>
</main>
</body>
</html>
"""
    out_path = output_dir / "experiment_progress_report.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def run(args):
    project_root = find_project_root(Path(__file__).resolve())
    results_root = resolve_results_root(project_root, args).resolve()
    masks_base_path = resolve_masks_base_path(project_root, args)
    global_expected = expected_image_ids_from_masks(masks_base_path)
    methods = args.methods or DEFAULT_METHODS

    experiment_dirs = discover_experiment_dirs(results_root, args.experiments)
    if not experiment_dirs:
        raise FileNotFoundError(f"No explanation runs found in: {results_root}")

    print(f"Results root: {results_root}")
    if masks_base_path:
        print(f"Masks base path: {masks_base_path}")
    if args.methods:
        print(f"Expected methods ({len(methods)}): {','.join(methods)}")
    else:
        print('Expected methods: read from each run_info.json (legacy defaults when absent)')
    if global_expected:
        print(f"Expected images from masks: {len(global_expected)}")

    summaries = []
    partial_rows = []
    for results_dir in experiment_dirs:
        if not results_dir.exists():
            print(f"Missing experiment folder: {results_dir}")
            continue
        expected_ids = expected_image_ids_for_experiment(results_dir, global_expected)
        experiment_methods = args.methods or read_json(results_dir / 'run_info.json').get('methods') or methods
        summary, rows = summarize_experiment(results_dir, expected_ids, experiment_methods, args.require_plots)
        summaries.append(summary)
        partial_rows.extend(rows)

    summary_df = pd.DataFrame(summaries)
    partial_df = pd.DataFrame(partial_rows)

    if summary_df.empty:
        raise RuntimeError("No experiment summaries could be created.")
    summary_df = order_summary_columns(summary_df)

    display_cols = [
        'experiment', 'partition', 'attribute', 'bg_type', 'max_evals',
        'completed_images', 'expected_images', 'pending_images', 'partial_images',
        'failed_images', 'completion_percent', 'avg_time_per_completed_image', 'estimated_time_required',
    ]
    print("\nExperiment progress:")
    print(summary_df[display_cols].to_string(index=False))

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        output_path(load_config(project_root, args.config), 'evaluations', project_root) / 'experiment_progress'
        if args.config else results_root / 'experiment_progress')
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "experiment_progress_summary.csv"
    partial_path = output_dir / "experiment_progress_pending.csv"
    summary_df.to_csv(summary_path, index=False)
    partial_df.to_csv(partial_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved pending details: {partial_path}")
    if args.html_report:
        html_path = write_html_report(summary_df, partial_df, output_dir)
        print(f"Saved HTML report: {html_path}")

    if args.show_pending and not partial_df.empty:
        print("\nPending / partial images:")
        print(partial_df.head(args.show_pending).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Config name from src/configs or path to YAML. Used to infer results and masks paths.")
    parser.add_argument("--results-root", default=None, help="Explanation root; scans run_info.json recursively for all dataset runners.")
    parser.add_argument("--experiments", nargs="*", default=None, help="Experiment folder names or absolute paths to check.")
    parser.add_argument("--masks-base-path", default=None, help="Folder containing saved partition .npy files. Used to infer expected image count.")
    parser.add_argument("--output-dir", default=None, help="Where to save progress CSVs. Default: results root.")
    parser.add_argument("--methods", nargs="*", default=None, help="Expected explanation methods. Default is the 7-method pipeline.")
    parser.add_argument("--require-plots", action="store_true", help="Also require XAI grid, AUC curve, and faithfulness curve images.")
    parser.add_argument("--html-report", action="store_true", help="Save a sortable HTML progress report.")
    parser.add_argument("--show-pending", type=int, default=25, help="Print first N pending/partial image rows. Use 0 to hide.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
