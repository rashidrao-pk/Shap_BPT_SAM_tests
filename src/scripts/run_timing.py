"""Consistent end-of-run wall-clock timing for explanation runners."""
import json
from pathlib import Path


def duration(seconds):
    seconds = float(seconds)
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f'{hours}h {minutes}m {remainder:.2f}s'
    if minutes:
        return f'{minutes}m {remainder:.2f}s'
    return f'{remainder:.2f}s'


def report_timing(run_dir, evaluation_dir, images, model_load_sec, total_run_sec, skipped=0, failed=0):
    """Report newly computed images only; resumed images never count as new work."""
    summary = {'model_load_sec': float(model_load_sec), 'total_run_sec': float(total_run_sec),
               'computed_images': len(images), 'skipped_images': int(skipped), 'failed_images': int(failed),
               'average_image_sec': (sum(row['total_time_sec'] for row in images)/len(images) if images else None),
               'images': images,
               'scope': 'Image time includes input/GT, partitions/BPT, explanations, evaluation and per-image output. '
                        'Model loading and aggregate report generation are outside image time. '
                        'Total run includes startup, model loading, image processing and aggregate reports.'}
    for folder in {Path(run_dir), Path(evaluation_dir)}:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'timing_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\nTiming summary (this run):')
    for row in images:
        print(f"  Image {row['image_id']}: {duration(row['total_time_sec'])} ({row['total_time_sec']:.2f} seconds)")
        print(f"    Explanations: {duration(row['explanation_time_sec'])}; "
              f"AUC/faithfulness: {duration(row['evaluation_time_sec'])}")
    if not images:
        print('  No new images completed; cached results are not timed as new work.')
    elif len(images) > 1:
        print(f"  Average per image: {duration(summary['average_image_sec'])}")
    print(f'  Model loading: {duration(model_load_sec)}')
    print(f'  Total run (including startup and reports): {duration(total_run_sec)}')
    print(f'  Saved: {Path(run_dir) / "timing_summary.json"}')
    return summary
