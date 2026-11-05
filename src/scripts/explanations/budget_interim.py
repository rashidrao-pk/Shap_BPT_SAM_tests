"""Descriptive checkpoints for a small first stage; never a validated budget selection."""
import csv
import json
from pathlib import Path
import statistics
from budget_selection import passes


def write_interim(folder, study, rows, diagnostics):
    folder = Path(folder)
    groups = []
    for method in study['methods']:
        for budget in study['budgets']:
            subset = [row for row in rows if row['method'] == method and row['budget'] == budget]
            images = sorted({row['image_id'] for row in subset})
            passed = sum(all(passes(row, study['criteria'], study['compute_auc'])
                             for row in subset if row['image_id'] == image) for image in images)
            entry = {'method': method, 'budget': budget, 'images': len(images), 'passed_images': passed}
            for key in ['spearman', 'top_jaccard', 'relative_l1', 'auc_ins_difference',
                        'auc_del_difference', 'explanation_sec', 'evaluation_sec']:
                values = [row[key] for row in subset if row.get(key) is not None]
                entry['mean_'+key] = statistics.mean(values) if values else None
            groups.append(entry)
    candidate = next((budget for budget in study['budgets'] if all(
        passes(row, study['criteria'], study['compute_auc']) for row in rows if row['budget'] >= budget)), None)
    result = {'status': 'interim_only', 'selected_budget': None,
              'descriptive_candidate': candidate, 'images': study['completed_images'],
              'background': study['background'], 'runner': study['runner'], 'summaries': groups,
              'reference_stable_groups': sum(item['stable'] for item in diagnostics),
              'reference_groups': len(diagnostics),
              'claim': 'Descriptive result on a few completed images, not independent validation or a universal budget.'}
    (folder/'interim_selection.json').write_text(json.dumps(result, indent=2)+'\n')
    with (folder/'interim_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0])); writer.writeheader(); writer.writerows(groups)
    with (folder/'interim_comparisons.csv').open('w', newline='') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    conclusion = (f"Descriptive candidate: {candidate}. It still needs a larger independent validation stage."
                  if candidate is not None else
                  'No tested budget met all declared checks on these images. This does not establish a minimum budget.')
    text = ('# Intermediate XAI budget conclusion\n\n'
        f"Scope: {study['dataset']}, {study['runner']}, {Path(study['inputs'][0]['partition_path'][0]).parent.name}, {study['background']} background. "
        f"Completed images: {len(study['completed_images'])}. Budgets: {study['budgets']}; reference: {study['reference_budget']}.\n\n"
        +conclusion+'\n\n**No production budget is validated by this small checkpoint.** '
        'The completed images follow the original fixed order; they are not a new random validation sample. '
        'Original calibration/validation assignments are preserved. Add images without changing settings to continue.\n\n'
        '| Method | Budget | Images passing all repeats | Mean Spearman | Mean relative L1 | Mean explanation seconds |\n'
        '| --- | ---: | ---: | ---: | ---: | ---: |\n')
    for row in groups:
        text += (f"| {row['method']} | {row['budget']} | {row['passed_images']}/{row['images']} | "
                 f"{row['mean_spearman']} | {row['mean_relative_l1']} | {row['mean_explanation_sec']:.2f} |\n")
    (folder/'interim_conclusion.md').write_text(text)
    print(conclusion, flush=True)
    print('Intermediate report:', folder/'interim_conclusion.md', flush=True)
    return result
