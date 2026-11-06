"""Explain low partition scores: why objects fail, which sizes and classes fail, and the worst images."""
from collections import Counter, defaultdict

import numpy as np

LOW_IOU = .5
SIZES = [('Tiny (<0.1%)', .1), ('Small (0.1-1%)', 1), ('Medium (1-10%)', 10), ('Large (>10%)', float('inf'))]
CAUSES = {'Merged': 'cells spill far outside the object, so it has no boundary of its own',
          'Partial': 'the cells that fit cover only part of the object',
          'Unassigned': 'most of the object is in unlabeled pixels (cell 0)'}
MIN_GROUP = 15


def size_band(share):
    """share is the object's area as a percentage of the image."""
    return next(name for name, limit in SIZES if share < limit)


def failure_cause(iou, precision, unassigned):
    """Why an object fell below LOW_IOU; empty when it did not."""
    if iou >= LOW_IOU:
        return ''
    if unassigned >= .5:
        return 'Unassigned'
    return 'Merged' if precision < .5 else 'Partial'


def image_table(rows):
    """One row per image, worst first."""
    images = defaultdict(list)
    for row in rows:
        images[row['image_id']].append(row)
    table = []
    for image_id, items in images.items():
        causes = Counter(item['cause'] for item in items if item['cause'])
        table.append({'config': items[0]['config'], 'image_id': image_id, 'mean_iou': float(np.mean([item['best_iou'] for item in items])),
                      'objects': len(items), 'low_objects': sum(causes.values()), 'cells': items[0]['cells'],
                      'smallest_object': min(item['gt_share'] for item in items),
                      'main_cause': causes.most_common(1)[0][0] if causes else ''})
    return sorted(table, key=lambda item: (item['mean_iou'], item['image_id']))


def diagnose(rows):
    ious = np.array([row['best_iou'] for row in rows])
    low = [row for row in rows if row['cause']]
    causes = Counter(row['cause'] for row in low)
    sizes = []
    for name, _ in SIZES:
        values = ious[[row['size'] == name for row in rows]]
        if values.size:
            sizes.append({'size': name, 'objects': int(values.size), 'share': float(values.size / len(rows)),
                          'median_iou': float(np.median(values)), 'low': float((values < LOW_IOU).mean())})
    groups = defaultdict(list)
    for row in rows:
        groups[row['object']].append(row['best_iou'])
    groups = [{'group': name, 'objects': len(values), 'median_iou': float(np.median(values)),
               'low': float(np.mean(np.array(values) < LOW_IOU))} for name, values in groups.items() if len(values) >= MIN_GROUP]
    larger = ious[[row['gt_share'] >= 1 for row in rows]]
    return {'objects': len(rows), 'low': len(low), 'causes': {name: causes[name] / max(len(low), 1) for name in CAUSES},
            'sizes': sizes, 'groups': sorted(groups, key=lambda item: item['median_iou']) if len(groups) > 1 else [],
            'larger_objects': int(larger.size), 'larger_median': float(np.median(larger)) if larger.size else 0.,
            'larger_iou_50': float((larger >= LOW_IOU).mean()) if larger.size else 0., 'images': image_table(rows)}


def finding(summary, result):
    """A one-paragraph explanation built from the measured results."""
    if not result['low']:
        return f"Every object reaches IoU {LOW_IOU}."
    cause, share = max(result['causes'].items(), key=lambda item: item[1])
    text = [f"{result['low']} of {result['objects']} objects ({result['low'] / result['objects']:.0%}) fall below IoU {LOW_IOU}. "
            f"The main cause is **{cause}** ({share:.0%}): {CAUSES[cause]}."]
    worst = max(result['sizes'], key=lambda item: item['low'] * item['objects'])
    text.append(f"Most low scores come from {worst['size'].lower()} objects, of which {worst['low']:.0%} score low.")
    if result['larger_objects'] and result['larger_objects'] < result['objects']:
        text.append(f"Objects covering at least 1% of the image reach a median IoU of {result['larger_median']:.2f}, "
                    f"and {result['larger_iou_50']:.0%} reach IoU {LOW_IOU}, against {summary['median_iou']:.2f} and "
                    f"{summary['iou_50']:.0%} overall.")
    if summary['gain'] < .1:
        text.append(f"The partitions lead a plain grid by only {summary['gain']:+.2f}, so they add little structure here.")
    return ' '.join(text)


def report(summaries, diagnoses, worst):
    lines = ['# Why partitions score low', '', f'An object counts as low when its best IoU is below {LOW_IOU}. Sizes are the share of the image.', '',
             '| Cause | Meaning |', '| --- | --- |', *[f'| {name} | {text} |' for name, text in CAUSES.items()], '']
    for summary in summaries:
        result = diagnoses.get(summary['config'])
        if result is None:
            continue
        lines += [f"## {summary['config']}: {summary['verdict']}", '', f"Verdict reason: {summary['reason']}.", '',
                  finding(summary, result), '', '| Object size | Objects | Share | Median IoU | Below IoU 0.5 |', '| --- | --- | --- | --- | --- |']
        lines += [f"| {item['size']} | {item['objects']} | {item['share']:.0%} | {item['median_iou']:.3f} | {item['low']:.0%} |"
                  for item in result['sizes']]
        if result['groups']:
            lines += ['', f'Lowest classes, with at least {MIN_GROUP} objects:', '', '| Class | Objects | Median IoU | Below IoU 0.5 |', '| --- | --- | --- | --- |']
            lines += [f"| {item['group']} | {item['objects']} | {item['median_iou']:.3f} | {item['low']:.0%} |" for item in result['groups'][:8]]
        lines += ['', 'Worst images:', '', '| Image | Mean IoU | Objects | Below IoU 0.5 | Smallest object | Main cause |', '| --- | --- | --- | --- | --- | --- |']
        lines += [f"| {item['image_id']} | {item['mean_iou']:.3f} | {item['objects']} | {item['low_objects']} | {item['smallest_object']:.2f}% | {item['main_cause'] or '-'} |"
                  for item in result['images'][:worst]]
        lines.append('')
    return '\n'.join(lines)
