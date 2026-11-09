"""RefCOCO (UNC split): referring expressions, their objects, and the image list.

The official refs(unc).p is a Python pickle. It is read with an unpickler that refuses every
class and function lookup, so only plain lists, dicts, strings and numbers can load and no code
from the file can run.

Each prepared expression is one explanation record, with the ID
<image stem>__ref<ref_id>_s<sent_id>. Partitions stay per image and are shared by its expressions.
"""
import io
import json
from pathlib import Path
import pickle

SPLITS = ('val',)
RECORD_FIELDS = ['record_id', 'image_stem', 'file_name', 'image_id', 'ref_id', 'ann_id', 'sent_id',
                 'category', 'expression', 'sentence_index']


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'refs file asks for {module}.{name}; only plain data is accepted')


def read_refs(source):
    """The list of ref dicts from refs(unc).p, given as a path or bytes."""
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    refs = DataOnlyUnpickler(io.BytesIO(data), encoding='latin1').load()
    if not isinstance(refs, list) or not all(isinstance(ref, dict) for ref in refs):
        raise ValueError('refs(unc).p does not hold a list of refs')
    return refs


def record_id(image_stem, ref_id, sent_id):
    return f'{image_stem}__ref{ref_id}_s{sent_id}'


def image_stem(file_name):
    return Path(file_name).stem


def build_expressions(refs, instances, split='val'):
    """One entry per sentence of every ref in the split, with the COCO image it belongs to.

    refs give file names such as COCO_train2014_000000581857_16.jpg (with the annotation ID
    appended); the image itself comes from instances.json.
    """
    if split not in SPLITS:
        raise ValueError(f'RefCOCO supports --split {", ".join(SPLITS)}')
    images = {item['id']: item for item in instances['images']}
    annotations = {item['id']: item for item in instances['annotations']}
    categories = {item['id']: item['name'] for item in instances['categories']}
    entries = []
    for ref in refs:
        if ref['split'] != split:
            continue
        if ref['ann_id'] not in annotations:
            raise ValueError(f"ref {ref['ref_id']}: annotation {ref['ann_id']} missing from instances.json")
        image = images[ref['image_id']]
        stem = image_stem(image['file_name'])
        for index, sentence in enumerate(ref['sentences']):
            entries.append({'record_id': record_id(stem, ref['ref_id'], sentence['sent_id']), 'image_stem': stem,
                            'file_name': image['file_name'], 'image_id': ref['image_id'], 'ref_id': ref['ref_id'],
                            'ann_id': ref['ann_id'], 'sent_id': sentence['sent_id'],
                            'category': categories[ref['category_id']], 'expression': sentence['sent'].strip(),
                            'sentence_index': index})
    return sorted(entries, key=lambda entry: (entry['image_stem'], entry['ref_id'], entry['sent_id']))


def coco_url(file_name):
    return f'http://images.cocodataset.org/train2014/{file_name}'


def write_expressions(path, entries, split, source):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {'dataset': 'RefCOCO', 'split_by': 'unc', 'split': split, 'source': str(source),
                'expressions': entries}
    text = json.dumps(document, indent=1) + '\n'
    if not path.is_file() or path.read_text() != text:
        path.write_text(text)


def load_expressions(path):
    document = json.loads(Path(path).read_text())
    entries = document['expressions'] if isinstance(document, dict) else document
    missing = [field for field in RECORD_FIELDS if entries and field not in entries[0]]
    if missing:
        raise ValueError(f'{path}: expressions lack {missing}; rerun download_datasets.py --dataset refcoco')
    return entries


def select_expressions(entries, all_sentences=False):
    """By default the first sentence of each ref, so every referred object is explained once."""
    return entries if all_sentences else [entry for entry in entries if entry['sentence_index'] == 0]


def referred_annotations(entries):
    """image stem -> {annotation ID: category} of every referred object."""
    result = {}
    for entry in entries:
        result.setdefault(entry['image_stem'], {})[entry['ann_id']] = entry['category']
    return result
