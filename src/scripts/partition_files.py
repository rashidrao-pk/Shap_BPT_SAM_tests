"""Names of the compact partition files.

One partition folder holds every compact selection made from the same SAM masks, each in its own file:

    <id>_sam.npy                 all SAM masks painted largest first (one per folder)
    <id>_compact_greedy.npy      --compact-strategy greedy
    <id>_compact_mso_0.5.npy     --compact-strategy mso --mso-alpha 0.5
    <id>_compact_split.npy       --compact-strategy split (split_<N> for --min-region-pixels N != 64); the generator
                                 writes greedy and split by default, and the configs read split
    <id>_results.json            metadata, including one entry per compact file

Readers choose a file with `data.compact_strategy` in the config (every config sets `split`; greedy if absent). Folders written before
this layout have a single `<id>_compact.npy`, made with greedy selection; it is read as `compact_greedy`.
"""
from pathlib import Path

DEFAULT_TAG = 'greedy'
DEFAULT_MIN_GAIN = 64
LEGACY_STAGE = 'compact'


def compact_tags(strategies, mso_alphas=(0.5,), min_gain=DEFAULT_MIN_GAIN):
    """File tags for the requested strategies, e.g. ['greedy', 'mso_0.3', 'mso_0.5', 'split']."""
    tags = []
    for strategy in strategies:
        if strategy == 'greedy':
            tags.append('greedy')
        elif strategy == 'mso':
            tags.extend(f'mso_{alpha:g}' for alpha in mso_alphas)
        elif strategy == 'split':
            tags.append('split' if min_gain == DEFAULT_MIN_GAIN else f'split_{min_gain}')
        else:
            raise ValueError(f"Unknown compact strategy {strategy!r}; use greedy, mso or split")
    return list(dict.fromkeys(tags))


def parse_tag(tag):
    """(strategy, mso alpha or None, minimum split gain or None) for a tag such as 'mso_0.5' or 'split_200'."""
    if tag == 'greedy':
        return 'greedy', None, None
    if tag.startswith('mso_'):
        return 'mso', float(tag[4:]), None
    if tag == 'split' or tag.startswith('split_'):
        return 'split', None, int(tag[6:]) if tag != 'split' else DEFAULT_MIN_GAIN
    raise ValueError(f'Unknown compact tag {tag!r}')


def stage(tag=DEFAULT_TAG):
    """Stage name (file suffix) of a compact tag: 'compact_greedy', 'compact_split', ..."""
    tag = str(tag or DEFAULT_TAG)
    return tag if tag.startswith(LEGACY_STAGE) else f'{LEGACY_STAGE}_{tag}'


def configured_tag(data):
    return str((data or {}).get('compact_strategy') or DEFAULT_TAG)


def configured_stage(data):
    return stage(configured_tag(data))


def tag_of(stage_name):
    """'compact_split' -> 'split'; the legacy 'compact' -> 'greedy'."""
    return DEFAULT_TAG if stage_name == LEGACY_STAGE else stage_name.removeprefix(f'{LEGACY_STAGE}_')


def alternatives(stage_name):
    """Stage names that hold the same partition: compact_greedy and the legacy compact are interchangeable."""
    if stage_name in (LEGACY_STAGE, stage(DEFAULT_TAG)):
        return [stage(DEFAULT_TAG), LEGACY_STAGE] if stage_name != LEGACY_STAGE else [LEGACY_STAGE, stage(DEFAULT_TAG)]
    return [stage_name]


def partition_file(folder, image_id, stage_name):
    """Path of one image's partition for a stage, falling back to the legacy name; the preferred path if absent."""
    folder = Path(folder)
    for name in alternatives(stage_name):
        path = folder / f'{image_id}_{name}.npy'
        if path.is_file():
            return path
    return folder / f'{image_id}_{alternatives(stage_name)[0]}.npy'


def stage_files(folder, stage_name, names=None):
    """{image id: path} of every file of this stage in a folder (legacy names included).
    `names` may pass a listing of the folder, which is faster on network storage."""
    folder = Path(folder)
    names = set(names) if names is not None else ({path.name for path in folder.iterdir() if path.is_file()}
                                                   if folder.is_dir() else set())
    found = {}
    for name in reversed(alternatives(stage_name)):  # Preferred name last, so it wins.
        suffix = f'_{name}.npy'
        for file_name in names:
            if file_name.endswith(suffix):
                found[file_name[:-len(suffix)]] = folder / file_name
    return found


def run_label(folder_name, tag):
    """Name for XAI output folders: the partition folder, plus the tag unless it is greedy."""
    return folder_name if tag == DEFAULT_TAG else f'{folder_name}_{tag}'


def tag_label(stage_name):
    """Short label for tables: '' for greedy and non-compact stages, 'MSO0.5', 'SPLIT', 'SPLIT200' otherwise."""
    if not str(stage_name).startswith(LEGACY_STAGE):
        return ''
    tag = tag_of(stage_name)
    if tag == DEFAULT_TAG:
        return ''
    strategy, alpha, gain = parse_tag(tag)
    return f'MSO{alpha:g}' if strategy == 'mso' else ('SPLIT' if gain == DEFAULT_MIN_GAIN else f'SPLIT{gain}')


def sibling(partition_path, stage_name, other_stage):
    """Path of another stage's file for the same image, e.g. the _sam.npy next to a compact file."""
    path = Path(partition_path)
    for name in alternatives(stage_name):
        if path.name.endswith(f'_{name}.npy'):
            return path.with_name(path.name[:-len(f'_{name}.npy')] + f'_{other_stage}.npy')
    return None



SAM_MODELS = ('sam1', 'sam2', 'medsam', 'medicosam')


def add_partition_arguments(parser):
    """--sam / --partition-folder / --compact-strategy, shared by the XAI runners (same rules as view_partitions)."""
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--sam', dest='partition_sam', choices=SAM_MODELS,
                        help="Partitions of this model: the config's data.mask_dir_final_<model>")
    choice.add_argument('--partition-folder', help='Partition folder name under output.partitions, or a full path')
    parser.add_argument('--compact-strategy', dest='partition_strategy', metavar='TAG',
                        help='Compact file to explain: greedy, split, mso_0.5, ... (default: data.compact_strategy)')
    return parser


def apply_partition_choice(config, sam=None, partition_folder=None, compact_strategy=None):
    """Point config['data'] at the chosen folder and compact file; returns a one-line description."""
    data = config['data']
    if partition_folder:
        data['mask_dir_final'] = str(partition_folder)
    elif sam:
        key = f'mask_dir_final_{sam}'
        if not data.get(key):
            raise ValueError(f'The config has no data.{key}; pass --partition-folder FOLDER_NAME instead')
        data['mask_dir_final'] = data[key]
    if compact_strategy:
        tag = tag_of(stage(compact_strategy))
        parse_tag(tag)  # Rejects unknown tags early.
        data['compact_strategy'] = tag
    return f"Partitions: {Path(data['mask_dir_final']).name} | compact file: <id>_{configured_stage(data)}.npy"
