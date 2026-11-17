"""Options of the SAM-guided BPT merge (T1), shared by every runner.

Defaults reproduce the v1 builder exactly (one 64-bit label word, priority * (|L| + 1) ** 2, label 0
counted). Non-default settings change three things, and only then:
  - the run folder gets a suffix, e.g. bg_full_B3000_W2_a4_e1;
  - with --explain_all, the guided method is named e.g. BPT+SAM_c@a4W2-3000 (no '-' inside the tag,
    because method names are split with rsplit('-', 1));
  - the partition cap rises to 64 * W - 1 labels.
"""
DEFAULTS = {'label_words': 1, 'guide_alpha': 2.0, 'guide_eps': 1.0, 'count_unassigned': True}
MAX_WORDS = 4


def add_arguments(parser):
    group = parser.add_argument_group('SAM-guided BPT merge (defaults reproduce v1)')
    group.add_argument('--label-words', choices=['1', '2', '4', 'auto'], default='1',
                       help='64-bit words per label set: 1 (labels < 64), 2 (< 128), 4 (< 256), or auto')
    group.add_argument('--guide-alpha', default='2',
                       help="Penalty exponent: priority * (|L| + eps) ** alpha. 0 = unguided tree; 'hard' = never "
                            'merge across a label until no other merge is left')
    group.add_argument('--guide-eps', type=float, default=1.0, help='Penalty offset eps (> 0)')
    group.add_argument('--no-count-unassigned', dest='count_unassigned', action='store_false',
                       help='Do not count label 0 (pixels without a SAM mask) in |L|')
    parser.set_defaults(count_unassigned=True)


def settings(args):
    """Normalised builder keyword arguments from parsed args (or DEFAULTS when the options are absent)."""
    words = str(getattr(args, 'label_words', '1'))
    alpha = str(getattr(args, 'guide_alpha', '2'))
    result = {'label_words': 'auto' if words == 'auto' else int(words),
              'guide_alpha': 'hard' if alpha == 'hard' else float(alpha),
              'guide_eps': float(getattr(args, 'guide_eps', 1.0)),
              'count_unassigned': bool(getattr(args, 'count_unassigned', True))}
    if result['label_words'] not in (1, 2, 4, 'auto'):
        raise ValueError('--label-words must be 1, 2, 4 or auto')
    if result['guide_alpha'] != 'hard' and result['guide_alpha'] < 0:
        raise ValueError("--guide-alpha must be >= 0 or 'hard'")
    if result['guide_eps'] <= 0:
        raise ValueError('--guide-eps must be > 0')
    return result


def is_default(args):
    return settings(args) == DEFAULTS


def tree_kwargs(args):
    """Keyword arguments for build_bpt_from_image(..., prebuilt_partitions=...); empty for the defaults."""
    return {} if is_default(args) else settings(args)


def max_labels(args):
    """Partition cap matching the label words: 63 by default, 255 for 4 words or auto."""
    words = settings(args)['label_words']
    return 64 * (MAX_WORDS if words == 'auto' else words) - 1


def _number(value):
    return value if isinstance(value, str) else f'{value:g}'


def folder_suffix(args):
    """'' for the defaults, else e.g. '_W2_a4_e1' (plus '_u0' when label 0 is not counted)."""
    if is_default(args):
        return ''
    value = settings(args)
    return (f"_W{value['label_words']}_a{_number(value['guide_alpha'])}_e{_number(value['guide_eps'])}"
            + ('' if value['count_unassigned'] else '_u0'))


def method_tag(args):
    """'' for the defaults, else e.g. '@a4W2' (only the settings that differ from the defaults)."""
    if is_default(args):
        return ''
    value = settings(args)
    parts = [f"a{_number(value['guide_alpha'])}" if value['guide_alpha'] != DEFAULTS['guide_alpha'] else '',
             f"W{value['label_words']}" if value['label_words'] != DEFAULTS['label_words'] else '',
             f"e{_number(value['guide_eps'])}" if value['guide_eps'] != DEFAULTS['guide_eps'] else '',
             'u0' if not value['count_unassigned'] else '']
    tag = '@' + ''.join(parts)
    if '-' in tag:
        raise ValueError(f'method tag {tag} must not contain "-"')
    return tag
