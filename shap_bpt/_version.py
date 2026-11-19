"""Version of this ShapBPT build (the SAM-guided fork). setup.py reads __version__ from this file.

History:
  1.0.0  2026-01-01  ShapBPT as published (AAAI 2026): colour/shape BPT, Owen-value explainer.
  1.1.0  2026-08-04  SAM guidance: one 64-bit label bitset per cluster (labels 1-63), merges that join
                     clusters with different SAM labels are penalised (BPT+SAM_c).
  1.2.0  2026-11-17  Wider label bitsets: label_words W = 1/2/4/'auto' (up to 63/127/255 labels, so 254-region
                     SAM partitions fit), configurable penalty guide_alpha (number or 'hard') and guide_eps,
                     penalty dtype fix (TO_DO T1).

Bump __version__ and add a line above whenever shap_bpt/bpt.pyx or shap_bpt/shap_bpt.py changes, then rebuild
the extension on each CPU type (ARM and x86).
"""
__version__ = '1.2.0'
__release_name__ = 'ShapBPT with SAM guidance 1.2.0 (wider label bitsets, configurable penalty), 2026-11-17'


def build_info():
    """Version, compiled extension and whether it is older than bpt.pyx (then it must be rebuilt)."""
    import os
    package = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(package, 'bpt.pyx')
    info = {'version': __version__, 'release': __release_name__, 'package_dir': package}
    try:
        from . import bpt
    except ImportError as error:
        return {**info, 'extension': None, 'error': f'{error} (run: pip install --no-deps --no-build-isolation -e .)'}
    info['extension'] = bpt.__file__
    if os.path.exists(source):
        info['extension_older_than_source'] = os.path.getmtime(bpt.__file__) < os.path.getmtime(source)
    return info
