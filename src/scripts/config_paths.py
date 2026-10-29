"""Shared output roots for generation, explanations, and evaluation reports."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value, project_root=None):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(project_root or PROJECT_ROOT) / path


def load_path_config(name, project_root=None):
    import yaml
    root = Path(project_root or PROJECT_ROOT)
    path = Path(name).expanduser()
    if not path.suffix:
        path = root / 'src/configs' / f'{name}.yaml'
    return yaml.safe_load(resolve_path(path, root).read_text())


def output_path(config, kind, project_root=None):
    if kind not in {'partitions', 'explanations', 'evaluations'}:
        raise ValueError(f'Unknown output kind: {kind}')
    output = config.get('output', {})
    if output.get(kind):
        return resolve_path(output[kind], project_root)
    # Compatibility for historical configs; new configs only need the three keys.
    if kind == 'partitions':
        data = config.get('data', {})
        if data.get('masks_path') is not None:
            return resolve_path(data['masks_path'], project_root) / data.get('mask_dir', '')
    elif output.get('dir') or output.get('folder'):
        return resolve_path(output.get('dir', '.'), project_root) / output.get('folder', '')
    raise ValueError(f'Config must define output.{kind}')


def partition_path(config, project_root=None):
    return output_path(config, 'partitions', project_root) / config['data']['mask_dir_final']


def evaluation_path(config, explanation_run, project_root=None):
    """Mirror a run's relative path under the evaluation root."""
    run = resolve_path(explanation_run, project_root)
    try:
        relative = run.relative_to(output_path(config, 'explanations', project_root))
    except ValueError:
        relative = Path(run.name)
    return output_path(config, 'evaluations', project_root) / relative
