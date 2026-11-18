"""Per-image timing of every XAI step, split into GPU, CPU and disk work, written as <id>_timing.json.

Steps nest: `method:BPT-1000` contains the model calls it makes (`yolo.predict` and its `inference (GPU)` part,
or `model.predict_batch`), and the masked-image composition (`compose_masked_images`). For each step the file
gives the total time (`seconds`, children included) and its own time (`self_seconds`). The `by_kind` totals add
up own times, so they partition the image's wall time:

- gpu   : model forward passes on the device (YOLO inference; adapters' predict_batch, which also includes their
          own tensor preprocessing; the GPU work has finished when the step ends, because results come back to
          the CPU inside it)
- cpu   : Python / NumPy work: masked-image composition, BPT construction and traversal, Shapley bookkeeping,
          YOLO pre- and postprocessing (letterbox, NMS), metrics, plots
- io    : reading images and partitions, writing results
- mixed : steps whose own time is mostly Python glue around children; counted as cpu in `by_kind`

The runner opens one timer per image (`start`); code anywhere in the call stack records into it through
`current()`, which is a no-op when no timer is open.
"""
import json
import platform
import time
from contextlib import contextmanager

KINDS = ('gpu', 'cpu', 'io', 'mixed')


class StepTimer:
    def __init__(self, image_id=None):
        self.image_id = image_id
        self.started = time.perf_counter()
        self.records = {}       # path -> [seconds, calls, kind]
        self.order = []
        self.stack = []
        self.counters = {}
        self.open = []          # (path, start) for begin/end

    def _path(self, name):
        return '/'.join([*self.stack, name])

    def _add(self, path, seconds, kind):
        if path not in self.records:
            self.records[path] = [0.0, 0, kind]
            self.order.append(path)
        record = self.records[path]
        record[0] += seconds
        record[1] += 1

    @contextmanager
    def step(self, name, kind='cpu'):
        self.begin(name, kind)
        try:
            yield
        finally:
            self.end()

    def begin(self, name, kind='cpu'):
        path = self._path(name)
        self.stack.append(name)
        self.open.append((path, kind, time.perf_counter()))

    def end(self):
        if not self.open:
            return
        path, kind, start = self.open.pop()
        self.stack.pop()
        self._add(path, time.perf_counter() - start, kind)

    def add(self, name, seconds, kind):
        """A part measured elsewhere (e.g. Ultralytics' own preprocess/inference/postprocess times)."""
        self._add(self._path(name), float(seconds), kind)

    def count(self, name, value=1):
        self.counters[name] = self.counters.get(name, 0) + value

    def summary(self, extra=None):
        while self.open:  # A step left open by an exception
            self.end()
        total = time.perf_counter() - self.started
        steps, by_kind = [], dict.fromkeys(('gpu', 'cpu', 'io'), 0.0)
        for path in self.order:
            seconds, calls, kind = self.records[path]
            depth = path.count('/') + 1
            children = sum(self.records[child][0] for child in self.order
                           if child.startswith(path + '/') and child.count('/') + 1 == depth + 1)
            own = max(seconds - children, 0.0)
            by_kind['cpu' if kind == 'mixed' else kind] += own
            steps.append({'step': path, 'kind': kind, 'calls': calls, 'seconds': round(seconds, 4),
                          'self_seconds': round(own, 4), 'share_of_image': round(seconds / total, 4) if total else 0})
        top_level = sum(self.records[path][0] for path in self.order if '/' not in path)
        by_kind['cpu'] += max(total - top_level, 0.0)  # Untimed glue between steps
        result = {
            'image_id': self.image_id, 'total_seconds': round(total, 3),
            'by_kind_seconds': {kind: round(value, 3) for kind, value in by_kind.items()},
            'by_kind_share': {kind: round(value / total, 3) if total else 0 for kind, value in by_kind.items()},
            'untimed_seconds': round(max(total - top_level, 0.0), 3),
            'counters': self.counters,
            'slowest_own_time': sorted(({'step': s['step'], 'kind': s['kind'], 'self_seconds': s['self_seconds']}
                                        for s in steps), key=lambda s: -s['self_seconds'])[:10],
            'steps': steps,
            'host': platform.node(),
        }
        if self.counters.get('model_images') and by_kind['gpu']:
            result['model_images_per_gpu_second'] = round(self.counters['model_images'] / by_kind['gpu'], 1)
        return {**result, **(extra or {})}

    def write(self, path, extra=None):
        summary = self.summary(extra)
        path.write_text(json.dumps(summary, indent=2) + '\n')
        return summary


class _NoTimer:
    def step(self, name, kind='cpu'):
        return _null()

    def begin(self, name, kind='cpu'):
        pass

    def end(self):
        pass

    def add(self, name, seconds, kind):
        pass

    def count(self, name, value=1):
        pass


@contextmanager
def _null():
    yield


_NO_TIMER = _NoTimer()
_current = None


def start(image_id=None):
    global _current
    _current = StepTimer(image_id)
    return _current


def stop():
    global _current
    _current = None


def current():
    return _current or _NO_TIMER


def device_info():
    """GPU name and batch-relevant facts for the timing file."""
    try:
        import torch
        if torch.cuda.is_available():
            return {'device': 'cuda', 'gpu': torch.cuda.get_device_name(0),
                    'gpu_memory_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    except ImportError:
        pass
    return {'device': 'cpu', 'gpu': None}
