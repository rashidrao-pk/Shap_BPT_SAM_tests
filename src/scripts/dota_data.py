"""DOTA-v1.0 labels and 1024-pixel tiles: parsing, polygon clipping, tiling and rasterising.

Tile label files hold one object per line, in tile pixel coordinates:

    x1 y1 x2 y2 x3 y3 x4 y4 <class name, may contain spaces> <difficult 0|1>

The four corners are the source box moved into the tile, not clipped, so a box cut by the tile
border keeps its shape; filling it inside the tile gives exactly the clipped part. difficult=1
marks an ignored region: an object the DOTA annotators flagged as difficult, or one with less
than --min-keep of its area inside the tile. Ignored regions are left out of GT masks and scores.
"""
import csv
from pathlib import Path
import random

import numpy as np

CLASSES = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
           'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter',
           'roundabout', 'soccer ball field', 'swimming pool']
IGNORED_LABEL = 65535  # Ignored-region pixels in a DOTA label map; never an instance number.
INDEX_FIELDS = ['tile_id', 'source_image', 'x0', 'y0', 'width', 'height', 'n_counted', 'n_ignored', 'selected']
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def normalize_name(name):
    """'small-vehicle', 'Small_Vehicle' and 'small vehicle' all become 'small vehicle'."""
    return ' '.join(str(name).lower().replace('-', ' ').replace('_', ' ').split())


def class_name(name, path=None):
    normalized = normalize_name(name)
    if normalized not in CLASSES:
        where = f' in {path}' if path else ''
        raise ValueError(f'Unknown DOTA class {name!r}{where}; expected one of {CLASSES}')
    return normalized


def _numbers(tokens, path, number):
    try:
        return np.asarray([float(value) for value in tokens], dtype=float).reshape(4, 2)
    except ValueError as error:
        raise ValueError(f'{path}:{number}: expected 8 coordinates, got {tokens}') from error


def parse_tile_line(line, path='<labels>', number=0):
    """One tile label line, read from the right because class names contain spaces."""
    tokens = line.split()
    if len(tokens) < 10 or tokens[-1] not in {'0', '1'}:
        raise ValueError(f'{path}:{number}: expected "x1 y1 ... x4 y4 <class> <0|1>", got {line.strip()!r}')
    return {'polygon': _numbers(tokens[:8], path, number), 'class_name': class_name(' '.join(tokens[8:-1]), path),
            'difficult': int(tokens[-1])}


def read_tile_labels(path):
    path = Path(path)
    return [parse_tile_line(line, path, number) for number, line in enumerate(path.read_text().splitlines(), 1)
            if line.strip()]


def format_tile_labels(objects):
    lines = []
    for item in objects:
        coordinates = ' '.join(f'{value:.2f}' for value in np.asarray(item['polygon'], dtype=float).ravel())
        lines.append(f"{coordinates} {item['class_name']} {int(item['difficult'])}")
    return '\n'.join(lines) + ('\n' if lines else '')


def read_source_labels(path, label_format, width, height):
    """Objects of one source image, in source pixels.

    dota-txt: the official labelTxt (pixel corners, hyphenated class, difficult flag); Ultralytics'
    DOTAv1.zip ships it as labels/<split>_original. yolo-obb: Ultralytics' labels/<split>
    (class index, corners normalised to 0-1, no difficult flag).
    """
    path = Path(path)
    objects = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        tokens = line.split()
        if not tokens or tokens[0].startswith(('imagesource:', 'gsd:')):
            continue
        if label_format == 'dota-txt':
            if len(tokens) < 10:
                raise ValueError(f'{path}:{number}: expected "x1 y1 ... x4 y4 <class> <difficult>"')
            difficult = int(tokens[-1]) if tokens[-1] in {'0', '1'} else 0
            name_tokens = tokens[8:-1] if tokens[-1] in {'0', '1'} else tokens[8:]
            objects.append({'polygon': _numbers(tokens[:8], path, number),
                            'class_name': class_name(' '.join(name_tokens), path), 'difficult': difficult})
        elif label_format == 'yolo-obb':
            if len(tokens) != 9:
                raise ValueError(f'{path}:{number}: expected "<class index> x1 y1 ... x4 y4"')
            index = int(tokens[0])
            if not 0 <= index < len(CLASSES):
                raise ValueError(f'{path}:{number}: class index {index} outside 0-{len(CLASSES) - 1}')
            polygon = _numbers(tokens[1:], path, number) * [width, height]
            objects.append({'polygon': polygon, 'class_name': CLASSES[index], 'difficult': 0})
        else:
            raise ValueError(f'Unknown DOTA label format: {label_format}')
    return objects


def polygon_area(points):
    """Shoelace formula."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def clip_polygon(points, left, top, right, bottom):
    """Sutherland-Hodgman clipping against an axis-aligned rectangle (exact for any polygon)."""
    output = [tuple(point) for point in np.asarray(points, dtype=float)]
    edges = [(lambda p: p[0] >= left, lambda a, b: _cross_x(a, b, left)),
             (lambda p: p[0] <= right, lambda a, b: _cross_x(a, b, right)),
             (lambda p: p[1] >= top, lambda a, b: _cross_y(a, b, top)),
             (lambda p: p[1] <= bottom, lambda a, b: _cross_y(a, b, bottom))]
    for inside, cross in edges:
        points, output = output, []
        for index, current in enumerate(points):
            previous = points[index - 1]
            if inside(current):
                if not inside(previous):
                    output.append(cross(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(cross(previous, current))
        if not output:
            break
    return np.asarray(output, dtype=float).reshape(-1, 2)


def _cross_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return x, a[1] + t * (b[1] - a[1])


def _cross_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return a[0] + t * (b[0] - a[0]), y


def window_starts(size, tile, stride):
    """0, stride, 2*stride, ...; the last window moves inward so every tile is full size."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def tile_objects(objects, x0, y0, tile, min_keep):
    """Objects of one tile in tile coordinates, or [] when none overlaps it."""
    result = []
    for item in objects:
        area = polygon_area(item['polygon'])
        if area <= 0:
            continue
        kept = polygon_area(clip_polygon(item['polygon'], x0, y0, x0 + tile, y0 + tile)) / area
        if kept <= 0:
            continue
        difficult = int(item['difficult'] or kept < min_keep)
        result.append({'polygon': np.asarray(item['polygon'], dtype=float) - [x0, y0],
                       'class_name': item['class_name'], 'difficult': difficult, 'kept': kept})
    return result


def plan_tiles(source_id, width, height, objects, tile=1024, stride=1024, min_keep=.7):
    """Every window of one source image with its objects."""
    plans = []
    for y0 in window_starts(height, tile, stride):
        for x0 in window_starts(width, tile, stride):
            inside = tile_objects(objects, x0, y0, tile, min_keep)
            plans.append({'tile_id': f'{source_id}__{x0}_{y0}', 'source_image': source_id, 'x0': x0, 'y0': y0,
                          'width': min(tile, width - x0), 'height': min(tile, height - y0), 'objects': inside,
                          'n_counted': sum(1 for item in inside if not item['difficult']),
                          'n_ignored': sum(1 for item in inside if item['difficult'])})
    return plans


def select_tiles(plans, min_objects=3, max_tiles=300, seed=0):
    """Candidates have at least min_objects counted objects; a seeded sample of them is kept."""
    candidates = sorted((plan for plan in plans if plan['n_counted'] >= min_objects), key=lambda plan: plan['tile_id'])
    ids = [plan['tile_id'] for plan in candidates]
    chosen = set(random.Random(seed).sample(ids, min(max_tiles, len(ids))))
    return candidates, chosen


def _write_if_changed(path, text):
    if not path.is_file() or path.read_text() != text:
        path.write_text(text)


def prepare_tiles(images_dir, labels_dir, label_format, output, tile=1024, stride=1024, min_keep=.7,
                  min_objects=3, max_tiles=300, seed=0):
    """Cut tiles from every source image, keep a seeded sample, and write images, labels and the index."""
    from PIL import Image
    images_dir, labels_dir, output = Path(images_dir), Path(labels_dir), Path(output)
    sources = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not sources:
        raise FileNotFoundError(f'No DOTA images in {images_dir}')
    plans, sizes = [], {}
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # Aerial scenes exceed Pillow's decompression-bomb guess; these files are pinned.
    try:
        for path in sources:
            label_path = labels_dir / f'{path.stem}.txt'
            if not label_path.is_file():
                raise FileNotFoundError(f'No labels for {path.name}: {label_path}')
            with Image.open(path) as source:
                sizes[path.stem] = source.size
            width, height = sizes[path.stem]
            objects = read_source_labels(label_path, label_format, width, height)
            plans.extend(plan_tiles(path.stem, width, height, objects, tile, stride, min_keep))
        candidates, chosen = select_tiles(plans, min_objects, max_tiles, seed)
        (output / 'images').mkdir(parents=True, exist_ok=True)
        (output / 'labels').mkdir(parents=True, exist_ok=True)
        by_source = {path.stem: path for path in sources}
        selected = [plan for plan in candidates if plan['tile_id'] in chosen]
        for plan in selected:
            _write_if_changed(output / 'labels' / f"{plan['tile_id']}.txt", format_tile_labels(plan['objects']))
            target = output / 'images' / f"{plan['tile_id']}.png"
            if target.is_file():
                continue
            with Image.open(by_source[plan['source_image']]) as source:
                crop = source.convert('RGB').crop((plan['x0'], plan['y0'], plan['x0'] + tile, plan['y0'] + tile))
            # crop() pads beyond the source edge with black; padding has no objects.
            temporary = target.with_name(target.stem + '.part.png')
            crop.save(temporary)
            temporary.replace(target)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    rows = [{**{key: plan[key] for key in INDEX_FIELDS[:-1]}, 'selected': int(plan['tile_id'] in chosen)}
            for plan in candidates]
    lines = [','.join(INDEX_FIELDS)] + [','.join(str(row[key]) for key in INDEX_FIELDS) for row in rows]
    _write_if_changed(output / 'tile_index.csv', '\n'.join(lines) + '\n')
    counts = {}
    for plan in selected:
        for item in plan['objects']:
            if not item['difficult']:
                counts[item['class_name']] = counts.get(item['class_name'], 0) + 1
    return {'source_images': len(sources), 'tiles': len(plans), 'candidates': len(candidates),
            'selected': len(selected), 'source_images_selected': len({plan['source_image'] for plan in selected}),
            'objects_per_class': dict(sorted(counts.items(), key=lambda item: -item[1]))}


def read_tile_index(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def rasterize(polygon, shape):
    """Boolean mask of one polygon, clipped to the image."""
    from PIL import Image, ImageDraw
    canvas = Image.new('1', (shape[1], shape[0]), 0)
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in np.asarray(polygon, dtype=float)], fill=1, outline=1)
    return np.asarray(canvas, dtype=bool)


def tile_instances(path, shape):
    """Every labelled object of one tile with its mask; raises when a box misses the image."""
    path = Path(path)
    instances = read_tile_labels(path)
    height, width = shape[:2]
    for item in instances:
        if polygon_area(clip_polygon(item['polygon'], 0, 0, width, height)) <= 0:
            raise ValueError(f'{path}: a {item["class_name"]} box lies outside the {width}x{height} image; '
                             'labels and images do not match')
        item['mask'] = rasterize(item['polygon'], shape)
        item['area'] = int(item['mask'].sum())
    return instances
