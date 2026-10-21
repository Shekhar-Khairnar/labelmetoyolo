"""Convert LabelMe JSON annotations into a YOLO dataset.

The output follows the layout Ultralytics expects (YOLOv5/v8/11/26)::

    output/
        images/{train,val,test}/<original image name>
        labels/{train,val,test}/<original image stem>.txt
        data.yaml

When no split is requested, images and labels go straight into
``images/`` and ``labels/`` without the subfolders.

File names are never changed: ``frame_0012.jpg`` stays ``frame_0012.jpg``
and its label file becomes ``frame_0012.txt``.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TASKS = ("detect", "segment", "obb")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Number of vertices used when a LabelMe circle is turned into a polygon.
CIRCLE_VERTICES = 32

Point = Tuple[float, float]


def convert(
    input_dir: str | Path,
    output_dir: str | Path = "yolo_dataset",
    task: str = "detect",
    classes: Optional[Sequence[str]] = None,
    split: Optional[Sequence[float]] = None,
    seed: int = 42,
) -> Path:
    """Convert a folder of LabelMe JSON files into a YOLO dataset.

    Args:
        input_dir: Folder containing the LabelMe ``.json`` files. Images are
            looked up through each file's ``imagePath``, then next to the
            JSON, and finally decoded from the embedded ``imageData``.
        output_dir: Where the YOLO dataset is written.
        task: ``"detect"`` for bounding boxes, ``"segment"`` for polygons or
            ``"obb"`` for oriented bounding boxes.
        classes: Class names in the order they should get their ids. Shapes
            whose label is not in this list are skipped. When omitted, every
            label found is used, sorted alphabetically.
        split: Optional ratios as ``(train, val)`` or ``(train, val, test)``.
            They do not need to add up to 1, so ``(80, 10, 10)`` works too.
            When omitted, everything is written without a split.
        seed: Random seed for the shuffle, so splits are reproducible.

    Returns:
        Path to the generated ``data.yaml``.
    """
    task = task.lower()
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    classes = list(classes) if classes else collect_classes(json_files)
    class_ids = {name: idx for idx, name in enumerate(classes)}

    groups = split_files(json_files, split, seed)
    unknown_labels: set = set()

    for split_name, files in groups.items():
        images_dir = output_dir / "images" / split_name
        labels_dir = output_dir / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for json_path in files:
            if _convert_file(json_path, images_dir, labels_dir, class_ids, task, unknown_labels):
                written += 1
        logger.info("%s: %d images", split_name or "all", written)

    if unknown_labels:
        logger.warning("Skipped labels not in class list: %s", ", ".join(sorted(unknown_labels)))

    yaml_path = _write_data_yaml(output_dir, list(groups), classes)
    logger.info("Dataset config saved to %s", yaml_path)
    return yaml_path


def collect_classes(json_files: Iterable[Path]) -> List[str]:
    """Return every label used across the given LabelMe files, sorted."""
    labels = set()
    for path in json_files:
        for shape in _read_json(path).get("shapes", []):
            if shape.get("label"):
                labels.add(shape["label"])
    return sorted(labels)


def split_files(
    files: Sequence[Path], ratios: Optional[Sequence[float]], seed: int = 42
) -> Dict[str, List[Path]]:
    """Shuffle and divide files into train/val(/test) groups.

    Without ratios a single group with an empty name is returned, which maps
    to writing directly into ``images/`` and ``labels/``.
    """
    if not ratios:
        return {"": list(files)}
    if len(ratios) not in (2, 3) or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError("split needs 2 or 3 non-negative ratios, e.g. 0.8 0.2 or 0.7 0.2 0.1")

    total = sum(ratios)
    files = list(files)
    random.Random(seed).shuffle(files)

    n_train = round(len(files) * ratios[0] / total)
    if len(ratios) == 2:
        return {"train": files[:n_train], "val": files[n_train:]}

    n_val = round(len(files) * ratios[1] / total)
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def shape_to_yolo(shape: dict, task: str, width: float, height: float) -> Optional[List[float]]:
    """Turn one LabelMe shape into normalized YOLO coordinates.

    Returns ``None`` for shapes that have no area (points, lines) or that
    end up empty after clipping to the image.
    """
    points = _shape_polygon(shape)
    if points is None:
        return None

    if task == "detect":
        xs = [min(max(x, 0.0), width) for x, _ in points]
        ys = [min(max(y, 0.0), height) for _, y in points]
        box_w, box_h = max(xs) - min(xs), max(ys) - min(ys)
        if box_w <= 0 or box_h <= 0:
            return None
        cx, cy = min(xs) + box_w / 2, min(ys) + box_h / 2
        return [cx / width, cy / height, box_w / width, box_h / height]

    if task == "obb":
        points = _min_area_rect(points)

    return _normalize(points, width, height)


def _convert_file(
    json_path: Path,
    images_dir: Path,
    labels_dir: Path,
    class_ids: Dict[str, int],
    task: str,
    unknown_labels: set,
) -> bool:
    """Copy one image and write its label file. Returns False if skipped."""
    data = _read_json(json_path)
    image_name = _place_image(json_path, data, images_dir)
    if image_name is None:
        logger.warning("No image found for %s, skipping", json_path.name)
        return False

    width, height = float(data["imageWidth"]), float(data["imageHeight"])
    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in class_ids:
            if label:
                unknown_labels.add(label)
            continue
        coords = shape_to_yolo(shape, task, width, height)
        if coords:
            lines.append(f"{class_ids[label]} " + " ".join(f"{v:.6f}" for v in coords))

    # An image without shapes still gets an empty label file, YOLO treats
    # it as a background image.
    label_path = labels_dir / f"{Path(image_name).stem}.txt"
    label_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return True


def _place_image(json_path: Path, data: dict, dest_dir: Path) -> Optional[str]:
    """Copy (or decode) the image for a LabelMe file into ``dest_dir``.

    Returns the image file name, which is the same as the original.
    """
    # LabelMe stores imagePath relative to the JSON, often with Windows
    # separators, so normalize them before building the path.
    image_path = (data.get("imagePath") or "").replace("\\", "/")

    candidates = [json_path.parent / image_path] if image_path else []
    candidates += [json_path.with_suffix(ext) for ext in IMAGE_EXTS]
    candidates += [json_path.with_suffix(ext.upper()) for ext in IMAGE_EXTS]

    for candidate in candidates:
        if candidate.is_file():
            shutil.copy2(candidate, dest_dir / candidate.name)
            return candidate.name

    # Fall back to the base64 image LabelMe can embed in the JSON itself.
    if data.get("imageData"):
        raw = base64.b64decode(data["imageData"])
        name = Path(image_path).name or json_path.stem + (".png" if raw[:4] == b"\x89PNG" else ".jpg")
        (dest_dir / name).write_bytes(raw)
        return name

    return None


def _shape_polygon(shape: dict) -> Optional[List[Point]]:
    """Return the outline of a shape as a list of (x, y) pixel points."""
    shape_type = shape.get("shape_type") or "polygon"
    points = [(float(x), float(y)) for x, y in shape.get("points", [])]

    if shape_type == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if shape_type == "circle" and len(points) == 2:
        # LabelMe stores a circle as its center plus one point on the edge.
        (cx, cy), (ex, ey) = points
        r = math.hypot(ex - cx, ey - cy)
        step = 2 * math.pi / CIRCLE_VERTICES
        return [(cx + r * math.cos(i * step), cy + r * math.sin(i * step)) for i in range(CIRCLE_VERTICES)]

    if shape_type == "polygon" and len(points) >= 3:
        return points

    return None


def _min_area_rect(points: List[Point]) -> List[Point]:
    """Smallest rotated rectangle around the points, as 4 corners.

    Uses the fact that the optimal rectangle shares a side with the
    convex hull, so only the hull edge angles need to be checked.
    """
    hull = _convex_hull(points)
    best = None
    for i in range(len(hull)):
        (x1, y1), (x2, y2) = hull[i], hull[(i + 1) % len(hull)]
        angle = math.atan2(y2 - y1, x2 - x1)
        c, s = math.cos(angle), math.sin(angle)
        # Rotate the hull so this edge lies flat, then take the bounding box.
        us = [x * c + y * s for x, y in hull]
        vs = [-x * s + y * c for x, y in hull]
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if best is None or area < best[0]:
            best = (area, c, s, min(us), max(us), min(vs), max(vs))

    _, c, s, u0, u1, v0, v1 = best
    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    return [(u * c - v * s, u * s + v * c) for u, v in corners]


def _convex_hull(points: List[Point]) -> List[Point]:
    """Convex hull using Andrew's monotone chain algorithm."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Point] = []
    upper: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _normalize(points: List[Point], width: float, height: float) -> List[float]:
    """Flatten points to [x1, y1, x2, y2, ...] scaled to 0-1 and clipped."""
    coords = []
    for x, y in points:
        coords.append(min(max(x / width, 0.0), 1.0))
        coords.append(min(max(y / height, 0.0), 1.0))
    return coords


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_data_yaml(output_dir: Path, splits: List[str], classes: List[str]) -> Path:
    """Write the dataset config that Ultralytics reads for training."""
    lines = [f"path: {json.dumps(output_dir.resolve().as_posix())}"]
    if splits == [""]:
        # No split: train and validate on the same folder.
        lines += ["train: images", "val: images"]
    else:
        lines += [f"{name}: images/{name}" for name in splits]

    lines.append(f"nc: {len(classes)}")
    lines.append("names:")
    # JSON strings are valid YAML, which keeps names with spaces or
    # special characters safe without needing PyYAML.
    lines += [f"  {idx}: {json.dumps(name, ensure_ascii=False)}" for idx, name in enumerate(classes)]

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path
