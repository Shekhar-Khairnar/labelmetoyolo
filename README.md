# labelmetoyolo

Convert [LabelMe](https://github.com/wkentaro/labelme) JSON annotations to YOLO format for Ultralytics (YOLOv5, v8, 11, 26).

Most converters out there rename your images to random IDs. This one doesn't. `IMG_0042.jpg` stays `IMG_0042.jpg` and its label is `IMG_0042.txt`, so you can always trace a label back to the original file.

- Keeps original image file names and class names
- Detection, segmentation and oriented bounding boxes (OBB)
- Optional train/val or train/val/test split with any ratio
- Generates a ready to use `data.yaml`
- No dependencies, just the Python standard library

## Install

```bash
pip install git+https://github.com/Shekhar-Khairnar/labelmetoyolo.git
```

Or from a local clone:

```bash
git clone https://github.com/Shekhar-Khairnar/labelmetoyolo.git
cd labelmetoyolo
pip install .
```

## Usage

Point it at the folder with your LabelMe `.json` files:

```bash
labelmetoyolo path/to/labelme
```

That's it. You get a `yolo_dataset/` folder with images, labels and `data.yaml`.

### Options

| Option            | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| `-o, --output`  | Output folder (default: `yolo_dataset`)                          |
| `-t, --task`    | `detect`, `segment` or `obb` (default: `detect`)           |
| `-s, --split`   | Split ratios, two values for train/val or three for train/val/test |
| `-c, --classes` | Class names in id order, or a `.txt` file with one name per line |
| `--seed`        | Random seed for the split (default: `42`)                        |

### Examples

```bash
# Bounding boxes, no split
labelmetoyolo data/ -o dataset

# Segmentation, 80% train / 20% val
labelmetoyolo data/ -t segment --split 0.8 0.2

# Train/val/test split, percentages work too
labelmetoyolo data/ --split 70 20 10

# Fix the class order (labels not listed here are ignored)
labelmetoyolo data/ --classes person car bike
labelmetoyolo data/ --classes classes.txt
```

If you don't pass `--classes`, all labels found in the JSON files are used and sorted alphabetically.

### Python

```python
from labelmetoyolo import convert

convert("data/", "dataset", task="segment", split=(0.8, 0.1, 0.1))
```

## Output

With a split:

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Without a split, images and labels go directly into `images/` and `labels/`.

Then train as usual:

```bash
yolo detect train data=dataset/data.yaml model=yolo11n.pt
```

## Notes

- Supported LabelMe shapes: `rectangle`, `polygon` and `circle`. Points and lines have no area and are skipped.
- Images are found through `imagePath` in the JSON, then by matching file name next to the JSON. If neither exists, the image embedded in `imageData` is used.
- Images with no shapes get an empty label file so YOLO uses them as background images.
- For OBB, polygons are converted to their minimum area rotated rectangle.

## License

MIT
