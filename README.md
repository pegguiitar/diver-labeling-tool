# Diver Labeling Tool

3D sonar + 2D camera annotation tool for diver scenes, built on top of [labelCloud](https://github.com/ch-sa/labelCloud).

Scenes are organized per annotator under `converted_labels/<Person>/<scene_id>/`, each
with the same layout as a source scene:

```
converted_labels/
  Person1/
    scene_0000/
      sonar/     frame_XXXXXX.bin   (3D point clouds)
      camera/    frame_XXXXXX.jpg   (2D images)
      labels/    frame_XXXXXX.json  (centroid_abs labelCloud labels)
      pairs.jsonl
    annotations/                    (aggregated per-scene JSON, written by the tool)
```

## Setup

### 1. Create and activate the conda environment

```bash
conda create -n labelcloud python=3.10 -y
conda activate labelcloud
pip install labelCloud
pip install "setuptools<70"   # required for pkg_resources compatibility
```

### 2. Apply patches

```bash
python patches/apply_patches.py
```

This patches three labelCloud files to add:
- Y-axis flip correction for sonar coordinate convention
- Integrated 2D camera annotation panel
- Sonar link ID support for cross-referencing sonar and camera labels

### 3. Configure paths

Edit `label_scene.py` (and `auto_track.py`) and update `LABELCLOUD_PYTHON` to point to
your conda env:

```python
LABELCLOUD_PYTHON = Path(r"path/to/envs/labelcloud/python.exe")   # Windows
LABELCLOUD_PYTHON = Path("/path/to/envs/labelcloud/bin/python")    # Linux/Mac
```

Both `label_scene.py` and `auto_track.py` resolve scenes under `converted_labels/`,
which is expected to sit next to this tool directory (`CONVERTED_DIR = DATA_DIR /
"converted_labels"`). Adjust that constant if your data lives elsewhere.

## Usage

Label a scene (opens labelCloud, then aggregates labels to annotation JSON). The first
argument is the person folder, the second is the scene:

```bash
python label_scene.py Person1 0            # Person1/scene_0000
python label_scene.py Person3 scene_0019   # explicit scene id
```

Convert existing labels without reopening the tool:

```bash
python label_scene.py Person4 65 --convert-only
```

### Semi-automated propagation (optional)

`auto_track.py` propagates a hand-labeled frame forward/backward within a single scene:

```bash
python3 auto_track.py Person1 49              # both modalities
python3 auto_track.py Person1 49 --dry-run
python3 auto_track.py Person1 49 --modality sonar
```

## Annotation format

Labels are read from and written back into each scene's `labels/` folder in labelCloud's
**centroid_abs** format (set in `labels/_classes.json`). The per-frame files are then
aggregated into `converted_labels/<Person>/annotations/<scene_id>.json`.

Each aggregated sonar object keeps **every raw label field** — full `rotations` (x/y/z),
`quaternion`, and any per-annotator flags (e.g. `_person3_reflip`, `_axis_flip_v2`) — with
the aggregation helpers `class` and `link_id` added on top:

```json
{
  "person": "Person3",
  "scene_id": "scene_0019",
  "frames": {
    "351": {
      "objects": [
        {
          "name": "Diver",
          "centroid": {"x": 1.44, "y": -0.94, "z": 0.15},
          "dimensions": {"length": 0.34, "width": 0.46, "height": 0.95},
          "rotations": {"x": 0.0, "y": 360.0, "z": 360.0},
          "quaternion": {"w": 1.0, "x": 0.0, "y": -1.22e-16, "z": -6.12e-17},
          "_body_frame": true,
          "_axis_flip_v2": true,
          "_person3_reflip": true,
          "class": "Diver-sonar",
          "link_id": 0
        },
        {
          "class": "Diver-camera",
          "link_id": 1,
          "bbox_2d": {"x1": 0.8, "y1": 0.14, "x2": 1.0, "y2": 0.85}
        }
      ]
    }
  }
}
```

Sonar and camera labels with the same `link_id` correspond to the same physical object.

## Object classes

Defined in `labels/_classes.json`. Default classes: Unassigned, Fish, Coral, Rock, Diver, Structure.
