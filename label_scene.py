"""
label_scene.py - Launch labelCloud for a converted_labels scene and sync labels
back to per-person annotations.

Data lives under converted_labels/<Person>/<scene_id>/ with the same layout as a
uScenes scene: sonar/ (point clouds), camera/ (images), labels/ (centroid_abs
labelCloud labels) and pairs.jsonl. Labels are read from and written back into
that labels/ folder in centroid_abs format; the per-frame files are then
aggregated into converted_labels/<Person>/annotations/<scene_id>.json.

Usage:
    python label_scene.py Person1 0            # label Person1/scene_0000
    python label_scene.py Person3 scene_0019   # same, explicit scene id
    python label_scene.py Person4 65 --convert-only   # convert only, no GUI
"""

import argparse
import configparser
import json
import subprocess
import sys
from pathlib import Path

TOOL_DIR  = Path(__file__).parent          # labeling_tool/
DATA_DIR  = TOOL_DIR.parent               # project root containing converted_labels/
CONVERTED_DIR    = DATA_DIR / "converted_labels"
CONFIG_PATH      = TOOL_DIR / "config.ini"
LABELCLOUD_PYTHON = Path("/home/eugene/anaconda3/envs/labelcloud/bin/python")


def parse_scene(arg: str) -> str:
    if arg.startswith("scene_"):
        return arg
    return f"scene_{int(arg):04d}"


def resolve_scene_dir(person: str, scene_id: str) -> Path:
    """Locate converted_labels/<person>/<scene_id>, erroring helpfully if absent."""
    person_dir = CONVERTED_DIR / person
    if not person_dir.is_dir():
        available = sorted(p.name for p in CONVERTED_DIR.iterdir() if p.is_dir())
        sys.exit(f"Person '{person}' not found under {CONVERTED_DIR}.\n"
                 f"  Available: {', '.join(available)}")
    scene_dir = person_dir / scene_id
    if not scene_dir.is_dir():
        available = sorted(p.name for p in person_dir.iterdir() if p.is_dir())
        sys.exit(f"Scene '{scene_id}' not found under {person_dir}.\n"
                 f"  Available: {', '.join(available)}")
    return scene_dir


def update_config(scene_dir: Path) -> None:
    sonar_dir = scene_dir / "sonar"
    labels_dir = scene_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    config["FILE"]["pointcloud_folder"] = str(sonar_dir)
    config["FILE"]["label_folder"] = str(labels_dir)
    config["FILE"]["class_definitions"] = str(TOOL_DIR / "labels" / "_classes.json")
    config["FILE"]["image_folder"] = str(scene_dir / "camera")
    with open(CONFIG_PATH, "w") as f:
        config.write(f)

    print(f"  point clouds : {sonar_dir}")
    print(f"  labels       : {labels_dir}")


def convert_labels(person: str, scene_id: str, scene_dir: Path) -> int:
    labels_dir = scene_dir / "labels"
    annotations_dir = CONVERTED_DIR / person / "annotations"
    annotation_path = annotations_dir / f"{scene_id}.json"

    if annotation_path.exists():
        with open(annotation_path) as f:
            annotation = json.load(f)
    else:
        annotation = {"person": person, "scene_id": scene_id, "frames": {}}

    # Collect all sonar label files (exclude _camera.json and _ids.json)
    sonar_files = sorted(
        f for f in labels_dir.glob("frame_*.json")
        if not f.name.endswith("_camera.json") and not f.name.endswith("_ids.json")
    )
    if not sonar_files:
        print("No label files found — nothing to convert.")
        return 0

    for sonar_file in sonar_files:
        stem      = sonar_file.stem                        # e.g. frame_000042
        frame_idx = str(int(stem.split("_")[1]))           # e.g. "42"

        with open(sonar_file) as f:
            lc = json.load(f)

        # Load sonar link IDs if present
        ids_file = labels_dir / (stem + "_ids.json")
        sonar_ids: dict = {}
        if ids_file.exists():
            with open(ids_file) as f:
                sonar_ids = {int(k): v for k, v in json.load(f).items()}

        # Preserve every raw label field (full x/y/z rotations, quaternion and
        # per-person flags such as _person3_reflip); only add the aggregation
        # helpers `class` and `link_id` on top.
        sonar_objects = [
            {
                **obj,
                "class": obj["name"] + "-sonar",
                "link_id": sonar_ids.get(i, 0),
            }
            for i, obj in enumerate(lc.get("objects", []))
        ]

        # Load camera labels if present
        cam_file = labels_dir / (stem + "_camera.json")
        camera_objects = []
        if cam_file.exists():
            with open(cam_file) as f:
                cam_data = json.load(f)
            camera_objects = [
                {
                    "class":   item["class"] + "-camera",
                    "link_id": item.get("link_id", 0),
                    "bbox_2d": {
                        "x1": item["x1"], "y1": item["y1"],
                        "x2": item["x2"], "y2": item["y2"],
                    },
                }
                for item in cam_data.get("labels", [])
            ]

        if frame_idx not in annotation["frames"]:
            annotation["frames"][frame_idx] = {}
        annotation["frames"][frame_idx]["objects"] = sonar_objects + camera_objects

    annotations_dir.mkdir(exist_ok=True)
    with open(annotation_path, "w") as f:
        json.dump(annotation, f, indent=2)

    n = len(sonar_files)
    print(f"Wrote {n} frame(s) -> {annotation_path}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("person", help="Person folder (e.g. Person1)")
    parser.add_argument("scene", help="Scene number or ID (e.g. 0 or scene_0042)")
    parser.add_argument("--convert-only", action="store_true",
                        help="Convert existing labels without launching labelCloud")
    args = parser.parse_args()

    scene_id = parse_scene(args.scene)
    scene_dir = resolve_scene_dir(args.person, scene_id)
    print(f"\nPerson: {args.person}    Scene: {scene_id}")

    if not args.convert_only:
        print("Configuring labelCloud...")
        update_config(scene_dir)
        print("Launching labelCloud  (close the window when done labeling)\n")
        subprocess.run([str(LABELCLOUD_PYTHON), "-m", "labelCloud"], cwd=str(TOOL_DIR))
        print()

    print("Converting labels -> annotations...")
    convert_labels(args.person, scene_id, scene_dir)


if __name__ == "__main__":
    main()
