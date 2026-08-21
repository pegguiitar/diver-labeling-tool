import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ...model import BBox
from . import BaseLabelFormat, abs2rel_rotation, rel2abs_rotation


def _euler_zyx_to_quaternion(x_deg: float, y_deg: float, z_deg: float) -> Dict[str, float]:
    """Quaternion (w, x, y, z) for the rotation R = Rz(z) @ Ry(y) @ Rx(x).

    Matches labelCloud's render/export Euler convention (glRotate z, then y,
    then x), so the quaternion is consistent with the stored `rotations` field
    and with the quaternions produced by the upstream conversion pipeline.
    """
    rx, ry, rz = np.deg2rad([x_deg, y_deg, z_deg]) / 2.0
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return {
        "w": cz * cy * cx + sz * sy * sx,
        "x": cz * cy * sx - sz * sy * cx,
        "y": cz * sy * cx + sz * cy * sx,
        "z": sz * cy * cx - cz * sy * sx,
    }


class CentroidFormat(BaseLabelFormat):
    FILE_ENDING = ".json"

    def import_labels(self, pcd_path: Path) -> List[BBox]:
        labels = []

        label_path = self.label_folder.joinpath(pcd_path.stem + self.FILE_ENDING)
        if label_path.is_file():
            with label_path.open("r") as read_file:
                data = json.load(read_file)

            for label in data["objects"]:
                x = label["centroid"]["x"]
                y = label["centroid"]["y"]
                z = label["centroid"]["z"]
                length = label["dimensions"]["length"]
                width = label["dimensions"]["width"]
                height = label["dimensions"]["height"]
                bbox = BBox(x, y, z, length, width, height)
                rotations = label["rotations"].values()
                if self.relative_rotation:
                    rotations = map(rel2abs_rotation, rotations)
                bbox.set_rotations(*rotations)
                bbox.set_classname(label["name"])
                labels.append(bbox)
            logging.info(
                "Imported %s labels from %s." % (len(data["objects"]), label_path)
            )
        return labels

    def _carry_forward_extra_fields(self, pcd_path: Path) -> Dict[str, Any]:
        """Collect extra `_`-prefixed fields from the frame's existing label file.

        The upstream conversion pipeline stamps per-annotator/per-scene flags
        (e.g. `_body_frame`, `_axis_flip_v2`, `_person3_reflip`) on every object.
        labelCloud's BBox model has no slot for them, so without this they would
        be dropped the moment a frame is re-saved. These flags are constant
        across a scene's objects, so we gather them from the existing file and
        re-apply them to every exported object.
        """
        extra: Dict[str, Any] = {}
        existing_path = self.label_folder.joinpath(pcd_path.stem + self.FILE_ENDING)
        if not existing_path.is_file():
            return extra
        try:
            with existing_path.open("r") as read_file:
                prev = json.load(read_file)
        except (json.JSONDecodeError, OSError):
            return extra
        for obj in prev.get("objects", []):
            for key, value in obj.items():
                if key.startswith("_"):
                    extra[key] = value
        return extra

    def export_labels(self, bboxes: List[BBox], pcd_path: Path) -> None:
        data: Dict[str, Any] = {}
        # Header
        data["folder"] = pcd_path.parent.name
        data["filename"] = pcd_path.name
        data["path"] = str(pcd_path)

        # Flags stamped by the upstream pipeline, preserved across re-saves.
        extra_fields = self._carry_forward_extra_fields(pcd_path)

        # Labels
        data["objects"] = []
        for bbox in bboxes:
            label: Dict[str, Any] = {}
            label["name"] = bbox.get_classname()
            label["centroid"] = {
                str(axis): self.round_dec(val)
                for axis, val in zip(["x", "y", "z"], bbox.get_center())
            }
            label["dimensions"] = {
                str(dim): self.round_dec(val)
                for dim, val in zip(
                    ["length", "width", "height"], bbox.get_dimensions()
                )
            }

            abs_rotations = bbox.get_rotations()  # absolute degrees (x, y, z)
            conv_rotations = abs_rotations
            if self.relative_rotation:
                conv_rotations = map(abs2rel_rotation, conv_rotations)  # type: ignore
            label["rotations"] = {
                str(axis): self.round_dec(angle)
                for axis, angle in zip(["x", "y", "z"], conv_rotations)
            }

            # Recompute the quaternion from the (absolute) Euler angles so it
            # survives edits instead of being dropped on save.
            quaternion = _euler_zyx_to_quaternion(*abs_rotations)
            label["quaternion"] = {
                axis: self.round_dec(value) for axis, value in quaternion.items()
            }

            # Re-attach preserved upstream flags.
            label.update(extra_fields)

            data["objects"].append(label)

        # Save to JSON
        label_path = self.save_label_to_file(pcd_path, data)
        logging.info(
            f"Exported {len(bboxes)} labels to {label_path} "
            f"in {self.__class__.__name__} formatting!"
        )
