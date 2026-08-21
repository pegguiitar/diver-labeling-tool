"""stamp_core.py - pure "stamp sonar box from camera" helper.

Only numpy beyond stdlib (json, pathlib) - no scipy/cv2 - so it can be
imported both from auto_track.py (run with a python that has cv2/scipy for
the other modes) and from inside labelCloud itself, which runs on the
labelcloud conda env that has numpy but not cv2/scipy. Used by
patches/gui.py's "Stamp Sonar from Camera" / "Re-stamp" buttons.

Algorithm: copy the seed frame's sonar box - centroid, dimensions, rotation,
all unchanged - into every following frame that already has a human-drawn
camera box for the same link_id, for as long as that camera box still
overlaps the seed's camera box well enough (IoU >= min_iou), AND the sonar
point cloud near that (frozen) position isn't being cut off by the sensor's
field of view. Never overwrites a frame that already has sonar content for
that link_id (unless overwrite=True).
"""

import json
import math
from pathlib import Path

import numpy as np

# Sonar field of view, measured empirically from the point clouds themselves
# (arctan2 of the raw x/y/z) across six scenes - every scene clips azimuth at
# exactly +/-45.0 deg and elevation at +/-27.24 deg, so these are the
# sensor's real aperture limits, not scene-specific.
SONAR_AZIMUTH_LIMIT_DEG = 45.0
SONAR_ELEVATION_LIMIT_DEG = 27.24


def load_points_xyz(bin_path: Path) -> np.ndarray:
    raw = np.fromfile(bin_path, dtype=np.float32)
    pts = raw.reshape((-1, 4 if raw.size % 4 == 0 else 3))[:, 0:3]
    return pts[~np.isnan(pts).any(axis=1)]


def fov_clip_reason(points: np.ndarray, centroid, gate_radius: float = 0.5,
                     edge_margin_deg: float = 1.5, min_fraction: float = 0.08):
    """Whether the point cloud right around `centroid` in *this* frame shows
    a hard cutoff at the sonar's FOV boundary - i.e. whether copying the
    seed's (frozen) position here would put part of the diver's body where
    the sensor physically can't see it, rather than the diver's body
    actually ending there. Returns a reason string, or None if not clipped."""
    if len(points) == 0:
        return None
    c = np.asarray(centroid, dtype=np.float64)
    d = np.linalg.norm(points - c, axis=1)
    nearby = points[d <= gate_radius]
    if len(nearby) < 10:
        return None
    az = np.degrees(np.arctan2(nearby[:, 1], nearby[:, 0]))
    el = np.degrees(np.arctan2(nearby[:, 2], nearby[:, 0]))
    el_edge_frac = float(np.mean(np.abs(np.abs(el) - SONAR_ELEVATION_LIMIT_DEG) <= edge_margin_deg))
    az_edge_frac = float(np.mean(np.abs(np.abs(az) - SONAR_AZIMUTH_LIMIT_DEG) <= edge_margin_deg))
    if el_edge_frac >= min_fraction:
        return (f"sonar cut off at the vertical/elevation FOV edge "
                f"({el_edge_frac:.0%} of nearby points sitting right at the "
                f"{SONAR_ELEVATION_LIMIT_DEG}° limit) - not copying this frame")
    if az_edge_frac >= min_fraction:
        return (f"sonar cut off at the horizontal/azimuth FOV edge "
                f"({az_edge_frac:.0%} of nearby points sitting right at the "
                f"{SONAR_AZIMUTH_LIMIT_DEG}° limit) - not copying this frame")
    return None


def stem(idx: int, width: int) -> str:
    return f"frame_{idx:0{width}d}"


def frame_width(sonar_dir: Path) -> int:
    sample = next(iter(sorted(sonar_dir.glob("frame_*.bin"))), None)
    if sample is None:
        raise ValueError(f"No point cloud files found in {sonar_dir}")
    return len(sample.stem.split("_")[1])


def frame_indices_of(sonar_dir: Path) -> list:
    return sorted(int(p.stem.split("_")[1]) for p in sonar_dir.glob("frame_*.bin"))


def load_sonar_file(labels_dir: Path, idx: int, width: int, sonar_dir: Path) -> dict:
    path = labels_dir / f"{stem(idx, width)}.json"
    if path.exists():
        return json.loads(path.read_text())
    s = stem(idx, width)
    return {"folder": "sonar", "filename": f"{s}.bin",
            "path": str(sonar_dir / f"{s}.bin"), "objects": []}


def load_sonar_ids(labels_dir: Path, idx: int, width: int) -> dict:
    path = labels_dir / f"{stem(idx, width)}_ids.json"
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def load_camera_file(labels_dir: Path, idx: int, width: int) -> dict:
    path = labels_dir / f"{stem(idx, width)}_camera.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"frame": stem(idx, width), "labels": []}


def save_sonar_file(labels_dir: Path, idx: int, width: int, sonar_dir: Path,
                     objects: list, link_ids: dict) -> None:
    s = stem(idx, width)
    data = {"folder": "sonar", "filename": f"{s}.bin",
            "path": str(sonar_dir / f"{s}.bin"), "objects": objects}
    (labels_dir / f"{s}.json").write_text(json.dumps(data, indent="\t"))
    if link_ids:
        (labels_dir / f"{s}_ids.json").write_text(json.dumps(link_ids))


def sonar_link_ids_at(labels_dir: Path, idx: int, width: int) -> set:
    """link_ids actually backed by an object at this frame - the _ids.json
    side-file can go stale relative to the main frame file (observed in
    scene_0068: labelCloud can leave it behind when a box is deleted)."""
    n_objects = len(load_sonar_file(labels_dir, idx, width, labels_dir).get("objects", []))
    ids = load_sonar_ids(labels_dir, idx, width)
    return {v for k, v in ids.items() if k < n_objects}


def resolve_seed_link_ids_sonar(labels_dir: Path, idx: int, width: int, objects: list) -> list:
    ids = load_sonar_ids(labels_dir, idx, width)
    link_ids = [ids.get(i, 0) for i in range(len(objects))]
    missing = [i for i, v in enumerate(link_ids) if not v]
    if not missing:
        return link_ids
    cam = load_camera_file(labels_dir, idx, width)
    available = [l.get("link_id") for l in cam.get("labels", []) if l.get("link_id")]
    if len(missing) == 1 and len(available) == 1:
        link_ids[missing[0]] = available[0]
    return link_ids


def bbox_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def camera_bbox_at(labels_dir: Path, idx: int, width: int, link_id: int):
    """Returns (x1, y1, x2, y2) with x1<=x2 and y1<=y2, normalized regardless
    of which corner the box was dragged from - camera_panel.py stores raw
    drag start/end, so ~13% of boxes have x1>x2 or y1>y2, which silently
    zeroes their area in bbox_iou if left unnormalized."""
    for l in load_camera_file(labels_dir, idx, width).get("labels", []):
        if l.get("link_id") == link_id:
            x1, y1, x2, y2 = l["x1"], l["y1"], l["x2"], l["y2"]
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return None


def run_stamp(sonar_dir: Path, labels_dir: Path, frame_indices: list, width: int,
              seed_idx: int, min_iou: float = 0.4, max_frames=None,
              skip_frames=None, dry_run: bool = False, overwrite: bool = False):
    """Returns (targets, error). targets is a list of per-link_id dicts with
    n_written/stop_frame/stop_reason, or None with an error string.

    overwrite=True replaces existing sonar content for this link_id going
    forward instead of stopping at the first already-labeled frame - for
    re-seeding partway through a stretch that was already stamped once but
    has drifted. It only ever touches this link_id's own entry in each
    frame (other objects in the same frame are left alone), and still only
    goes as far as the camera box keeps agreeing with the new seed."""
    skip_frames = skip_frames or set()
    seed_objs = load_sonar_file(labels_dir, seed_idx, width, sonar_dir).get("objects", [])
    if not seed_objs:
        return None, f"Frame {seed_idx} has no sonar object - draw one first."
    link_ids = resolve_seed_link_ids_sonar(labels_dir, seed_idx, width, seed_objs)

    targets = []
    for obj, link_id in zip(seed_objs, link_ids):
        if not link_id:
            continue
        seed_bbox = camera_bbox_at(labels_dir, seed_idx, width, link_id)
        if seed_bbox is None:
            continue
        targets.append({
            "link_id": link_id, "name": obj["name"], "dimensions": obj["dimensions"],
            "rotations": obj["rotations"], "centroid": obj["centroid"],
            "seed_bbox": seed_bbox, "active": True, "n_written": 0,
            "stop_frame": None, "stop_reason": None,
        })
    if not targets:
        return None, (f"Frame {seed_idx}'s sonar box(es) have no matching camera box "
                       f"(same link_id) to compare against.")

    forward = [i for i in frame_indices if i > seed_idx and i not in skip_frames]
    for idx in forward:
        if not any(t["active"] for t in targets):
            break
        if max_frames is not None and max(t["n_written"] for t in targets) >= max_frames:
            break
        existing_sonar_ids = sonar_link_ids_at(labels_dir, idx, width)
        results = []
        points = None
        for t in targets:
            if not t["active"]:
                continue
            if not overwrite and t["link_id"] in existing_sonar_ids:
                t["active"] = False
                t["stop_frame"], t["stop_reason"] = idx, "reached a previously labeled frame"
                continue
            cur_bbox = camera_bbox_at(labels_dir, idx, width, t["link_id"])
            if cur_bbox is None:
                continue

            if points is None:
                bin_path = sonar_dir / f"{stem(idx, width)}.bin"
                points = load_points_xyz(bin_path) if bin_path.exists() else np.zeros((0, 3))
            c = t["centroid"]
            clip_reason = fov_clip_reason(points, (c["x"], c["y"], c["z"]))
            if clip_reason:
                t["active"] = False
                t["stop_frame"] = idx
                t["stop_reason"] = clip_reason
                continue

            iou = bbox_iou(t["seed_bbox"], cur_bbox)
            results.append(t)
            if iou < min_iou:
                # still write this last frame - it's a rough starting point
                # to fix by hand, faster than drawing one from scratch - but
                # do not go any further than this.
                t["active"] = False
                t["stop_frame"] = idx
                t["stop_reason"] = (f"camera box drifted too far (IoU={iou:.2f} < {min_iou}) "
                                     f"- wrote it anyway as a starting point to fix")
        if results:
            existing = load_sonar_file(labels_dir, idx, width, sonar_dir)
            obj_list = list(existing.get("objects", []))
            id_map = {k: v for k, v in load_sonar_ids(labels_dir, idx, width).items()
                      if k < len(obj_list)}
            # drop any existing entry for the link_ids we're about to write,
            # so overwrite replaces in place instead of duplicating
            new_link_ids = {t["link_id"] for t in results}
            keep = [i for i in range(len(obj_list)) if id_map.get(i) not in new_link_ids]
            obj_list = [obj_list[i] for i in keep]
            id_map = {new_i: id_map[old_i] for new_i, old_i in enumerate(keep)}
            next_i = len(obj_list)
            for t in results:
                obj_list.append({"name": t["name"], "centroid": t["centroid"],
                                  "dimensions": t["dimensions"], "rotations": t["rotations"]})
                id_map[next_i] = t["link_id"]
                next_i += 1
                t["n_written"] += 1
            if not dry_run:
                save_sonar_file(labels_dir, idx, width, sonar_dir, obj_list,
                                 {str(k): v for k, v in id_map.items()})
    return targets, None


# ------------------------------------------------------------- interpolate --
#
# Fill the frames *between* two already-labeled ("anchor") sonar frames by
# interpolating position/rotation/dimensions between them - weighted by how
# far the camera box actually moved over that span rather than by frame
# count, then lightly snapped to nearby points. Always overwrites whatever
# is currently at the in-between frames, since the point is to redo a
# stretch that turned out wrong (e.g. copy-drift from --stamp). Each frame
# is anchored independently to its own interpolated prior rather than to
# the previous frame's own output, so a bad frame can't propagate error
# into the next one the way frame-to-frame tracking used to.

def get_sonar_object(labels_dir: Path, idx: int, width: int, sonar_dir: Path, link_id: int):
    data = load_sonar_file(labels_dir, idx, width, sonar_dir)
    objs = data.get("objects", [])
    ids = load_sonar_ids(labels_dir, idx, width)
    for i, obj in enumerate(objs):
        if ids.get(i) == link_id:
            return obj
    return None


def angle_lerp_deg(a_deg: float, b_deg: float, t: float) -> float:
    """Interpolate an angle along the shorter way around, e.g. 350 -> 10
    passes through 0/360, not backwards through 180."""
    diff = ((b_deg - a_deg + 180) % 360) - 180
    return (a_deg + diff * t) % 360


def local_snap(points: np.ndarray, prior_centroid, gate_radius: float = 0.2,
               min_points: int = 15):
    """Mean position of points within a tight radius of `prior_centroid` -
    deliberately not full clustering: the gate is small and anchored to an
    independent per-frame prior (the interpolated position), not the
    previous frame's own possibly-wrong output, which is what kept the old
    frame-to-frame tracker compounding errors. Returns (position, n_nearby);
    position is None if too few points were found to trust the snap."""
    if len(points) == 0:
        return None, 0
    c = np.asarray(prior_centroid, dtype=np.float64)
    d = np.linalg.norm(points - c, axis=1)
    nearby = points[d <= gate_radius]
    if len(nearby) < min_points:
        return None, len(nearby)
    return nearby.mean(axis=0), len(nearby)


def write_sonar_objects(labels_dir: Path, idx: int, width: int, sonar_dir: Path,
                         writes: list, dry_run: bool = False) -> None:
    """writes: list of (link_id, object_dict). Replaces any existing entry
    for these link_ids in this frame in place; leaves other objects alone."""
    existing = load_sonar_file(labels_dir, idx, width, sonar_dir)
    obj_list = list(existing.get("objects", []))
    id_map = {k: v for k, v in load_sonar_ids(labels_dir, idx, width).items()
              if k < len(obj_list)}
    new_link_ids = {link_id for link_id, _ in writes}
    keep = [i for i in range(len(obj_list)) if id_map.get(i) not in new_link_ids]
    obj_list = [obj_list[i] for i in keep]
    id_map = {new_i: id_map[old_i] for new_i, old_i in enumerate(keep)}
    next_i = len(obj_list)
    for link_id, obj in writes:
        obj_list.append(obj)
        id_map[next_i] = link_id
        next_i += 1
    if not dry_run:
        save_sonar_file(labels_dir, idx, width, sonar_dir, obj_list,
                         {str(k): v for k, v in id_map.items()})


def run_interpolate(sonar_dir: Path, labels_dir: Path, frame_indices: list, width: int,
                     anchor_frames: list, snap_gate: float = 0.2,
                     min_snap_points: int = 15, dry_run: bool = False,
                     link_ids: set = None):
    """Returns (reports, error). reports is a list of one summary string per
    (anchor pair x link_id) segment processed, or None with an error string.

    link_ids, if given, restricts processing to just those link_ids (e.g. so
    each diver in a multi-diver scene can be filled on its own anchor set) -
    anchors that don't have one of these link_ids at both ends are skipped
    rather than erroring, same as the "no common link_id" case below.
    """
    anchors = sorted(set(anchor_frames))
    if len(anchors) < 2:
        return None, "Need at least 2 anchor frames."
    frame_set = set(frame_indices)
    for a in anchors:
        if a not in frame_set:
            return None, f"Frame {a} has no point cloud in this scene."

    reports = []
    for a, b in zip(anchors, anchors[1:]):
        if b <= a + 1:
            continue  # adjacent anchors, nothing in between to fill
        ids_a = sonar_link_ids_at(labels_dir, a, width)
        ids_b = sonar_link_ids_at(labels_dir, b, width)
        common = sorted(ids_a & ids_b)
        if link_ids is not None:
            common = [i for i in common if i in link_ids]
        if not common:
            reports.append(f"frame {a}->{b}: no matching link_id has a sonar box at both ends - skipped")
            continue
        for link_id in common:
            obj_a = get_sonar_object(labels_dir, a, width, sonar_dir, link_id)
            obj_b = get_sonar_object(labels_dir, b, width, sonar_dir, link_id)
            pA = np.array([obj_a["centroid"]["x"], obj_a["centroid"]["y"], obj_a["centroid"]["z"]])
            pB = np.array([obj_b["centroid"]["x"], obj_b["centroid"]["y"], obj_b["centroid"]["z"]])

            # camera-motion-weighted interpolation parameter t per frame:
            # distributes position along pA->pB by how far the camera box
            # actually moved by that point, not by raw frame count.
            cam_pos = {}
            for idx in range(a, b + 1):
                bbox = camera_bbox_at(labels_dir, idx, width, link_id)
                if bbox:
                    cam_pos[idx] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            frames_with_cam = sorted(cam_pos)
            t_of_frame = {}
            if len(frames_with_cam) >= 2:
                cum = [0.0]
                for i in range(1, len(frames_with_cam)):
                    p0, p1 = cam_pos[frames_with_cam[i - 1]], cam_pos[frames_with_cam[i]]
                    cum.append(cum[-1] + math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                total = cum[-1] if cum[-1] > 1e-9 else 1.0
                for i, idx in enumerate(frames_with_cam):
                    t_of_frame[idx] = cum[i] / total
            else:
                # no (or one) camera box in the whole span - fall back to
                # plain frame-count interpolation
                for idx in range(a, b + 1):
                    t_of_frame[idx] = (idx - a) / (b - a)

            n_written = n_snapped = n_interp_only = n_fov_skip = n_no_camera = 0
            for idx in range(a + 1, b):
                if idx not in t_of_frame:
                    n_no_camera += 1
                    continue
                t = t_of_frame[idx]
                pos = pA + t * (pB - pA)
                rot = dict(obj_a["rotations"])
                rot["z"] = angle_lerp_deg(obj_a["rotations"]["z"], obj_b["rotations"]["z"], t)
                dims = {k: obj_a["dimensions"][k] + t * (obj_b["dimensions"][k] - obj_a["dimensions"][k])
                        for k in obj_a["dimensions"]}

                bin_path = sonar_dir / f"{stem(idx, width)}.bin"
                points = load_points_xyz(bin_path) if bin_path.exists() else np.zeros((0, 3))
                clip = fov_clip_reason(points, pos)
                if clip:
                    n_fov_skip += 1
                    continue

                snapped, _ = local_snap(points, pos, snap_gate, min_snap_points)
                final_pos = snapped if snapped is not None else pos
                n_snapped += snapped is not None
                n_interp_only += snapped is None

                obj = {"name": obj_a["name"],
                       "centroid": {"x": float(final_pos[0]), "y": float(final_pos[1]),
                                    "z": float(final_pos[2])},
                       "dimensions": dims, "rotations": rot}
                write_sonar_objects(labels_dir, idx, width, sonar_dir, [(link_id, obj)], dry_run)
                n_written += 1

            reports.append(
                f"frame {a}->{b} link_id={link_id}: wrote {n_written} "
                f"({n_snapped} point-confirmed, {n_interp_only} interpolation-only), "
                f"{n_fov_skip} FOV-skipped, {n_no_camera} no-camera-skip"
            )
    return reports, None
