#!/usr/bin/env python3
"""
auto_track.py - semi-automated forward label propagation for converted_labels scenes.

Standalone, read/write-compatible helper for the labeling tool. Scenes live
under converted_labels/<Person>/<scene_id>/ (same layout as a uScenes scene). It does
NOT import or modify label_scene.py, patches/, config.ini, or anything installed
by labelCloud. It only reads and writes the same native per-frame label files
that labelCloud itself reads/writes inside <scene>/labels/:

    frame_XXXXXX.json          sonar (3D) label for a frame
    frame_XXXXXX_camera.json   camera (2D) label for a frame
    frame_XXXXXX_ids.json      sonar object index -> sonar_link_id map

Workflow:
    1. Label whichever single frame is easiest to judge by hand in labelCloud
       (does not have to be at either end of the scene). A frame may contain
       several divers, each with its own link_id (1, 2, 3, ...) - link_id
       only pairs a sonar box with a camera box of the same frame, it is not
       a persistent cross-frame track id.
    2. Run this script. For each modality (sonar / camera) it finds the
       most recently edited frame that has label(s), and tracks every object
       in it both forward and backward frame by frame (point-cloud
       clustering for sonar, OpenCV tracker for camera) independently, until
       a given object's body is no longer well visible in that direction, at
       which point that object (and only that object, in that direction)
       stops. Each direction also stops immediately if it reaches a frame
       that already has a label (never overwrites existing work).
    3. Open labelCloud, jump to where it stopped, fix/verify boxes.
    4. Re-run this script to continue expanding from the newest edit.

"Body no longer well visible" is approximated per tracked object as its
current size (camera box area / sonar cluster point count) dropping below
--min-size-ratio of its own baseline size (measured at the frame where it was
last hand-labeled) - a relative rather than absolute threshold, since divers
of very different apparent size still shrink by a similar fraction when they
become mostly occluded or edge-on. Camera boxes that end up mostly outside
the image, or below --min-box-px pixels, are treated the same way. This is a
heuristic, not the labeler's judgement - always review in labelCloud.

It never overwrites a frame that already has a non-empty label - only frames
that are currently empty get filled in.

Usage:
    python3 auto_track.py Person1 49              # Person1/scene_0049, both modalities
    python3 auto_track.py Person1 scene_0049 --dry-run
    python3 auto_track.py Person1 49 --modality sonar
    python3 auto_track.py Person1 49 --max-frames 50
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

TOOL_DIR = Path(__file__).parent
DATA_DIR = TOOL_DIR.parent
CONVERTED_DIR = DATA_DIR / "converted_labels"

TRACKER_FACTORY = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "mil": cv2.TrackerMIL_create,
}


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


def frame_width(sonar_dir: Path) -> int:
    sample = next(iter(sorted(sonar_dir.glob("frame_*.bin"))), None)
    if sample is None:
        raise SystemExit(f"No point cloud files found in {sonar_dir}")
    digits = sample.stem.split("_")[1]
    return len(digits)


def stem(idx: int, width: int) -> str:
    return f"frame_{idx:0{width}d}"


def parse_frame_ranges(text: str) -> set:
    """'82-84,120' -> {82,83,84,120}. Empty string -> empty set."""
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


# ---------------------------------------------------------------- sonar I/O --

def load_points(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32)
    pts = pts.reshape((-1, 4 if pts.size % 4 == 0 else 3))[:, 0:3]
    return pts[~np.isnan(pts).any(axis=1)]


def load_sonar_file(labels_dir: Path, idx: int, width: int, sonar_dir: Path) -> dict:
    path = labels_dir / f"{stem(idx, width)}.json"
    if path.exists():
        return json.loads(path.read_text())
    s = stem(idx, width)
    return {
        "folder": "sonar",
        "filename": f"{s}.bin",
        "path": str(sonar_dir / f"{s}.bin"),
        "objects": [],
    }


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
    data = {
        "folder": "sonar",
        "filename": f"{s}.bin",
        "path": str(sonar_dir / f"{s}.bin"),
        "objects": objects,
    }
    (labels_dir / f"{s}.json").write_text(json.dumps(data, indent="\t"))
    if link_ids:
        (labels_dir / f"{s}_ids.json").write_text(json.dumps(link_ids))


def save_camera_file(labels_dir: Path, idx: int, width: int, labels: list) -> None:
    s = stem(idx, width)
    data = {"frame": s, "labels": labels}
    (labels_dir / f"{s}_camera.json").write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------- clustering --

def estimate_local_eps(points: np.ndarray, centroid: np.ndarray, gate_radius: float,
                        multiplier: float, floor: float = 0.03, ceiling: float = 0.6) -> float:
    """Point density (and so the spacing between returns on the same object)
    varies a lot with range from the sensor - a diver 10m out is far sparser
    than one 1.5m out. Rather than one fixed clustering radius for every
    scene, measure the actual nearest-neighbour spacing among the points
    near this target's seed position and scale from that."""
    d = np.linalg.norm(points - centroid, axis=1)
    gated = points[d <= gate_radius]
    if len(gated) < 2:
        return floor
    tree = cKDTree(gated)
    dists, _ = tree.query(gated, k=2)
    median_spacing = float(np.median(dists[:, 1]))
    return float(np.clip(median_spacing * multiplier, floor, ceiling))


def cluster_labels(points: np.ndarray, eps: float) -> np.ndarray:
    """Simple radius-graph connected-components clustering (DBSCAN-lite)."""
    n = len(points)
    if n == 0:
        return np.array([], dtype=int)
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n)
    row, col = pairs[:, 0], pairs[:, 1]
    graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def track_sonar_step(points: np.ndarray, prev_centroid: np.ndarray,
                      gate_radius: float, eps: float, min_points: int):
    """Return (new_centroid, n_points) of the best matching cluster nearest
    prev_centroid, or None if nothing big enough was found within range."""
    d = np.linalg.norm(points - prev_centroid, axis=1)
    gated = points[d <= gate_radius]
    if len(gated) < min_points:
        return None
    labels = cluster_labels(gated, eps)
    best = None
    for lbl in np.unique(labels):
        idx = labels == lbl
        size = int(idx.sum())
        if size < min_points:
            continue
        if best is None or size > best[1]:
            best = (gated[idx].mean(axis=0), size)
    return best


# ------------------------------------------------------- unseeded suggestions --
#
# Point-cloud-only "find the diver with no prior seed" was tried as a fully
# autonomous detector (rank clusters of low-reflectance points by how well
# their size matches a diver) and validated against frames with a known,
# hand-labelled diver position - it found the right cluster in most of those.
# But run against frames a human had already confirmed had NO diver, it
# produced a confident-looking candidate every time too (most likely fish,
# vegetation, or structure with similarly low reflectance) - so it cannot
# reliably answer "is a diver here at all" on its own. It's kept here only as
# a *suggestion* tool: it proposes candidate positions in a single frame for
# a human to glance at and confirm/reject (against the camera image, say),
# it never writes a label by itself.

# empirical (length, width, height) prior from every hand-labelled sonar
# object across the whole uScenes dataset (n=1814): (median, std)
DIVER_SIZE_PRIOR = {"length": (1.2, 0.3), "width": (0.86, 0.17), "height": (0.9, 0.14)}


def load_raw_points(bin_path: Path) -> np.ndarray:
    """Like load_points, but keeps the reflectance/intensity column."""
    raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return raw[~np.isnan(raw).any(axis=1)]


def oriented_extents(cluster: np.ndarray):
    """PCA-oriented (length, width) in the XY plane plus axis-aligned height."""
    xy = cluster[:, :2] - cluster[:, :2].mean(axis=0)
    cov = np.cov(xy.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    proj = xy @ evecs[:, order]
    length = float(proj[:, 0].max() - proj[:, 0].min())
    width = float(proj[:, 1].max() - proj[:, 1].min())
    height = float(cluster[:, 2].max() - cluster[:, 2].min())
    return length, width, height


def diver_size_score(length: float, width: float, height: float) -> float:
    """Higher is more diver-shaped; roughly a negative squared z-score sum
    against the empirical size prior above."""
    zL = (length - DIVER_SIZE_PRIOR["length"][0]) / DIVER_SIZE_PRIOR["length"][1]
    zW = (width - DIVER_SIZE_PRIOR["width"][0]) / DIVER_SIZE_PRIOR["width"][1]
    zH = (height - DIVER_SIZE_PRIOR["height"][0]) / DIVER_SIZE_PRIOR["height"][1]
    return -(zL ** 2 + zW ** 2 + zH ** 2)


def find_candidate_clusters(bin_path: Path, refl_thresh: float = 0.2, eps: float = 0.1,
                             min_points: int = 15, top_k: int = 3) -> list:
    """Rank the largest low-reflectance point clusters in a frame by how
    diver-shaped they are. Read-only - never writes anything."""
    raw = load_raw_points(bin_path)
    xyz, refl = raw[:, :3], raw[:, 3]
    pts = xyz[refl <= refl_thresh]
    if len(pts) < min_points:
        return []
    labels = cluster_labels(pts, eps)
    sizes = np.bincount(labels)
    candidates = []
    for lbl in range(len(sizes)):
        if sizes[lbl] < min_points:
            continue
        cluster = pts[labels == lbl]
        length, width, height = oriented_extents(cluster)
        candidates.append({
            "centroid": cluster.mean(axis=0),
            "dimensions": (length, width, height),
            "n_points": int(sizes[lbl]),
            "score": diver_size_score(length, width, height),
        })
    candidates.sort(key=lambda c: -c["score"])
    return candidates[:top_k]


def render_candidates(all_points: np.ndarray, candidates: list, out_path: Path,
                       px_per_m: float = 60) -> None:
    """Top-down (X-Y) scatter of the whole point cloud with candidates
    circled and ranked, so a human can glance at one image instead of
    orbiting the 3D viewer to find the diver by eye."""
    xs, ys = all_points[:, 0], all_points[:, 1]
    pad = 0.5
    x0, x1 = xs.min() - pad, xs.max() + pad
    y0, y1 = ys.min() - pad, ys.max() + pad
    w = max(int((x1 - x0) * px_per_m), 50)
    h = max(int((y1 - y0) * px_per_m), 50)
    img = np.full((h, w, 3), 30, dtype=np.uint8)

    def to_px(x, y):
        return int((x - x0) * px_per_m), int(h - (y - y0) * px_per_m)

    for x, y in zip(xs, ys):
        cv2.circle(img, to_px(x, y), 1, (90, 90, 90), -1)
    colors = [(0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 0, 0)]
    for rank, cand in enumerate(candidates):
        cx, cy = to_px(cand["centroid"][0], cand["centroid"][1])
        color = colors[rank % len(colors)]
        radius = max(int(max(cand["dimensions"][:2]) / 2 * px_per_m), 6)
        cv2.circle(img, (cx, cy), radius, color, 2)
        cv2.putText(img, f"#{rank + 1}", (cx + radius + 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(out_path), img)


# Sonar field of view, measured empirically from the point clouds themselves
# (arctan2 of raw x/y/z) across six scenes - every scene clips azimuth at
# exactly +/-45.0 deg and elevation at +/-27.24 deg, so these are the
# sensor's real aperture limits, not scene-specific. Used by --stamp to skip
# frames where the frozen seed position would be sitting where the sensor
# physically can't see (see also stamp_core.py, used by the GUI button).
SONAR_AZIMUTH_LIMIT_DEG = 45.0
SONAR_ELEVATION_LIMIT_DEG = 27.24


def fov_clip_reason(points: np.ndarray, centroid, gate_radius: float = 0.5,
                     edge_margin_deg: float = 1.5, min_fraction: float = 0.08):
    """Whether the point cloud right around `centroid` in *this* frame shows
    a hard cutoff at the sonar's FOV boundary. Returns a reason string, or
    None if not clipped."""
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


def clipped_area_fraction(bx, by, bw, bh, w, h) -> float:
    """Fraction of a pixel box's area that actually lies inside the image."""
    if bw <= 0 or bh <= 0:
        return 0.0
    cx1, cy1 = max(bx, 0), max(by, 0)
    cx2, cy2 = min(bx + bw, w), min(by + bh, h)
    inter = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return inter / (bw * bh)


# --------------------------------------------------------------- discovery --

def find_seed_by_mtime(labels_dir: Path, frame_indices: list, width: int, camera: bool):
    """Return (frame_idx, items) of the most *recently edited* frame with
    non-empty content for this modality, or None. items is the raw 'labels'
    or 'objects' list.

    The seed is picked by file mtime rather than frame index so you can label
    whichever single frame is easiest to judge - anywhere in the scene - and
    have tracking expand outward from exactly that edit, in both directions.
    """
    best = None  # (mtime, idx, items)
    for idx in frame_indices:
        path = (labels_dir / f"{stem(idx, width)}_camera.json" if camera
                else labels_dir / f"{stem(idx, width)}.json")
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        items = data.get("labels" if camera else "objects", [])
        if not items:
            continue
        mtime = path.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, idx, items)
    return None if best is None else (best[1], best[2])


def find_highest_index_seed(labels_dir: Path, frame_indices: list, width: int, camera: bool):
    """Return (frame_idx, items) of the highest-*index* frame with non-empty
    content for this modality, or None. Used for camera-guided mode, which
    is a linear forward extension of one contiguous run rather than the
    "resume from wherever I just edited" pattern find_seed_by_mtime serves -
    repeated forward/backward runs can leave an earlier frame with the
    newest mtime, which would be the wrong place to resume from here."""
    best = None  # (idx, items)
    for idx in frame_indices:
        path = (labels_dir / f"{stem(idx, width)}_camera.json" if camera
                else labels_dir / f"{stem(idx, width)}.json")
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        items = data.get("labels" if camera else "objects", [])
        if items:
            best = (idx, items)
    return best


def resolve_seed_link_ids_camera(labels_dir: Path, idx: int, width: int, labels: list) -> list:
    """Fill in missing (0) link_ids on the seed frame's camera labels.
    link_id only pairs a sonar box with a camera box of the *same* frame
    (see module docstring) - so the only safe place to recover a missing one
    is that same frame's sonar labels, and only when the mapping is
    unambiguous (single missing label, single available id)."""
    missing = [i for i, l in enumerate(labels) if not l.get("link_id")]
    if not missing:
        return labels
    available = [v for v in load_sonar_ids(labels_dir, idx, width).values() if v]
    fixed = list(labels)
    if len(missing) == 1 and len(available) == 1:
        fixed[missing[0]] = {**fixed[missing[0]], "link_id": available[0]}
    else:
        for i in missing:
            print(f"[camera] warning: frame {idx} label #{i} "
                  f"(class={fixed[i]['class']}) has no link_id and it could not "
                  f"be resolved unambiguously - set it by hand in labelCloud "
                  f"before tracking, otherwise it is skipped.")
    return fixed


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
    else:
        for i in missing:
            print(f"[sonar] warning: frame {idx} object #{i} "
                  f"(name={objects[i]['name']}) has no link_id and it could not "
                  f"be resolved unambiguously - set it by hand in labelCloud "
                  f"before tracking, otherwise it is skipped.")
    return link_ids


def sonar_link_ids_at(labels_dir: Path, idx: int, width: int) -> set:
    """link_ids actually backed by an object at this frame. The _ids.json
    side-file can go stale relative to the main frame file (e.g. objects
    list cleared without touching companion) - only trust ids for indices
    that still have a real object, rather than blindly trusting the file."""
    n_objects = len(load_sonar_file(labels_dir, idx, width, labels_dir).get("objects", []))
    ids = load_sonar_ids(labels_dir, idx, width)
    return {v for k, v in ids.items() if k < n_objects}


def camera_link_ids_at(labels_dir: Path, idx: int, width: int) -> set:
    return {l.get("link_id") for l in load_camera_file(labels_dir, idx, width).get("labels", [])}


def camera_bbox_center_at(labels_dir: Path, idx: int, width: int, link_id: int):
    """Normalized (cx, cy) of this link_id's camera box at this frame, or None."""
    for l in load_camera_file(labels_dir, idx, width).get("labels", []):
        if l.get("link_id") == link_id:
            return ((l["x1"] + l["x2"]) / 2, (l["y1"] + l["y2"]) / 2)
    return None


# -------------------------------------------------------------------- run --

def run_sonar_direction(sonar_dir: Path, labels_dir: Path, width: int,
                         base_targets: list, indices: list, args, direction: str) -> int:
    targets = [{**t, "prev_centroid": t["prev_centroid"].copy(), "active": True}
               for t in base_targets]
    n_written_frames = 0
    for idx in indices:
        # link_id already present at this exact frame -> that target reached
        # previously reviewed territory (do not touch it), but other targets
        # sharing this frame range may still have room to extend into it.
        existing_link_ids = sonar_link_ids_at(labels_dir, idx, width)
        newly_conflicted = [t for t in targets if t["active"] and t["link_id"] in existing_link_ids]
        for t in newly_conflicted:
            print(f"[sonar:{direction}] link_id={t['link_id']} already labeled at "
                  f"frame {idx} - stopping this target (reached previously reviewed territory).")
            t["active"] = False
        if not any(t["active"] for t in targets):
            print(f"[sonar:{direction}] no active targets left by frame {idx} - stopping.")
            break
        if args.max_frames is not None and n_written_frames >= args.max_frames:
            print(f"[sonar:{direction}] hit --max-frames {args.max_frames} at frame {idx}.")
            break

        bin_path = sonar_dir / f"{stem(idx, width)}.bin"
        if not bin_path.exists():
            print(f"[sonar:{direction}] no point cloud for frame {idx} - stopping.")
            break
        points = load_points(bin_path)

        results = []
        for t in targets:
            if not t["active"]:
                continue
            result = track_sonar_step(points, t["prev_centroid"], args.gate_radius,
                                       t["eps"], args.min_points)
            reason = None
            if result is None:
                reason = "no cluster found nearby"
            else:
                new_centroid, n_pts = result
                if n_pts < args.min_size_ratio * t["baseline_size"]:
                    reason = (f"cluster shrank to {n_pts}pts "
                              f"({n_pts / t['baseline_size']:.0%} of baseline) "
                              f"- body likely not well visible")
            if reason:
                print(f"[sonar:{direction}] link_id={t['link_id']} lost at frame {idx}: "
                      f"{reason} - stopping this target.")
                t["active"] = False
                continue
            t["prev_centroid"] = new_centroid
            results.append((t, new_centroid, n_pts))

        if not any(t["active"] for t in targets):
            print(f"[sonar:{direction}] all targets lost by frame {idx} - stopping.")
            break
        if results:
            existing = load_sonar_file(labels_dir, idx, width, sonar_dir)
            obj_list = list(existing.get("objects", []))
            id_map = {str(k): v for k, v in load_sonar_ids(labels_dir, idx, width).items()}
            next_i = len(obj_list)
            for t, centroid, n_pts in results:
                obj_list.append({
                    "name": t["name"],
                    "centroid": {"x": float(centroid[0]), "y": float(centroid[1]),
                                 "z": float(centroid[2])},
                    "dimensions": t["dimensions"],
                    "rotations": t["rotations"],
                })
                id_map[str(next_i)] = t["link_id"]
                next_i += 1
            if not args.dry_run:
                save_sonar_file(labels_dir, idx, width, sonar_dir, obj_list, id_map)
            ids_str = ",".join(str(t["link_id"]) for t, _, _ in results)
            print(f"[sonar:{direction}] frame {idx}: {len(obj_list)} object(s) "
                  f"(link_id {ids_str}){' [dry-run]' if args.dry_run else ''}")
            n_written_frames += 1
    return n_written_frames


def run_sonar(scene_dir: Path, labels_dir: Path, frame_indices: list, width: int,
              args) -> None:
    sonar_dir = scene_dir / "sonar"
    seed = find_seed_by_mtime(labels_dir, frame_indices, width, camera=False)
    if seed is None:
        print("[sonar] no labeled frame found yet - nothing to track from. "
              "Label at least one frame in labelCloud first.")
        return
    seed_idx, seed_objs = seed
    link_ids = resolve_seed_link_ids_sonar(labels_dir, seed_idx, width, seed_objs)
    seed_points = load_points(sonar_dir / f"{stem(seed_idx, width)}.bin")

    base_targets = []
    for obj, link_id in zip(seed_objs, link_ids):
        if not link_id:
            continue
        centroid = np.array([obj["centroid"]["x"], obj["centroid"]["y"], obj["centroid"]["z"]])
        if args.eps is not None:
            eps, result = args.eps, track_sonar_step(
                seed_points, centroid, args.gate_radius, args.eps, args.min_points)
        else:
            # the diver's own return points sometimes split into a couple of
            # pieces at the base multiplier (self-shadowing, sparse angle) -
            # widen the clustering radius step by step until they merge into
            # one cluster that clears min_points, instead of giving up.
            eps, result = None, None
            for mult in (args.eps_multiplier * f for f in (1.0, 1.4, 1.8, 2.4)):
                eps = estimate_local_eps(seed_points, centroid, args.gate_radius, mult)
                result = track_sonar_step(seed_points, centroid, args.gate_radius, eps, args.min_points)
                if result is not None:
                    break
        if result is None:
            print(f"[sonar] warning: link_id={link_id} at seed frame {seed_idx} has no "
                  f"point cluster within {args.gate_radius}m even after widening eps up to "
                  f"{eps:.3f} - the manual box may not line up with the point cloud. "
                  f"Skipping this target.")
            continue
        baseline_size = result[1]
        base_targets.append({
            "link_id": link_id,
            "name": obj["name"],
            "dimensions": obj["dimensions"],
            "rotations": obj["rotations"],
            "prev_centroid": centroid,
            "baseline_size": baseline_size,
            "eps": eps,
        })
    if not base_targets:
        print(f"[sonar] seed frame {seed_idx} has no object with a resolvable "
              f"link_id and a matching point cluster - nothing to track.")
        return
    summary = ", ".join(f"link_id={t['link_id']} baseline={t['baseline_size']}pts eps={t['eps']:.3f}"
                         for t in base_targets)
    print(f"[sonar] seed frame {seed_idx} (most recently edited): "
          f"{len(base_targets)} target(s) ({summary})")

    forward_indices = [i for i in frame_indices if i > seed_idx]
    backward_indices = sorted((i for i in frame_indices if i < seed_idx), reverse=True)
    n_fwd = run_sonar_direction(sonar_dir, labels_dir, width, base_targets,
                                 forward_indices, args, "forward")
    n_bwd = run_sonar_direction(sonar_dir, labels_dir, width, base_targets,
                                 backward_indices, args, "backward")

    print(f"[sonar] {'would write' if args.dry_run else 'wrote'} "
          f"{n_fwd} frame(s) forward and {n_bwd} frame(s) backward from seed frame {seed_idx}.")


def run_sonar_camera_guided(scene_dir: Path, labels_dir: Path, frame_indices: list, width: int,
                             args) -> None:
    """Track sonar centroids only across frames that already carry a
    human-drawn camera label for that link_id - the camera label is treated
    as ground truth for "is this diver actually in this frame", so unlike
    run_sonar() this never stops just because the point-cluster match
    momentarily weakens (min-size-ratio does not apply here). It only stops
    a target after several consecutive camera-confirmed frames in a row with
    no matching cluster at all (a real, sustained sonar dropout), widening
    the search gate a little for each frame skipped in between.

    Position only. Dimensions/rotation are carried from the seed - rotation
    can't be reliably derived automatically from either modality (see the
    PCA calibration attempt in this project's history), so re-anchor it by
    hand at frames where the camera shows the diver's pose has clearly
    changed.

    Cross-checked against the camera box: a sonar step is only accepted if
    either the jump is small, or the camera box center moved by a
    comparable amount over the same span - point clusters occasionally jump
    onto a nearby similarly-sized rigid object (e.g. a calibration board)
    while the diver has barely moved, and the camera box (drawn by a human)
    calls that out immediately since it did not also jump.
    """
    sonar_dir = scene_dir / "sonar"
    if args.from_frame is not None:
        seed_idx = args.from_frame
        data = load_sonar_file(labels_dir, seed_idx, width, sonar_dir)
        seed_objs = data.get("objects", [])
        if not seed_objs:
            print(f"[sonar:cam-guided] --from-frame {seed_idx} has no sonar object - "
                  f"label it first.")
            return
        seed = (seed_idx, seed_objs)
    else:
        seed = find_highest_index_seed(labels_dir, frame_indices, width, camera=False)
    if seed is None:
        print("[sonar:cam-guided] no sonar seed found yet - label one frame first.")
        return
    seed_idx, seed_objs = seed
    link_ids = resolve_seed_link_ids_sonar(labels_dir, seed_idx, width, seed_objs)
    seed_points = load_points(sonar_dir / f"{stem(seed_idx, width)}.bin")

    targets = []
    for obj, link_id in zip(seed_objs, link_ids):
        if not link_id:
            continue
        centroid = np.array([obj["centroid"]["x"], obj["centroid"]["y"], obj["centroid"]["z"]])
        eps = args.eps if args.eps is not None else estimate_local_eps(
            seed_points, centroid, args.gate_radius, args.eps_multiplier)
        targets.append({
            "link_id": link_id, "name": obj["name"], "dimensions": obj["dimensions"],
            "rotations": obj["rotations"], "prev_centroid": centroid, "eps": eps,
            "active": True, "consecutive_gaps": 0,
            "last_cam_pos": camera_bbox_center_at(labels_dir, seed_idx, width, link_id),
        })
    if not targets:
        print("[sonar:cam-guided] seed frame has no resolvable target.")
        return
    summary = ", ".join(f"link_id={t['link_id']} eps={t['eps']:.3f}" for t in targets)
    print(f"[sonar:cam-guided] seed frame {seed_idx}: {len(targets)} target(s) ({summary})")

    forward = [i for i in frame_indices if i > seed_idx and i not in args.skip_frames]
    if args.skip_frames:
        print(f"[sonar:cam-guided] skipping frame(s) by request: {sorted(args.skip_frames)}")
    n_written, gap_frames = 0, []
    for idx in forward:
        if not any(t["active"] for t in targets):
            break
        if args.max_frames is not None and n_written >= args.max_frames:
            print(f"[sonar:cam-guided] hit --max-frames {args.max_frames} at frame {idx}.")
            break

        cam_ids = camera_link_ids_at(labels_dir, idx, width)
        existing_sonar_ids = sonar_link_ids_at(labels_dir, idx, width)
        bin_path = sonar_dir / f"{stem(idx, width)}.bin"
        points = None
        results = []
        for t in targets:
            if not t["active"] or t["link_id"] in existing_sonar_ids:
                if t["active"] and t["link_id"] in existing_sonar_ids:
                    print(f"[sonar:cam-guided] link_id={t['link_id']} already labeled at "
                          f"frame {idx} - stopping this target.")
                    t["active"] = False
                continue
            if t["link_id"] not in cam_ids:
                continue  # camera confirms nothing to track here for this target
            if points is None:
                if not bin_path.exists():
                    print(f"[sonar:cam-guided] no point cloud for frame {idx} - stopping.")
                    for t2 in targets:
                        t2["active"] = False
                    break
                points = load_points(bin_path)
            gate = min(args.gate_radius * (1 + 0.5 * t["consecutive_gaps"]), args.gate_radius * 4)
            result = track_sonar_step(points, t["prev_centroid"], gate, t["eps"], args.min_points)
            if result is None:
                t["consecutive_gaps"] += 1
                gap_frames.append((t["link_id"], idx))
                if t["consecutive_gaps"] >= 10:
                    print(f"[sonar:cam-guided] link_id={t['link_id']} lost - no cluster for "
                          f"10 consecutive camera-confirmed frames (through {idx}) - "
                          f"stopping this target, needs a manual re-seed.")
                    t["active"] = False
                continue
            new_centroid, n_pts = result
            sonar_jump = float(np.linalg.norm(new_centroid - t["prev_centroid"]))
            cam_pos = camera_bbox_center_at(labels_dir, idx, width, t["link_id"])
            cam_jump = (math.hypot(cam_pos[0] - t["last_cam_pos"][0], cam_pos[1] - t["last_cam_pos"][1])
                        if cam_pos and t["last_cam_pos"] else None)
            if (sonar_jump > args.max_jump and cam_jump is not None
                    and cam_jump < args.cam_still_thresh):
                print(f"[sonar:cam-guided] link_id={t['link_id']} rejected candidate at "
                      f"frame {idx}: sonar jumped {sonar_jump:.2f}m but camera box barely "
                      f"moved ({cam_jump:.3f}) - likely latched onto a nearby object "
                      f"(e.g. the calibration board), not the diver moving. Skipping frame.")
                t["consecutive_gaps"] += 1
                gap_frames.append((t["link_id"], idx))
                if t["consecutive_gaps"] >= 10:
                    print(f"[sonar:cam-guided] link_id={t['link_id']} lost - no accepted "
                          f"cluster for 10 consecutive camera-confirmed frames (through "
                          f"{idx}) - stopping this target, needs a manual re-seed.")
                    t["active"] = False
                continue
            t["prev_centroid"], t["consecutive_gaps"] = new_centroid, 0
            if cam_pos:
                t["last_cam_pos"] = cam_pos
            results.append((t, new_centroid, n_pts))

        if results:
            existing = load_sonar_file(labels_dir, idx, width, sonar_dir)
            obj_list = list(existing.get("objects", []))
            id_map = {str(k): v for k, v in load_sonar_ids(labels_dir, idx, width).items()}
            next_i = len(obj_list)
            for t, centroid, n_pts in results:
                obj_list.append({
                    "name": t["name"],
                    "centroid": {"x": float(centroid[0]), "y": float(centroid[1]),
                                 "z": float(centroid[2])},
                    "dimensions": t["dimensions"],
                    "rotations": t["rotations"],
                })
                id_map[str(next_i)] = t["link_id"]
                next_i += 1
            if not args.dry_run:
                save_sonar_file(labels_dir, idx, width, sonar_dir, obj_list, id_map)
            ids_str = ",".join(str(t["link_id"]) for t, _, _ in results)
            print(f"[sonar:cam-guided] frame {idx}: {len(results)} object(s) "
                  f"(link_id {ids_str}){' [dry-run]' if args.dry_run else ''}")
            n_written += 1

    if gap_frames:
        shown = [f for _, f in gap_frames[:20]]
        print(f"[sonar:cam-guided] {len(gap_frames)} camera-confirmed frame(s) had no "
              f"matching sonar cluster (likely brief sonar dropout) and were left empty: "
              f"{shown}{'...' if len(gap_frames) > 20 else ''}")
    print(f"[sonar:cam-guided] {'would write' if args.dry_run else 'wrote'} "
          f"{n_written} frame(s) forward from seed frame {seed_idx}.")


# ------------------------------------------------------------------ stamp --

def bbox_iou(a, b) -> float:
    """IoU between two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def camera_bbox_at(labels_dir: Path, idx: int, width: int, link_id: int):
    """Returns (x1, y1, x2, y2) with x1<=x2 and y1<=y2 - camera_panel.py
    stores raw drag start/end, so ~13% of boxes have x1>x2 or y1>y2, which
    silently zeroes their area in bbox_iou if left unnormalized."""
    for l in load_camera_file(labels_dir, idx, width).get("labels", []):
        if l.get("link_id") == link_id:
            x1, y1, x2, y2 = l["x1"], l["y1"], l["x2"], l["y2"]
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return None


def show_popup(title: str, message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()
    except Exception as e:
        print(f"[popup] could not show a popup ({e}) - see the summary above instead.")


def run_sonar_stamp(scene_dir: Path, labels_dir: Path, frame_indices: list, width: int,
                     args) -> None:
    """The simplest possible mode: copy the seed's sonar box - centroid,
    dimensions, rotation, all unchanged - into every subsequent
    camera-confirmed frame, for as long as that frame's camera box still
    overlaps the seed's camera box well enough (IoU >= --min-iou). No
    point-cloud loading, no clustering, no per-frame confidence heuristics -
    the human-drawn camera label is the only signal used to decide how far
    to stamp, and exactly where it stopped is reported (and popped up).
    """
    sonar_dir = scene_dir / "sonar"
    if args.from_frame is not None:
        seed_idx = args.from_frame
        seed_objs = load_sonar_file(labels_dir, seed_idx, width, sonar_dir).get("objects", [])
        if not seed_objs:
            print(f"[sonar:stamp] --from-frame {seed_idx} has no sonar object - label it first.")
            return
        seed = (seed_idx, seed_objs)
    else:
        seed = find_highest_index_seed(labels_dir, frame_indices, width, camera=False)
    if seed is None:
        print("[sonar:stamp] no sonar seed found - label one frame first.")
        return
    seed_idx, seed_objs = seed
    link_ids = resolve_seed_link_ids_sonar(labels_dir, seed_idx, width, seed_objs)

    targets = []
    for obj, link_id in zip(seed_objs, link_ids):
        if not link_id:
            continue
        seed_bbox = camera_bbox_at(labels_dir, seed_idx, width, link_id)
        if seed_bbox is None:
            print(f"[sonar:stamp] warning: link_id={link_id} has no camera box at seed "
                  f"frame {seed_idx} to compare against - skipping.")
            continue
        targets.append({
            "link_id": link_id, "name": obj["name"], "dimensions": obj["dimensions"],
            "rotations": obj["rotations"], "centroid": obj["centroid"],
            "seed_bbox": seed_bbox, "active": True,
            "stop_frame": None, "stop_reason": None,
        })
    if not targets:
        print("[sonar:stamp] nothing to stamp.")
        return
    ids_summary = ", ".join(f"link_id={t['link_id']}" for t in targets)
    print(f"[sonar:stamp] seed frame {seed_idx}: {ids_summary}")

    forward = [i for i in frame_indices if i > seed_idx and i not in args.skip_frames]
    n_written = 0
    for idx in forward:
        if not any(t["active"] for t in targets):
            break
        if args.max_frames is not None and n_written >= args.max_frames:
            print(f"[sonar:stamp] hit --max-frames {args.max_frames} at frame {idx}.")
            break

        existing_sonar_ids = sonar_link_ids_at(labels_dir, idx, width)
        results = []
        points = None
        for t in targets:
            if not t["active"]:
                continue
            if not args.overwrite and t["link_id"] in existing_sonar_ids:
                t["active"] = False
                t["stop_frame"], t["stop_reason"] = idx, "reached a previously labeled frame"
                continue
            cur_bbox = camera_bbox_at(labels_dir, idx, width, t["link_id"])
            if cur_bbox is None:
                continue  # camera doesn't confirm this diver here - nothing to stamp

            if points is None:
                bin_path = sonar_dir / f"{stem(idx, width)}.bin"
                points = load_points(bin_path) if bin_path.exists() else np.zeros((0, 3))
            c = t["centroid"]
            clip_reason = fov_clip_reason(points, (c["x"], c["y"], c["z"]))
            if clip_reason:
                t["active"] = False
                t["stop_frame"] = idx
                t["stop_reason"] = clip_reason
                print(f"[sonar:stamp] link_id={t['link_id']} stopped at frame {idx}: "
                      f"{clip_reason}")
                continue

            iou = bbox_iou(t["seed_bbox"], cur_bbox)
            results.append(t)
            if iou < args.min_iou:
                # still write this last frame - a rough starting point to
                # fix by hand - but do not go any further than this.
                t["active"] = False
                t["stop_frame"] = idx
                t["stop_reason"] = (f"camera box drifted too far (IoU={iou:.2f} < "
                                     f"{args.min_iou}) - wrote it anyway as a starting "
                                     f"point to fix")
                print(f"[sonar:stamp] link_id={t['link_id']} stopped at frame {idx} "
                      f"(wrote it anyway to fix by hand): {t['stop_reason']}")

        if results:
            existing = load_sonar_file(labels_dir, idx, width, sonar_dir)
            obj_list = list(existing.get("objects", []))
            id_map = {k: v for k, v in load_sonar_ids(labels_dir, idx, width).items()
                      if k < len(obj_list)}
            # drop any existing entry for the link_ids we're about to write,
            # so --overwrite replaces in place instead of duplicating
            new_link_ids = {t["link_id"] for t in results}
            keep = [i for i in range(len(obj_list)) if id_map.get(i) not in new_link_ids]
            obj_list = [obj_list[i] for i in keep]
            id_map = {new_i: id_map[old_i] for new_i, old_i in enumerate(keep)}
            next_i = len(obj_list)
            for t in results:
                obj_list.append({
                    "name": t["name"], "centroid": t["centroid"],
                    "dimensions": t["dimensions"], "rotations": t["rotations"],
                })
                id_map[next_i] = t["link_id"]
                next_i += 1
            if not args.dry_run:
                save_sonar_file(labels_dir, idx, width, sonar_dir, obj_list,
                                 {str(k): v for k, v in id_map.items()})
            n_written += 1

    lines = [f"Scene {scene_dir.name}: seed frame {seed_idx} -> "
             f"{'would write' if args.dry_run else 'wrote'} {n_written} frame(s)."]
    for t in targets:
        if t["stop_frame"] is not None:
            lines.append(f"  link_id={t['link_id']}: stopped at frame {t['stop_frame']} "
                         f"({t['stop_reason']})")
        else:
            lines.append(f"  link_id={t['link_id']}: reached end of range without stopping")
    message = "\n".join(lines)
    print(message)
    if args.popup and not args.dry_run:
        show_popup("auto_track.py - stamp done", message)


def run_camera_direction(camera_dir: Path, labels_dir: Path, width: int, seed_idx: int,
                          seed_img, seed_labels: list, indices: list, args, direction: str) -> int:
    h, w = seed_img.shape[:2]
    targets = []
    for lbl in seed_labels:
        if not lbl.get("link_id"):
            continue
        # normalize in case x1/x2 or y1/y2 were saved swapped, and skip
        # anything that still has no real extent instead of handing cv2 a
        # degenerate box (it throws a hard C++ assertion, not a Python error).
        x1, x2 = sorted((lbl["x1"], lbl["x2"]))
        y1, y2 = sorted((lbl["y1"], lbl["y2"]))
        px_bbox = (int(round(x1 * w)), int(round(y1 * h)),
                   int(round((x2 - x1) * w)), int(round((y2 - y1) * h)))
        if px_bbox[2] <= 0 or px_bbox[3] <= 0:
            print(f"[camera:{direction}] warning: link_id={lbl['link_id']} seed box at "
                  f"frame {seed_idx} is degenerate ({px_bbox[2]}x{px_bbox[3]}px) - skipping.")
            continue
        tracker = TRACKER_FACTORY[args.tracker]()
        tracker.init(seed_img, px_bbox)
        targets.append({
            "link_id": lbl["link_id"],
            "class": lbl["class"],
            "tracker": tracker,
            "baseline_area": max(px_bbox[2] * px_bbox[3], 1),
            "active": True,
        })

    n_written_frames = 0
    for idx in indices:
        existing_link_ids = camera_link_ids_at(labels_dir, idx, width)
        newly_conflicted = [t for t in targets if t["active"] and t["link_id"] in existing_link_ids]
        for t in newly_conflicted:
            print(f"[camera:{direction}] link_id={t['link_id']} already labeled at "
                  f"frame {idx} - stopping this target (reached previously reviewed territory).")
            t["active"] = False
        if not any(t["active"] for t in targets):
            print(f"[camera:{direction}] no active targets left by frame {idx} - stopping.")
            break
        if args.max_frames is not None and n_written_frames >= args.max_frames:
            print(f"[camera:{direction}] hit --max-frames {args.max_frames} at frame {idx}.")
            break

        img_path = camera_dir / f"{stem(idx, width)}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[camera:{direction}] no image for frame {idx} - stopping.")
            break

        frame_labels = []
        for t in targets:
            if not t["active"]:
                continue
            ok, box = t["tracker"].update(img)
            bx, by, bw, bh = box
            area = bw * bh
            visible_frac = clipped_area_fraction(bx, by, bw, bh, w, h)
            reason = None
            if not ok:
                reason = "tracker lost target"
            elif bw < args.min_box_px or bh < args.min_box_px:
                reason = f"box too small ({bw:.0f}x{bh:.0f}px)"
            elif area < args.min_size_ratio * t["baseline_area"]:
                reason = (f"box area shrank to {area / t['baseline_area']:.0%} "
                          f"of baseline - body likely not well visible")
            elif visible_frac < 0.4:
                reason = f"mostly out of frame ({visible_frac:.0%} visible)"
            if reason:
                print(f"[camera:{direction}] link_id={t['link_id']} lost at frame {idx}: "
                      f"{reason} - stopping this target.")
                t["active"] = False
                continue
            nx1, ny1 = bx / w, by / h
            nx2, ny2 = (bx + bw) / w, (by + bh) / h
            frame_labels.append({
                "class": t["class"],
                "link_id": t["link_id"],
                "x1": float(max(0.0, min(1.0, nx1))),
                "y1": float(max(0.0, min(1.0, ny1))),
                "x2": float(max(0.0, min(1.0, nx2))),
                "y2": float(max(0.0, min(1.0, ny2))),
            })

        if not any(t["active"] for t in targets):
            print(f"[camera:{direction}] all targets lost by frame {idx} - stopping.")
            break
        if frame_labels:
            existing = load_camera_file(labels_dir, idx, width)
            merged = list(existing.get("labels", [])) + frame_labels
            if not args.dry_run:
                save_camera_file(labels_dir, idx, width, merged)
            ids_str = ",".join(str(l["link_id"]) for l in frame_labels)
            print(f"[camera:{direction}] frame {idx}: {len(frame_labels)} label(s) "
                  f"(link_id {ids_str}){' [dry-run]' if args.dry_run else ''}")
            n_written_frames += 1
    return n_written_frames


def run_camera(scene_dir: Path, labels_dir: Path, frame_indices: list, width: int,
               args) -> None:
    camera_dir = scene_dir / "camera"
    seed = find_seed_by_mtime(labels_dir, frame_indices, width, camera=True)
    if seed is None:
        print("[camera] no labeled frame found yet - nothing to track from. "
              "Label at least one frame in labelCloud first.")
        return
    seed_idx, seed_labels = seed
    seed_labels = resolve_seed_link_ids_camera(labels_dir, seed_idx, width, seed_labels)
    seed_labels = [l for l in seed_labels if l.get("link_id")]

    seed_img_path = camera_dir / f"{stem(seed_idx, width)}.jpg"
    seed_img = cv2.imread(str(seed_img_path))
    if seed_img is None:
        print(f"[camera] could not read {seed_img_path} - stopping.")
        return
    if not seed_labels:
        print(f"[camera] seed frame {seed_idx} has no label with a resolvable "
              f"link_id - nothing to track.")
        return
    summary = ", ".join(f"link_id={l['link_id']}" for l in seed_labels)
    print(f"[camera] seed frame {seed_idx} (most recently edited): "
          f"{len(seed_labels)} target(s) ({summary})")

    forward_indices = [i for i in frame_indices if i > seed_idx]
    backward_indices = sorted((i for i in frame_indices if i < seed_idx), reverse=True)
    n_fwd = run_camera_direction(camera_dir, labels_dir, width, seed_idx, seed_img,
                                  seed_labels, forward_indices, args, "forward")
    n_bwd = run_camera_direction(camera_dir, labels_dir, width, seed_idx, seed_img,
                                  seed_labels, backward_indices, args, "backward")

    print(f"[camera] {'would write' if args.dry_run else 'wrote'} "
          f"{n_fwd} frame(s) forward and {n_bwd} frame(s) backward from seed frame {seed_idx}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("person", help="Person folder (e.g. Person1)")
    parser.add_argument("scene", help="Scene number or ID (e.g. 49 or scene_0049)")
    parser.add_argument("--modality", choices=["both", "sonar", "camera"], default="both")
    parser.add_argument("--max-frames", type=int, default=30,
                         help="Cap on number of frames to auto-fill per modality/direction "
                              "this run (default 30). The confidence checks do not reliably "
                              "catch a visual tracker drifting onto the wrong diver in busy "
                              "or turbid scenes, so long unattended runs risk writing "
                              "silently-wrong boxes - review in labelCloud and re-run to "
                              "continue rather than raising this a lot. Pass 0 for unlimited.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would be tracked without writing any files")
    # visibility / confidence gating
    parser.add_argument("--min-size-ratio", type=float, default=0.4,
                         help="Stop a target once its size (cluster points / box area) "
                              "drops below this fraction of its baseline size")
    # sonar tuning
    parser.add_argument("--gate-radius", type=float, default=0.3,
                         help="Max allowed centroid displacement between frames (m)")
    parser.add_argument("--eps", type=float, default=None,
                         help="Point clustering neighbor radius (m). Default: auto-estimated "
                              "per target from local point spacing at its seed frame, since "
                              "point density drops a lot with range from the sensor")
    parser.add_argument("--eps-multiplier", type=float, default=2.5,
                         help="Auto eps = median nearest-neighbour spacing near the seed "
                              "x this multiplier")
    parser.add_argument("--min-points", type=int, default=20,
                         help="Absolute floor on cluster point count to accept it at all")
    # camera tuning
    parser.add_argument("--tracker", choices=list(TRACKER_FACTORY), default="csrt")
    parser.add_argument("--min-box-px", type=float, default=15,
                         help="Absolute floor on tracked box width/height in pixels")
    # unseeded suggestions (read-only, writes nothing)
    parser.add_argument("--suggest", type=int, metavar="FRAME",
                         help="Suggest candidate diver positions for this sonar frame from "
                              "point-cloud shape alone (no seed needed) and save a top-down "
                              "image to review. Does NOT write any label - not reliable enough "
                              "to trust unattended, see module docstring.")
    parser.add_argument("--suggest-out", type=Path, default=None,
                         help="Where to save the --suggest preview image "
                              "(default: <scene>_suggest_<frame>.png in the current directory)")
    parser.add_argument("--camera-guided", action="store_true",
                         help="Sonar-only mode: track position across every frame that "
                              "already has a human camera label, using the camera label as "
                              "ground truth for whether a diver is present instead of the "
                              "sonar-only confidence heuristics. See run_sonar_camera_guided().")
    parser.add_argument("--max-jump", type=float, default=0.15,
                         help="[camera-guided] Reject a sonar step bigger than this (m) if "
                              "the camera box barely moved over the same span - it likely "
                              "latched onto a different nearby object")
    parser.add_argument("--cam-still-thresh", type=float, default=0.1,
                         help="[camera-guided] Camera box center movement (normalized image "
                              "units) below this counts as 'barely moved' for --max-jump")
    parser.add_argument("--from-frame", type=int, default=None,
                         help="[camera-guided] Use this exact frame as the seed instead of "
                              "the highest-index labeled frame - for filling a gap you just "
                              "re-seeded in the middle of the scene")
    parser.add_argument("--skip-frames", type=parse_frame_ranges, default=set(),
                         help="[camera-guided/stamp] Comma-separated frame(s)/ranges to leave "
                              "untouched even if a camera label exists there, e.g. "
                              "'82-84,120'")
    parser.add_argument("--stamp", action="store_true",
                         help="Sonar-only mode: copy the seed's sonar box as-is (no "
                              "point-cloud tracking at all) into every following "
                              "camera-confirmed frame until the camera box has drifted too "
                              "far from the seed's camera box. See run_sonar_stamp().")
    parser.add_argument("--min-iou", type=float, default=0.4,
                         help="[stamp] Stop once the current frame's camera box overlaps "
                              "the seed's camera box less than this (IoU, 0-1)")
    parser.add_argument("--overwrite", action="store_true",
                         help="[stamp] Replace existing sonar content for this link_id "
                              "going forward instead of stopping at the first "
                              "already-labeled frame - for re-seeding partway through a "
                              "stretch that was already stamped once but has drifted.")
    parser.add_argument("--popup", dest="popup", action="store_true", default=True,
                         help="[stamp] Show a desktop popup summarizing where it stopped "
                              "(default on)")
    parser.add_argument("--no-popup", dest="popup", action="store_false",
                         help="[stamp] Disable the popup, print-only")
    parser.add_argument("--interpolate", type=str, default=None, metavar="F1,F2,F3,...",
                         help="Sonar-only mode: comma-separated already-labeled anchor "
                              "frames (e.g. '45,120,300'). Fills every frame strictly "
                              "between each consecutive pair by interpolating position "
                              "(weighted by camera-box motion, not frame count) plus "
                              "rotation/dimensions, then lightly snapping to nearby points. "
                              "Always overwrites whatever is currently at those in-between "
                              "frames. See stamp_core.run_interpolate().")
    parser.add_argument("--link-id", type=str, default=None, metavar="ID,ID,...",
                         help="[interpolate] Restrict to these link_id(s) only, so divers "
                              "with different anchor points can be filled one at a time. "
                              "Default: every link_id with a sonar box at both ends of "
                              "each anchor pair.")
    args = parser.parse_args()
    if args.max_frames == 0:
        args.max_frames = None

    scene_id = parse_scene(args.scene)
    scene_dir = resolve_scene_dir(args.person, scene_id)
    labels_dir = scene_dir / "labels"
    sonar_dir = scene_dir / "sonar"
    if not sonar_dir.exists():
        raise SystemExit(f"Sonar folder not found: {sonar_dir}")

    width = frame_width(sonar_dir)

    if args.suggest is not None:
        bin_path = sonar_dir / f"{stem(args.suggest, width)}.bin"
        if not bin_path.exists():
            raise SystemExit(f"No point cloud at {bin_path}")
        candidates = find_candidate_clusters(bin_path)
        if not candidates:
            print(f"[suggest] no candidate found in frame {args.suggest}.")
            return
        for i, c in enumerate(candidates):
            L, W, H = c["dimensions"]
            print(f"[suggest] #{i + 1}: centroid=({c['centroid'][0]:.3f},"
                  f"{c['centroid'][1]:.3f},{c['centroid'][2]:.3f}) "
                  f"size=({L:.2f}x{W:.2f}x{H:.2f}) n_points={c['n_points']} "
                  f"score={c['score']:.2f}")
        out_path = args.suggest_out or Path(f"{scene_id}_suggest_{args.suggest}.png")
        all_points = load_points(bin_path)
        render_candidates(all_points, candidates, out_path)
        print(f"[suggest] preview saved to {out_path} - check against the camera image "
              f"before trusting any of these; this does not write a label.")
        return

    if args.interpolate is not None:
        import stamp_core
        anchors = [int(x) for x in args.interpolate.split(",") if x.strip()]
        link_ids = None
        if args.link_id:
            link_ids = {int(x) for x in args.link_id.split(",") if x.strip()}
        frame_indices = sorted(int(p.stem.split("_")[1]) for p in sonar_dir.glob("frame_*.bin"))
        reports, error = stamp_core.run_interpolate(
            sonar_dir, labels_dir, frame_indices, width, anchors, dry_run=args.dry_run,
            link_ids=link_ids,
        )
        if error:
            print(f"[interpolate] {error}")
            return
        for line in reports:
            print(f"[interpolate] {line}")
        return

    labels_dir.mkdir(exist_ok=True)
    frame_indices = sorted(int(p.stem.split("_")[1]) for p in sonar_dir.glob("frame_*.bin"))
    print(f"Scene: {scene_id}  ({len(frame_indices)} frames, width={width})"
          f"{'  [DRY RUN]' if args.dry_run else ''}")

    if args.modality in ("both", "sonar"):
        if args.stamp:
            run_sonar_stamp(scene_dir, labels_dir, frame_indices, width, args)
        elif args.camera_guided:
            run_sonar_camera_guided(scene_dir, labels_dir, frame_indices, width, args)
        else:
            run_sonar(scene_dir, labels_dir, frame_indices, width, args)
    if args.modality in ("both", "camera") and not args.camera_guided and not args.stamp:
        run_camera(scene_dir, labels_dir, frame_indices, width, args)


if __name__ == "__main__":
    main()
