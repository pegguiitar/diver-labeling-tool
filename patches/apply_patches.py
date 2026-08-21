"""
apply_patches.py - Copy patched labelCloud files into the active conda environment.

Run once after installing labelCloud:
    python patches/apply_patches.py
"""

import shutil
import sys
from pathlib import Path

try:
    import labelCloud
except ImportError:
    sys.exit("labelCloud is not installed in this Python environment. Run: pip install labelCloud")

pkg = Path(labelCloud.__file__).parent
patches = Path(__file__).parent

files = {
    "numpy_handler.py": pkg / "io"      / "pointclouds" / "numpy.py",
    "gui.py":           pkg / "view"     / "gui.py",
    "controller.py":    pkg / "control"  / "controller.py",
    "centroid.py":      pkg / "io"       / "labels" / "centroid.py",
}

for src_name, dst in files.items():
    src = patches / src_name
    shutil.copy2(src, dst)
    print(f"  patched {dst.relative_to(pkg.parent)}")

print("Done.")
