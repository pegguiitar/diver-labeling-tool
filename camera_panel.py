"""
camera_panel.py - 2D camera annotation panel injected into labelCloud.

Provides click-drag bounding box drawing on camera images, synchronized with
the 3D sonar frame via controller hooks. Labels saved as frame_XXXXXX_camera.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QColor, QImageReader, QPainter, QPen, QBrush, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QListWidget, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

# Keep in sync with labels/_classes.json
CLASS_COLORS = {
    "Unassigned":        QColor(0x9d, 0xa2, 0xab),
    "Fish":              QColor(0x4d, 0xa6, 0xff),
    "Coral":             QColor(0xff, 0x6b, 0x35),
    "Rock":              QColor(0x8b, 0x73, 0x55),
    "Diver":             QColor(0xff, 0x33, 0x66),
    "Structure":         QColor(0x33, 0xcc, 0x33),
    "ROV":               QColor(0xff, 0xe6, 0x00),
    "Calibration Board": QColor(0xcc, 0x00, 0xff),
}


@dataclass
class CameraBox:
    class_name: str
    link_id: int   # 0 = no link
    x1: float      # normalized [0,1]
    y1: float
    x2: float
    y2: float


class ImageCanvas(QWidget):
    """Displays a camera image and lets the user draw 2D bounding boxes."""

    box_committed = QtCore.pyqtSignal()
    box_selected  = QtCore.pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.pixmap:       Optional[QPixmap] = None
        self.boxes:        List[CameraBox]   = []
        self.selected_idx: int               = -1
        self._drawing    = False
        self._drag_start: Optional[QPoint]   = None
        self._drag_cur:   Optional[QPoint]   = None
        self.current_class   = "Unassigned"
        self.current_link_id = 0
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    # ------------------------------------------------------------------ layout
    def _image_rect(self) -> QRect:
        if not self.pixmap:
            return QRect(0, 0, self.width(), self.height())
        iw, ih = self.pixmap.width(), self.pixmap.height()
        scale  = min(self.width() / iw, self.height() / ih)
        w, h   = int(iw * scale), int(ih * scale)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _to_norm(self, pt: QPoint):
        r = self._image_rect()
        x = max(0.0, min(1.0, (pt.x() - r.x()) / r.width()))
        y = max(0.0, min(1.0, (pt.y() - r.y()) / r.height()))
        return x, y

    def _to_px(self, nx: float, ny: float) -> QPoint:
        r = self._image_rect()
        return QPoint(int(r.x() + nx * r.width()), int(r.y() + ny * r.height()))

    # ------------------------------------------------------------------ paint
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(40, 40, 40))

        if self.pixmap:
            p.drawPixmap(self._image_rect(), self.pixmap)
            for i, box in enumerate(self.boxes):
                color    = CLASS_COLORS.get(box.class_name, CLASS_COLORS["Unassigned"])
                selected = (i == self.selected_idx)
                pen = QPen(color, 3 if selected else 2,
                           Qt.DashLine if selected else Qt.SolidLine)
                p.setPen(pen)
                p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 35)))
                tl = self._to_px(min(box.x1, box.x2), min(box.y1, box.y2))
                br = self._to_px(max(box.x1, box.x2), max(box.y1, box.y2))
                p.drawRect(QRect(tl, br))
                tag = box.class_name + (f"  [ID:{box.link_id}]" if box.link_id else "")
                p.setPen(QPen(color))
                p.drawText(tl.x() + 3, tl.y() - 4, tag)

        if self._drawing and self._drag_start and self._drag_cur:
            p.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            p.setBrush(QBrush(QColor(255, 255, 0, 30)))
            p.drawRect(QRect(self._drag_start, self._drag_cur).normalized())

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() != Qt.LeftButton or not self.pixmap:
            return
        nx, ny = self._to_norm(ev.pos())
        for i, box in enumerate(self.boxes):
            if (min(box.x1,box.x2) <= nx <= max(box.x1,box.x2) and
                    min(box.y1,box.y2) <= ny <= max(box.y1,box.y2)):
                self.selected_idx = i
                self.box_selected.emit(i)
                self.update()
                return
        self._drawing    = True
        self._drag_start = ev.pos()
        self._drag_cur   = ev.pos()
        self.selected_idx = -1

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._drawing:
            self._drag_cur = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if not (self._drawing and ev.button() == Qt.LeftButton):
            return
        self._drawing = False
        if self._drag_start and self._drag_cur:
            x1, y1 = self._to_norm(self._drag_start)
            x2, y2 = self._to_norm(self._drag_cur)
            if abs(x2 - x1) > 0.01 and abs(y2 - y1) > 0.01:
                self.boxes.append(CameraBox(
                    class_name=self.current_class,
                    link_id=self.current_link_id,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
                self.selected_idx = len(self.boxes) - 1
                self.box_committed.emit()
        self._drag_start = self._drag_cur = None
        self.update()

    # ------------------------------------------------------------------ public
    def load_image(self, path: Path) -> None:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        img = reader.read()
        self.pixmap = QPixmap.fromImage(img) if not img.isNull() else None
        self.update()

    def clear(self) -> None:
        self.pixmap = None
        self.boxes  = []
        self.selected_idx = -1
        self.update()


class CameraPanel(QWidget):
    """Full camera-annotation panel: image canvas + label list + controls."""

    SUFFIX = "_camera.json"

    def __init__(
        self,
        camera_folder: Path,
        labels_folder: Path,
        classes: List[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.camera_folder = camera_folder
        self.labels_folder = labels_folder
        self.classes       = classes
        self._stem: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(4)

        hdr = QLabel("Camera Labels")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 2px;")
        lay.addWidget(hdr)

        # Class + ID row
        row = QHBoxLayout()
        row.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.classes)
        self.class_combo.currentTextChanged.connect(self._on_class_changed)
        row.addWidget(self.class_combo, 1)
        row.addWidget(QLabel("Link ID:"))
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 99)
        self.id_spin.setSpecialValueText("—")
        self.id_spin.setToolTip("Assign the same ID to a sonar and camera label to link them")
        self.id_spin.valueChanged.connect(self._on_id_changed)
        row.addWidget(self.id_spin)
        lay.addLayout(row)

        # Canvas
        self.canvas = ImageCanvas()
        self.canvas.box_committed.connect(self._refresh_list)
        self.canvas.box_selected.connect(self._on_canvas_select)
        lay.addWidget(self.canvas, 1)

        # Camera label list
        self.cam_list = QListWidget()
        self.cam_list.setMaximumHeight(110)
        self.cam_list.currentRowChanged.connect(self._on_list_select)
        lay.addWidget(self.cam_list)

        # Delete button
        btn_row = QHBoxLayout()
        btn_del = QPushButton("Delete selected")
        btn_del.clicked.connect(self._delete_selected)
        btn_row.addStretch()
        btn_row.addWidget(btn_del)
        lay.addLayout(btn_row)

    # ------------------------------------------------------------------ slots
    def _on_class_changed(self, name: str) -> None:
        self.canvas.current_class = name
        idx = self.canvas.selected_idx
        if 0 <= idx < len(self.canvas.boxes):
            self.canvas.boxes[idx].class_name = name
            self._refresh_list()
            self.canvas.update()

    def _on_id_changed(self, value: int) -> None:
        self.canvas.current_link_id = value
        idx = self.canvas.selected_idx
        if 0 <= idx < len(self.canvas.boxes):
            self.canvas.boxes[idx].link_id = value
            self._refresh_list()
            self.canvas.update()

    def _on_canvas_select(self, idx: int) -> None:
        self.cam_list.setCurrentRow(idx)
        self._sync_controls(idx)

    def _on_list_select(self, idx: int) -> None:
        self.canvas.selected_idx = idx
        self.canvas.update()
        self._sync_controls(idx)

    def _sync_controls(self, idx: int) -> None:
        if 0 <= idx < len(self.canvas.boxes):
            b = self.canvas.boxes[idx]
            self.class_combo.setCurrentText(b.class_name)
            self.id_spin.setValue(b.link_id)

    def _delete_selected(self) -> None:
        idx = self.canvas.selected_idx
        if 0 <= idx < len(self.canvas.boxes):
            self.canvas.boxes.pop(idx)
            self.canvas.selected_idx = -1
            self._refresh_list()
            self.canvas.update()
            self._save()

    def _refresh_list(self) -> None:
        self.cam_list.clear()
        for b in self.canvas.boxes:
            tag = b.class_name + (f"  [ID:{b.link_id}]" if b.link_id else "")
            self.cam_list.addItem(tag)
        if 0 <= self.canvas.selected_idx < len(self.canvas.boxes):
            self.cam_list.setCurrentRow(self.canvas.selected_idx)
        self._save()

    # ------------------------------------------------------------------ I/O
    def update_frame(self, stem: str) -> None:
        """Called by controller whenever the sonar frame changes."""
        if self._stem:
            self._save()
        self._stem = stem
        self._load()

        img_path: Optional[Path] = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            c = self.camera_folder / (stem + ext)
            if c.exists():
                img_path = c
                break
        if img_path:
            self.canvas.load_image(img_path)
        else:
            self.canvas.clear()
            self.cam_list.clear()

    def _save(self) -> None:
        if not self._stem:
            return
        data = [
            {"class": b.class_name, "link_id": b.link_id,
             "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}
            for b in self.canvas.boxes
        ]
        path = self.labels_folder / (self._stem + self.SUFFIX)
        with open(path, "w") as f:
            json.dump({"frame": self._stem, "labels": data}, f, indent=2)

    def _load(self) -> None:
        self.canvas.boxes = []
        self.canvas.selected_idx = -1
        if not self._stem:
            return
        path = self.labels_folder / (self._stem + self.SUFFIX)
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            for item in raw.get("labels", []):
                self.canvas.boxes.append(CameraBox(
                    class_name=item.get("class", "Unassigned"),
                    link_id=item.get("link_id", 0),
                    x1=item["x1"], y1=item["y1"],
                    x2=item["x2"], y2=item["y2"],
                ))
        self._refresh_list()
        self.canvas.update()
