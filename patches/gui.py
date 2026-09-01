import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set

import pkg_resources
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QColorDialog,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMessageBox,
)

from ..control.config_manager import config
from ..definitions import Color3f, LabelingMode
from ..io.labels.config import LabelConfig
from ..io.pointclouds import BasePointCloudHandler
from ..labeling_strategies import PickingStrategy, SpanningStrategy
from ..model.point_cloud import PointCloud
from .settings_dialog import SettingsDialog  # type: ignore
from .startup.dialog import StartupDialog
from .status_manager import StatusManager
from .viewer import GLWidget

if TYPE_CHECKING:
    from ..control.controller import Controller


def string_is_float(string: str, recect_negative: bool = False) -> bool:
    """Returns True if string can be converted to float"""
    try:
        decimal = float(string)
    except ValueError:
        return False
    if recect_negative and decimal < 0:
        return False
    return True


def set_floor_visibility(state: bool) -> None:
    logging.info(
        "%s floor grid (SHOW_FLOOR: %s).",
        "Activated" if state else "Deactivated",
        state,
    )
    config.set("USER_INTERFACE", "show_floor", str(state))


def set_orientation_visibility(state: bool) -> None:
    config.set("USER_INTERFACE", "show_orientation", str(state))


def set_zrotation_only(state: bool) -> None:
    config.set("USER_INTERFACE", "z_rotation_only", str(state))


def set_color_with_label(state: bool) -> None:
    config.set("POINTCLOUD", "color_with_label", str(state))


def set_keep_perspective(state: bool) -> None:
    config.set("USER_INTERFACE", "keep_perspective", str(state))


def set_propagate_labels(state: bool) -> None:
    config.set("LABEL", "propagate_labels", str(state))


# CSS file paths need to be set dynamically
STYLESHEET = """
    * {{
        background-color: #FFF;
        font-family: "DejaVu Sans", Arial;
    }}

    QMenu::item:selected {{
        background-color: #0000DD;
    }}

    QListWidget#label_list::item {{
        padding-left: 22px;
        padding-top: 7px;
        padding-bottom: 7px;
        background: url("{icons_dir}/cube-outline.svg") center left no-repeat;
    }}

    QListWidget#label_list::item:selected {{
        color: #FFF;
        border: none;
        background: rgb(0, 0, 255);
        background: url("{icons_dir}/cube-outline_white.svg") center left no-repeat, #0000ff;
    }}

    QComboBox#current_class_dropdown::item:checked{{
        color: gray;
    }}

    QComboBox#current_class_dropdown::item:selected {{
        color: #FFFFFF;
    }}

    QComboBox#current_class_dropdown{{
        selection-background-color: #0000FF;
    }}
"""


class GUI(QtWidgets.QMainWindow):
    def __init__(self, control: "Controller") -> None:
        super(GUI, self).__init__()
        uic.loadUi(
            pkg_resources.resource_filename(
                "labelCloud.resources.interfaces", "interface.ui"
            ),
            self,
        )
        self.resize(1500, 900)
        self.setWindowTitle("labelCloud")
        self.setStyleSheet(
            STYLESHEET.format(
                icons_dir=str(
                    Path(__file__)
                    .resolve()
                    .parent.parent.joinpath("resources")
                    .joinpath("icons")
                    .as_posix()
                )
            )
        )

        # MENU BAR
        # File
        self.act_set_pcd_folder: QtWidgets.QAction
        self.act_set_label_folder: QtWidgets.QAction

        # Labels
        self.act_delete_all_labels: QtWidgets.QAction
        self.act_set_default_class: QtWidgets.QMenu
        self.actiongroup_default_class = QActionGroup(self.act_set_default_class)
        self.act_propagate_labels: QtWidgets.QAction

        # Settings
        self.act_z_rotation_only: QtWidgets.QAction
        self.act_color_with_label: QtWidgets.QAction
        self.act_show_floor: QtWidgets.QAction
        self.act_show_orientation: QtWidgets.QAction
        self.act_save_perspective: QtWidgets.QAction
        self.act_align_pcd: QtWidgets.QAction
        self.act_change_settings: QtWidgets.QAction

        # STATUS BAR
        self.status_bar: QtWidgets.QStatusBar
        self.status_manager = StatusManager(self.status_bar)

        # CENTRAL WIDGET
        self.gl_widget: GLWidget

        # LEFT PANEL
        # point cloud management
        self.label_current_pcd: QtWidgets.QLabel
        self.button_prev_pcd: QtWidgets.QPushButton
        self.button_next_pcd: QtWidgets.QPushButton
        self.button_set_pcd: QtWidgets.QPushButton
        self.progressbar_pcds: QtWidgets.QProgressBar

        # bbox control section
        self.button_bbox_up: QtWidgets.QPushButton
        self.button_bbox_down: QtWidgets.QPushButton
        self.button_bbox_left: QtWidgets.QPushButton
        self.button_bbox_right: QtWidgets.QPushButton
        self.button_bbox_forward: QtWidgets.QPushButton
        self.button_bbox_backward: QtWidgets.QPushButton
        self.dial_bbox_z_rotation: QtWidgets.QDial
        self.button_bbox_decrease_dimension: QtWidgets.QPushButton
        self.button_bbox_increase_dimension: QtWidgets.QPushButton

        # 2d image viewer
        self.button_show_image: QtWidgets.QPushButton
        self.button_show_image.setVisible(
            config.getboolean("USER_INTERFACE", "show_2d_image")
        )

        # label mode selection
        self.button_pick_bbox: QtWidgets.QPushButton
        self.button_span_bbox: QtWidgets.QPushButton
        self.button_save_label: QtWidgets.QPushButton

        # RIGHT PANEL
        self.label_list: QtWidgets.QListWidget
        self.current_class_dropdown: QtWidgets.QComboBox
        self.button_deselect_label: QtWidgets.QPushButton
        self.button_delete_label: QtWidgets.QPushButton
        self.button_assign_label: QtWidgets.QPushButton

        # label list actions
        # self.act_rename_class = QtWidgets.QAction("Rename class") #TODO: Implement!
        self.act_change_class_color = QtWidgets.QAction("Change class color")
        self.act_delete_class = QtWidgets.QAction("Delete label")
        self.act_crop_pointcloud_inside = QtWidgets.QAction("Save points inside as")
        self.label_list.addActions(
            [
                self.act_change_class_color,
                self.act_delete_class,
                self.act_crop_pointcloud_inside,
            ]
        )
        self.label_list.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        # BOUNDING BOX PARAMETER EDITS
        self.edit_pos_x: QtWidgets.QLineEdit
        self.edit_pos_y: QtWidgets.QLineEdit
        self.edit_pos_z: QtWidgets.QLineEdit

        self.edit_length: QtWidgets.QLineEdit
        self.edit_width: QtWidgets.QLineEdit
        self.edit_height: QtWidgets.QLineEdit

        self.edit_rot_x: QtWidgets.QLineEdit
        self.edit_rot_y: QtWidgets.QLineEdit
        self.edit_rot_z: QtWidgets.QLineEdit

        self.all_line_edits = [
            self.edit_pos_x,
            self.edit_pos_y,
            self.edit_pos_z,
            self.edit_length,
            self.edit_width,
            self.edit_height,
            self.edit_rot_x,
            self.edit_rot_y,
            self.edit_rot_z,
        ]

        self.label_volume: QtWidgets.QLabel

        self.controller = control

        # Connect all events to functions
        self.connect_events()
        self.set_checkbox_states()  # tick in menu

        # Run startup dialog
        self.startup_dialog = StartupDialog()
        if self.startup_dialog.exec():
            pass
        else:
            sys.exit()
        # Segmentation only functionalities
        if LabelConfig().type == LabelingMode.OBJECT_DETECTION:
            self.button_assign_label.setVisible(False)
            self.act_color_with_label.setVisible(False)

        # Connect with controller
        self.controller.startup(self)

        # Inject camera panel and sonar link-ID controls
        self._inject_camera_panel()

        # Start event cycle
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(20)  # period, in milliseconds
        self.timer.timeout.connect(self.controller.loop_gui)
        self.timer.start()

    # ------------------------------------------------------------------ camera
    def _inject_camera_panel(self) -> None:
        import sys, os, json as _json
        sys.path.insert(0, os.getcwd())
        from camera_panel import CameraPanel

        # Rename the sonar section header
        self.labels_label.setText("Sonar Labels")

        # Add Link ID row for sonar labels under the bbox params grid
        sonar_id_row = QtWidgets.QHBoxLayout()
        sonar_id_row.addWidget(QLabel("Sonar Link ID:"))
        self.sonar_id_spin = QtWidgets.QSpinBox()
        self.sonar_id_spin.setRange(0, 99)
        self.sonar_id_spin.setSpecialValueText("—")
        self.sonar_id_spin.setToolTip("Assign matching IDs to link a sonar box with a camera box")
        self.sonar_id_spin.valueChanged.connect(self._on_sonar_id_changed)
        sonar_id_row.addWidget(self.sonar_id_spin)
        sonar_id_row.addStretch()
        # Insert into the right-panel frame layout (verticalLayout_4)
        self.verticalLayout_4.addLayout(sonar_id_row)

        # Show how many point-cloud points fall inside the active box
        point_count_row = QtWidgets.QHBoxLayout()
        point_count_row.addWidget(QLabel("Points inside:"))
        self.label_point_count = QLabel("—")
        self.label_point_count.setToolTip(
            "Number of point-cloud points that fall inside the selected box"
        )
        point_count_row.addWidget(self.label_point_count)
        point_count_row.addStretch()
        self.verticalLayout_4.addLayout(point_count_row)

        # Stamp Sonar from Camera: copy the current frame's sonar box as-is
        # into following frames for as long as the camera box (ground truth)
        # still overlaps where it started, stopping (and popping up) the
        # instant it drifts too far or hits already-labeled territory.
        stamp_row = QtWidgets.QHBoxLayout()
        self.button_stamp_sonar = QtWidgets.QPushButton("Stamp Sonar from Camera →")
        self.button_stamp_sonar.setToolTip(
            "Saves this frame, then copies its sonar box unchanged into every "
            "following frame whose camera box still overlaps this one "
            "(IoU ≥ 0.4), stopping the instant it doesn't (that stopping "
            "frame still gets a copy too, as a starting point to fix)."
        )
        self.button_stamp_sonar.clicked.connect(self._on_stamp_sonar_clicked)
        stamp_row.addWidget(self.button_stamp_sonar)
        self.verticalLayout_4.addLayout(stamp_row)

        # Re-stamp (overwrite): same thing, but for re-seeding partway
        # through a stretch that was already stamped once and has drifted -
        # replaces existing sonar content for this link_id going forward
        # instead of stopping at the first already-labeled frame.
        restamp_row = QtWidgets.QHBoxLayout()
        self.button_restamp_sonar = QtWidgets.QPushButton("Re-stamp Sonar (overwrite) →")
        self.button_restamp_sonar.setStyleSheet("color: #b30000;")
        self.button_restamp_sonar.setToolTip(
            "Like Stamp Sonar from Camera, but REPLACES existing sonar boxes "
            "for this link_id going forward instead of stopping at the first "
            "one it meets. Use when you've re-seeded in the middle of an "
            "already-stamped stretch because it had drifted."
        )
        self.button_restamp_sonar.clicked.connect(
            lambda: self._on_stamp_sonar_clicked(overwrite=True)
        )
        restamp_row.addWidget(self.button_restamp_sonar)
        self.verticalLayout_4.addLayout(restamp_row)

        # Fill Between Sonar Anchors: type in several already-labeled anchor
        # frames, interpolate (camera-motion-weighted) + snap between each
        # consecutive pair, overwriting whatever's currently there.
        fill_row = QtWidgets.QHBoxLayout()
        self.button_fill_anchors = QtWidgets.QPushButton("Fill Between Sonar Anchors…")
        self.button_fill_anchors.setToolTip(
            "Type in the frame numbers of several already-labeled sonar boxes "
            "(same link_id). Fills every frame strictly between each "
            "consecutive pair by interpolating position (weighted by how far "
            "the camera box moved) and rotation, then snapping lightly to "
            "nearby points - overwriting whatever's currently there. You'll "
            "then be asked which link_id(s) to fill, so divers with "
            "different anchor points can be done one at a time."
        )
        self.button_fill_anchors.clicked.connect(self._on_fill_anchors_clicked)
        fill_row.addWidget(self.button_fill_anchors)
        self.verticalLayout_4.addLayout(fill_row)

        # Build and attach the camera panel below label_list
        cam_folder    = Path(config.get("FILE", "image_folder"))
        labels_folder = Path(config.get("FILE", "label_folder"))
        classes       = list(LabelConfig().get_classes())
        self.camera_panel = CameraPanel(cam_folder, labels_folder, classes)
        self.camera_panel.setMinimumHeight(300)
        self.verticalLayout_3.addWidget(self.camera_panel, stretch=2)

        # Sonar ID storage: {bbox_index: link_id}, keyed by frame stem
        self._sonar_ids: dict = {}
        self._sonar_id_stem: str = ""

        # Sync camera panel and sonar IDs to the frame already loaded at startup
        if self.controller.pcd_manager.pcd_path:
            stem = self.controller.pcd_manager.pcd_path.stem
            self._sonar_id_stem = stem
            self._load_sonar_ids()
            self.camera_panel.update_frame(stem)

        # Resize window to give the camera panel room
        self.resize(1500, 1150)

    def _on_sonar_list_row_changed(self, idx: int) -> None:
        if hasattr(self, 'sonar_id_spin') and hasattr(self, '_sonar_ids'):
            self.sonar_id_spin.setValue(self._sonar_ids.get(idx, 0))

    def _on_sonar_id_changed(self, value: int) -> None:
        idx = self.label_list.currentRow()
        if idx < 0 or not hasattr(self, '_sonar_ids'):
            return
        self._sonar_ids[idx] = value
        self._save_sonar_ids()

    def _on_stamp_sonar_clicked(self, overwrite: bool = False) -> None:
        import sys, os
        sys.path.insert(0, os.getcwd())
        import stamp_core

        title = "Re-stamp Sonar (overwrite)" if overwrite else "Stamp Sonar from Camera"
        self.controller.save()  # flush the seed box (and its link id) to disk first

        sonar_dir = Path(config.get("FILE", "pointcloud_folder"))
        labels_dir = Path(config.get("FILE", "label_folder"))
        pcd_path = self.controller.pcd_manager.pcd_path
        if pcd_path is None:
            return
        try:
            width = stamp_core.frame_width(sonar_dir)
            seed_idx = int(pcd_path.stem.split("_")[1])
            frame_indices = stamp_core.frame_indices_of(sonar_dir)
            targets, error = stamp_core.run_stamp(
                sonar_dir, labels_dir, frame_indices, width, seed_idx,
                min_iou=0.4, overwrite=overwrite,
            )
        except Exception as e:
            QMessageBox.critical(self, title, f"Failed: {e.__class__.__name__}: {e}")
            return

        if error:
            QMessageBox.warning(self, title, error)
            return

        lines = [f"Seed frame: {seed_idx}"]
        for t in targets:
            if t["stop_frame"] is not None:
                lines.append(f"link_id={t['link_id']}: wrote {t['n_written']} frame(s), "
                             f"stopped at frame {t['stop_frame']}\n  ({t['stop_reason']})")
            else:
                lines.append(f"link_id={t['link_id']}: wrote {t['n_written']} frame(s), "
                             f"reached the end without stopping")
        QMessageBox.information(self, title, "\n".join(lines))

    def _on_fill_anchors_clicked(self) -> None:
        import sys, os
        sys.path.insert(0, os.getcwd())
        import stamp_core

        self.controller.save()  # flush the current frame, in case it's meant to be an anchor

        text, ok = QInputDialog.getText(
            self, "Fill Between Sonar Anchors",
            "Already-labeled anchor frame numbers, comma-separated (need at least 2), "
            "e.g. 45,120,300,450:",
        )
        if not ok or not text.strip():
            return
        try:
            anchors = [int(x.strip()) for x in text.split(",") if x.strip()]
        except ValueError:
            QMessageBox.warning(self, "Fill Between Sonar Anchors",
                                 "Couldn't parse that as a comma-separated list of frame numbers.")
            return

        id_text, ok = QInputDialog.getText(
            self, "Fill Between Sonar Anchors",
            "Link ID(s) to fill, comma-separated (blank = every diver present "
            "at all the anchors above):",
        )
        if not ok:
            return
        link_ids = None
        if id_text.strip():
            try:
                link_ids = {int(x.strip()) for x in id_text.split(",") if x.strip()}
            except ValueError:
                QMessageBox.warning(self, "Fill Between Sonar Anchors",
                                     "Couldn't parse that as a comma-separated list of link IDs.")
                return

        sonar_dir = Path(config.get("FILE", "pointcloud_folder"))
        labels_dir = Path(config.get("FILE", "label_folder"))
        try:
            width = stamp_core.frame_width(sonar_dir)
            frame_indices = stamp_core.frame_indices_of(sonar_dir)
            reports, error = stamp_core.run_interpolate(
                sonar_dir, labels_dir, frame_indices, width, anchors, link_ids=link_ids
            )
        except Exception as e:
            QMessageBox.critical(self, "Fill Between Sonar Anchors",
                                  f"Failed: {e.__class__.__name__}: {e}")
            return

        if error:
            QMessageBox.warning(self, "Fill Between Sonar Anchors", error)
            return
        QMessageBox.information(self, "Fill Between Sonar Anchors", "\n".join(reports))

    def notify_frame_changed(self, stem: str) -> None:
        """Called by controller whenever the active sonar frame changes."""
        if hasattr(self, 'camera_panel'):
            self.camera_panel.update_frame(stem)
        if hasattr(self, '_sonar_ids'):
            self._save_sonar_ids()
            self._sonar_ids = {}
            self._sonar_id_stem = stem
            self._load_sonar_ids()

    def _save_sonar_ids(self) -> None:
        if not getattr(self, '_sonar_id_stem', None) or not self._sonar_ids:
            return
        labels_folder = Path(config.get("FILE", "label_folder"))
        path = labels_folder / (self._sonar_id_stem + "_ids.json")
        with open(path, "w") as f:
            import json as _json
            _json.dump(self._sonar_ids, f)

    def _load_sonar_ids(self) -> None:
        if not getattr(self, '_sonar_id_stem', None):
            return
        labels_folder = Path(config.get("FILE", "label_folder"))
        path = labels_folder / (self._sonar_id_stem + "_ids.json")
        if path.exists():
            import json as _json
            with open(path) as f:
                raw = _json.load(f)
            self._sonar_ids = {int(k): v for k, v in raw.items()}
        # Refresh spinbox for current selection
        idx = self.label_list.currentRow()
        if hasattr(self, 'sonar_id_spin'):
            self.sonar_id_spin.setValue(self._sonar_ids.get(idx, 0))

    # Event connectors
    def connect_events(self) -> None:
        # POINTCLOUD CONTROL
        self.button_next_pcd.clicked.connect(
            lambda: self.controller.next_pcd(save=True)
        )
        self.button_prev_pcd.clicked.connect(self.controller.prev_pcd)

        # BBOX CONTROL
        self.button_bbox_up.pressed.connect(
            lambda: self.controller.bbox_controller.translate_along_z()
        )
        self.button_bbox_down.pressed.connect(
            lambda: self.controller.bbox_controller.translate_along_z(down=True)
        )
        self.button_bbox_left.pressed.connect(
            lambda: self.controller.bbox_controller.translate_along_x(left=True)
        )
        self.button_bbox_right.pressed.connect(
            self.controller.bbox_controller.translate_along_x
        )
        self.button_bbox_forward.pressed.connect(
            lambda: self.controller.bbox_controller.translate_along_y(forward=True)
        )
        self.button_set_pcd.pressed.connect(lambda: self.ask_custom_index())
        self.button_bbox_backward.pressed.connect(
            lambda: self.controller.bbox_controller.translate_along_y()
        )

        self.dial_bbox_z_rotation.valueChanged.connect(
            lambda x: self.controller.bbox_controller.rotate_around_z(x, absolute=True)
        )
        self.button_bbox_decrease_dimension.clicked.connect(
            lambda: self.controller.bbox_controller.scale(decrease=True)
        )
        self.button_bbox_increase_dimension.clicked.connect(
            lambda: self.controller.bbox_controller.scale()
        )

        # LABELING CONTROL
        self.current_class_dropdown.currentTextChanged.connect(
            self.controller.bbox_controller.set_classname
        )
        self.button_deselect_label.clicked.connect(
            self.controller.bbox_controller.deselect_bbox
        )
        self.button_delete_label.clicked.connect(
            self.controller.bbox_controller.delete_current_bbox
        )
        self.label_list.currentRowChanged.connect(
            self.controller.bbox_controller.set_active_bbox
        )
        self.label_list.currentRowChanged.connect(self._on_sonar_list_row_changed)
        self.button_assign_label.clicked.connect(
            self.controller.bbox_controller.assign_point_label_in_active_box
        )
        # context menu
        self.act_delete_class.triggered.connect(
            self.controller.bbox_controller.delete_current_bbox
        )
        self.act_crop_pointcloud_inside.triggered.connect(
            self.controller.crop_pointcloud_inside_active_bbox
        )
        self.act_change_class_color.triggered.connect(self.change_label_color)

        # open_2D_img
        self.button_show_image.pressed.connect(lambda: self.show_2d_image())

        # LABEL CONTROL
        self.button_pick_bbox.clicked.connect(
            lambda: self.controller.drawing_mode.set_drawing_strategy(
                PickingStrategy(self)
            )
        )
        self.button_span_bbox.clicked.connect(
            lambda: self.controller.drawing_mode.set_drawing_strategy(
                SpanningStrategy(self)
            )
        )
        self.button_save_label.clicked.connect(self.controller.save)

        # BOUNDING BOX PARAMETER
        self.edit_pos_x.editingFinished.connect(
            lambda: self.update_bbox_parameter("pos_x")
        )
        self.edit_pos_y.editingFinished.connect(
            lambda: self.update_bbox_parameter("pos_y")
        )
        self.edit_pos_z.editingFinished.connect(
            lambda: self.update_bbox_parameter("pos_z")
        )

        self.edit_length.editingFinished.connect(
            lambda: self.update_bbox_parameter("length")
        )
        self.edit_width.editingFinished.connect(
            lambda: self.update_bbox_parameter("width")
        )
        self.edit_height.editingFinished.connect(
            lambda: self.update_bbox_parameter("height")
        )

        self.edit_rot_x.editingFinished.connect(
            lambda: self.update_bbox_parameter("rot_x")
        )
        self.edit_rot_y.editingFinished.connect(
            lambda: self.update_bbox_parameter("rot_y")
        )
        self.edit_rot_z.editingFinished.connect(
            lambda: self.update_bbox_parameter("rot_z")
        )

        # MENU BAR
        self.act_set_pcd_folder.triggered.connect(self.change_pointcloud_folder)
        self.act_set_label_folder.triggered.connect(self.change_label_folder)
        self.actiongroup_default_class.triggered.connect(
            self.change_default_object_class
        )
        self.act_delete_all_labels.triggered.connect(
            self.controller.bbox_controller.reset
        )
        self.act_propagate_labels.toggled.connect(set_propagate_labels)
        self.act_z_rotation_only.toggled.connect(set_zrotation_only)
        self.act_color_with_label.toggled.connect(set_color_with_label)
        self.act_show_floor.toggled.connect(set_floor_visibility)
        self.act_show_orientation.toggled.connect(set_orientation_visibility)
        self.act_save_perspective.toggled.connect(set_keep_perspective)
        self.act_align_pcd.toggled.connect(self.controller.align_mode.change_activation)
        self.act_change_settings.triggered.connect(self.show_settings_dialog)

    def set_checkbox_states(self) -> None:
        self.act_propagate_labels.setChecked(
            config.getboolean("LABEL", "propagate_labels")
        )
        self.act_show_floor.setChecked(
            config.getboolean("USER_INTERFACE", "show_floor")
        )
        self.act_show_orientation.setChecked(
            config.getboolean("USER_INTERFACE", "show_orientation")
        )
        self.act_z_rotation_only.setChecked(
            config.getboolean("USER_INTERFACE", "z_rotation_only")
        )
        self.act_color_with_label.setChecked(
            config.getboolean("POINTCLOUD", "color_with_label")
        )

    # Collect, filter and forward events to viewer
    def eventFilter(self, event_object, event) -> bool:
        # Keyboard Events
        if event.type() == QEvent.KeyPress:
            # Left/Right = prev/next frame is a labelCloud default (see
            # controller.key_press_event), but it only fired when focus was
            # on the main window or label_list - the camera panel and its
            # widgets (added by patches) silently ate the key otherwise.
            # Make it work regardless of focus, except while actually typing
            # in a field, where left/right should move the cursor instead.
            is_nav_key = event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right)
            editing_widget = isinstance(
                self.focusWidget(),
                (QtWidgets.QLineEdit, QtWidgets.QSpinBox, QtWidgets.QComboBox),
            )
            if event_object in [self, self.label_list] or (
                is_nav_key and not editing_widget
            ):
                self.controller.key_press_event(event)
                self.update_bbox_stats(self.controller.bbox_controller.get_active_bbox())
                return True  # TODO: Recheck pyqt behaviour
        elif event.type() == QEvent.KeyRelease:
            self.controller.key_release_event(event)

        # Mouse Events
        elif (event.type() == QEvent.MouseMove) and (event_object == self.gl_widget):
            self.controller.mouse_move_event(event)
            self.update_bbox_stats(self.controller.bbox_controller.get_active_bbox())
        elif (event.type() == QEvent.Wheel) and (event_object == self.gl_widget):
            self.controller.mouse_scroll_event(event)
            self.update_bbox_stats(self.controller.bbox_controller.get_active_bbox())
        elif event.type() == QEvent.MouseButtonDblClick and (
            event_object == self.gl_widget
        ):
            self.controller.mouse_double_clicked(event)
            return True
        elif (event.type() == QEvent.MouseButtonPress) and (
            event_object == self.gl_widget
        ):
            self.controller.mouse_clicked(event)
            self.update_bbox_stats(self.controller.bbox_controller.get_active_bbox())
        elif (event.type() == QEvent.MouseButtonPress) and (
            event_object != self.current_class_dropdown
        ):
            self.current_class_dropdown.clearFocus()
            self.update_bbox_stats(self.controller.bbox_controller.get_active_bbox())
        return False

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        logging.info("Closing window after saving ...")
        self.controller.save()
        self.timer.stop()
        a0.accept()

    def show_settings_dialog(self) -> None:
        dialog = SettingsDialog(self)
        dialog.exec()

    def show_2d_image(self):
        """Searches for a 2D image with the point cloud name and displays it in a new window."""
        image_folder = config.getpath("FILE", "image_folder")

        # Look for image files with the name of the point cloud
        pcd_name = self.controller.pcd_manager.pcd_path.stem
        image_file_pattern = re.compile(
            f"{pcd_name}+(\\.(?i:(jpe?g|png|gif|bmp|tiff)))"
        )

        try:
            image_name = next(
                filter(image_file_pattern.search, os.listdir(image_folder))
            )
        except StopIteration:
            QMessageBox.information(
                self,
                "No 2D Image File",
                (
                    f"Could not find a related image in the image folder ({image_folder}).\n"
                    "Check your path to the folder or if an image for this point cloud exists."
                ),
                QMessageBox.Ok,
            )
        else:
            image_path = image_folder.joinpath(image_name)
            image = QtGui.QImage(QtGui.QImageReader(str(image_path)).read())
            self.imageLabel = QLabel()
            self.imageLabel.setWindowTitle(f"2D Image ({image_name})")
            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.imageLabel.show()

    def show_no_pointcloud_dialog(
        self, pcd_folder: Path, pcd_extensions: Set[str]
    ) -> None:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setText(
            "<b>labelCloud could not find any valid point cloud files inside the "
            "specified folder.</b>"
        )
        msg.setInformativeText(
            f"Please copy all your point clouds into <code>{pcd_folder.resolve()}</code> or update "
            "the point cloud folder location. labelCloud supports the following point "
            f"cloud file formats:\n {', '.join(pcd_extensions)}."
        )
        msg.setWindowTitle("No Point Clouds Found")
        msg.exec_()

    # VISUALIZATION METHODS

    def set_pcd_label(self, pcd_name: str) -> None:
        self.label_current_pcd.setText("Current: <em>%s</em>" % pcd_name)

    def init_progress(self, min_value, max_value):
        self.progressbar_pcds.setMinimum(min_value)
        self.progressbar_pcds.setMaximum(max_value)

    def update_progress(self, value) -> None:
        self.progressbar_pcds.setValue(value)

    def update_current_class_dropdown(self) -> None:
        self.controller.pcd_manager.populate_class_dropdown()

    def update_bbox_stats(self, bbox) -> None:
        viewing_precision = config.getint("USER_INTERFACE", "viewing_precision")

        # Number of point-cloud points inside the selected box (read-only, so
        # updated even while a parameter field is being edited).
        if hasattr(self, "label_point_count"):
            pcd = self.controller.pcd_manager.pointcloud
            if bbox and pcd is not None and pcd.points is not None:
                self.label_point_count.setText(str(int(bbox.is_inside(pcd.points).sum())))
            else:
                self.label_point_count.setText("—")

        if bbox and not self.line_edited_activated():
            self.edit_pos_x.setText(str(round(bbox.get_center()[0], viewing_precision)))
            self.edit_pos_y.setText(str(round(bbox.get_center()[1], viewing_precision)))
            self.edit_pos_z.setText(str(round(bbox.get_center()[2], viewing_precision)))

            self.edit_length.setText(
                str(round(bbox.get_dimensions()[0], viewing_precision))
            )
            self.edit_width.setText(
                str(round(bbox.get_dimensions()[1], viewing_precision))
            )
            self.edit_height.setText(
                str(round(bbox.get_dimensions()[2], viewing_precision))
            )

            self.edit_rot_x.setText(str(round(bbox.get_x_rotation(), 1)))
            self.edit_rot_y.setText(str(round(bbox.get_y_rotation(), 1)))
            self.edit_rot_z.setText(str(round(bbox.get_z_rotation(), 1)))

            self.label_volume.setText(str(round(bbox.get_volume(), viewing_precision)))

    def update_bbox_parameter(self, parameter: str) -> None:
        str_value = None
        self.setFocus()  # Changes the focus from QLineEdit to the window

        if parameter == "pos_x":
            str_value = self.edit_pos_x.text()
        if parameter == "pos_y":
            str_value = self.edit_pos_y.text()
        if parameter == "pos_z":
            str_value = self.edit_pos_z.text()
        if str_value and string_is_float(str_value):
            self.controller.bbox_controller.update_position(parameter, float(str_value))
            return

        if parameter == "length":
            str_value = self.edit_length.text()
        if parameter == "width":
            str_value = self.edit_width.text()
        if parameter == "height":
            str_value = self.edit_height.text()
        if str_value and string_is_float(str_value, recect_negative=True):
            self.controller.bbox_controller.update_dimension(
                parameter, float(str_value)
            )
            return

        if parameter == "rot_x":
            str_value = self.edit_rot_x.text()
        if parameter == "rot_y":
            str_value = self.edit_rot_y.text()
        if parameter == "rot_z":
            str_value = self.edit_rot_z.text()
        if str_value and string_is_float(str_value):
            self.controller.bbox_controller.update_rotation(parameter, float(str_value))
            return

    # Enables, disables the draw mode
    def activate_draw_modes(self, state: bool) -> None:
        self.button_pick_bbox.setEnabled(state)
        self.button_span_bbox.setEnabled(state)

    def line_edited_activated(self) -> bool:
        for line_edit in self.all_line_edits:
            if line_edit.hasFocus():
                return True
        return False

    def change_pointcloud_folder(self) -> None:
        path_to_folder = Path(
            QFileDialog.getExistingDirectory(
                self,
                "Change Point Cloud Folder",
                directory=config.get("FILE", "pointcloud_folder"),
            )
        )
        if not path_to_folder.is_dir():
            logging.warning("Please specify a valid folder path.")
        else:
            self.controller.pcd_manager.pcd_folder = path_to_folder
            self.controller.pcd_manager.read_pointcloud_folder()
            self.controller.pcd_manager.get_next_pcd()
            logging.info("Changed point cloud folder to %s!" % path_to_folder)

    def change_label_folder(self) -> None:
        path_to_folder = Path(
            QFileDialog.getExistingDirectory(
                self,
                "Change Label Folder",
                directory=config.get("FILE", "label_folder"),
            )
        )
        if not path_to_folder.is_dir():
            logging.warning("Please specify a valid folder path.")
        else:
            self.controller.pcd_manager.label_manager.label_folder = path_to_folder
            self.controller.pcd_manager.label_manager.label_strategy.update_label_folder(
                path_to_folder
            )
            logging.info("Changed label folder to %s!" % path_to_folder)

    def update_default_object_class_menu(
        self, new_classes: Optional[Set[str]] = None
    ) -> None:
        object_classes = set(LabelConfig().get_classes())

        object_classes.update(new_classes or [])
        existing_classes = {
            action.text() for action in self.actiongroup_default_class.actions()
        }
        for object_class in object_classes.difference(existing_classes):
            action = self.actiongroup_default_class.addAction(
                object_class
            )  # TODO: Add limiter for number of classes
            action.setCheckable(True)
            if object_class == LabelConfig().get_default_class_name():
                action.setChecked(True)

        self.act_set_default_class.addActions(self.actiongroup_default_class.actions())

    def change_default_object_class(self, action: QAction) -> None:
        LabelConfig().set_default_class(action.text())
        logging.info("Changed default object class to %s.", action.text())

    def ask_custom_index(self):
        input_d = QInputDialog(self)
        self.input_pcd = input_d
        input_d.setInputMode(QInputDialog.IntInput)
        input_d.setWindowTitle("labelCloud")
        input_d.setLabelText("Insert Point Cloud number: ()")
        input_d.setIntMaximum(len(self.controller.pcd_manager.pcds) - 1)
        input_d.intValueChanged.connect(lambda val: self.update_dialog_pcd(val))
        input_d.intValueSelected.connect(lambda val: self.controller.custom_pcd(val))
        input_d.open()
        self.update_dialog_pcd(0)

    def update_dialog_pcd(self, value: int) -> None:
        pcd_path = self.controller.pcd_manager.pcds[value]
        self.input_pcd.setLabelText(f"Insert Point Cloud number: {pcd_path.name}")

    def change_label_color(self):
        bbox = self.controller.bbox_controller.get_active_bbox()
        LabelConfig().set_class_color(
            bbox.classname, Color3f.from_qcolor(QColorDialog.getColor())
        )

    @staticmethod
    def save_point_cloud_as(pointcloud: PointCloud) -> None:
        extensions = BasePointCloudHandler.get_supported_extensions()
        make_filter = " ".join(["*" + extension for extension in extensions])
        file_filter = f"Point Cloud File ({make_filter})"
        file_name, _ = QFileDialog.getSaveFileName(
            caption="Select a file name to save the point cloud",
            directory=str(pointcloud.path.parent),
            filter=file_filter,
            initialFilter=file_filter,
        )
        if file_name == "":
            logging.warning("No file path provided. Ignored.")
            return

        try:
            path = Path(file_name)
            handler = BasePointCloudHandler.get_handler(path.suffix)
            handler.write_point_cloud(path, pointcloud)
        except Exception as e:
            msg = QMessageBox()
            msg.setWindowTitle("Failed to save a point cloud")
            msg.setText(e.__class__.__name__)
            msg.setInformativeText(traceback.format_exc())
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Cancel)
            msg.exec_()
