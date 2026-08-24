import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import Qt as Keys

from ..definitions import BBOX_SIDES, Colors, Context, LabelingMode
from ..io.labels.config import LabelConfig
from ..utils import oglhelper
from ..view.gui import GUI
from .alignmode import AlignMode
from .bbox_controller import BoundingBoxController
from .config_manager import config
from .drawing_manager import DrawingManager
from .pcd_manager import PointCloudManger


# --------------------------------------------------------------------------- #
# Gimbal-lock-safe rotation helpers.
#
# labelCloud stores an orientation as three Euler angles (x, y, z) that are
# applied, at render/export time, in the order Z -> Y -> X (see BBox.draw_bbox:
# glRotate(z,Z); glRotate(y,Y); glRotate(x,X)). Incrementing one Euler component
# in isolation is not a true rotation about a fixed axis once the other two are
# non-zero, and it degenerates completely at y = +/-90 deg (gimbal lock).
#
# To stay well-behaved we instead compose the increment as a real rotation
# matrix onto the box's current orientation (body frame) and decompose the
# result back into the same Z-Y-X Euler triple, using a singularity-robust
# decomposition. Interaction never locks up; only the stored representation is
# ambiguous exactly at the poles, and that is resolved deterministically.
# --------------------------------------------------------------------------- #
def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _euler_zyx_to_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """Build R = Rz(z) @ Ry(y) @ Rx(x) from degrees (labelCloud's render order)."""
    return (
        _rot_z(np.deg2rad(z_deg))
        @ _rot_y(np.deg2rad(y_deg))
        @ _rot_x(np.deg2rad(x_deg))
    )


def _matrix_to_euler_zyx(R: np.ndarray) -> tuple:
    """Decompose R = Rz(z) @ Ry(y) @ Rx(x) back to (x, y, z) degrees, pole-safe."""
    sy = min(1.0, max(-1.0, -R[2, 0]))
    cy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if cy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(sy, cy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock (y = +/-90): fix yaw at 0 and fold it into roll.
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(sy, cy)
        z = 0.0
    return np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)


class Controller:
    MOVEMENT_THRESHOLD = 0.1

    def __init__(self) -> None:
        """Initializes all controllers and managers."""
        self.view: "GUI"
        self.pcd_manager = PointCloudManger()
        self.bbox_controller = BoundingBoxController()

        # Drawing states
        self.drawing_mode = DrawingManager(self.bbox_controller)
        self.align_mode = AlignMode(self.pcd_manager)

        # Control states
        self.curr_cursor_pos: Optional[QPoint] = None  # updated by mouse movement
        self.last_cursor_pos: Optional[QPoint] = None  # updated by mouse click
        self.ctrl_pressed = False
        self.scroll_mode = False  # to enable the side-pulling

        # Correction states
        self.side_mode = False
        self.selected_side: Optional[str] = None

    def startup(self, view: "GUI") -> None:
        """Sets the view in all controllers and dependent modules; Loads labels from file."""
        self.view = view
        self.bbox_controller.set_view(self.view)
        self.pcd_manager.set_view(self.view)
        self.drawing_mode.set_view(self.view)
        self.align_mode.set_view(self.view)
        self.view.gl_widget.set_bbox_controller(self.bbox_controller)
        self.bbox_controller.pcd_manager = self.pcd_manager

        # Read labels from folders
        self.pcd_manager.read_pointcloud_folder()
        self.next_pcd(save=False)

    def loop_gui(self) -> None:
        """Function collection called during each event loop iteration."""
        self.set_crosshair()
        self.set_selected_side()
        self.view.gl_widget.updateGL()

    # POINT CLOUD METHODS
    def next_pcd(self, save: bool = True) -> None:
        if save:
            self.save()
        if self.pcd_manager.pcds_left():
            previous_bboxes = self.bbox_controller.bboxes
            self.pcd_manager.get_next_pcd()
            self.reset()
            self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())

            if not self.bbox_controller.bboxes and config.getboolean(
                "LABEL", "propagate_labels"
            ):
                self.bbox_controller.set_bboxes(previous_bboxes)
            self.bbox_controller.set_active_bbox(0)
            self.view.notify_frame_changed(self.pcd_manager.pcd_path.stem)
        else:
            self.view.update_progress(len(self.pcd_manager.pcds))
            self.view.button_next_pcd.setEnabled(False)

    def prev_pcd(self) -> None:
        self.save()
        if self.pcd_manager.current_id > 0:
            self.pcd_manager.get_prev_pcd()
            self.reset()
            self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())
            self.bbox_controller.set_active_bbox(0)
            self.view.notify_frame_changed(self.pcd_manager.pcd_path.stem)

    def custom_pcd(self, custom: int) -> None:
        self.save()
        self.pcd_manager.get_custom_pcd(custom)
        self.reset()
        self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())
        self.view.notify_frame_changed(self.pcd_manager.pcd_path.stem)

    # CONTROL METHODS
    def save(self) -> None:
        """Saves all bounding boxes and optionally segmentation labels in the label file."""
        self.pcd_manager.save_labels_into_file(self.bbox_controller.bboxes)
        if hasattr(self.view, '_save_sonar_ids'):
            self.view._save_sonar_ids()

        if LabelConfig().type == LabelingMode.SEMANTIC_SEGMENTATION:
            assert self.pcd_manager.pointcloud is not None
            self.pcd_manager.pointcloud.save_segmentation_labels()

    def reset(self) -> None:
        """Resets the controllers and bounding boxes from the current screen."""
        self.bbox_controller.reset()
        self.drawing_mode.reset()
        self.align_mode.reset()

    # CORRECTION METHODS
    def set_crosshair(self) -> None:
        """Sets the crosshair position in the glWidget to the current cursor position."""
        if self.curr_cursor_pos:
            self.view.gl_widget.crosshair_col = Colors.GREEN.value
            self.view.gl_widget.crosshair_pos = (
                self.curr_cursor_pos.x(),
                self.curr_cursor_pos.y(),
            )

    def set_selected_side(self) -> None:
        """Sets the currently hovered bounding box side in the glWidget."""
        if (
            (not self.side_mode)
            and self.curr_cursor_pos
            and self.bbox_controller.has_active_bbox()
            and (not self.scroll_mode)
        ):
            _, self.selected_side = oglhelper.get_intersected_sides(
                self.curr_cursor_pos.x(),
                self.curr_cursor_pos.y(),
                self.bbox_controller.get_active_bbox(),  # type: ignore
                self.view.gl_widget.modelview,
                self.view.gl_widget.projection,
            )
        if (
            self.selected_side
            and (not self.ctrl_pressed)
            and self.bbox_controller.has_active_bbox()
        ):
            self.view.gl_widget.crosshair_col = Colors.RED.value
            side_vertices = self.bbox_controller.get_active_bbox().get_vertices()  # type: ignore
            self.view.gl_widget.selected_side_vertices = side_vertices[
                BBOX_SIDES[self.selected_side]
            ]
            self.view.status_manager.set_message(
                "Scroll to change the bounding box dimension.",
                context=Context.SIDE_HOVERED,
            )
        else:
            self.view.gl_widget.selected_side_vertices = np.array([])
            self.view.status_manager.clear_message(Context.SIDE_HOVERED)

    # EVENT PROCESSING
    def mouse_clicked(self, a0: QtGui.QMouseEvent) -> None:
        """Triggers actions when the user clicks the mouse."""
        self.last_cursor_pos = a0.pos()

        if (
            self.drawing_mode.is_active()
            and (a0.buttons() & Keys.LeftButton)
            and (not self.ctrl_pressed)
        ):
            self.drawing_mode.register_point(a0.x(), a0.y(), correction=True)

        elif self.align_mode.is_active and (not self.ctrl_pressed):
            self.align_mode.register_point(
                self.view.gl_widget.get_world_coords(a0.x(), a0.y(), correction=False)
            )

        elif self.selected_side:
            self.side_mode = True

    def mouse_double_clicked(self, a0: QtGui.QMouseEvent) -> None:
        """Triggers actions when the user double clicks the mouse."""
        self.bbox_controller.select_bbox_by_ray(a0.x(), a0.y())

    def mouse_move_event(self, a0: QtGui.QMouseEvent) -> None:
        """Triggers actions when the user moves the mouse."""
        self.curr_cursor_pos = a0.pos()  # Updates the current mouse cursor position

        # Methods that use absolute cursor position
        if self.drawing_mode.is_active() and (not self.ctrl_pressed):
            self.drawing_mode.register_point(
                a0.x(), a0.y(), correction=True, is_temporary=True
            )

        elif self.align_mode.is_active and (not self.ctrl_pressed):
            self.align_mode.register_tmp_point(
                self.view.gl_widget.get_world_coords(a0.x(), a0.y(), correction=False)
            )

        if self.last_cursor_pos:
            dx = (
                self.last_cursor_pos.x() - a0.x()
            ) / 5  # Calculate relative movement from last click position
            dy = (self.last_cursor_pos.y() - a0.y()) / 5

            if (
                self.ctrl_pressed
                and (not self.drawing_mode.is_active())
                and (not self.align_mode.is_active)
            ):
                if a0.buttons() & Keys.LeftButton:  # bbox rotation
                    self.bbox_controller.rotate_with_mouse(-dx, -dy)
                elif a0.buttons() & Keys.RightButton:  # bbox translation
                    new_center = self.view.gl_widget.get_world_coords(
                        a0.x(), a0.y(), correction=True
                    )
                    self.bbox_controller.set_center(*new_center)  # absolute positioning
            else:
                if a0.buttons() & Keys.LeftButton:  # pcd rotation
                    self.pcd_manager.rotate_around_x(dy)
                    self.pcd_manager.rotate_around_z(dx)
                elif a0.buttons() & Keys.RightButton:  # pcd translation
                    self.pcd_manager.translate_along_x(dx)
                    self.pcd_manager.translate_along_y(dy)

            # Reset scroll locks of "side scrolling" for significant cursor movements
            if dx > Controller.MOVEMENT_THRESHOLD or dy > Controller.MOVEMENT_THRESHOLD:
                if self.side_mode:
                    self.side_mode = False
                else:
                    self.scroll_mode = False
        self.last_cursor_pos = a0.pos()

    def mouse_scroll_event(self, a0: QtGui.QWheelEvent) -> None:
        """Triggers actions when the user scrolls the mouse wheel."""
        if self.selected_side:
            self.side_mode = True

        if (
            self.drawing_mode.is_active()
            and (not self.ctrl_pressed)
            and self.drawing_mode.drawing_strategy is not None
        ):
            self.drawing_mode.drawing_strategy.register_scrolling(a0.angleDelta().y())
        elif self.side_mode and self.bbox_controller.has_active_bbox():
            self.bbox_controller.get_active_bbox().change_side(  # type: ignore
                self.selected_side, -a0.angleDelta().y() / 4000  # type: ignore
            )  # ToDo implement method
        else:
            self.pcd_manager.zoom_into(a0.angleDelta().y())
            self.scroll_mode = True

    def key_press_event(self, a0: QtGui.QKeyEvent) -> None:
        """Triggers actions when the user presses a key."""

        # Reset position to intial value
        if a0.key() == Keys.Key_Control:
            self.ctrl_pressed = True
            self.view.status_manager.set_message(
                "Hold right mouse button to translate or left mouse button to rotate "
                "the bounding box.",
                context=Context.CONTROL_PRESSED,
            )
        # Reset point cloud pose to intial rotation and translation
        elif a0.key() in [Keys.Key_P, Keys.Key_Home]:
            self.pcd_manager.reset_transformations()
            logging.info("Reseted position to default.")

        elif a0.key() == Keys.Key_Delete:  # Delete active bbox
            self.bbox_controller.delete_current_bbox()

        # Save labels to file
        elif a0.key() == Keys.Key_S and self.ctrl_pressed:
            self.save()

        elif a0.key() == Keys.Key_Escape:
            if self.drawing_mode.is_active():
                self.drawing_mode.reset()
                logging.info("Resetted drawn points!")
            elif self.align_mode.is_active:
                self.align_mode.reset()
                logging.info("Resetted selected points!")

        # BBOX MANIPULATION
        elif a0.key() == Keys.Key_Z:
            # z rotate counterclockwise (gimbal-safe, body frame)
            self.rotate_active_bbox_safe("z")
        elif a0.key() == Keys.Key_X:
            # z rotate clockwise
            self.rotate_active_bbox_safe("z", clockwise=True)
        elif a0.key() == Keys.Key_C:
            # y rotate counterclockwise (gimbal-safe, body frame)
            self.rotate_active_bbox_safe("y")
        elif a0.key() == Keys.Key_V:
            # y rotate clockwise
            self.rotate_active_bbox_safe("y", clockwise=True)
        elif a0.key() == Keys.Key_B:
            # x rotate counterclockwise (gimbal-safe, body frame)
            self.rotate_active_bbox_safe("x")
        elif a0.key() == Keys.Key_N:
            # x rotate clockwise
            self.rotate_active_bbox_safe("x", clockwise=True)
        elif a0.key() == Keys.Key_W:
            # move backward
            self.bbox_controller.translate_along_y()
        elif a0.key() == Keys.Key_S:
            # move forward
            self.bbox_controller.translate_along_y(forward=True)
        elif a0.key() == Keys.Key_A:
            # move left
            self.bbox_controller.translate_along_x(left=True)
        elif a0.key() == Keys.Key_D:
            # move right
            self.bbox_controller.translate_along_x()
        elif a0.key() == Keys.Key_Q:
            # move up
            self.bbox_controller.translate_along_z()
        elif a0.key() == Keys.Key_E:
            # move down
            self.bbox_controller.translate_along_z(down=True)

        # BBOX Scaling
        elif a0.key() == Keys.Key_I:
            # increase length
            self.bbox_controller.scale_along_length()
        elif a0.key() == Keys.Key_O:
            # decrease length
            self.bbox_controller.scale_along_length(decrease=True)
        elif a0.key() == Keys.Key_K:
            # increase width
            self.bbox_controller.scale_along_width()
        elif a0.key() == Keys.Key_L:
            # decrease width
            self.bbox_controller.scale_along_width(decrease=True)
        elif a0.key() == Keys.Key_Comma:
            # increase height
            self.bbox_controller.scale_along_height()
        elif a0.key() == Keys.Key_Period:
            # decrease height
            self.bbox_controller.scale_along_height(decrease=True)

        elif a0.key() in [Keys.Key_R, Keys.Key_Left]:
            # load previous sample
            self.prev_pcd()
        elif a0.key() in [Keys.Key_F, Keys.Key_Right]:
            # load next sample
            self.next_pcd()
        elif a0.key() == Keys.Key_T:
            # copy the labels from the most recent earlier labeled frame
            self.copy_previous_frame_labels()
        elif a0.key() == Keys.Key_Up:
            # select previous bbox
            self.select_relative_bbox(-1)
        elif a0.key() in [Keys.Key_G, Keys.Key_Down]:
            # select previous bbox
            self.select_relative_bbox(1)
        elif a0.key() == Keys.Key_Y:
            # change bbox class to previous available class
            self.select_relative_class(-1)
        elif a0.key() == Keys.Key_H:
            # change bbox class to next available class
            self.select_relative_class(1)
        elif a0.key() in list(range(49, 58)):
            # select bboxes with 1-9 digit keys
            self.bbox_controller.set_active_bbox(int(a0.key()) - 49)

    def select_relative_class(self, step: int):
        if step == 0:
            return
        curr_class = self.bbox_controller.get_active_bbox().get_classname()  # type: ignore
        new_class = LabelConfig().get_relative_class(curr_class, step)
        self.bbox_controller.get_active_bbox().set_classname(new_class)  # type: ignore
        self.bbox_controller.update_all()  # updates UI in SelectBox

    def select_relative_bbox(self, step: int):
        if step == 0:
            return
        max_id = len(self.bbox_controller.bboxes) - 1
        curr_id = self.bbox_controller.active_bbox_id
        new_id = curr_id + step
        corner_case_id = 0 if step > 0 else max_id
        new_id = new_id if new_id in range(max_id + 1) else corner_case_id
        self.bbox_controller.set_active_bbox(new_id)

    def rotate_active_bbox_safe(self, axis: str, clockwise: bool = False) -> None:
        """Rotate the active bbox about its own current local axis, gimbal-safe.

        Used by all six rotation keys (z/x, c/v, b/n). Composes the increment
        as a rotation matrix onto the box's current orientation (body frame)
        and writes the result back as Euler angles. Doing this for all three
        axes - not just x/y - matters once more than one axis is non-zero:
        directly incrementing a raw Euler field (the old z/x behavior) only
        spins the box about its own axis while the other two angles are zero.
        Past that, and especially once y approaches +/-90 deg (gimbal lock),
        the raw z field stops corresponding to the box's actual local Z axis,
        so "rotating z" visibly spins a different axis instead. Composing as a
        matrix keeps every key meaning "spin about my own current axis",
        continuously, regardless of the other two angles.

        Deliberately bypasses labelCloud's z_rotation_only guard, since c/v/b/n
        exist precisely to adjust the x/y axes.
        """
        bbox = self.bbox_controller.get_active_bbox()
        if bbox is None:
            logging.warning("There is currently no active bounding box to rotate.")
            return

        dangle = config.getfloat("LABEL", "std_rotation")
        if clockwise:
            dangle = -dangle

        x, y, z = bbox.get_rotations()
        R = _euler_zyx_to_matrix(x, y, z)
        drad = np.deg2rad(dangle)
        dR = {"x": _rot_x, "y": _rot_y, "z": _rot_z}[axis](drad)
        nx, ny, nz = _matrix_to_euler_zyx(R @ dR)  # body-frame increment
        bbox.set_rotations(nx % 360, ny % 360, nz % 360)
        self.bbox_controller.update_all()

    def copy_previous_frame_labels(self) -> None:
        """Copy the labels from the most recent earlier labeled frame into this one.

        Scans backward from the current frame for the first frame whose label
        file has any objects, imports those boxes (fresh copies) and appends
        them to the current frame without disturbing existing boxes. Sonar link
        IDs of the copied boxes are carried over as well.
        """
        pm = self.pcd_manager
        for idx in range(pm.current_id - 1, -1, -1):
            src_stem = pm.pcds[idx].stem
            candidates = pm.label_manager.import_labels(pm.pcds[idx])
            if candidates:
                base_index = len(self.bbox_controller.bboxes)
                for bbox in candidates:
                    self.bbox_controller.add_bbox(bbox)
                self._copy_sonar_ids(src_stem, base_index, len(candidates))
                logging.info(
                    "Copied %d label(s) from %s into the current frame.",
                    len(candidates),
                    src_stem,
                )
                self.view.status_manager.set_message(
                    f"Copied {len(candidates)} label(s) from {src_stem}.",
                    context=Context.DEFAULT,
                )
                return
        logging.warning("No earlier labeled frame found to copy from.")
        self.view.status_manager.set_message(
            "No earlier labeled frame to copy from.", context=Context.DEFAULT
        )

    def _copy_sonar_ids(self, src_stem: str, base_index: int, count: int) -> None:
        """Carry sonar link IDs from a source frame onto freshly copied boxes.

        The copied boxes occupy indices base_index .. base_index+count-1 in the
        current frame; map each to the source frame's link ID at the same
        relative index. No-op if the camera-panel link-ID machinery is absent
        or the source frame has no link IDs.
        """
        view = self.view
        if not hasattr(view, "_sonar_ids"):
            return
        src_ids_path = Path(config.get("FILE", "label_folder")) / (src_stem + "_ids.json")
        if not src_ids_path.is_file():
            return
        try:
            with open(src_ids_path) as f:
                src_ids = {int(k): v for k, v in json.load(f).items()}
        except (json.JSONDecodeError, OSError, ValueError):
            return
        for i in range(count):
            link_id = src_ids.get(i, 0)
            if link_id:
                view._sonar_ids[base_index + i] = link_id
        if hasattr(view, "_save_sonar_ids"):
            view._save_sonar_ids()
        if hasattr(view, "sonar_id_spin"):
            cur = view.label_list.currentRow()
            view.sonar_id_spin.setValue(view._sonar_ids.get(cur, 0))

    def key_release_event(self, a0: QtGui.QKeyEvent) -> None:
        """Triggers actions when the user releases a key."""
        if a0.key() == Keys.Key_Control:
            self.ctrl_pressed = False
            self.view.status_manager.clear_message(Context.CONTROL_PRESSED)

    def crop_pointcloud_inside_active_bbox(self) -> None:
        bbox = self.bbox_controller.get_active_bbox()
        assert bbox is not None
        assert self.pcd_manager.pointcloud is not None
        points_inside = bbox.is_inside(self.pcd_manager.pointcloud.points)
        pointcloud = self.pcd_manager.pointcloud.get_filtered_pointcloud(points_inside)
        if pointcloud is None:
            logging.warning("No points found inside the box. Ignored.")
            return
        self.view.save_point_cloud_as(pointcloud)
