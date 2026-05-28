import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.setdefault("controller", SimpleNamespace(Robot=object))

from recess_recognition import TopTrayRecessRecognizer
import top_tray_recess_controller


def _external_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def test_recognizer_can_be_constructed():
    recognizer = TopTrayRecessRecognizer()

    assert isinstance(recognizer, TopTrayRecessRecognizer)


def test_build_top_tray_mask_matches_current_camera_pose():
    recognizer = TopTrayRecessRecognizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = recognizer.build_top_tray_mask(frame)
    assert mask[28, 165] == 255
    assert mask[460, 165] == 255
    assert mask[90, 382] == 255
    assert mask[120, 410] == 0


def test_top_tray_slot_map_has_18_slots():
    recognizer = TopTrayRecessRecognizer()
    slots = recognizer.get_slot_map()
    assert len(slots) == 18
    assert slots[0]["slot_index"] == 0
    assert slots[-1]["slot_index"] == 17


def test_extract_slot_contour_returns_dominant_recess():
    recognizer = TopTrayRecessRecognizer()
    frame = np.full((480, 640, 3), 205, dtype=np.uint8)
    slot = {"slot_index": 0, "slot_rect": (172, 34, 56, 70), "expected_labels": ("cross",)}
    cv2.rectangle(frame, (190, 48), (208, 84), (232, 232, 232), -1)
    cv2.rectangle(frame, (182, 60), (216, 72), (232, 232, 232), -1)
    contour = recognizer.extract_slot_contour(frame, slot)
    assert contour is not None


def test_smooth_slot_contour_keeps_valid_contour():
    recognizer = TopTrayRecessRecognizer()
    contour = np.array([[[10, 10]], [[40, 12]], [[39, 40]], [[12, 39]]], dtype=np.int32)
    smoothed = recognizer._smooth_slot_contour(contour)
    assert smoothed is not None
    assert cv2.contourArea(smoothed) > 0


def test_classify_recess_contours_for_slot_shapes():
    recognizer = TopTrayRecessRecognizer()

    square_mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(square_mask, (28, 28), (92, 92), 255, -1)

    parallelogram_mask = np.zeros((120, 120), dtype=np.uint8)
    pts = np.array([[36, 28], [94, 28], [78, 92], [20, 92]], dtype=np.int32)
    cv2.fillPoly(parallelogram_mask, [pts], 255)

    leaf_mask = np.zeros((120, 120), dtype=np.uint8)
    for center in [(42, 42), (78, 42), (42, 78), (78, 78)]:
        cv2.circle(leaf_mask, center, 18, 255, -1)

    assert recognizer.classify_slot_contour(_external_contour(square_mask), ("square",))["label"] == "square"
    assert recognizer.classify_slot_contour(_external_contour(parallelogram_mask), ("parallelogram",))["label"] == "parallelogram"
    assert recognizer.classify_slot_contour(_external_contour(leaf_mask), ("four_leaf",))["label"] == "four_leaf"


def test_template_first_classification_handles_small_shape_bias():
    recognizer = TopTrayRecessRecognizer()
    skewed_square_mask = np.zeros((120, 120), dtype=np.uint8)
    pts = np.array([[30, 28], [92, 32], [88, 92], [26, 88]], dtype=np.int32)
    cv2.fillPoly(skewed_square_mask, [pts], 255)
    result = recognizer.classify_slot_contour(_external_contour(skewed_square_mask), ("square",))
    assert result["label"] == "square"


def test_detect_recesses_returns_detection_fields():
    recognizer = TopTrayRecessRecognizer()
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    cv2.rectangle(frame, (318, 48), (358, 88), (245, 245, 245), -1)

    detections = recognizer.detect_recesses(frame)

    assert len(detections) == 1
    detection = detections[0]
    assert detection["label"] == "square"
    assert detection["slot_index"] == 2
    assert isinstance(detection["confidence"], float)
    assert 0.0 <= detection["confidence"] <= 1.0
    assert isinstance(detection["centroid"], tuple)
    assert len(detection["centroid"]) == 2
    assert isinstance(detection["bbox"], tuple)
    assert len(detection["bbox"]) == 4
    assert isinstance(detection["contour"], np.ndarray)
    x, y, w, h = detection["bbox"]
    cx, cy = detection["centroid"]
    assert w > 0 and h > 0
    assert x <= cx <= x + w
    assert y <= cy <= y + h


def test_draw_detections_modifies_output_frame():
    recognizer = TopTrayRecessRecognizer()
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    detections = [
        {
            "label": "square",
            "slot_index": 2,
            "confidence": 0.95,
            "centroid": (60, 60),
            "bbox": (30, 30, 40, 40),
            "contour": np.array([[[30, 30]], [[70, 30]], [[70, 70]], [[30, 70]]], dtype=np.int32),
        }
    ]

    output = recognizer.draw_detections(frame, detections)

    assert output is not frame
    assert np.count_nonzero(output) > 0
    assert not np.array_equal(output, frame)


def test_controller_main_uses_camera_and_displays_frames(monkeypatch):
    displayed_frames = []

    class FakeCamera:
        def __init__(self):
            self.enabled_with = None

        def enable(self, timestep):
            self.enabled_with = timestep

        def getWidth(self):
            return 8

        def getHeight(self):
            return 6

        def getImage(self):
            image = np.zeros((6, 8, 4), dtype=np.uint8)
            image[..., 0] = 10
            image[..., 1] = 20
            image[..., 2] = 30
            image[..., 3] = 255
            return image.tobytes()

    class FakeRobot:
        def __init__(self):
            self.camera = FakeCamera()
            self.steps = iter([0, -1])

        def getBasicTimeStep(self):
            return 32

        def getDevice(self, name):
            assert name == "camera"
            return self.camera

        def step(self, timestep):
            assert timestep == 32
            return next(self.steps)

    monkeypatch.setattr(top_tray_recess_controller, "Robot", FakeRobot)
    monkeypatch.setattr(
        top_tray_recess_controller,
        "TopTrayRecessRecognizer",
        lambda: SimpleNamespace(
            detect_recesses=lambda frame: [],
            draw_detections=lambda frame, detections: frame.copy(),
        ),
    )
    monkeypatch.setattr(top_tray_recess_controller.cv2, "imshow", lambda name, frame: displayed_frames.append((name, frame.copy())))
    monkeypatch.setattr(top_tray_recess_controller.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(top_tray_recess_controller.cv2, "destroyAllWindows", lambda: displayed_frames.append(("destroy", None)))

    top_tray_recess_controller.main()

    assert displayed_frames
    assert displayed_frames[0][0] == "top_tray_recess"
    assert displayed_frames[0][1].shape == (6, 8, 3)
    assert displayed_frames[-1][0] == "destroy"


def test_detect_recesses_returns_expected_fields():
    recognizer = TopTrayRecessRecognizer()
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    detections = recognizer.detect_recesses(frame)
    assert isinstance(detections, list)
    if detections:
        detection = detections[0]
        assert set(detection.keys()) >= {"slot_index", "label", "confidence", "centroid", "contour", "bbox"}


def test_draw_detections_marks_output_frame():
    recognizer = TopTrayRecessRecognizer()
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    detections = [{
        "slot_index": 0,
        "label": "cross",
        "confidence": 0.95,
        "centroid": (30, 30),
        "contour": np.array([[[20, 20]], [[40, 20]], [[40, 40]], [[20, 40]]], dtype=np.int32),
        "bbox": (20, 20, 20, 20),
    }]
    output = recognizer.draw_detections(frame, detections)
    assert not np.array_equal(frame, output)


def test_draw_detections_draws_external_contour_visible_in_window():
    recognizer = TopTrayRecessRecognizer()
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    detections = [{
        "slot_index": 0,
        "label": "cross",
        "confidence": 0.95,
        "centroid": (30, 30),
        "contour": np.array([[[20, 20]], [[40, 20]], [[40, 40]], [[20, 40]]], dtype=np.int32),
        "bbox": (20, 20, 20, 20),
    }]
    output = recognizer.draw_detections(frame, detections)
    assert tuple(output[20, 30]) != (255, 255, 255)


def test_draw_detections_shows_slot_index_not_label_text():
    recognizer = TopTrayRecessRecognizer()
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    detections = [{
        "slot_index": 7,
        "label": "parallelogram",
        "confidence": 0.95,
        "centroid": (30, 30),
        "contour": np.array([[[20, 20]], [[40, 20]], [[40, 40]], [[20, 40]]], dtype=np.int32),
        "bbox": (20, 20, 20, 20),
    }]
    output = recognizer.draw_detections(frame, detections)
    assert not np.array_equal(frame, output)


def test_draw_layout_marks_roi_and_slots():
    recognizer = TopTrayRecessRecognizer()
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    output = recognizer.draw_layout(frame)
    assert not np.array_equal(frame, output)


def test_save_window_screenshot_only_every_fifteen_frames(monkeypatch, tmp_path):
    calls = []

    def fake_imwrite(path, frame):
        calls.append(path)
        return True

    monkeypatch.setattr(top_tray_recess_controller.cv2, "imwrite", fake_imwrite)

    top_tray_recess_controller._save_window_screenshot(tmp_path, 14, np.zeros((10, 10, 3), dtype=np.uint8))
    top_tray_recess_controller._save_window_screenshot(tmp_path, 15, np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(calls) == 1
    assert calls[0].endswith("000015_window.png")


def test_write_debug_images_saves_multiple_views(monkeypatch, tmp_path):
    calls = []

    def fake_imwrite(path, frame):
        calls.append(path)
        return True

    recognizer = TopTrayRecessRecognizer()
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    recognizer.detect_recesses(frame)
    monkeypatch.setattr(top_tray_recess_controller.cv2, "imwrite", fake_imwrite)

    top_tray_recess_controller._write_debug_images(tmp_path, 15, frame, recognizer, frame)

    assert any(path.endswith("000015_raw.png") for path in calls)
    assert any(path.endswith("000015_display.png") for path in calls)
    assert any(path.endswith("000015_tray_mask.png") for path in calls)
    assert any(path.endswith("000015_object_mask.png") for path in calls)


def test_main_prints_slot_index_label_confidence_and_centroid(monkeypatch, capsys):
    class FakeCamera:
        def enable(self, timestep):
            self.enabled_with = timestep

        def getWidth(self):
            return 4

        def getHeight(self):
            return 3

        def getImage(self):
            image = np.zeros((3, 4, 4), dtype=np.uint8)
            image[..., 3] = 255
            return image.tobytes()

    class FakeRobot:
        def __init__(self):
            self.camera = FakeCamera()
            self.steps = iter([0, -1])

        def getBasicTimeStep(self):
            return 32

        def getDevice(self, name):
            return self.camera

        def step(self, timestep):
            return next(self.steps)

    monkeypatch.setattr(top_tray_recess_controller, "Robot", FakeRobot)
    monkeypatch.setattr(
        top_tray_recess_controller,
        "TopTrayRecessRecognizer",
        lambda: SimpleNamespace(
            detect_recesses=lambda frame: [
                {"label": "square", "confidence": 0.96, "centroid": (1, 2), "contour": np.zeros((4, 1, 2), dtype=np.int32), "bbox": (0, 0, 1, 1), "slot_index": 2}
            ],
            draw_detections=lambda frame, detections: frame.copy(),
        ),
    )
    monkeypatch.setattr(top_tray_recess_controller.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(top_tray_recess_controller.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(top_tray_recess_controller.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(top_tray_recess_controller, "_save_window_screenshot", lambda *args, **kwargs: None)

    top_tray_recess_controller.main()
    captured = capsys.readouterr().out
    assert "2" in captured
    assert "square" in captured
    assert "0.96" in captured
    assert "(1, 2)" in captured
    assert captured.strip().count("\n") == 0
