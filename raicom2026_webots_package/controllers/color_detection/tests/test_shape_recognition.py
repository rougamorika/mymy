import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shape_recognition import ShapeRecognizer, TrackerState, draw_detections, update_tracks


def test_shape_recognizer_can_be_constructed():
    recognizer = ShapeRecognizer()
    assert recognizer.class_names == (
        "cross",
        "rectangle",
        "square",
        "star",
        "pentagon",
        "triangle",
        "four_leaf",
        "parallelogram",
        "circle",
    )


def test_tray_roi_mask_limits_detection_area():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    recognizer = ShapeRecognizer()
    mask = recognizer.build_tray_roi_mask(frame)
    assert mask.shape == (480, 640)
    assert mask[100, 200] == 255
    assert mask[20, 20] == 0


def test_build_object_mask_rejects_white_background():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (220, 150), (300, 220), (0, 0, 255), -1)
    recognizer = ShapeRecognizer()
    mask = recognizer.build_object_mask(frame)
    assert cv2.countNonZero(mask) > 0
    assert mask[20, 20] == 0


def test_build_color_core_mask_is_more_selective():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (220, 150), (300, 220), (0, 0, 255), -1)
    recognizer = ShapeRecognizer()
    object_mask = recognizer.build_object_mask(frame)
    core_mask = recognizer.build_color_core_mask(frame)
    assert cv2.countNonZero(core_mask) <= cv2.countNonZero(object_mask)


def test_build_color_core_mask_prefers_brighter_top_face_pixels():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[150:185, 220:300] = (40, 40, 255)
    frame[185:220, 220:300] = (20, 20, 150)
    recognizer = ShapeRecognizer()
    core_mask = recognizer.build_color_core_mask(frame)
    top_count = cv2.countNonZero(core_mask[150:185, 220:300])
    side_count = cv2.countNonZero(core_mask[185:220, 220:300])
    assert top_count >= side_count


def test_detect_shapes_finds_colored_triangle():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    pts = np.array([[[220, 140]], [[280, 140]], [[250, 200]]], dtype=np.int32)
    cv2.drawContours(frame, [pts], -1, (0, 0, 255), -1)
    recognizer = ShapeRecognizer()
    detections = recognizer.detect_shapes(frame)
    assert any(d["raw_label"] == "triangle" for d in detections)


def test_update_tracks_assigns_ids():
    tracker = TrackerState(max_history=3)
    detections = [{
        "label": "square",
        "raw_label": "square",
        "confidence": 0.95,
        "centroid": (100, 100),
        "contour": None,
        "bbox": (0, 0, 10, 10),
    }]
    events = update_tracks(tracker, detections)
    assert detections[0]["id"] == 1
    assert events[0]["type"] == "appeared"


def test_draw_detections_marks_output():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    detections = [{
        "id": 1,
        "contour": np.array([[[20, 20]], [[40, 20]], [[40, 40]], [[20, 40]]], dtype=np.int32),
        "label": "square",
        "raw_label": "square",
        "confidence": 0.96,
        "centroid": (30, 30),
        "bbox": (20, 20, 20, 20),
    }]
    output = draw_detections(frame, detections)
    assert not np.array_equal(frame, output)
