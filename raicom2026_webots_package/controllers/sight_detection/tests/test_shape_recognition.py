import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shape_recognition import ShapeRecognizer, TrackerState, contour_from_points, draw_detections, update_tracks


def test_shape_recognizer_imports():
    recognizer = ShapeRecognizer()
    assert recognizer is not None


def test_tray_roi_mask_limits_detection_area():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    recognizer = ShapeRecognizer()
    mask = recognizer.build_tray_roi_mask(frame)
    assert mask.shape == (480, 640)
    assert mask[100, 200] == 255
    assert mask[20, 20] == 0


def test_top_face_mask_is_more_selective_than_object_mask():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    recognizer = ShapeRecognizer()
    object_mask = recognizer.build_object_mask(frame)
    top_face_mask = recognizer.build_top_face_mask(frame)
    assert top_face_mask.shape == object_mask.shape
    assert cv2.countNonZero(top_face_mask) <= cv2.countNonZero(object_mask)


def test_top_face_mask_keeps_large_colored_shapes():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    recognizer = ShapeRecognizer()
    cv2.rectangle(frame, (220, 150), (300, 220), (0, 0, 255), -1)
    star = np.array([[250, 80], [265, 120], [305, 122], [274, 145], [284, 185], [250, 162], [216, 185], [226, 145], [195, 122], [235, 120]], dtype=np.int32)
    cv2.fillPoly(frame, [star], (255, 0, 0))
    mask = recognizer.build_top_face_mask(frame)
    assert cv2.countNonZero(mask) > 0


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
