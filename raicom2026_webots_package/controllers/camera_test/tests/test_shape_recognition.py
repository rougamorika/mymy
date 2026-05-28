import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shape_recognition import (
    ShapeRecognizer,
    TrackerState,
    UNCERTAIN_THRESHOLD,
    contour_from_points,
    draw_detections,
    extract_features,
    normalize_contour,
    update_tracks,
)


def _find_external_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert contours
    return contours[0]


def _affine_transform(contour: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = contour.reshape((-1, 2)).astype(np.float32)
    transformed = cv2.transform(points.reshape((-1, 1, 2)), matrix.astype(np.float32))
    return transformed


def _warp_board(board: np.ndarray, destination: np.ndarray, output_shape=(360, 420)) -> np.ndarray:
    source = np.array(
        [
            [0.0, 0.0],
            [board.shape[1] - 1.0, 0.0],
            [board.shape[1] - 1.0, board.shape[0] - 1.0],
            [0.0, board.shape[0] - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
    return cv2.warpPerspective(board, matrix, (output_shape[1], output_shape[0]))


def _paste_nonzero(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    result = base.copy()
    mask = np.any(overlay != 0, axis=2)
    result[mask] = overlay[mask]
    return result


def test_recognizer_can_be_constructed():
    recognizer = ShapeRecognizer()
    tracker = TrackerState()
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
    assert tracker.max_history == 5


def test_recognizer_has_no_bottom_tray_api():
    recognizer = ShapeRecognizer()
    assert not hasattr(recognizer, "detect_bottom_tray_slots")
    assert not hasattr(recognizer, "get_bottom_tray_rect")
    assert not hasattr(recognizer, "get_bottom_tray_cells")


def test_fixed_roi_matches_new_camera_position():
    recognizer = ShapeRecognizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = recognizer.build_left_tray_frame_mask(frame)
    assert mask is not None
    assert mask[28, 165] == 255
    assert mask[460, 165] == 255
    assert mask[90, 382] == 255
    assert mask[120, 410] == 0


def test_normalize_contour_centers_and_scales_points():
    contour = contour_from_points([(10, 10), (30, 10), (30, 30), (10, 30)])
    normalized = normalize_contour(contour)
    xs = normalized[:, 0, 0]
    ys = normalized[:, 0, 1]
    assert np.isclose(xs.mean(), 0.0)
    assert np.isclose(ys.mean(), 0.0)
    assert np.max(np.abs(normalized)) <= 1.0


def test_extract_features_reports_quadrilateral_properties():
    contour = contour_from_points([(0, 0), (40, 0), (40, 20), (0, 20)])
    features = extract_features(contour)
    assert features["vertex_count"] == 4
    assert features["is_convex"] is True
    assert features["aspect_ratio"] > 1.5
    assert features["circularity"] < 1.0


def test_classify_square_from_rectangle():
    recognizer = ShapeRecognizer()
    square = contour_from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    rectangle = contour_from_points([(0, 0), (36, 0), (36, 20), (0, 20)])
    assert recognizer.classify_contour(square)["label"] == "square"
    assert recognizer.classify_contour(rectangle)["label"] == "rectangle"


def test_classify_parallelogram_separately_from_rectangle():
    recognizer = ShapeRecognizer()
    contour = contour_from_points([(8, 0), (36, 0), (28, 20), (0, 20)])
    assert recognizer.classify_contour(contour)["label"] == "parallelogram"


def test_classify_triangle_and_pentagon():
    recognizer = ShapeRecognizer()
    triangle = contour_from_points([(0, 20), (20, 0), (40, 20)])
    pentagon = contour_from_points([(20, 0), (38, 14), (31, 36), (9, 36), (2, 14)])
    assert recognizer.classify_contour(triangle)["label"] == "triangle"
    assert recognizer.classify_contour(pentagon)["label"] == "pentagon"


def test_classify_circle_from_dense_round_contour():
    recognizer = ShapeRecognizer()
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 20, 255, -1)
    contour = _find_external_contour(mask)
    assert recognizer.classify_contour(contour)["label"] == "circle"


def test_detect_shapes_on_perspective_skewed_tray():
    recognizer = ShapeRecognizer()
    board = np.full((240, 240, 3), 240, dtype=np.uint8)
    cv2.rectangle(board, (0, 0), (239, 239), (40, 40, 40), 4)
    cv2.rectangle(board, (24, 24), (84, 84), (0, 0, 0), -1)
    triangle = np.array([[150, 86], [180, 30], [210, 86]], dtype=np.int32)
    cv2.fillPoly(board, [triangle], (0, 0, 0))
    cv2.circle(board, (182, 182), 28, (0, 0, 0), -1)
    for center in [(42, 164), (74, 164), (42, 196), (74, 196)]:
        cv2.circle(board, center, 20, (0, 0, 0), -1)

    destination = np.array([[80, 28], [320, 48], [300, 286], [56, 258]], dtype=np.float32)
    frame = _warp_board(board, destination)

    detections = recognizer.detect_shapes(frame)
    labels = sorted(d["raw_label"] for d in detections)
    assert labels == ["circle", "four_leaf", "square", "triangle"]


def test_classify_non_convex_templates():
    recognizer = ShapeRecognizer()

    mask_cross = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(mask_cross, (45, 20), (75, 100), 255, -1)
    cv2.rectangle(mask_cross, (20, 45), (100, 75), 255, -1)

    mask_star = np.zeros((120, 120), dtype=np.uint8)
    star_points = np.array(
        [[60, 10], [72, 45], [108, 45], [80, 67], [92, 104], [60, 82], [28, 104], [40, 67], [12, 45], [48, 45]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask_star, [star_points], 255)

    mask_leaf = np.zeros((120, 120), dtype=np.uint8)
    for center in [(40, 40), (80, 40), (40, 80), (80, 80)]:
        cv2.circle(mask_leaf, center, 22, 255, -1)

    cross = _find_external_contour(mask_cross)
    star = _find_external_contour(mask_star)
    leaf = _find_external_contour(mask_leaf)

    assert recognizer.classify_contour(cross)["label"] == "cross"
    assert recognizer.classify_contour(star)["label"] == "star"
    assert recognizer.classify_contour(leaf)["label"] == "four_leaf"


def test_clipped_square_rectangle_and_cross_should_still_be_rectilinear():
    recognizer = ShapeRecognizer()

    square = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(square, (30, 30), (90, 90), 255, -1)
    cv2.rectangle(square, (30, 30), (46, 46), 0, -1)

    rectangle = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(rectangle, (24, 36), (96, 84), 255, -1)
    cv2.rectangle(rectangle, (24, 36), (38, 50), 0, -1)

    cross = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(cross, (50, 20), (70, 100), 255, -1)
    cv2.rectangle(cross, (20, 50), (100, 70), 255, -1)
    cv2.rectangle(cross, (20, 50), (28, 70), 0, -1)

    square_contour = _find_external_contour(square)
    rectangle_contour = _find_external_contour(rectangle)
    cross_contour = _find_external_contour(cross)

    assert recognizer.classify_contour(square_contour)["label"] in {"square", "rectangle"}
    assert recognizer.classify_contour(rectangle_contour)["label"] in {"square", "rectangle"}
    assert recognizer.classify_contour(cross_contour)["label"] == "cross"


def test_uncertain_threshold_is_point_eight():
    assert UNCERTAIN_THRESHOLD == 0.80


def test_mid_confidence_candidate_is_not_uncertain_at_point_eight_threshold():
    recognizer = ShapeRecognizer()
    noisy_quad = contour_from_points([(0, 0), (30, 2), (34, 21), (3, 20)])
    result = recognizer.classify_contour(noisy_quad)
    assert result["confidence"] >= 0.80
    assert result["label"] != "uncertain"


def test_detect_shapes_returns_multiple_targets():
    recognizer = ShapeRecognizer()
    frame = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (70, 70), (0, 0, 0), -1)
    cv2.circle(frame, (170, 60), 25, (0, 0, 0), -1)
    pts = np.array([[110, 180], [150, 120], [190, 180]], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (0, 0, 0))
    detections = recognizer.detect_shapes(frame)
    labels = sorted(d["raw_label"] for d in detections)
    assert labels == ["circle", "square", "triangle"]


def test_detect_shapes_covers_two_vertical_recess_groups_in_fixed_roi():
    recognizer = ShapeRecognizer()
    frame = np.full((480, 640, 3), 70, dtype=np.uint8)
    tray_color = np.array([220, 205, 165], dtype=np.uint8)
    frame[28:464, 164:385] = tray_color

    cv2.rectangle(frame, (193, 46), (207, 90), (240, 240, 240), -1)
    cv2.rectangle(frame, (178, 61), (222, 75), (240, 240, 240), -1)
    cv2.rectangle(frame, (236, 52), (296, 86), (240, 240, 240), -1)
    cv2.circle(frame, (348, 69), 22, (240, 240, 240), -1)

    cv2.rectangle(frame, (193, 258), (207, 302), (240, 240, 240), -1)
    cv2.rectangle(frame, (178, 273), (222, 287), (240, 240, 240), -1)
    triangle = np.array([[286, 294], [254, 346], [318, 346]], dtype=np.int32)
    cv2.fillPoly(frame, [triangle], (240, 240, 240))
    mask_leaf = np.zeros((480, 640), dtype=np.uint8)
    for center in [(344, 276), (364, 276), (344, 296), (364, 296)]:
        cv2.circle(mask_leaf, center, 14, 255, -1)
    frame[mask_leaf > 0] = (240, 240, 240)

    detections = recognizer.detect_shapes(frame)
    labels = sorted(d["raw_label"] for d in detections)
    assert labels == ["circle", "cross", "cross", "four_leaf", "rectangle", "triangle"]


def test_tracker_stabilizes_label_before_reporting_change():
    tracker = TrackerState(max_history=5)
    first = [{"raw_label": "rectangle", "label": "rectangle", "confidence": 0.95, "centroid": (40, 40), "bbox": (20, 20, 40, 20)}]
    second = [{"raw_label": "square", "label": "uncertain", "confidence": 0.62, "centroid": (41, 40), "bbox": (20, 20, 40, 20)}]
    third = [{"raw_label": "rectangle", "label": "rectangle", "confidence": 0.96, "centroid": (42, 40), "bbox": (20, 20, 40, 20)}]
    events1 = update_tracks(tracker, first)
    events2 = update_tracks(tracker, second)
    events3 = update_tracks(tracker, third)
    assert any(event["type"] == "appeared" for event in events1)
    assert events2 == []
    assert events3 == []


def test_update_tracks_assigns_ids_shared_by_display_and_console():
    tracker = TrackerState(max_history=5)
    detections = [
        {"raw_label": "square", "label": "square", "confidence": 0.96, "centroid": (30, 30), "bbox": (20, 20, 20, 20)},
        {"raw_label": "circle", "label": "circle", "confidence": 0.95, "centroid": (90, 30), "bbox": (80, 20, 20, 20)},
    ]

    events = update_tracks(tracker, detections)

    assert [detection["id"] for detection in detections] == [1, 2]
    assert [event["id"] for event in events if event["type"] == "appeared"] == [1, 2]


def test_draw_detections_marks_centroid_and_bbox():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    detections = [{
        "id": 3,
        "contour": np.array([[[20, 20]], [[40, 20]], [[40, 40]], [[20, 40]]], dtype=np.int32),
        "label": "square",
        "raw_label": "square",
        "confidence": 0.96,
        "centroid": (30, 30),
        "bbox": (20, 20, 20, 20),
    }]
    output = draw_detections(frame, detections)
    assert not np.array_equal(frame, output)
