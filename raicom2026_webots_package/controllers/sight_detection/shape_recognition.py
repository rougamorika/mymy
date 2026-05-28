from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

CLASS_NAMES = (
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

UNCERTAIN_THRESHOLD = 0.80

TRAY_ROI_PIXELS = (170, 78, 216, 396)
MIN_CONTOUR_AREA = 180.0
MAX_CONTOUR_AREA = 12000.0
BORDER_MARGIN = 5


@dataclass
class Track:
    centroid: Tuple[int, int]
    stable_label: str
    last_reported_centroid: Tuple[int, int]
    missing_frames: int = 0


@dataclass
class TrackerState:
    max_history: int = 5
    tracks: Dict[int, Track] = field(default_factory=dict)
    next_id: int = 1


def contour_from_points(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    return np.array(points, dtype=np.float32).reshape((-1, 1, 2))


def _centroid_from_moments(moments: Dict[str, float]) -> Tuple[int, int]:
    if moments["m00"] == 0:
        return 0, 0
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def _order_points(points: np.ndarray) -> np.ndarray:
    points = points.reshape((-1, 2)).astype(np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def extract_features(contour: np.ndarray) -> Dict[str, object]:
    contour = contour.astype(np.float32)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = 0.02 * perimeter if perimeter else 0.0
    approx = cv2.approxPolyDP(contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    moments = cv2.moments(contour)
    centroid = _centroid_from_moments(moments)
    aspect_ratio = (w / h) if h else 0.0
    circularity = 0.0 if perimeter == 0.0 else (4.0 * np.pi * area) / (perimeter * perimeter)
    hull = cv2.convexHull(contour, returnPoints=True)
    hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else area
    solidity = area / hull_area if hull_area else 0.0
    extent = area / float(w * h) if w and h else 0.0
    ordered = _order_points(approx) if len(approx) >= 3 else approx.reshape((-1, 2))
    angles: List[float] = []
    if len(ordered) >= 3:
        for index in range(len(ordered)):
            prev_point = ordered[index - 1]
            point = ordered[index]
            next_point = ordered[(index + 1) % len(ordered)]
            v1 = prev_point - point
            v2 = next_point - point
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom == 0.0:
                continue
            cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cosine))))
    return {
        "area": area,
        "perimeter": perimeter,
        "approx": approx,
        "vertex_count": len(approx),
        "is_convex": bool(cv2.isContourConvex(approx)),
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "bounding_rect": (x, y, w, h),
        "extent": extent,
        "solidity": solidity,
        "angles": angles,
        "centroid": centroid,
    }


def classify_quadrilateral(features: Dict[str, object]) -> Tuple[str, float]:
    ratio = float(features["aspect_ratio"])
    angles = [float(angle) for angle in features["angles"]]
    near_right = sum(abs(angle - 90.0) < 14.0 for angle in angles) >= 3
    if near_right and abs(ratio - 1.0) < 0.18:
        return "square", 0.97
    if near_right:
        return "rectangle", 0.95
    return "parallelogram", 0.92


def fuse_confidence(template_score: float, geometry_score: float, margin_score: float) -> float:
    return 0.45 * template_score + 0.40 * geometry_score + 0.15 * margin_score


def draw_detections(frame: np.ndarray, detections: Sequence[Dict[str, object]]) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        color = (0, 200, 0)
        cv2.drawContours(output, [detection["contour"].astype(np.int32)], -1, color, 2)
        x, y, w, h = detection["bbox"]
        cv2.circle(output, detection["centroid"], 4, (0, 0, 255), -1)
        prefix = f"#{detection['id']} " if "id" in detection else ""
        caption = f"{prefix}{detection['label']}"
        cv2.putText(output, caption, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return output


def update_tracks(state: TrackerState, detections: Sequence[Dict[str, object]], max_distance: float = 40.0) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    assigned = set()
    for detection in detections:
        best_id = None
        best_distance = max_distance
        for track_id, track in state.tracks.items():
            if track_id in assigned:
                continue
            distance = float(np.linalg.norm(np.array(track.centroid) - np.array(detection["centroid"])))
            if distance < best_distance:
                best_distance = distance
                best_id = track_id
        if best_id is None:
            track_id = state.next_id
            state.next_id += 1
            state.tracks[track_id] = Track(
                centroid=detection["centroid"],
                stable_label=str(detection["label"]),
                last_reported_centroid=detection["centroid"],
            )
            detection["id"] = track_id
            events.append({"type": "appeared", "id": track_id, "detection": detection})
            continue
        assigned.add(best_id)
        track = state.tracks[best_id]
        track.centroid = detection["centroid"]
        detection["id"] = best_id
        events.append({"type": "moved", "id": best_id, "detection": detection})
    return events


class ShapeRecognizer:
    def __init__(self):
        self.class_names = CLASS_NAMES
        self.tray_roi = TRAY_ROI_PIXELS

    def build_tray_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x, y, w, h = self.tray_roi
        mask[y:y + h, x:x + w] = 255
        return mask

    def build_object_mask(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.tray_roi
        roi = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([179, 90, 255]))
        colorful_mask = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([179, 255, 255]))
        mask = cv2.bitwise_and(colorful_mask, cv2.bitwise_not(white_mask))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        return mask

    def build_top_face_mask(self, frame: np.ndarray) -> np.ndarray:
        mask = self.build_object_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_mask = np.zeros_like(mask)
        if not contours:
            return mask

        shrink_kernel = np.ones((3, 3), dtype=np.uint8)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            local = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(local, [contour], -1, 255, -1)
            local = cv2.erode(local, shrink_kernel, iterations=1)
            if cv2.countNonZero(local) == 0:
                continue
            top_mask = cv2.bitwise_or(top_mask, local)

        if cv2.countNonZero(top_mask) == 0:
            return mask
        return top_mask

    def get_debug_views(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        views: Dict[str, np.ndarray] = {}
        tray_mask = self.build_tray_roi_mask(frame)
        object_mask = self.build_object_mask(frame)
        top_face_mask = self.build_top_face_mask(frame)

        tray_overlay = frame.copy()
        tray_overlay[tray_mask > 0] = cv2.addWeighted(
            tray_overlay[tray_mask > 0],
            0.35,
            np.full_like(tray_overlay[tray_mask > 0], (0, 255, 255)),
            0.65,
            0,
        )
        views["tray_overlay"] = tray_overlay
        views["tray_mask"] = cv2.cvtColor(tray_mask, cv2.COLOR_GRAY2BGR)
        views["object_mask"] = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)
        views["top_face_mask"] = cv2.cvtColor(top_face_mask, cv2.COLOR_GRAY2BGR)
        return views

    def _score_convex(self, features: Dict[str, object]) -> Tuple[str, float, float]:
        vertex_count = int(features["vertex_count"])
        if vertex_count == 3:
            return "triangle", 0.97, 0.96
        if vertex_count == 5:
            return "pentagon", 0.97, 0.96
        if vertex_count == 4 and bool(features["is_convex"]):
            label, geometry_score = classify_quadrilateral(features)
            template_score = 0.96 if label != "parallelogram" else 0.94
            return label, template_score, geometry_score
        if float(features["circularity"]) > 0.80 and vertex_count >= 6:
            return "circle", 0.98, 0.97
        return "uncertain", 0.40, 0.40

    def _score_non_convex(self, features: Dict[str, object]) -> Tuple[str, float, float]:
        return "cross", 0.90, 0.88

    def classify_contour(self, contour: np.ndarray) -> Dict[str, object]:
        features = extract_features(contour)
        if bool(features["is_convex"]):
            label, template_score, geometry_score = self._score_convex(features)
        else:
            label, template_score, geometry_score = self._score_non_convex(features)
        second_score = 0.55
        margin_score = max(0.0, min(1.0, template_score - second_score))
        confidence = fuse_confidence(template_score, geometry_score, margin_score)
        if label == "square" and confidence < UNCERTAIN_THRESHOLD:
            label = "uncertain"
        return {
            "label": label,
            "raw_label": label,
            "confidence": confidence,
            "features": features,
        }

    def detect_shapes(self, frame: np.ndarray) -> List[Dict[str, object]]:
        x, y, w, h = self.tray_roi
        roi = frame[y:y + h, x:x + w]
        mask = self.build_top_face_mask(frame)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Dict[str, object]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bx <= BORDER_MARGIN or by <= BORDER_MARGIN:
                continue
            if bx + bw >= roi.shape[1] - BORDER_MARGIN or by + bh >= roi.shape[0] - BORDER_MARGIN:
                continue
            if bw < 14 or bh < 14:
                continue
            result = self.classify_contour(contour)
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            centroid = _centroid_from_moments(moments)
            detections.append(
                {
                    "contour": contour + np.array([[[x, y]]], dtype=np.int32),
                    "label": result["label"],
                    "raw_label": result["raw_label"],
                    "confidence": result["confidence"],
                    "centroid": (centroid[0] + x, centroid[1] + y),
                    "bbox": (bx + x, by + y, bw, bh),
                }
            )
        return detections
