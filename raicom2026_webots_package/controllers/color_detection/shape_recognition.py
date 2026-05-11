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

TRAY_ROI_PIXELS = (206, 78, 196, 396)
MIN_CONTOUR_AREA = 180.0
MAX_CONTOUR_AREA = 12000.0
BORDER_MARGIN = 5
MOTION_REPORT_THRESHOLD = 6.0
TOP_FACE_H_TOLERANCE = 6
TOP_FACE_S_TOLERANCE = 42
TOP_FACE_V_DROP = 18
TOP_FACE_RGB_DISTANCE = 48.0
TOP_FACE_SEED_INSET = 0.24


@dataclass
class Track:
    history: deque
    centroid_history: deque
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


def radial_signature(contour: np.ndarray, bins: int = 36) -> np.ndarray:
    points = contour.reshape((-1, 2)).astype(np.float32)
    center = points.mean(axis=0)
    vectors = points - center
    radii = np.linalg.norm(vectors, axis=1)
    angles = (np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) + 360.0) % 360.0
    sums = np.zeros(bins, dtype=np.float32)
    counts = np.zeros(bins, dtype=np.float32)
    for radius, angle in zip(radii, angles):
        idx = int(angle / 360.0 * bins) % bins
        sums[idx] += radius
        counts[idx] += 1.0
    counts[counts == 0.0] = 1.0
    signature = sums / counts
    max_value = float(signature.max()) if len(signature) else 0.0
    if max_value > 0.0:
        signature = signature / max_value
    return signature


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


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
    hull_indices = cv2.convexHull(contour.astype(np.int32), returnPoints=False)
    defect_count = 0
    defect_depth_max = 0.0
    if hull_indices is not None and len(hull_indices) >= 3 and len(contour) >= 4:
        try:
            defects = cv2.convexityDefects(contour.astype(np.int32), hull_indices)
        except cv2.error:
            defects = None
        if defects is not None:
            defect_count = int(len(defects))
            defect_depth_max = float(np.max(defects[:, 0, 3])) / 256.0
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
    signature = radial_signature(contour)
    radial_std = float(np.std(signature))
    radial_peaks = int(np.sum((signature > np.roll(signature, 1)) & (signature > np.roll(signature, -1)) & (signature > 0.82)))
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
        "defect_count": defect_count,
        "defect_depth_max": defect_depth_max,
        "radial_std": radial_std,
        "angles": angles,
        "radial_peaks": radial_peaks,
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
                history=deque(maxlen=state.max_history),
                centroid_history=deque(maxlen=state.max_history),
                centroid=detection["centroid"],
                stable_label=str(detection["label"]),
                last_reported_centroid=detection["centroid"],
            )
            state.tracks[track_id].history.append((detection["raw_label"], detection["label"], detection["confidence"]))
            state.tracks[track_id].centroid_history.append(detection["centroid"])
            detection["id"] = track_id
            events.append({"type": "appeared", "id": track_id, "detection": detection})
            continue
        assigned.add(best_id)
        track = state.tracks[best_id]
        previous_centroid = track.centroid
        track.centroid = detection["centroid"]
        track.history.append((detection["raw_label"], detection["label"], detection["confidence"]))
        track.centroid_history.append(detection["centroid"])
        detection["id"] = best_id
        stable_candidates = [item[1] for item in track.history if item[1] != "uncertain"]
        if stable_candidates:
            track.stable_label = Counter(stable_candidates).most_common(1)[0][0]
        smoothed_centroid = (
            int(round(np.mean([point[0] for point in track.centroid_history]))),
            int(round(np.mean([point[1] for point in track.centroid_history]))),
        )
        moved_distance = float(np.linalg.norm(np.array(previous_centroid) - np.array(smoothed_centroid)))
        if moved_distance > MOTION_REPORT_THRESHOLD:
            track.last_reported_centroid = smoothed_centroid
            detection["centroid"] = smoothed_centroid
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
        rgb = roi.astype(np.int16)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([179, 90, 255]))
        colorful_mask = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([179, 255, 255]))
        color_balance = np.max(rgb, axis=2) - np.min(rgb, axis=2)
        rgb_mask = np.where(color_balance >= 18, 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(colorful_mask, rgb_mask)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_mask))
        return mask

    def build_color_core_mask(self, frame: np.ndarray, core_distance: float = 2.5) -> np.ndarray:
        x, y, w, h = self.tray_roi
        roi = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        base_mask = self.build_object_mask(frame)
        if cv2.countNonZero(base_mask) == 0:
            return base_mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(base_mask, connectivity=8)
        core_mask = np.zeros_like(base_mask)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < MIN_CONTOUR_AREA:
                continue

            component = labels == label
            ys, xs = np.where(component)
            if len(xs) == 0:
                continue

            component_hsv = hsv[ys, xs]
            component_rgb = roi[ys, xs].astype(np.float32)

            # Use the central region of the component as the top-face color seed.
            # This avoids locking onto a single bright highlight that can look triangular.
            left = int(round(stats[label, cv2.CC_STAT_LEFT]))
            top = int(round(stats[label, cv2.CC_STAT_TOP]))
            width = int(round(stats[label, cv2.CC_STAT_WIDTH]))
            height = int(round(stats[label, cv2.CC_STAT_HEIGHT]))
            inset_x = max(1, int(round(width * TOP_FACE_SEED_INSET)))
            inset_y = max(1, int(round(height * TOP_FACE_SEED_INSET)))
            inner_x0 = left + inset_x
            inner_y0 = top + inset_y
            inner_x1 = max(inner_x0 + 1, left + width - inset_x)
            inner_y1 = max(inner_y0 + 1, top + height - inset_y)

            ys_all, xs_all = np.where(component)
            inner_selector = (
                (xs_all >= inner_x0)
                & (xs_all < inner_x1)
                & (ys_all >= inner_y0)
                & (ys_all < inner_y1)
            )
            if np.any(inner_selector):
                seed_hsv = np.median(component_hsv[inner_selector], axis=0).astype(np.float32)
                seed_rgb = np.median(component_rgb[inner_selector], axis=0).astype(np.float32)
            else:
                seed_hsv = np.median(component_hsv, axis=0).astype(np.float32)
                seed_rgb = np.median(component_rgb, axis=0).astype(np.float32)

            h_values = component_hsv[:, 0].astype(np.float32)
            s_values = component_hsv[:, 1].astype(np.float32)
            v_values = component_hsv[:, 2].astype(np.float32)
            hue_delta = np.abs(h_values - seed_hsv[0])
            hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
            hsv_close = (
                (hue_delta <= TOP_FACE_H_TOLERANCE)
                & (np.abs(s_values - seed_hsv[1]) <= TOP_FACE_S_TOLERANCE)
                & (v_values >= seed_hsv[2] - TOP_FACE_V_DROP)
            )
            rgb_distance = np.linalg.norm(component_rgb - seed_rgb, axis=1)
            rgb_close = rgb_distance <= TOP_FACE_RGB_DISTANCE
            # Prioritize hue/saturation similarity with only a lower brightness bound.
            # RGB distance stays as a soft fallback so specular highlights on top faces survive.
            keep = hsv_close & ((rgb_close) | (v_values >= seed_hsv[2]))
            if not np.any(keep):
                keep = np.ones_like(v_values, dtype=bool)

            filtered = np.zeros_like(base_mask)
            filtered[ys[keep], xs[keep]] = 255
            filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
            if cv2.countNonZero(filtered) == 0:
                filtered[ys[keep], xs[keep]] = 255
            core_mask = cv2.bitwise_or(core_mask, filtered)

        if cv2.countNonZero(core_mask) == 0:
            return base_mask
        return core_mask

    def get_debug_views(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        views: Dict[str, np.ndarray] = {}
        tray_mask = self.build_tray_roi_mask(frame)
        object_mask = self.build_object_mask(frame)
        core_mask = self.build_color_core_mask(frame)
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
        views["core_mask"] = cv2.cvtColor(core_mask, cv2.COLOR_GRAY2BGR)
        return views

    def _score_convex(self, features: Dict[str, object]) -> Tuple[str, float, float]:
        vertex_count = int(features["vertex_count"])
        circularity = float(features["circularity"])
        if circularity > 0.84 and vertex_count >= 6:
            return "circle", 0.98, 0.96
        if vertex_count == 3 and bool(features["is_convex"]):
            return "triangle", 0.97, 0.96
        if vertex_count == 5 and bool(features["is_convex"]):
            return "pentagon", 0.97, 0.96
        if vertex_count == 4 and bool(features["is_convex"]):
            label, geometry_score = classify_quadrilateral(features)
            template_score = 0.96 if label != "parallelogram" else 0.94
            return label, template_score, geometry_score
        return "uncertain", 0.40, 0.40

    def _score_non_convex(self, features: Dict[str, object]) -> Tuple[str, float, float]:
        extent = float(features["extent"])
        radial_std = float(features["radial_std"])
        defect_count = int(features["defect_count"])
        peaks = int(features["radial_peaks"])
        circularity = float(features["circularity"])
        solidity = float(features["solidity"])
        if defect_count >= 5 and peaks >= 5 and circularity < 0.42:
            return "star", 0.96, 0.95
        if defect_count >= 4 and solidity > 0.84 and circularity > 0.50:
            return "four_leaf", 0.95, 0.95
        if defect_count >= 4 and extent < 0.70 and solidity < 0.84:
            return "cross", 0.94, 0.93
        if defect_count >= 4 and extent < 0.75:
            return "cross", 0.90, 0.88
        return "four_leaf", 0.90, 0.89

    def classify_contour(self, contour: np.ndarray) -> Dict[str, object]:
        features = extract_features(contour)
        if (
            int(features["defect_count"]) >= 12
            and float(features["circularity"]) > 0.50
            and float(features["solidity"]) > 0.84
            and int(features["radial_peaks"]) == 4
        ):
            candidates = [
                {"label": "four_leaf", "score": 0.96, "template_score": 0.96, "geometry_score": 0.95},
                {"label": "square", "score": 0.10, "template_score": 0.10, "geometry_score": 0.10},
                {"label": "rectangle", "score": 0.10, "template_score": 0.10, "geometry_score": 0.10},
                {"label": "parallelogram", "score": 0.10, "template_score": 0.10, "geometry_score": 0.10},
            ]
        elif float(features["circularity"]) > 0.78 and int(features["vertex_count"]) >= 6:
            candidates = [
                {"label": "circle", "score": 0.98, "template_score": 0.98, "geometry_score": 0.97},
                {"label": "four_leaf", "score": 0.20, "template_score": 0.20, "geometry_score": 0.20},
                {"label": "square", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12},
                {"label": "rectangle", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12},
                {"label": "parallelogram", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12},
            ]
        elif int(features["vertex_count"]) >= 6 and float(features["solidity"]) > 0.85 and int(features["radial_peaks"]) >= 5:
            candidates = [
                {"label": "four_leaf", "score": 0.95, "template_score": 0.95, "geometry_score": 0.94},
                {"label": "square", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12},
                {"label": "rectangle", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12},
                {"label": "parallelogram", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12},
            ]
        elif not bool(features["is_convex"]):
            label, template_score, geometry_score = self._score_non_convex(features)
            candidates = [{"label": label, "score": (template_score + geometry_score) / 2.0, "template_score": template_score, "geometry_score": geometry_score}]
            candidates.extend([
                {"label": "square", "score": 0.15, "template_score": 0.15, "geometry_score": 0.15},
                {"label": "rectangle", "score": 0.15, "template_score": 0.15, "geometry_score": 0.15},
                {"label": "parallelogram", "score": 0.15, "template_score": 0.15, "geometry_score": 0.15},
            ])
            candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        else:
            label, template_score, geometry_score = self._score_convex(features)
            candidates = [{"label": label, "score": (template_score + geometry_score) / 2.0, "template_score": template_score, "geometry_score": geometry_score}]

        if candidates[0]["label"] in {"square", "rectangle", "parallelogram"}:
            ratio = float(features["aspect_ratio"])
            diff = abs(ratio - 1.0)
            candidates.extend([
                {"label": "square", "score": max(0.0, 0.9 - diff), "template_score": max(0.0, 0.9 - diff), "geometry_score": max(0.0, 0.88 - diff)},
                {"label": "rectangle", "score": min(0.89, 0.72 + diff), "template_score": min(0.89, 0.70 + diff), "geometry_score": min(0.89, 0.74 + diff)},
                {"label": "parallelogram", "score": 0.55 if candidates[0]["label"] != "parallelogram" else 0.88, "template_score": 0.55, "geometry_score": 0.55},
            ])
        elif candidates[0]["label"] == "circle":
            candidates.append({"label": "pentagon", "score": 0.35, "template_score": 0.35, "geometry_score": 0.35})
        elif candidates[0]["label"] == "triangle":
            candidates.append({"label": "pentagon", "score": 0.30, "template_score": 0.30, "geometry_score": 0.30})
        elif candidates[0]["label"] == "pentagon":
            candidates.append({"label": "circle", "score": 0.35, "template_score": 0.35, "geometry_score": 0.35})
        else:
            candidates.append({"label": "uncertain", "score": 0.2, "template_score": 0.2, "geometry_score": 0.2})

        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else {"score": 0.0}
        margin_score = max(0.0, min(1.0, float(best["score"]) - float(second["score"])))
        confidence = fuse_confidence(float(best["template_score"]), float(best["geometry_score"]), margin_score)
        if best["label"] == "square":
            angles = [float(angle) for angle in features["angles"]]
            if max(abs(angle - 90.0) for angle in angles) <= 5.0 and abs(float(features["aspect_ratio"]) - 1.0) <= 0.10:
                confidence = max(confidence, 0.97)
        elif best["label"] == "rectangle":
            angles = [float(angle) for angle in features["angles"]]
            max_angle_error = max(abs(angle - 90.0) for angle in angles)
            aspect_ratio = float(features["aspect_ratio"])
            if max_angle_error <= 5.0 and aspect_ratio >= 1.25:
                confidence = max(confidence, 0.95)
            else:
                confidence = min(confidence, 0.84)
        elif best["label"] == "parallelogram":
            angles = [float(angle) for angle in features["angles"]]
            max_angle_error = max(abs(angle - 90.0) for angle in angles)
            if max_angle_error >= 10.0:
                confidence = max(confidence, 0.92)
            else:
                confidence = min(confidence, 0.84)
        elif best["label"] == "four_leaf":
            confidence = max(confidence, 0.94 if float(features["solidity"]) > 0.84 else 0.90)
        label = str(best["label"]) if confidence >= UNCERTAIN_THRESHOLD else "uncertain"
        return {
            "label": label,
            "raw_label": str(best["label"]),
            "confidence": confidence,
            "features": features,
        }

    def detect_shapes(self, frame: np.ndarray) -> List[Dict[str, object]]:
        x, y, w, h = self.tray_roi
        top_mask = self.build_color_core_mask(frame)
        contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Dict[str, object]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bx <= BORDER_MARGIN or by <= BORDER_MARGIN:
                continue
            if bx + bw >= top_mask.shape[1] - 2 or by + bh >= top_mask.shape[0] - 2:
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
