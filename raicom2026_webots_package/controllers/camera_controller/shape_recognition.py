from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
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

MIN_CONTOUR_AREA = 150.0
UNCERTAIN_THRESHOLD = 0.80
RIGHT_ANGLE_TOLERANCE = 14.0
SQUARE_RATIO_TOLERANCE = 0.18
ASSOCIATION_DISTANCE = 40.0
MOTION_REPORT_THRESHOLD = 4.0
BOARD_MIN_AREA_RATIO = 0.18
WARPED_BOARD_SIZE = (320, 320)
FIXED_BOARD_QUADS_NORMALIZED = (
    np.array([[0.4651, 0.0531], [0.8098, 0.0531], [0.8098, 0.5343], [0.4651, 0.5343]], dtype=np.float32),
)
FIXED_LEFT_TRAY_RECT_PIXELS = (157, 24, 230, 219)
FIXED_BOTTOM_TRAY_RECT_PIXELS = (184, 250, 194, 184)
FIXED_BOTTOM_TRAY_SLOT_CENTERS = (
    (212, 288), (274, 288), (338, 288),
    (212, 349), (276, 349), (338, 349),
    (211, 410), (274, 410), (338, 410),
)
BOTTOM_TRAY_TEMPLATE_LABELS = (
    "cross",
    "square",
    "rectangle",
    "star",
    "pentagon",
    "triangle",
    "four_leaf",
    "parallelogram",
    "circle",
)
BOTTOM_TRAY_SLOT_BOX_SIZE = (56, 54)
BOTTOM_TRAY_CELL_PADDING = 6
BOTTOM_TRAY_CENTER_CROP_SCALE = 0.64
BOTTOM_SLOT_BINARY_THRESHOLD = 210
YELLOW_LOWER = np.array([15, 70, 80], dtype=np.uint8)
YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)
MAX_CONTOUR_AREA = 12000.0
MIN_EXTENT_FOR_LARGE_CONTOURS = 0.35
MAX_BORDER_TOUCH = 6
TRAY_COLOR_DISTANCE_THRESHOLD = 18.0
DARK_OBJECT_THRESHOLD = 170


@dataclass
class Track:
    history: deque
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


def normalize_contour(contour: np.ndarray) -> np.ndarray:
    points = contour.astype(np.float32).reshape((-1, 2))
    centered = points - points.mean(axis=0)
    scale = float(np.max(np.abs(centered)))
    if scale == 0.0:
        scale = 1.0
    normalized = centered / scale
    return normalized.reshape((-1, 1, 2))


def ordered_points_from_approx(approx: np.ndarray) -> np.ndarray:
    points = approx.reshape((-1, 2)).astype(np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def polygon_side_lengths(points: np.ndarray) -> np.ndarray:
    rolled = np.roll(points, -1, axis=0)
    return np.linalg.norm(rolled - points, axis=1)


def polygon_angles(points: np.ndarray) -> List[float]:
    result: List[float] = []
    total = len(points)
    for index in range(total):
        prev_point = points[index - 1]
        point = points[index]
        next_point = points[(index + 1) % total]
        v1 = prev_point - point
        v2 = next_point - point
        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denominator == 0.0:
            result.append(0.0)
            continue
        cosine = float(np.dot(v1, v2) / denominator)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        result.append(float(np.degrees(np.arccos(cosine))))
    return result


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


def _convexity_defects(contour: np.ndarray, hull_indices: np.ndarray) -> np.ndarray | None:
    if hull_indices is None or len(hull_indices) < 3 or len(contour) < 4:
        return None
    try:
        return cv2.convexityDefects(contour.astype(np.int32), hull_indices)
    except cv2.error:
        return None


def _centroid_from_moments(moments: Dict[str, float]) -> Tuple[int, int]:
    if moments["m00"] == 0:
        return 0, 0
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    points = points.reshape((4, 2)).astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _scale_polygon(points: np.ndarray, scale: float) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    scaled = (points - center) * scale + center
    return scaled.astype(np.float32)


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
    defects = _convexity_defects(contour.astype(np.int32), hull_indices)
    defect_depths: List[float] = []
    if defects is not None:
        for defect in defects[:, 0]:
            defect_depths.append(float(defect[3]) / 256.0)
    ordered_points = ordered_points_from_approx(approx) if len(approx) >= 3 else approx.reshape((-1, 2))
    side_lengths = polygon_side_lengths(ordered_points) if len(ordered_points) >= 2 else np.array([], dtype=np.float32)
    angles = polygon_angles(ordered_points) if len(ordered_points) >= 3 else []
    side_ratio = float(side_lengths.max() / side_lengths.min()) if len(side_lengths) and side_lengths.min() > 0 else 0.0
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
        "angles": angles,
        "side_ratio": side_ratio,
        "defect_count": len(defect_depths),
        "defect_depth_mean": float(np.mean(defect_depths)) if defect_depths else 0.0,
        "defect_depth_max": float(np.max(defect_depths)) if defect_depths else 0.0,
        "radial_std": radial_std,
        "radial_signature": signature,
        "radial_peaks": radial_peaks,
        "centroid": centroid,
    }


def fuse_confidence(template_score: float, geometry_score: float, margin_score: float) -> float:
    return 0.45 * template_score + 0.40 * geometry_score + 0.15 * margin_score


def classify_quadrilateral(features: Dict[str, object]) -> Tuple[str, float]:
    ratio = float(features["aspect_ratio"])
    angles = [float(angle) for angle in features["angles"]]
    near_right = sum(abs(angle - 90.0) < RIGHT_ANGLE_TOLERANCE for angle in angles) >= 3
    side_ratio = float(features["side_ratio"])
    if near_right and abs(ratio - 1.0) < SQUARE_RATIO_TOLERANCE and abs(side_ratio - 1.0) < 0.12:
        return "square", 0.97
    if near_right:
        return "rectangle", 0.95
    return "parallelogram", 0.92


def draw_detections(frame: np.ndarray, detections: Sequence[Dict[str, object]]) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        label = str(detection["label"])
        color = (0, 200, 0) if label != "uncertain" else (0, 180, 255)
        cv2.drawContours(output, [detection["contour"].astype(np.int32)], -1, color, 2)
        x, y, w, h = detection["bbox"]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cx, cy = detection["centroid"]
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
        prefix = f"#{detection['id']} " if "id" in detection else ""
        caption = f"{prefix}{label} {float(detection['confidence']):.2f}"
        cv2.putText(output, caption, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return output


def update_tracks(
    state: TrackerState,
    detections: Sequence[Dict[str, object]],
    max_distance: float = ASSOCIATION_DISTANCE,
    motion_threshold: float = MOTION_REPORT_THRESHOLD,
) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    unmatched_tracks = set(state.tracks.keys())

    for detection in detections:
        matched_id = None
        best_distance = max_distance
        for track_id, track in state.tracks.items():
            distance = float(np.linalg.norm(np.array(track.centroid) - np.array(detection["centroid"])))
            if distance < best_distance:
                best_distance = distance
                matched_id = track_id

        if matched_id is None:
            track_id = state.next_id
            state.next_id += 1
            state.tracks[track_id] = Track(
                history=deque(maxlen=state.max_history),
                centroid=detection["centroid"],
                stable_label=str(detection["label"]),
                last_reported_centroid=detection["centroid"],
            )
            state.tracks[track_id].history.append((detection["raw_label"], detection["label"], detection["confidence"]))
            detection["id"] = track_id
            events.append({"type": "appeared", "id": track_id, "detection": detection})
            continue

        unmatched_tracks.discard(matched_id)
        track = state.tracks[matched_id]
        track.centroid = detection["centroid"]
        track.missing_frames = 0
        track.history.append((detection["raw_label"], detection["label"], detection["confidence"]))
        detection["id"] = matched_id

        stable_candidates = [item[1] for item in track.history if item[1] != "uncertain"]
        if stable_candidates:
            winner = Counter(stable_candidates).most_common(1)[0][0]
            track.stable_label = winner

        moved_distance = float(np.linalg.norm(np.array(track.centroid) - np.array(track.last_reported_centroid)))
        if moved_distance > motion_threshold:
            events.append({"type": "moved", "id": matched_id, "detection": detection})
            track.last_reported_centroid = track.centroid

    for track_id in list(unmatched_tracks):
        track = state.tracks[track_id]
        track.missing_frames += 1
        if track.missing_frames >= 2:
            del state.tracks[track_id]
            events.append({"type": "disappeared", "id": track_id})

    return events


class ShapeRecognizer:
    def __init__(self):
        self.class_names = CLASS_NAMES
        self.last_debug: Dict[str, np.ndarray | None] = {}
        self.bottom_tray_templates = self._load_bottom_tray_templates()

    def _bottom_tray_template_dir(self) -> Path:
        return Path(r"D:\hlh1\codex-local\bottom_tray_templates")

    def _load_bottom_tray_templates(self) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        template_dir = self._bottom_tray_template_dir()
        for label in BOTTOM_TRAY_TEMPLATE_LABELS:
            path = template_dir / f"{label}.png"
            if not path.exists():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            templates[label] = binary
        return templates

    def build_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        _, otsu_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive_mask = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            4,
        )
        grayscale_mask = cv2.bitwise_or(otsu_mask, adaptive_mask)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        mask = cv2.bitwise_or(grayscale_mask, yellow_mask)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def find_board_quad(self, frame: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), dtype=np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(frame.shape[0] * frame.shape[1])
        best_quad = None
        best_area = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < image_area * BOARD_MIN_AREA_RATIO:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) != 4:
                continue
            if area > best_area:
                best_area = area
                best_quad = _order_quad_points(approx.reshape((4, 2)))
        return best_quad

    def rectify_board(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
        quad = self.find_board_quad(frame)
        if quad is None:
            return None
        width, height = WARPED_BOARD_SIZE
        target = np.array(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(quad, target)
        warped = cv2.warpPerspective(frame, matrix, WARPED_BOARD_SIZE)
        inverse = cv2.getPerspectiveTransform(target, quad)
        return warped, inverse

    def fixed_board_rectifications(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        rectifications: List[Tuple[np.ndarray, np.ndarray]] = []
        width = float(frame.shape[1])
        height = float(frame.shape[0])
        if width < 1000 or height < 700:
            return rectifications
        target = np.array(
            [[0.0, 0.0], [WARPED_BOARD_SIZE[0] - 1.0, 0.0], [WARPED_BOARD_SIZE[0] - 1.0, WARPED_BOARD_SIZE[1] - 1.0], [0.0, WARPED_BOARD_SIZE[1] - 1.0]],
            dtype=np.float32,
        )
        for normalized_quad in FIXED_BOARD_QUADS_NORMALIZED:
            quad = normalized_quad.copy()
            quad[:, 0] *= width
            quad[:, 1] *= height
            matrix = cv2.getPerspectiveTransform(quad, target)
            warped = cv2.warpPerspective(frame, matrix, WARPED_BOARD_SIZE)
            inverse = cv2.getPerspectiveTransform(target, quad)
            rectifications.append((warped, inverse))
        return rectifications

    def build_left_tray_frame_mask(self, frame: np.ndarray, inset_scale: float = 0.94) -> np.ndarray | None:
        height, width = frame.shape[:2]
        if (width, height) == (640, 480):
            x, y, w, h = FIXED_LEFT_TRAY_RECT_PIXELS
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            return mask
        width = float(width)
        height = float(height)
        if width < 1000 or height < 700:
            return None
        normalized_quad = FIXED_BOARD_QUADS_NORMALIZED[0].copy()
        normalized_quad[:, 0] *= width
        normalized_quad[:, 1] *= height
        quad = _scale_polygon(normalized_quad, inset_scale)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        return mask

    def build_left_tray_roi_mask(self, frame: np.ndarray) -> np.ndarray | None:
        return self.build_left_tray_frame_mask(frame, inset_scale=0.72)

    def build_left_tray_object_mask(self, frame: np.ndarray) -> np.ndarray | None:
        tray_mask = self.build_left_tray_roi_mask(frame)
        if tray_mask is None:
            self.last_debug = {"tray_mask": None, "object_mask": None}
            return None

        masked_frame = cv2.bitwise_and(frame, frame, mask=tray_mask)
        tray_pixels = masked_frame[tray_mask > 0]
        if tray_pixels.size == 0:
            return None

        tray_reference = np.median(tray_pixels, axis=0).astype(np.float32)
        color_distance = np.linalg.norm(frame.astype(np.float32) - tray_reference, axis=2)
        color_mask = np.where(color_distance >= TRAY_COLOR_DISTANCE_THRESHOLD, 255, 0).astype(np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, DARK_OBJECT_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

        mask = cv2.bitwise_or(color_mask, dark_mask)
        mask = cv2.bitwise_or(mask, yellow_mask)
        mask = cv2.bitwise_and(mask, tray_mask)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.last_debug = {"tray_mask": tray_mask, "object_mask": mask}
        return mask

    def get_bottom_tray_rect(self, frame: np.ndarray) -> Tuple[int, int, int, int] | None:
        height, width = frame.shape[:2]
        if (width, height) != (640, 480):
            return None
        return FIXED_BOTTOM_TRAY_RECT_PIXELS

    def get_bottom_tray_cells(self, frame: np.ndarray) -> List[Dict[str, Tuple[int, int, int, int] | Tuple[int, int] | int]]:
        rect = self.get_bottom_tray_rect(frame)
        if rect is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        slot_w, slot_h = BOTTOM_TRAY_SLOT_BOX_SIZE
        cells: List[Dict[str, Tuple[int, int, int, int] | Tuple[int, int] | int]] = []

        for slot_index, center in enumerate(FIXED_BOTTOM_TRAY_SLOT_CENTERS):
            center_x, center_y = center
            cell_x = max(0, center_x - slot_w // 2)
            cell_y = max(0, center_y - slot_h // 2)
            cell_w = min(slot_w, frame_w - cell_x)
            cell_h = min(slot_h, frame_h - cell_y)
            inner_rect = (
                cell_x + BOTTOM_TRAY_CELL_PADDING,
                cell_y + BOTTOM_TRAY_CELL_PADDING,
                max(8, cell_w - 2 * BOTTOM_TRAY_CELL_PADDING),
                max(8, cell_h - 2 * BOTTOM_TRAY_CELL_PADDING),
            )
            cells.append(
                {
                    "slot_index": slot_index,
                    "cell_rect": (cell_x, cell_y, cell_w, cell_h),
                    "inner_rect": inner_rect,
                    "center_pixel": center,
                }
            )
        return cells

    def _bottom_slot_crop(self, frame: np.ndarray, inner_rect: Tuple[int, int, int, int]) -> Tuple[int, int, np.ndarray]:
        x, y, w, h = inner_rect
        crop_w = max(8, int(w * BOTTOM_TRAY_CENTER_CROP_SCALE))
        crop_h = max(8, int(h * BOTTOM_TRAY_CENTER_CROP_SCALE))
        crop_x = x + (w - crop_w) // 2
        crop_y = y + (h - crop_h) // 2
        roi = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        return crop_x, crop_y, roi

    def build_bottom_slot_mask(self, frame: np.ndarray, inner_rect: Tuple[int, int, int, int]) -> np.ndarray:
        _, _, roi = self._bottom_slot_crop(frame, inner_rect)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, BOTTOM_SLOT_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def match_bottom_slot_template(self, slot_mask: np.ndarray) -> Tuple[str, float]:
        if not self.bottom_tray_templates:
            return "uncertain", 0.0

        best_label = "uncertain"
        best_score = 0.0
        slot_binary = slot_mask > 0

        for label, template in self.bottom_tray_templates.items():
            resized = cv2.resize(template, (slot_mask.shape[1], slot_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            template_binary = resized > 0
            union = np.logical_or(slot_binary, template_binary).sum()
            if union == 0:
                score = 0.0
            else:
                intersection = np.logical_and(slot_binary, template_binary).sum()
                score = float(intersection / union)
            if score > best_score:
                best_score = score
                best_label = label

        if best_score < 0.20:
            return "uncertain", best_score
        return best_label, best_score

    def classify_bottom_slot(self, frame: np.ndarray, cell: Dict[str, object]) -> Dict[str, object]:
        inner_rect = cell["inner_rect"]
        x, y, w, h = inner_rect
        crop_x, crop_y, roi = self._bottom_slot_crop(frame, inner_rect)
        mask = self.build_bottom_slot_mask(frame, inner_rect)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "slot_index": int(cell["slot_index"]),
                "label": "uncertain",
                "confidence": 0.0,
                "center_pixel": cell["center_pixel"],
                "bbox": (crop_x, crop_y, roi.shape[1], roi.shape[0]),
            }

        contour = max(contours, key=cv2.contourArea)
        contour = contour + np.array([[[crop_x, crop_y]]], dtype=np.int32)
        label, score = self.match_bottom_slot_template(mask)
        return {
            "slot_index": int(cell["slot_index"]),
            "label": label,
            "confidence": score,
            "center_pixel": cell["center_pixel"],
            "bbox": (crop_x, crop_y, roi.shape[1], roi.shape[0]),
            "contour": contour,
        }

    def detect_bottom_tray_slots(self, frame: np.ndarray) -> List[Dict[str, object]]:
        cells = self.get_bottom_tray_cells(frame)
        return [self.classify_bottom_slot(frame, cell) for cell in cells]

    def get_debug_views(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        views: Dict[str, np.ndarray] = {}
        tray_mask = self.last_debug.get("tray_mask")
        object_mask = self.last_debug.get("object_mask")
        if tray_mask is not None:
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
        if object_mask is not None:
            views["object_mask"] = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)
        bottom_rect = self.get_bottom_tray_rect(frame)
        if bottom_rect is not None:
            x, y, w, h = bottom_rect
            bottom_overlay = frame.copy()
            cv2.rectangle(bottom_overlay, (x, y), (x + w, y + h), (255, 180, 0), 2)
            for cell in self.get_bottom_tray_cells(frame):
                cx, cy = cell["center_pixel"]
                cell_x, cell_y, cell_w, cell_h = cell["cell_rect"]
                cv2.rectangle(bottom_overlay, (cell_x, cell_y), (cell_x + cell_w, cell_y + cell_h), (0, 255, 255), 1)
                cv2.circle(bottom_overlay, (cx, cy), 3, (0, 0, 255), -1)
            views["bottom_tray_overlay"] = bottom_overlay
        return views

    def _project_detection(self, detection: Dict[str, object], inverse: np.ndarray) -> Dict[str, object]:
        contour = detection["contour"].astype(np.float32)
        projected = cv2.perspectiveTransform(contour.reshape((-1, 1, 2)), inverse)
        moments = cv2.moments(projected)
        centroid = _centroid_from_moments(moments)
        x, y, w, h = cv2.boundingRect(projected.astype(np.int32))
        updated = detection.copy()
        updated["contour"] = projected.astype(np.int32)
        updated["centroid"] = centroid
        updated["bbox"] = (x, y, w, h)
        return updated

    def _detect_from_mask(self, frame: np.ndarray, mask: np.ndarray) -> List[Dict[str, object]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Dict[str, object]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if x <= MAX_BORDER_TOUCH or y <= MAX_BORDER_TOUCH:
                continue
            if x + w >= frame.shape[1] - MAX_BORDER_TOUCH or y + h >= frame.shape[0] - MAX_BORDER_TOUCH:
                continue
            extent = area / float(w * h) if w and h else 0.0
            if area > 5000.0 and extent < MIN_EXTENT_FOR_LARGE_CONTOURS:
                continue
            result = self.classify_contour(contour)
            centroid = _centroid_from_moments(moments)
            detections.append(
                {
                    "contour": contour,
                    "label": result["label"],
                    "raw_label": result["raw_label"],
                    "confidence": result["confidence"],
                    "centroid": centroid,
                    "bbox": (x, y, w, h),
                }
            )
        return detections

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

    def score_candidates(self, features: Dict[str, object], contour: np.ndarray) -> List[Dict[str, float | str]]:
        candidates: List[Dict[str, float | str]] = []
        if (
            int(features["defect_count"]) >= 12
            and float(features["circularity"]) > 0.50
            and float(features["solidity"]) > 0.84
            and int(features["radial_peaks"]) == 4
        ):
            candidates.append({"label": "four_leaf", "score": 0.96, "template_score": 0.96, "geometry_score": 0.95})
            candidates.append({"label": "square", "score": 0.10, "template_score": 0.10, "geometry_score": 0.10})
            candidates.append({"label": "rectangle", "score": 0.10, "template_score": 0.10, "geometry_score": 0.10})
            candidates.append({"label": "parallelogram", "score": 0.10, "template_score": 0.10, "geometry_score": 0.10})
            return candidates
        if float(features["circularity"]) > 0.78 and int(features["vertex_count"]) >= 6:
            candidates.append({"label": "circle", "score": 0.98, "template_score": 0.98, "geometry_score": 0.97})
            candidates.append({"label": "four_leaf", "score": 0.20, "template_score": 0.20, "geometry_score": 0.20})
            candidates.append({"label": "square", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12})
            candidates.append({"label": "rectangle", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12})
            candidates.append({"label": "parallelogram", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12})
            return candidates
        if int(features["vertex_count"]) >= 6 and float(features["solidity"]) > 0.85 and int(features["radial_peaks"]) >= 5:
            candidates.append({"label": "four_leaf", "score": 0.95, "template_score": 0.95, "geometry_score": 0.94})
            candidates.append({"label": "square", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12})
            candidates.append({"label": "rectangle", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12})
            candidates.append({"label": "parallelogram", "score": 0.12, "template_score": 0.12, "geometry_score": 0.12})
            return candidates
        if not bool(features["is_convex"]):
            label, template_score, geometry_score = self._score_non_convex(features)
            candidates.append({"label": label, "score": (template_score + geometry_score) / 2.0, "template_score": template_score, "geometry_score": geometry_score})
            candidates.append({"label": "square", "score": 0.15, "template_score": 0.15, "geometry_score": 0.15})
            candidates.append({"label": "rectangle", "score": 0.15, "template_score": 0.15, "geometry_score": 0.15})
            candidates.append({"label": "parallelogram", "score": 0.15, "template_score": 0.15, "geometry_score": 0.15})
            candidates.sort(key=lambda item: float(item["score"]), reverse=True)
            return candidates
        if bool(features["is_convex"]):
            label, template_score, geometry_score = self._score_convex(features)
            candidates.append({"label": label, "score": (template_score + geometry_score) / 2.0, "template_score": template_score, "geometry_score": geometry_score})
        else:
            label, template_score, geometry_score = self._score_non_convex(features)
            candidates.append({"label": label, "score": (template_score + geometry_score) / 2.0, "template_score": template_score, "geometry_score": geometry_score})

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
        return candidates

    def classify_contour(self, contour: np.ndarray) -> Dict[str, object]:
        features = extract_features(contour)
        candidates = self.score_candidates(features, contour)
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
        left_tray_object_mask = self.build_left_tray_object_mask(frame)
        if left_tray_object_mask is not None:
            return self._detect_from_mask(frame, left_tray_object_mask)
        rectified = self.rectify_board(frame)
        if rectified is not None:
            warped, inverse = rectified
            warped_mask = self.build_mask(warped)
            detections = self._detect_from_mask(warped, warped_mask)
            return [self._project_detection(detection, inverse) for detection in detections]
        mask = self.build_mask(frame)
        return self._detect_from_mask(frame, mask)
