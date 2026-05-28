from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


TOP_TRAY_RECT_PIXELS = (150, 22, 245, 464)
TOP_TRAY_SLOT_MAP = (
    {"slot_index": 0, "slot_rect": (168, 32, 62, 72), "expected_labels": ("cross",)},
    {"slot_index": 1, "slot_rect": (232, 32, 72, 72), "expected_labels": ("rectangle",)},
    {"slot_index": 2, "slot_rect": (306, 32, 72, 72), "expected_labels": ("square",)},
    {"slot_index": 3, "slot_rect": (168, 104, 62, 76), "expected_labels": ("star",)},
    {"slot_index": 4, "slot_rect": (232, 104, 72, 76), "expected_labels": ("pentagon",)},
    {"slot_index": 5, "slot_rect": (306, 104, 72, 76), "expected_labels": ("triangle",)},
    {"slot_index": 6, "slot_rect": (168, 180, 62, 82), "expected_labels": ("four_leaf",)},
    {"slot_index": 7, "slot_rect": (232, 180, 72, 82), "expected_labels": ("parallelogram",)},
    {"slot_index": 8, "slot_rect": (306, 180, 72, 82), "expected_labels": ("circle",)},
    {"slot_index": 9, "slot_rect": (168, 252, 62, 74), "expected_labels": ("cross",)},
    {"slot_index": 10, "slot_rect": (232, 252, 72, 74), "expected_labels": ("rectangle",)},
    {"slot_index": 11, "slot_rect": (306, 252, 72, 74), "expected_labels": ("square",)},
    {"slot_index": 12, "slot_rect": (168, 326, 62, 78), "expected_labels": ("star",)},
    {"slot_index": 13, "slot_rect": (232, 326, 72, 78), "expected_labels": ("pentagon",)},
    {"slot_index": 14, "slot_rect": (306, 326, 72, 78), "expected_labels": ("triangle",)},
    {"slot_index": 15, "slot_rect": (168, 404, 62, 76), "expected_labels": ("four_leaf",)},
    {"slot_index": 16, "slot_rect": (232, 404, 72, 76), "expected_labels": ("parallelogram",)},
    {"slot_index": 17, "slot_rect": (306, 404, 72, 76), "expected_labels": ("circle",)},
)


def _template_contour_for_label(label: str) -> np.ndarray:
    templates = {
        "square": np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32),
        "rectangle": np.array([[15, 25], [85, 25], [85, 75], [15, 75]], dtype=np.float32),
        "parallelogram": np.array([[20, 25], [80, 25], [70, 75], [10, 75]], dtype=np.float32),
        "four_leaf": np.array(
            [[50, 10], [62, 28], [80, 40], [62, 52], [50, 70], [38, 52], [20, 40], [38, 28]],
            dtype=np.float32,
        ),
    }
    points = templates.get(label)
    if points is None:
        return np.array([[[20, 20]], [[80, 20]], [[80, 80]], [[20, 80]]], dtype=np.int32)
    return points.reshape((-1, 1, 2)).astype(np.int32)


class TopTrayRecessRecognizer:
    def __init__(self):
        self.class_names = (
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
        self.last_debug: Dict[str, np.ndarray | None] = {}

    def build_top_tray_mask(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = TOP_TRAY_RECT_PIXELS
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        return mask

    def draw_layout(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        x, y, w, h = TOP_TRAY_RECT_PIXELS
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        for slot in self.get_slot_map():
            sx, sy, sw, sh = slot["slot_rect"]
            cv2.rectangle(output, (sx, sy), (sx + sw, sy + sh), (180, 180, 180), 1)
            cv2.putText(output, str(slot["slot_index"]), (sx + 2, sy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        return output

    def _expanded_slot_rect(self, slot: dict) -> Tuple[int, int, int, int]:
        x, y, w, h = slot["slot_rect"]
        left_pad = 10
        right_pad = 6
        top_pad = 4
        bottom_pad = 12
        x = max(0, x - left_pad)
        y = max(0, y - top_pad)
        w = w + left_pad + right_pad
        h = h + top_pad + bottom_pad
        return x, y, w, h

    def _smooth_slot_contour(self, contour: np.ndarray) -> np.ndarray:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 40.0 or perimeter == 0.0:
            return contour

        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
        mask = np.zeros((bbox_h + 8, bbox_w + 8), dtype=np.uint8)
        shifted = contour.astype(np.int32) - np.array([[[bbox_x - 4, bbox_y - 4]]], dtype=np.int32)
        cv2.drawContours(mask, [shifted], -1, 255, -1)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return contour
        smoothed = max(contours, key=cv2.contourArea) + np.array([[[bbox_x - 4, bbox_y - 4]]], dtype=np.int32)
        return smoothed

    def get_slot_map(self):
        return list(TOP_TRAY_SLOT_MAP)

    def extract_slot_contour(self, frame: np.ndarray, slot: dict):
        x, y, w, h = self._expanded_slot_rect(slot)
        roi = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        bright_threshold = int(min(245, max(170, float(np.mean(blur)) + 12.0)))
        _, global_mask = cv2.threshold(blur, bright_threshold, 255, cv2.THRESH_BINARY)
        adaptive_mask = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            -3,
        )
        mask = cv2.bitwise_or(global_mask, adaptive_mask)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 40.0:
            return None
        contour = contour + np.array([[[x, y]]], dtype=np.int32)
        return self._smooth_slot_contour(contour)

    def classify_slot_contour(self, contour: np.ndarray, expected_labels: tuple[str, ...]) -> dict:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        epsilon = 0.02 * perimeter if perimeter else 0.0
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h else 0.0
        circularity = 0.0 if perimeter == 0 else (4.0 * np.pi * area) / (perimeter * perimeter)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else area
        solidity = area / hull_area if hull_area else 0.0

        best_label = expected_labels[0]
        best_similarity = -1.0
        for label in expected_labels:
            template = _template_contour_for_label(label)
            similarity = float(np.exp(-cv2.matchShapes(contour, template, cv2.CONTOURS_MATCH_I1, 0.0)))
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label

        if best_label == "square" and len(approx) == 4 and 0.80 <= aspect_ratio <= 1.20:
            return {"label": "square", "confidence": max(0.90, best_similarity)}
        if best_label == "rectangle" and len(approx) == 4:
            return {"label": "rectangle", "confidence": max(0.88, best_similarity)}
        if best_label == "parallelogram" and len(approx) == 4:
            return {"label": "parallelogram", "confidence": max(0.88, best_similarity)}
        if best_label == "four_leaf" and solidity > 0.76 and circularity > 0.40:
            return {"label": "four_leaf", "confidence": max(0.88, best_similarity)}
        return {"label": best_label, "confidence": max(0.75, best_similarity)}

    def detect_recesses(self, frame: np.ndarray) -> list[dict]:
        detections = []
        tray_mask = self.build_top_tray_mask(frame)
        masked_frame = cv2.bitwise_and(frame, frame, mask=tray_mask)
        tray_pixels = masked_frame[tray_mask > 0]
        if tray_pixels.size > 0:
            tray_reference = np.median(tray_pixels, axis=0).astype(np.float32)
            color_distance = np.linalg.norm(frame.astype(np.float32) - tray_reference, axis=2)
            object_mask = np.where((color_distance >= 16.0) & (tray_mask > 0), 255, 0).astype(np.uint8)
            kernel = np.ones((3, 3), dtype=np.uint8)
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        else:
            object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.last_debug = {"tray_mask": tray_mask, "object_mask": object_mask}
        for slot in self.get_slot_map():
            contour = self.extract_slot_contour(frame, slot)
            if contour is None:
                continue
            result = self.classify_slot_contour(contour, tuple(slot["expected_labels"]))
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            centroid = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                {
                    "slot_index": slot["slot_index"],
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "centroid": centroid,
                    "contour": contour,
                    "bbox": (x, y, w, h),
                }
            )
        return detections

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        output = self.draw_layout(frame)
        for detection in detections:
            contour = detection["contour"]
            x, y, w, h = detection["bbox"]
            cx, cy = detection["centroid"]
            cv2.drawContours(output, [contour], -1, (0, 0, 255), 3)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                output,
                f"{detection['slot_index']}",
                (x, max(16, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return output

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
        return views
