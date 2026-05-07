from pathlib import Path

import cv2
import numpy as np

from shape_recognition import (
    BOTTOM_SLOT_BINARY_THRESHOLD,
    BOTTOM_TRAY_CENTER_CROP_SCALE,
    BOTTOM_TRAY_TEMPLATE_LABELS,
    FIXED_BOTTOM_TRAY_SLOT_CENTERS,
    BOTTOM_TRAY_SLOT_BOX_SIZE,
)


SOURCE_IMAGE = Path(__file__).resolve().parent / "debug_frames" / "000195_raw.png"
OUTPUT_DIR = Path(r"D:\hlh1\codex-local\bottom_tray_templates")


def slot_crop(frame: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    slot_w, slot_h = BOTTOM_TRAY_SLOT_BOX_SIZE
    center_x, center_y = center
    cell_x = max(0, center_x - slot_w // 2)
    cell_y = max(0, center_y - slot_h // 2)
    crop_w = max(8, int((slot_w - 12) * BOTTOM_TRAY_CENTER_CROP_SCALE))
    crop_h = max(8, int((slot_h - 12) * BOTTOM_TRAY_CENTER_CROP_SCALE))
    crop_x = cell_x + ((slot_w - 12) - crop_w) // 2 + 6
    crop_y = cell_y + ((slot_h - 12) - crop_h) // 2 + 6
    return frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]


def slot_mask(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, BOTTOM_SLOT_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def main():
    frame = cv2.imread(str(SOURCE_IMAGE))
    if frame is None:
        raise FileNotFoundError(SOURCE_IMAGE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, center in zip(BOTTOM_TRAY_TEMPLATE_LABELS, FIXED_BOTTOM_TRAY_SLOT_CENTERS):
        roi = slot_crop(frame, center)
        mask = slot_mask(roi)
        cv2.imwrite(str(OUTPUT_DIR / f"{label}.png"), mask)
        print(f"wrote {label}.png")


if __name__ == "__main__":
    main()
