import cv2
import numpy as np
from controller import Robot
from pathlib import Path

from recess_recognition import TopTrayRecessRecognizer


def _camera_image_to_bgr(camera) -> np.ndarray:
    width = camera.getWidth()
    height = camera.getHeight()
    image = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((height, width, 4))
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def _save_window_screenshot(debug_dir: Path, frame_index: int, frame: np.ndarray) -> None:
    if frame_index % 15 != 0:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{frame_index:06d}_window.png"), frame)


def _write_debug_images(debug_dir: Path, frame_index: int, frame: np.ndarray, recognizer: TopTrayRecessRecognizer, display: np.ndarray) -> None:
    if frame_index % 15 != 0:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{frame_index:06d}_raw.png"), frame)
    cv2.imwrite(str(debug_dir / f"{frame_index:06d}_display.png"), display)
    for name, image in recognizer.get_debug_views(frame).items():
        cv2.imwrite(str(debug_dir / f"{frame_index:06d}_{name}.png"), image)


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    camera = robot.getDevice("top_tray_camera")
    if camera is None:
        camera = robot.getDevice("camera")
    if camera is None:
        raise RuntimeError("No camera device found")
    camera.enable(timestep)
    recognizer = TopTrayRecessRecognizer()
    debug_dir = Path(__file__).resolve().parent / "debug_frames"
    frame_index = 0
    last_reported = None
    while robot.step(timestep) != -1:
        frame_index += 1
        frame = _camera_image_to_bgr(camera)
        detections = recognizer.detect_recesses(frame)
        report = [(d["slot_index"], d["label"], round(float(d["confidence"]), 2), d["centroid"]) for d in detections]
        if report != last_reported:
            for slot_index, label, confidence, centroid in report:
                print(f"{slot_index} {label} {confidence:.2f} {centroid}")
            last_reported = report
        display_frame = recognizer.draw_detections(frame, detections)
        _save_window_screenshot(debug_dir, frame_index, display_frame)
        _write_debug_images(debug_dir, frame_index, frame, recognizer, display_frame)
        cv2.imshow("top_tray_recess", display_frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
