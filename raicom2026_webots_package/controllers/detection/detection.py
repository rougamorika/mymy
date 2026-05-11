from controller import Robot

import cv2
import numpy as np
from pathlib import Path

from shape_recognition import ShapeRecognizer, TrackerState, draw_detections, update_tracks


def _camera_frame(camera):
    width = camera.getWidth()
    height = camera.getHeight()
    image = camera.getImage()
    frame = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def _get_camera(robot):
    for camera_name in ("detection_camera", "camera"):
        try:
            camera = robot.getDevice(camera_name)
            camera.getWidth()
            return camera
        except Exception:
            continue

    for index in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(index)
        if device.getNodeTypeName() == "Camera":
            return device

    raise RuntimeError("No camera device found for detection controller")


def _write_debug_images(debug_dir: Path, frame_index: int, frame: np.ndarray, recognizer: ShapeRecognizer, display: np.ndarray):
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{frame_index:06d}_raw.png"), frame)
    cv2.imwrite(str(debug_dir / f"{frame_index:06d}_display.png"), display)
    for name, image in recognizer.get_debug_views(frame).items():
        cv2.imwrite(str(debug_dir / f"{frame_index:06d}_{name}.png"), image)


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera = _get_camera(robot)
    camera.enable(timestep)

    recognizer = ShapeRecognizer()
    tracker = TrackerState(max_history=5)
    debug_dir = Path(__file__).resolve().parent / "debug_frames"
    frame_index = 0

    cv2.namedWindow("RoboDyno Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RoboDyno Camera", 960, 720)

    last_reported = {}
    while robot.step(timestep) != -1:
        frame_index += 1
        frame = _camera_frame(camera)
        detections = recognizer.detect_shapes(frame)
        events = update_tracks(tracker, detections)

        for event in events:
            if event["type"] not in {"appeared", "moved", "disappeared"}:
                continue
            if event["type"] == "disappeared":
                print(f"[target {event['id']}] disappeared")
                last_reported.pop(event["id"], None)
                continue
            detection = event["detection"]
            centroid = detection["centroid"]
            signature = (detection["label"], round(float(detection["confidence"]), 2), centroid)
            if last_reported.get(event["id"]) != signature:
                last_reported[event["id"]] = signature
                print(
                    f"[target {event['id']}] {detection['label']} "
                    f"conf={float(detection['confidence']):.2f} centroid=({centroid[0]}, {centroid[1]})"
                )

        display = draw_detections(frame, detections)
        if frame_index % 15 == 0:
            _write_debug_images(debug_dir, frame_index, frame, recognizer, display)
        display = cv2.resize(display, (960, 720), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("RoboDyno Camera", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
