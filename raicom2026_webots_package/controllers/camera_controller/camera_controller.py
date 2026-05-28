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


def _write_debug_images(debug_dir: Path, frame_index: int, frame: np.ndarray, recognizer: ShapeRecognizer, display: np.ndarray):
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{frame_index:06d}_raw.png"), frame)
    cv2.imwrite(str(debug_dir / f"{frame_index:06d}_display.png"), display)
    for name, image in recognizer.get_debug_views(frame).items():
        cv2.imwrite(str(debug_dir / f"{frame_index:06d}_{name}.png"), image)


def _draw_bottom_slots(frame: np.ndarray, slots):
    output = frame.copy()
    for slot in slots:
        cx, cy = slot["center_pixel"]
        label = slot["label"]
        slot_index = slot["slot_index"]
        x, y, w, h = slot["bbox"]
        color = (255, 180, 0) if label != "uncertain" else (0, 180, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            output,
            f"S{slot_index} {label}",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def _slot_report_payload(slots):
    return [
        {
            "slot_index": slot["slot_index"],
            "label": slot["label"],
            "center_pixel": slot["center_pixel"],
            "confidence": round(float(slot["confidence"]), 2),
        }
        for slot in slots
    ]


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    recognizer = ShapeRecognizer()
    tracker = TrackerState(max_history=5)
    debug_dir = Path(__file__).resolve().parent / "debug_frames"
    frame_index = 0

    cv2.namedWindow("RoboDyno Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RoboDyno Camera", 960, 720)

    last_reported = {}
    last_slot_report = None

    while robot.step(timestep) != -1:
        frame_index += 1
        frame = _camera_frame(camera)
        detections = recognizer.detect_shapes(frame)
        slots = recognizer.detect_bottom_tray_slots(frame)
        events = update_tracks(tracker, detections)
        for detection in detections:
            for track_id, track in tracker.tracks.items():
                if track.centroid == detection["centroid"]:
                    detection["id"] = track_id
                    break

        slot_report = _slot_report_payload(slots)
        if slot_report != last_slot_report:
            print(f"[bottom_slots] {slot_report}")
            last_slot_report = slot_report

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
        display = _draw_bottom_slots(display, slots)
        if frame_index % 15 == 0:
            _write_debug_images(debug_dir, frame_index, frame, recognizer, display)
        display = cv2.resize(display, (960, 720), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("RoboDyno Camera", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
