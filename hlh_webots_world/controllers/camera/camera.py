from controller import Robot
import numpy as np
import cv2
import tempfile
import json
import os
import time
# =========================
# 1. 初始化
# =========================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

width = camera.getWidth()
height = camera.getHeight()

temp_dir = tempfile.gettempdir()
file_path = os.path.join(temp_dir, "pixel_coords.json")

has_written = False   # 是否已经写过
last_points = None    # 上一次检测结果（防重复）

print("Camera started!")

# =========================
# 2. 主循环
# =========================
while robot.step(timestep) != -1:

    image = camera.getImage()
    if image is None:
        continue

    # 转 numpy
    img = np.frombuffer(image, np.uint8).reshape((height, width, 4)).copy()
    img = img[:, :, :3].copy()

    # =========================
    # 3. HSV 颜色过滤（只保留“亮红色顶面”）
    # =========================
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 170, 210])# 由[0, 160, 180]改进，只识别亮红色顶面
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 170, 210])# 由[170, 160, 180]改进，只识别亮红色顶面
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # 去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # =========================
    # 4. 找轮廓
    # =========================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # =========================
    # 5. 遍历所有目标
    # =========================
    pixel_list = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 800 or area > 10000:
            continue

        rect = cv2.minAreaRect(contour)
        (w, h) = rect[1]

        if w == 0 or h == 0:
            continue

        ratio = max(w, h) / min(w, h)
        if ratio > 1.5:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        extent = area / (bw * bh)

        if extent < 0.7:
            continue

        # 画框
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        cx = int(rect[0][0])
        cy = int(rect[0][1])

        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

        pixel_list.append({"x": cx, "y": cy})
    # =========================
    #  写入临时文件（只写一次）
    # =========================
    if len(pixel_list) > 0:

        # 和上一次结果对比（防止重复写）
        if pixel_list != last_points:

            if not has_written:

                temp_file = file_path + ".tmp"

                data = {
                    "points": pixel_list,
                    "timestamp": time.time()
                }

                with open(temp_file, 'w') as f:
                    json.dump(data, f)

                os.replace(temp_file, file_path)

                print("✅ 已写入一次坐标:", pixel_list)

                has_written = True   # 锁住，不再写

            last_points = pixel_list

    # =========================
    # 8. 显示图像
    # =========================
    cv2.imshow("Red Cube Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()