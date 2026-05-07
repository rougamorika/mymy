from robodyno.interfaces import Webots
from robodyno.components import Motor
from robodyno.components import SliderModule
import math
import tempfile
import json
import os
import time
import cv2
import numpy as np
l1 = 0.21 #第一个臂长
l2 = 0.21 #第二个臂长

temp_dir = tempfile.gettempdir()
file_path = os.path.join(temp_dir, "pixel_coords.json")

# 读取像素坐标函数
def read_pixels():
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except:
        return None

# 图像像素坐标
image_points = np.array([
    [521, 59],
    [257, 101],
    [257, 143],
    [507, 171],
    [382, 212],
    [285, 268],
    [479, 309],
    [313, 392],
    [562, 406],
    [216, 434]
], dtype=np.float32)

# 实际坐标（单位：米）
world_points = np.array([
    [-0.15, -0.255],
    [-0.12, -0.445],
    [-0.09, -0.445],
    [-0.07, -0.265],
    [-0.04, -0.355],
    [0, -0.425],
    [0.03, -0.285],
    [0.09, -0.405],
    [0.1, -0.225],
    [0.12, -0.475]
], dtype=np.float32)

# 计算单应性矩阵 H
H, status = cv2.findHomography(image_points, world_points, cv2.RANSAC)

# print("Homography matrix:\n", H)

# 由像素点计算空间位置
def pixel_to_world(u, v, H):
    point = np.array([u, v, 1])
    world = np.dot(H, point)

    X = world[0] / world[2]
    Y = world[1] / world[2]

    return X, Y

# 由空间位置计算机械臂参数
def aim_postion(x, y, z):
    R_angle = math.atan(x/y)
    R_2 = x**2 + y**2
    Tri1_angle = math.acos((l1**2 + R_2 - l2**2)/(2*l1*math.sqrt(R_2)))
    Tri2_angle = math.acos((l1**2 + l2**2 - R_2)/(2*l1*l2))
    if y > 0 :
        Alpha = - math.pi/2 + R_angle + Tri1_angle
        Beta = - math.pi + Tri2_angle 
        Gamma = math.pi - Alpha - Beta
    else :
        Alpha = math.pi/2 - R_angle - Tri1_angle
        Beta = math.pi - Tri2_angle 
        Gamma = math.pi - Alpha - Beta
    return Alpha, Beta, Gamma
def put_box(number):
    webots.sleep(2+number)

    height=-0.139+0.04*number
    
    Alpha, Beta, Gamma = aim_postion(0.31, 0.1, 0)
    ready_position(Alpha, Beta, Gamma)

    webots.sleep(2)

    slider.set_pos(height)

    webots.sleep(4)

    vacuum.turnOff()

    slider.set_pos(height+0.03)

    webots.sleep(2)

def seize_box(number):
    webots.sleep(3)

    slider.set_pos(-0.109)

    webots.sleep(5+number)

    vacuum.turnOn()
    slider.set_pos(-0.1+0.04*number)
def ready_position(Alpha, Beta, Gamma):
    motor11.set_pos(-Alpha)
    motor12.set_pos(Beta)
    motor13.set_pos(Gamma)

# 初始化
webots = Webots()
webots.sleep(1)
vacuum = webots.robot.getDevice("0x1A")
motor11 = Motor(webots, 0x12)
motor12 = Motor(webots, 0x13)
motor13 = Motor(webots, 0x14)
slider = SliderModule(webots, 0x11)

slider.enable()
motor11.enable()
motor12.enable()
motor13.enable()

# 电机控制参数()
motor11.position_track_mode(1, 2, 2)
motor12.position_track_mode(1, 2, 2)
motor13.position_track_mode(1, 2, 2)

last_data = None



#输出机械臂状态参数
last_timestamp = None

while True:
    webots.sleep(0.05)

    data = read_pixels()

    if data is None:
        continue

    # 防止重复执行
    if data["timestamp"] == last_timestamp:
        continue

    print("✅ 收到视觉数据:", data)

    pixel_points = data["points"]

    posture_data = []

    # =========================
    # 1像素 → 世界坐标 → 机械臂角度
    # =========================
    for p in pixel_points:
        u = p["x"]
        v = p["y"]

        x, y = pixel_to_world(u, v, H)

        Alpha, Beta, Gamma = aim_postion(x + 0.065, y, 0)

        posture_data.append((Alpha, Beta, Gamma))

    # =========================
    # 依次抓取
    # =========================
    for i, (Alpha, Beta, Gamma) in enumerate(posture_data):

        print(f"抓取第{i+1}个目标")

        ready_position(Alpha, Beta, Gamma)

        seize_box(i+1)

        put_box(i+1)

    # =========================
    # 标记已处理
    # =========================
    last_timestamp = data["timestamp"]

    # 删除文件（让视觉可以写下一批）
    os.remove(file_path)