import numpy as np
import itertools
import matplotlib.pyplot as plt

#球体障碍物
def is_in_sphere(obstacle_center, radius, point):
    x, y, z = point
    xc, yc, zc = obstacle_center
    return (x - xc)**2 + (y - yc)**2 + (z - zc)**2 <= radius**2
#长方体障碍物
def is_in_box(obstacle_center, size, point):
    x, y, z = point
    xc, yc, zc = obstacle_center
    lx, ly, lz = size
    return (xc - lx/2 <= x <= xc + lx/2) and \
           (yc - ly/2 <= y <= yc + ly/2) and \
           (zc - lz/2 <= z <= zc + lz/2)

obstacles = [
    {'type':'sphere', 'center':(0.2, 0.1, -0.125), 'radius':0.05},
    {'type':'box', 'center':(0.35, -0.05, -0.122), 'size':(0.1,0.1,0.05)}
]

# 参数
L1 = 0.21
L2 = 0.21
x2, y2 = 0.065, 0  # 第二个关节坐标
z_range = np.linspace(-0.13, -0.12, 10)
theta1_range = np.linspace(-np.pi/2, np.pi/2, 36)
theta2_range = np.linspace(-3*np.pi/4, 3*np.pi/4, 36)

points = []

for z, theta1, theta2 in itertools.product(z_range, theta1_range, theta2_range):
    x = x2 + L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = y2 + L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    point = [x, y, z]
    
    collision = False
    for obs in obstacles:
        if obs['type'] == 'sphere' and is_in_sphere(obs['center'], obs['radius'], point):
            collision = True
            break
        elif obs['type'] == 'box' and is_in_box(obs['center'], obs['size'], point):
            collision = True
            break
    
    if not collision:
        points.append(point)

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制安全点
ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c='blue', label='Safe')

# 绘制障碍物（球）
for obs in obstacles:
    if obs['type'] == 'sphere':
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xc, yc, zc = obs['center']
        r = obs['radius']
        xs = xc + r * np.cos(u) * np.sin(v)
        ys = yc + r * np.sin(u) * np.sin(v)
        zs = zc + r * np.cos(v)
        ax.plot_surface(xs, ys, zs, color='red', alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()