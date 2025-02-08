import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# 生成随机点集
np.random.seed(42)
points = np.random.rand(30, 2)  # 30个随机点

# 计算凸包
hull = ConvexHull(points)

# 提取凸包的顶点
hull_points = points[hull.vertices]

# 可视化
plt.plot(points[:, 0], points[:, 1], 'o', label="Points")  # 绘制所有点
plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2, label="Convex Hull")  # 绘制凸包
plt.plot(np.append(hull_points[:, 0], hull_points[0, 0]),  # 闭合凸包
         np.append(hull_points[:, 1], hull_points[0, 1]), 'r--', lw=2)
plt.legend()
plt.title("Convex Hull of Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()