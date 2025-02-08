import numpy as np
from scipy.spatial import ConvexHull
from meshpy.triangle import MeshInfo, build
import pandas as pd
import os
import meshio
import matplotlib.pyplot as plt  # 导入 matplotlib 用于可视化

# 网格生成参数
MESH_SETTINGS = {
    "max_volume": 0.001,  # 控制网格密度
    "min_angle": 20.0,  # 最小单元角度
    "allow_boundary_steiner": True,  # 允许在边界上添加点
}

# 从 CSV 文件中读取点数据和属性
def read_points_from_csv(filepath):
    data = pd.read_csv(filepath, header=0)  # header=0 表示第一行是表头
    points = data.iloc[:, 1:3].values  # 忽略第一列（序号），读取第二列和第三列
    attributes = data.iloc[:, 3].values  # 读取第四列属性
    return points, attributes

# 在强调的点之间增加更多点
def add_points_on_enforced_boundary(points, attributes, num_points=5):
    new_points = points.tolist()  # 转换为列表以便添加新点
    enforced_indices = np.where(attributes == 1)[0]  # 找出所有强调的点

    # 将强调的点按顺序连接起来
    for i in range(len(enforced_indices) - 1):  # 只连接一次，不闭合
        start_idx = enforced_indices[i]
        end_idx = enforced_indices[i + 1]  # 只连接相邻的点
        start_point = points[start_idx]
        end_point = points[end_idx]
        # 在 start_point 和 end_point 之间增加更多点（不包括起点和终点）
        for t in np.linspace(0, 1, num_points + 2)[1:-1]:  # 去掉起点和终点
            x = start_point[0] + t * (end_point[0] - start_point[0])
            y = start_point[1] + t * (end_point[1] - start_point[1])
            new_points.append([x, y])

    # 去除重复的点
    unique_points = []
    seen = set()
    for point in new_points:
        point_tuple = tuple(point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            unique_points.append(point)

    return np.array(unique_points)

# 定义几何边界
def create_mesh(points, attributes):
    mesh_info = MeshInfo()

    # 设置所有点
    mesh_info.set_points(points)

    # 计算凸包
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]  # 凸包的顶点

    # 生成外边界边
    facets = []
    for i in range(len(hull_points)):
        start_idx = np.where((points == hull_points[i]).all(axis=1))[0][0]  # 找到凸包点的索引
        end_idx = np.where((points == hull_points[(i + 1) % len(hull_points)]).all(axis=1))[0][0]  # 下一个凸包点的索引
        facets.append([start_idx, end_idx])

    # 找出强调的点
    enforced_indices = np.where(attributes == 1)[0]

    # 将强调的点之间的连线添加到 facets 中
    for i in range(len(enforced_indices) - 1):  # 只连接一次，不闭合
        start_idx = enforced_indices[i]
        end_idx = enforced_indices[i + 1]
        facets.append([start_idx, end_idx])

    # 设置边界边
    mesh_info.set_facets(facets)

    # 生成网格
    mesh = build(mesh_info, **MESH_SETTINGS)  # 使用集中管理的参数
    return mesh

# 可视化网格
def visualize_mesh(points, attributes, mesh_points, triangles):
    plt.figure(figsize=(10, 10))

    # 绘制原始点
    plt.plot(points[:, 0], points[:, 1], 'bo', label="Original Points")

    # 绘制凸包
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2, label="Convex Hull")
    plt.plot(np.append(hull_points[:, 0], hull_points[0, 0]),
             np.append(hull_points[:, 1], hull_points[0, 1]), 'r--', lw=2)

    # 绘制强调的边界线
    enforced_indices = np.where(attributes == 1)[0]
    for i in range(len(enforced_indices) - 1):
        start_idx = enforced_indices[i]
        end_idx = enforced_indices[i + 1]
        plt.plot([points[start_idx, 0], points[end_idx, 0]],
                 [points[start_idx, 1], points[end_idx, 1]], 'g-', lw=2, label="Enforced Line" if i == 0 else "")

    # 绘制网格
    plt.triplot(mesh_points[:, 0], mesh_points[:, 1], triangles, 'k-', lw=0.5, label="Mesh")

    # 设置图形属性
    plt.legend()
    plt.title("2D Mesh with Convex Hull and Enforced Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴比例相等
    plt.show()

# 保存网格为 VTK 文件
def save_mesh_to_vtk(mesh_points, triangles, filepath):
    # 将二维点转换为三维点（添加 z=0）
    if mesh_points.shape[1] == 2:  # 检查是否是二维点
        mesh_points = np.hstack([mesh_points, np.zeros((mesh_points.shape[0], 1))])  # 添加 z=0

    cells = [("triangle", triangles)]  # 定义单元类型
    mesh = meshio.Mesh(
        points=mesh_points,  # 点坐标
        cells=cells,  # 单元
    )
    mesh.write(filepath)  # 写入文件

# 主程序
if __name__ == "__main__":
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(__file__)
    # 构建 data/boundary.csv 的路径
    filepath = os.path.join(script_dir, "..", "data", "boundary.csv")

    # 从 CSV 文件中读取点数据和属性
    points, attributes = read_points_from_csv(filepath)

    # 在强调的点之间增加更多点
    num_points = 5  # 设置 num_points 的值
    points = add_points_on_enforced_boundary(points, attributes, num_points=num_points)

    # 生成网格
    mesh = create_mesh(points, attributes)

    # 提取网格信息
    mesh_points = np.array(mesh.points)
    triangles = np.array(mesh.elements)

    # 可视化网格
    visualize_mesh(points, attributes, mesh_points, triangles)

    # 保存网格为 VTK 文件
    output_file = "../output/mesh.vtk"
    save_mesh_to_vtk(mesh_points, triangles, output_file)
    print(f"Mesh saved to {output_file}")