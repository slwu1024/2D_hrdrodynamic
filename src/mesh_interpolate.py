import numpy as np
import pandas as pd
import meshio
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取地形文件
def read_topography(filepath):
    """
    从 CSV 文件中读取地形数据，返回 x、y、z 坐标数组
    :param filepath: 地形数据 CSV 文件的路径
    :return: x、y、z 坐标数组
    """
    data = pd.read_csv(filepath, header=0)
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    z = data.iloc[:, 2].values
    return x, y, z

# 读取网格文件
def read_mesh(filepath):
    """
    从 VTK 文件中读取网格信息，返回网格点坐标和三角形单元信息
    :param filepath: 网格 VTK 文件的路径
    :return: 网格点坐标数组和三角形单元数组
    """
    mesh = meshio.read(filepath)
    mesh_points = mesh.points[:, :2]  # 提取二维网格点坐标
    triangles = mesh.cells_dict["triangle"]
    return mesh_points, triangles

# 插值函数
def interpolate_elevation(x, y, z, mesh_points, method):
    """
    根据指定的插值方法对网格点进行高程插值
    :param x: 地形数据的 x 坐标数组
    :param y: 地形数据的 y 坐标数组
    :param z: 地形数据的 z 坐标数组
    :param mesh_points: 网格点的二维坐标数组
    :param method: 插值方法，可选值为 'nearest', 'linear', 'natural', 'kriging'
    :return: 插值后的高程数组
    """
    if method == 'nearest':
        elevations = griddata((x, y), z, (mesh_points[:, 0], mesh_points[:, 1]), method='nearest')
    elif method == 'linear':
        elevations = griddata((x, y), z, (mesh_points[:, 0], mesh_points[:, 1]), method='linear')
    elif method == 'natural':
        elevations = griddata((x, y), z, (mesh_points[:, 0], mesh_points[:, 1]), method='cubic')  # 近似自然邻域
    elif method == 'kriging':
        OK = OrdinaryKriging(x, y, z, variogram_model='linear')
        elevations, _ = OK.execute('points', mesh_points[:, 0], mesh_points[:, 1])
    else:
        raise ValueError("Unsupported interpolation method. Choose from 'nearest', 'linear', 'natural', 'kriging'.")
    return elevations

# 保存网格为 VTK 文件
def save_mesh_to_vtk(mesh_points, triangles, elevations, filepath):
    """
    将插值后的网格信息保存为 VTK 文件
    :param mesh_points: 网格点的二维坐标数组
    :param triangles: 三角形单元数组
    :param elevations: 插值后的高程数组
    :param filepath: 保存的 VTK 文件路径
    """
    # 将二维点转换为三维点（添加 z=elevation）
    if mesh_points.shape[1] == 2:  # 检查是否是二维点
        mesh_points = np.hstack([mesh_points, elevations.reshape(-1, 1)])  # 添加 z=elevation

    cells = [("triangle", triangles)]  # 定义单元类型
    mesh = meshio.Mesh(
        points=mesh_points,  # 点坐标
        cells=cells,  # 单元
    )
    mesh.write(filepath)  # 写入文件

# 三维可视化函数
def visualize_3d(mesh_points, triangles, elevations):
    """
    三维可视化插值后的网格地形和高程
    :param mesh_points: 网格点的二维坐标数组
    :param triangles: 三角形单元数组
    :param elevations: 插值后的高程数组
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 将二维点转换为三维点（添加 z=elevation）
    mesh_points_3d = np.hstack([mesh_points, elevations.reshape(-1, 1)])

    # 绘制三维网格
    ax.plot_trisurf(mesh_points_3d[:, 0], mesh_points_3d[:, 1], mesh_points_3d[:, 2], triangles=triangles, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    ax.set_title('Interpolated Mesh Topography')

    plt.show()

# 主程序
if __name__ == "__main__":
    # 读取地形文件
    topography_filepath = "../data/topography.csv"
    x, y, z = read_topography(topography_filepath)

    # 读取网格文件
    mesh_filepath = "../output/mesh.vtk"
    mesh_points, triangles = read_mesh(mesh_filepath)

    # 选择插值方法
    available_methods = ['nearest', 'linear', 'natural', 'kriging']
    print("Available interpolation methods:")
    for i, method in enumerate(available_methods, 1):
        print(f"{i}. {method}")
    choice = int(input("Enter the number of the interpolation method you want to use: ")) - 1
    selected_method = available_methods[choice]

    # 插值高程
    elevations = interpolate_elevation(x, y, z, mesh_points, selected_method)

    # 保存网格为 VTK 文件
    output_file = f"../output/mesh_{selected_method}.vtk"
    save_mesh_to_vtk(mesh_points, triangles, elevations, output_file)
    print(f"Mesh with {selected_method} interpolation saved to {output_file}")

    # 三维可视化
    visualize_3d(mesh_points, triangles, elevations)