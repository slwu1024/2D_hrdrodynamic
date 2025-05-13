# prepare_mesh_data.py
import numpy as np  # NumPy 用于数值运算
import os  # os 模块用于操作系统相关功能，如路径操作
import math
import shutil  # shutil 模块用于文件操作，如复制
import triangle  # triangle 库用于生成符合Delaunay条件的二维网格
import matplotlib.pyplot as plt  # matplotlib 用于绘图
import yaml

from parse_poly_file import parse_poly_file  # 从自定义模块导入 .poly 文件解析函数
# --- 添加中文字体设置 ---
import matplotlib  # 导入 matplotlib 库
# --- 从原 mesh_interpolate.py 移入的函数 ---
import pandas as pd  # pandas 用于数据处理，特别是 CSV 文件
from scipy.interpolate import griddata  # scipy.interpolate.griddata 用于插值
from pykrige.ok import OrdinaryKriging  # pykrige.ok.OrdinaryKriging 用于克里金插值


try:
    # 你可以尝试不同的中文字体名称，直到找到一个你系统上存在的
    # 常见的有 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif' (作为通用后备)
    # 'WenQuanYi Micro Hei' (Linux)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 尝试使用 SimHei 字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决 Matplotlib 显示负号为方块的问题
    print("信息: 已尝试设置 Matplotlib 中文字体为 SimHei。")  # 打印设置字体成功的提示信息
except Exception as e:  # 捕获设置字体时可能发生的异常
    print(f"警告: 设置 Matplotlib 中文字体失败: {e}。图表中的中文可能无法正常显示。")  # 打印设置字体失败的警告信息
# --- 中文字体设置结束 ---

# --- 从 yaml 文件加载配置 ---
config_path = '../config.yaml' # 定义配置文件路径
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) # 加载 yaml 配置
    print(f"配置已从 {config_path} 加载。")
except FileNotFoundError:
    print(f"错误: 配置文件 {config_path} 未找到。")
    exit()
except yaml.YAMLError as e:
    print(f"错误: 解析配置文件 {config_path} 失败: {e}")
    exit()
except Exception as e:
     print(f"加载配置文件时发生未知错误: {e}")
     exit()


# --- 从配置中获取路径和参数 ---
file_paths = config.get('file_paths', {}) # 获取文件路径配置
mesh_gen_config = config.get('mesh_generation', {}) # 获取网格生成配置

POLY_FILE = file_paths.get('poly_file') # **从配置获取 poly 文件路径**
NODE_FINAL_FILE = file_paths.get('node_file') # **从配置获取 node 输出路径**
CELL_FINAL_FILE = file_paths.get('cell_file') # **从配置获取 cell 输出路径**
EDGE_FINAL_FILE = file_paths.get('edge_file') # **从配置获取 edge 输出路径**
# --- 修改 OUTPUT_VTK_VIS 的定义 ---
# 假设 NODE_FINAL_FILE, CELL_FINAL_FILE, EDGE_FINAL_FILE 都在同一个目录下，例如 "mesh/"
# 我们可以从 NODE_FINAL_FILE 获取这个目录
if NODE_FINAL_FILE: # 确保 NODE_FINAL_FILE 已定义
    mesh_output_directory = os.path.dirname(NODE_FINAL_FILE) # 获取 .node 文件所在的目录
    # 确保目录存在 (虽然保存 .node 文件时也会创建，但这里再次确保)
    os.makedirs(mesh_output_directory, exist_ok=True) # 创建目录，如果已存在则不报错
    OUTPUT_VTK_VIS = os.path.join(mesh_output_directory, "generated_mesh_with_terrain.vtk") # 构建VTK文件路径
else: # 如果 NODE_FINAL_FILE 未定义，则使用备用路径
    print("警告: NODE_FINAL_FILE 未在配置中定义，VTK可视化文件将保存到默认的 output_vis 目录。") # 打印警告
    default_vis_dir = file_paths.get('output_directory', 'output_vis') # 获取默认可视化目录
    os.makedirs(default_vis_dir, exist_ok=True) # 创建目录
    OUTPUT_VTK_VIS = os.path.join(default_vis_dir, "generated_mesh_with_terrain.vtk") # 构建备用VTK文件路径

TRIANGLE_OPTS = mesh_gen_config.get('triangle_opts', "pq30a1.0ez") # **从配置获取 Triangle 选项**
ELEVATION_SOURCE_METHOD = mesh_gen_config.get('elevation_source_method', 'interpolation') # **从配置获取高程方法**

# 检查关键路径是否已配置
if not POLY_FILE or not NODE_FINAL_FILE or not CELL_FINAL_FILE or not EDGE_FINAL_FILE:
    print("错误: yaml 文件中 file_paths 部分缺少 poly_file, node_file, cell_file 或 edge_file 配置。")
    exit()

print(f"使用的 Poly 文件: {POLY_FILE}")
print(f"输出 Node 文件: {NODE_FINAL_FILE}")
print(f"输出 Cell 文件: {CELL_FINAL_FILE}")
print(f"输出 Edge 文件: {EDGE_FINAL_FILE}")
print(f"Triangle 选项: {TRIANGLE_OPTS}")
print(f"高程来源方法: {ELEVATION_SOURCE_METHOD}")

# --- 地形函数 (仅当 ELEVATION_SOURCE_METHOD = "function" 时使用) ---
# --- 这个函数需要保持在这里，因为配置指定为 function 时会调用它 ---
def bed_elevation_func(x, y): # x 和 y 是 NumPy 数组
    """计算给定 (x, y) 处的底高程。对于溃坝算例，我们使用平底。"""
    # 对于平底，返回一个与 x (或 y) 形状相同的全零数组
    return np.zeros_like(x) # 返回与 x 相同形状的全零数组，表示底高程为0
# def bed_elevation_func(x, y): # x 和 y 是 NumPy 数组
#     """计算给定 (x, y) 处的底高程，包含一个高斯驼峰 (用于静水测试)。"""
#     x_center = 12.5 # 驼峰中心 x
#     y_center = 2.5  # 驼峰中心 y
#     bump_height = 0.2 # 驼峰高度
#     bump_width_sigma = 2.0 # 驼峰宽度
#
#     dist_sq = (x - x_center)**2 + (y - y_center)**2 # dist_sq 是 NumPy 数组
#
#     # --- 使用 numpy.exp() 而不是 math.exp() ---
#     z_bed = bump_height * np.exp(-dist_sq / (2 * bump_width_sigma**2)) # 对数组进行指数运算
#
#     return z_bed # 返回计算得到的底高程数组


# --- 原 mesh_interpolate.py 的地形读取和插值函数 ---
def read_topography_csv(filepath):  # 定义读取地形数据CSV文件的函数
    """
    从 CSV 文件中读取地形数据，返回 x、y、z 坐标数组
    :param filepath: 地形数据 CSV 文件的路径
    :return: x、y、z 坐标数组 (numpy arrays)
    """
    print(f"从 {filepath} 读取地形散点数据...")  # 打印开始读取地形数据文件的消息
    try:
        data = pd.read_csv(filepath, header=0)  # 使用 pandas 读取 CSV 文件，假设第一行为表头
        x = data.iloc[:, 0].values  # 提取第一列作为 x 坐标
        y = data.iloc[:, 1].values  # 提取第二列作为 y 坐标
        z = data.iloc[:, 2].values  # 提取第三列作为 z 坐标
        print(f"地形数据读取成功，共 {len(x)} 个点。")  # 打印地形数据读取成功的消息及点数
        return x, y, z  # 返回 x, y, z 坐标数组
    except FileNotFoundError:  # 捕获文件未找到的异常
        print(f"错误: 地形文件 {filepath} 未找到。")  # 打印文件未找到的错误消息
        return None, None, None  # 返回 None
    except Exception as e:  # 捕获其他读取文件时可能发生的异常
        print(f"读取地形文件 {filepath} 时出错: {e}")  # 打印读取文件出错的错误消息
        return None, None, None  # 返回 None


def interpolate_elevation(topo_x, topo_y, topo_z, mesh_points_xy, method):  # 定义高程插值函数
    """
    根据指定的插值方法对网格点进行高程插值
    :param topo_x: 地形数据的 x 坐标数组
    :param topo_y: 地形数据的 y 坐标数组
    :param topo_z: 地形数据的 z 坐标数组
    :param mesh_points_xy: 网格点的二维坐标数组 (N, 2)
    :param method: 插值方法，可选值为 'nearest', 'linear', 'natural', 'kriging'
    :return: 插值后的高程数组 (N,)
    """
    print(f"开始对网格节点进行高程插值，方法: {method}...")  # 打印开始插值的消息和使用的方法
    num_mesh_points = mesh_points_xy.shape[0]  # 获取网格点的数量
    if topo_x is None or len(topo_x) == 0:  # 检查地形数据是否为空
        print("错误: 地形数据为空，无法进行插值。将返回全零高程。")  # 打印地形数据为空的错误消息
        return np.zeros(num_mesh_points)  # 返回全零高程数组

    try:
        if method == 'nearest':  # 如果插值方法是 'nearest'
            elevations = griddata((topo_x, topo_y), topo_z, (mesh_points_xy[:, 0], mesh_points_xy[:, 1]),
                                  method='nearest')  # 使用 scipy 的 griddata 进行最近邻插值
        elif method == 'linear':  # 如果插值方法是 'linear'
            elevations = griddata((topo_x, topo_y), topo_z, (mesh_points_xy[:, 0], mesh_points_xy[:, 1]),
                                  method='linear')  # 使用 scipy 的 griddata 进行线性插值
        elif method == 'natural':  # 如果插值方法是 'natural' (在griddata中用 'cubic' 近似)
            elevations = griddata((topo_x, topo_y), topo_z, (mesh_points_xy[:, 0], mesh_points_xy[:, 1]),
                                  method='cubic')  # 使用 scipy 的 griddata 进行三次样条插值（近似自然邻域）
        elif method == 'kriging':  # 如果插值方法是 'kriging'
            # 注意: 克里金插值对输入参数较为敏感，可能需要根据数据调整 variogram_model 等参数
            print("  执行克里金插值 (这可能需要一些时间)...")  # 打印执行克里金插值的提示信息
            OK = OrdinaryKriging(topo_x, topo_y, topo_z, variogram_model='linear', verbose=False,
                                 enable_plotting=False)  # 初始化普通克里金模型
            elevations, variances = OK.execute('points', mesh_points_xy[:, 0], mesh_points_xy[:, 1])  # 执行插值，获取高程和方差
            print(f"  克里金插值完成。示例方差 (前5个): {variances[:5]}")  # 打印克里金插值完成的消息和示例方差
            # 处理可能的 NaN 值 (例如，如果网格点远超地形数据范围)
            if np.isnan(elevations).any():  # 检查插值结果中是否有 NaN 值
                print(
                    f"警告: 克里金插值结果包含 {np.isnan(elevations).sum()} 个 NaN 值。将尝试用最近邻填充...")  # 打印 NaN 值警告信息
                nan_indices = np.isnan(elevations)  # 获取 NaN 值的索引
                # 对于NaN值，尝试用最近邻再次插值，或者赋一个默认值（如0或平均高程）
                fallback_elevations = griddata((topo_x, topo_y), topo_z,
                                               (mesh_points_xy[nan_indices, 0], mesh_points_xy[nan_indices, 1]),
                                               method='nearest')  # 对 NaN 值使用最近邻插值进行填充
                elevations[nan_indices] = fallback_elevations  # 将填充后的高程赋给原数组
                if np.isnan(elevations).any():  # 再次检查是否仍有 NaN 值
                    print(
                        f"警告: 最近邻填充后仍有 {np.isnan(elevations).sum()} 个 NaN 值。这些将被设为0。")  # 打印仍有 NaN 值的警告信息
                    elevations[np.isnan(elevations)] = 0  # 将剩余的 NaN 值设为0
        else:  # 如果插值方法不受支持
            raise ValueError(
                f"不支持的插值方法: {method}. 请选择 'nearest', 'linear', 'natural', 'kriging'.")  # 抛出值错误异常

        print(f"高程插值完成。共处理 {len(elevations)} 个节点。")  # 打印高程插值完成的消息
        # 确保 elevations 是一维数组
        return np.asarray(elevations).ravel()  # 返回确保为一维的插值高程数组
    except Exception as e:  # 捕获插值过程中可能发生的异常
        print(f"高程插值时发生错误: {e}")  # 打印插值错误的错误消息
        print("将返回全零高程作为备用。")  # 打印返回全零高程的提示信息
        return np.zeros(num_mesh_points)  # 返回全零高程数组


# --- 文件保存函数 (与之前基本一致) ---
def save_node_file_with_z(filepath, nodes_xy, z_coords, markers):  # 定义保存带Z坐标的节点文件的函数
    num_nodes = len(nodes_xy)  # 获取节点数量
    has_marker = 1  # 假设节点文件中包含标记列
    print(f"保存最终节点文件 (含Z坐标) 到 {filepath}...")  # 打印开始保存节点文件的消息
    try:
        with open(filepath, 'w', encoding='utf-8') as f:  # 以写入模式打开文件，使用UTF-8编码
            f.write(f"{num_nodes} 2 0 {has_marker}\n")  # 写入文件头：节点数、维度(2D)、属性数(0)、是否有标记(1)
            for i in range(num_nodes):  # 遍历所有节点
                # 写入每行节点数据：节点ID、x坐标、y坐标、z坐标(高程)、标记
                f.write(
                    f"{i} {nodes_xy[i, 0]:.10f} {nodes_xy[i, 1]:.10f} {z_coords[i]:.10f} {int(markers[i].item()) if markers is not None and i < len(markers) else 0}\n")
        print("最终节点文件保存成功.")  # 打印节点文件保存成功的消息
        return True  # 返回 True 表示成功
    except Exception as e:  # 捕获保存文件时可能发生的异常
        print(f"保存最终节点文件 {filepath} 时出错: {e}")  # 打印保存文件出错的错误消息
        return False  # 返回 False 表示失败


def save_cell_file(filepath, triangles):  # 定义保存单元文件的函数
    print(f"保存单元文件到 {filepath}...")  # 打印开始保存单元文件的消息
    try:
        with open(filepath, 'w', encoding='utf-8') as f:  # 以写入模式打开文件，使用UTF-8编码
            f.write(f"{len(triangles)} 3 0\n")  # 写入文件头：单元数、每个单元的节点数(3)、属性数(0)
            for i, tri in enumerate(triangles):  # 遍历所有三角形单元
                f.write(f"{i} {tri[0]} {tri[1]} {tri[2]}\n")  # 写入每行单元数据：单元ID、节点1 ID、节点2 ID、节点3 ID
        print("单元文件保存成功.")  # 打印单元文件保存成功的消息
        return True  # 返回 True 表示成功
    except Exception as e:  # 捕获保存文件时可能发生的异常
        print(f"保存单元文件 {filepath} 时出错: {e}")  # 打印保存文件出错的错误消息
        return False  # 返回 False 表示失败


def save_edge_file(filepath, edges, edge_markers):  # 定义保存边文件的函数
    num_edges = len(edges)  # 获取边的数量
    # 检查边标记是否有效且数量匹配
    has_marker = 1 if edge_markers is not None and len(edge_markers) == num_edges else 0  # 判断是否包含边标记列
    if has_marker == 0 and edge_markers is not None and len(edge_markers) != num_edges:  # 如果标记数量不匹配
        print(f"警告: 边标记数量({len(edge_markers)})与边数({num_edges})不匹配，将不保存标记。")  # 打印警告信息
    print(f"保存边文件到 {filepath} (标记: {bool(has_marker)})...")  # 打印开始保存边文件的消息
    try:
        with open(filepath, 'w', encoding='utf-8') as f:  # 以写入模式打开文件，使用UTF-8编码
            f.write(f"{num_edges} {has_marker}\n")  # 写入文件头：边数、是否有标记
            for i, edge in enumerate(edges):  # 遍历所有边
                marker_str = f" {int(edge_markers[i].item())}" if has_marker else ""  # 如果有标记，则格式化标记字符串
                f.write(f"{i} {edge[0]} {edge[1]}{marker_str}\n")  # 写入每行边数据：边ID、节点1 ID、节点2 ID、[标记]
        print("边文件保存成功.")  # 打印边文件保存成功的消息
        return True  # 返回 True 表示成功
    except Exception as e:  # 捕获保存文件时可能发生的异常
        print(f"保存边文件 {filepath} 时出错: {e}")  # 打印保存文件出错的错误消息
        return False  # 返回 False 表示失败


def save_vtk_for_visualization(filepath, points_3d, triangles):  # 定义保存VTK可视化文件的函数
    try:  # 尝试导入 meshio 库
        import meshio  # 导入 meshio 库，用于读写多种网格文件格式
    except ImportError:  # 如果导入失败
        print("警告: 未找到 'meshio' 库，无法保存VTK文件。请尝试 'pip install meshio'。")  # 打印未找到 meshio 库的警告信息
        return  # 直接返回
    print(f"保存 VTK 可视化文件到 {filepath}...")  # 打印开始保存VTK文件的消息
    # 确保 points_3d 是 Nx3 的数组
    if points_3d.shape[1] == 2:  # 如果输入点是二维的
        print("警告: VTK可视化接收到2D点，将自动添加Z=0。")  # 打印警告信息
        points_3d = np.hstack([points_3d, np.zeros((points_3d.shape[0], 1))])  # 将二维点扩展为三维点，Z坐标设为0
    elif points_3d.shape[1] != 3:  # 如果点的维度不是2或3
        print(f"错误: VTK的可视化点应为3D，但接收到 {points_3d.shape[1]}D 点。无法保存。")  # 打印维度错误的错误信息
        return  # 直接返回

    cells = [("triangle", triangles.astype(int))]  # 定义单元类型为三角形，并将节点索引转换为整数
    mesh_to_save = meshio.Mesh(points_3d[:, :3], cells)  # 创建 meshio 的 Mesh 对象，只使用点的前三列 (X, Y, Z)
    try:
        mesh_to_save.write(filepath)  # 写入VTK文件
        print("VTK 文件保存成功。")  # 打印VTK文件保存成功的消息
    except Exception as e:  # 捕获写入文件时可能发生的异常
        print(f"保存 VTK 文件 {filepath} 时出错: {e}")  # 打印保存VTK文件出错的错误消息


def visualize_mesh_2d(parsed_poly_data, generated_mesh_points_xy, generated_triangles):  # 定义二维网格可视化函数
    plt.figure(figsize=(10, 8))  # 创建一个新的图形窗口，设置大小
    if generated_mesh_points_xy is not None and generated_triangles is not None:  # 如果有生成的网格数据
        plt.triplot(generated_mesh_points_xy[:, 0], generated_mesh_points_xy[:, 1], generated_triangles, 'k-', lw=0.3,
                    label="生成网格")  # 绘制生成的三角形网格

    if parsed_poly_data and 'points' in parsed_poly_data and 'segments' in parsed_poly_data:  # 如果有解析的 .poly 数据
        poly_points = parsed_poly_data['points']  # 获取 .poly 文件中的点
        poly_segments = parsed_poly_data['segments']  # 获取 .poly 文件中的线段
        # 绘制原始定义的线段
        if len(poly_points) > 0 and len(poly_segments) > 0:  # 确保点和线段数据有效
            for seg_idx, seg in enumerate(poly_segments):  # 遍历所有线段
                # 检查线段索引是否有效
                if seg[0] < len(poly_points) and seg[1] < len(poly_points) and seg[0] >= 0 and seg[1] >= 0:
                    # 获取线段标记，如果存在
                    seg_marker = parsed_poly_data['segment_markers'][
                        seg_idx] if 'segment_markers' in parsed_poly_data and seg_idx < len(
                        parsed_poly_data['segment_markers']) else 0  # 获取线段标记
                    color = 'r-'  # 默认颜色为红色
                    # 可以根据 seg_marker 设置不同颜色
                    # if seg_marker == 1: color = 'b-'
                    # elif seg_marker == 2: color = 'g-'
                    label_text = "原始边界/约束线段" if seg_idx == 0 else None  # 只为第一条线段添加图例标签
                    plt.plot(poly_points[seg, 0], poly_points[seg, 1], color, lw=1.5, label=label_text)  # 绘制线段
                else:
                    print(f"警告: 可视化时跳过无效线段索引: {seg} (节点数: {len(poly_points)})")  # 打印无效线段索引的警告
        # 单独添加一个图例项，避免重复标签
        # 检查图例是否已存在红色线条的标签
        legend_handles, legend_labels = plt.gca().get_legend_handles_labels()  # 获取当前图例的句柄和标签
        if not any(label == "原始边界/约束线段" for label in legend_labels):  # 如果还没有"原始边界/约束线段"的图例
            plt.plot([], [], 'r-', label="原始边界/约束线段")  # 添加一个空的红色线条用于图例显示

    plt.title(f"生成的二维网格 (triangle 库) - 高程来源: {ELEVATION_SOURCE_METHOD}")  # 设置图表标题，包含高程来源信息
    plt.xlabel("X 坐标")  # 设置X轴标签
    plt.ylabel("Y 坐标")  # 设置Y轴标签
    plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴比例相等
    plt.legend()  # 显示图例
    plt.grid(True, linestyle='--', alpha=0.6)  # 显示网格线
    plt.tight_layout()  # 调整布局以防止标签重叠
    plt.show()  # 显示图表


# --- 主程序 (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    # --- 步骤 1: 解析 .poly 文件 (路径来自配置) ---
    print("--- 步骤 1: 解析 .poly 文件 ---")
    poly_data = parse_poly_file('../' + POLY_FILE)  # 使用配置中的路径
    if poly_data is None: exit()

    # --- 步骤 2: 构建 triangle 输入字典 (不变) ---
    print("\n--- 步骤 2: 构建 triangle 库输入字典 ---")
    # ... (代码不变) ...
    triangle_input = {}
    if 'points' in poly_data and len(poly_data['points']) > 0: triangle_input['vertices'] = poly_data['points']
    if 'point_markers' in poly_data and len(poly_data['point_markers']) > 0: triangle_input['vertex_markers'] = \
    poly_data['point_markers']
    if 'segments' in poly_data and len(poly_data['segments']) > 0:
        triangle_input['segments'] = poly_data['segments']
        if 'segment_markers' in poly_data and len(poly_data['segment_markers']) == len(poly_data['segments']):
            triangle_input['segment_markers'] = poly_data['segment_markers']
    if 'holes' in poly_data and len(poly_data['holes']) > 0: triangle_input['holes'] = poly_data['holes']
    # 区域处理（如果需要）
    # if 'regions' in poly_data and len(poly_data['regions']) > 0:
    #     triangle_input['regions'] = poly_data['regions'].tolist()

    # --- 步骤 3: 调用 triangle.triangulate 生成网格 (不变) ---
    print("\n--- 步骤 3: 调用 triangle.triangulate 生成网格 ---")
    print(f"使用选项字符串: '{TRIANGLE_OPTS}'")
    try:
        mesh_data_dict = triangle.triangulate(triangle_input, TRIANGLE_OPTS)  # 使用配置中的选项
        print("网格生成成功。")
    except Exception as e:
        print(f"调用 triangle.triangulate 时出错: {e}")
        exit()

    # --- 步骤 4: 提取生成的网格数据 (不变) ---
    print("\n--- 步骤 4: 提取生成的网格数据 ---")
    # ... (代码不变) ...
    generated_nodes_xy = mesh_data_dict.get('vertices')
    generated_node_markers = mesh_data_dict.get('vertex_markers')
    generated_triangles = mesh_data_dict.get('triangles')
    generated_edges = mesh_data_dict.get('edges')
    generated_edge_markers = mesh_data_dict.get('edge_markers')
    if generated_nodes_xy is None or generated_triangles is None: exit()
    if generated_edges is None: print("警告: 未获取到边列表。")
    if generated_edge_markers is None and generated_edges is not None:
        generated_edge_markers = np.zeros(len(generated_edges), dtype=int)
        print("警告: 未获取到边标记，将使用默认标记 0。")

    print(f"生成网格包含 {len(generated_nodes_xy)} 个节点, {len(generated_triangles)} 个单元。")
    if generated_edges is not None: print(f"包含 {len(generated_edges)} 条边。")

    # --- 步骤 5: 计算/获取网格节点的底高程 Z_bed (根据配置决定) ---
    print("\n--- 步骤 5: 计算/获取网格节点底高程 ---")
    print(f"高程来源方法: {ELEVATION_SOURCE_METHOD}")
    if ELEVATION_SOURCE_METHOD == "function":
        # **确保 bed_elevation_func 函数已在此脚本中定义**
        mesh_z_bed = bed_elevation_func(generated_nodes_xy[:, 0], generated_nodes_xy[:, 1])
        print(f"已为 {len(mesh_z_bed)} 个节点通过函数 '{bed_elevation_func.__name__}' 计算底高程。")
    elif ELEVATION_SOURCE_METHOD == "interpolation":
        TOPOGRAPHY_FILE = file_paths.get('topography_file')  # 从配置获取地形文件路径
        INTERPOLATION_METHOD = mesh_gen_config.get('interpolation_method', 'kriging')  # 从配置获取插值方法
        if not TOPOGRAPHY_FILE:
            print("错误: 高程来源为 'interpolation' 但未在 yaml 的 file_paths 中配置 topography_file。")
            exit()
        print(f"  使用插值方法: {INTERPOLATION_METHOD}, 地形文件: {TOPOGRAPHY_FILE}")
        topo_x, topo_y, topo_z = read_topography_csv('../' + TOPOGRAPHY_FILE)
        if topo_x is not None and len(topo_x) > 0:
            mesh_z_bed = interpolate_elevation(topo_x, topo_y, topo_z, generated_nodes_xy, INTERPOLATION_METHOD)
            print(f"已为 {len(mesh_z_bed)} 个节点通过插值获取底高程。")
        else:
            print(f"错误: 无法从 {TOPOGRAPHY_FILE} 读取地形数据进行插值。将使用全零高程。")
            mesh_z_bed = np.zeros(len(generated_nodes_xy))
    else:
        print(f"错误: 不支持的高程来源方法 '{ELEVATION_SOURCE_METHOD}'。将使用全零高程。")
        mesh_z_bed = np.zeros(len(generated_nodes_xy))

    # 检查高程数组
    if not isinstance(mesh_z_bed, np.ndarray) or mesh_z_bed.ndim != 1 or len(mesh_z_bed) != len(generated_nodes_xy):
        print(f"错误: 生成的底高程数组类型或形状不正确。将使用全零高程。")
        mesh_z_bed = np.zeros(len(generated_nodes_xy))

    # 节点标记使用 .poly 文件中的定义
    final_node_markers = generated_node_markers if generated_node_markers is not None else np.zeros(
        len(generated_nodes_xy), dtype=int)

    # --- 步骤 6: 确保输出目录存在 ---
    # print("\n--- 步骤 6: 创建输出目录 (如果不存在) ---")
    # # ... (确保输出目录存在) ...
    # output_dir_base = os.path.dirname(NODE_FINAL_FILE)  # 获取mesh文件目录
    # output_dir_sim = file_paths.get('output_directory', 'output_sim')  # 获取模拟输出目录
    # os.makedirs(output_dir_base, exist_ok=True)
    # os.makedirs(output_dir_sim, exist_ok=True)
    # vtk_vis_dir = os.path.dirname(OUTPUT_VTK_VIS)  # 获取VTK可视化目录
    # os.makedirs(vtk_vis_dir, exist_ok=True)
    # print("输出目录检查/创建完毕。")

    print("\n--- 步骤 7: 保存最终模型输入文件 ---")
    save_node_file_with_z('../' + NODE_FINAL_FILE, generated_nodes_xy, mesh_z_bed, final_node_markers)
    save_cell_file('../' + CELL_FINAL_FILE, generated_triangles)
    if generated_edges is not None:
        save_edge_file('../' + EDGE_FINAL_FILE, generated_edges, generated_edge_markers)

    print("\n--- 步骤 8: 保存 VTK 可视化文件 ---")
    mesh_points_3d = np.hstack([generated_nodes_xy, mesh_z_bed.reshape(-1, 1)])
    save_vtk_for_visualization('../' + OUTPUT_VTK_VIS, mesh_points_3d, generated_triangles)  # 保存到模拟输出目录下的 vtk 文件

    print(f"\n网格数据准备流程完成。")
    # print(f"网格文件输出到: {output_dir_base}")
    print(f"VTK 可视化文件: {OUTPUT_VTK_VIS}")
    # --- 步骤 9: (可选) 可视化二维网格 ---
    print("\n--- 步骤 9: 可视化二维网格 ---")  # 打印步骤9的标题
    visualize_mesh_2d(poly_data, generated_nodes_xy, generated_triangles)  # 调用函数可视化二维网格

    print("\n基于 triangle 库的网格数据准备流程完成。")  # 打印流程完成的消息
    print(f"最终节点文件: {NODE_FINAL_FILE}")  # 打印最终节点文件路径
    print(f"最终单元文件: {CELL_FINAL_FILE}")  # 打印最终单元文件路径
    if generated_edges is not None: print(f"最终边文件: {EDGE_FINAL_FILE}")  # 如果有边文件，打印其路径
    print(f"VTK可视化文件: {OUTPUT_VTK_VIS}")  # 打印VTK文件路径