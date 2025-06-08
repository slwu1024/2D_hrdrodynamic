# prepare_mesh_data.py
import numpy as np  # NumPy 用于数值运算
import os  # os 模块用于操作系统相关功能，如路径操作
import math
import shutil  # shutil 模块用于文件操作，如复制
import triangle  # triangle 库用于生成符合Delaunay条件的二维网格
import matplotlib.pyplot as plt  # matplotlib 用于绘图
from mpl_toolkits.mplot3d import Axes3D # 导入3D绘图工具
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
config_path = 'config.yaml'  # 定义配置文件路径
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
    """计算给定 (x, y) 处的底高程。"""
    global config  # 声明使用全局的config变量
    test_case_name = config.get('mesh_generation', {}).get('test_case_name', '')  # 获取测试用例名称

    if test_case_name == "混合流缓流":  # 如果是混合流缓流算例
        print("  地形函数: 为 '混合流缓流' 计算抛物线型底坎。")  # 打印信息
        # b = max[0, 0.2 - 0.05 * (x - 10)^2]
        val_in_parabola = 0.2 - 0.05 * (x - 10.0) ** 2  # 计算抛物线内的值
        z_bed = np.maximum(0.0, val_in_parabola)  # 取它和0之间的较大值
        return z_bed  # 返回计算的底高程
    elif test_case_name == "Bernetti台阶溃坝":  # 如果是Bernetti算例
        print("  地形函数: 为 'Bernetti台阶溃坝' 计算底高程。")  # 打印信息
        z_bed = np.zeros_like(x)  # 初始化底高程为0
        z_bed[x >= 10.0] = 0.2  # x >= 10m 的区域，底高程为 0.2m
        return z_bed  # 返回计算的底高程
    elif test_case_name == "Stoker一维溃坝":  # 如果是Stoker算例
        print("  地形函数: 为 'Stoker一维溃坝' (平底) 返回0.")  # 打印信息
        return np.zeros_like(x)  # 返回0
    elif test_case_name == "三角堰溃坝":  # 如果是三角堰溃坝算例
        print("  地形函数: 为 '三角堰溃坝' 计算底高程 (含三角堰)。")  # 打印信息
        z_bed = np.zeros_like(x)  # 主河道是平底，高程为0

        # 三角堰参数
        dam_loc_x = 15.5  # 大坝位置
        weir_dist_from_dam = 13.0  # 堰顶距大坝下游的距离
        weir_crest_x = dam_loc_x + weir_dist_from_dam  # 三角堰堰顶的x坐标 (28.5 m)
        weir_base_length = 6.0  # 三角堰底边长
        weir_height = 0.4  # 三角堰高度

        weir_start_x = weir_crest_x - weir_base_length / 2.0  # 堰底坡脚起点x坐标 (25.5 m)
        weir_end_x = weir_crest_x + weir_base_length / 2.0  # 堰底坡脚终点x坐标 (31.5 m)

        # 计算三角堰上游坡面上的高程
        mask_weir_upstream_slope = (x >= weir_start_x) & (x < weir_crest_x)  # 定义上游坡面掩码
        # 确保分母不为零
        if (weir_crest_x - weir_start_x) > 1e-9:  # 检查堰顶和起点的x坐标差是否大于一个极小值
            z_bed[mask_weir_upstream_slope] = \
                (x[mask_weir_upstream_slope] - weir_start_x) * weir_height / (
                            weir_crest_x - weir_start_x)  # 线性插值计算上游坡面高程

        # 计算三角堰下游坡面上的高程
        mask_weir_downstream_slope = (x >= weir_crest_x) & (x <= weir_end_x)  # 注意这里用 <= 包含堰尾点, 定义下游坡面掩码
        # 确保分母不为零
        if (weir_end_x - weir_crest_x) > 1e-9:  # 检查终点和堰顶的x坐标差是否大于一个极小值
            z_bed[mask_weir_downstream_slope] = \
                weir_height - (x[mask_weir_downstream_slope] - weir_crest_x) * weir_height / (
                            weir_end_x - weir_crest_x)  # 线性插值计算下游坡面高程

        # 确保其他区域（非堰体区域）的z_bed保持为0，并且堰体计算结果不会意外变成负数
        z_bed = np.maximum(0, z_bed)  # 确保底高程非负

        return z_bed  # 返回计算的底高程
    elif test_case_name == "梯形堰溃坝":  # 如果是梯形堰溃坝算例
        print("  地形函数: 为 '梯形堰溃坝' 计算底高程。")  # 打印信息
        z_bed = np.zeros_like(x)  # 初始化底高程为0

        # 定义第一个梯形堰 (Hump A) 的参数
        p2_x, p2_z = 2.00, 0.000
        p3_x, p3_z = 2.20, 0.108  # 堰坡起点
        # P4 (2.25, 0.108) 是大坝位置，也是堰顶的一部分
        p5_x, p5_z = 2.30, 0.108  # 堰坡终点
        p6_x, p6_z = 2.50, 0.000

        # 定义第二个梯形堰 (Hump B) 的参数
        p7_x, p7_z = 6.10, 0.000
        p8_x, p8_z = 6.50, 0.213  # 堰坡起点
        # P9 (6.60, 0.213) 是堰顶的一部分
        p10_x, p10_z = 7.00, 0.000

        # 条件掩码
        # Hump A 上坡 (P2-P3)
        mask_a_up = (x >= p2_x) & (x < p3_x)
        z_bed[mask_a_up] = p2_z + (x[mask_a_up] - p2_x) * (p3_z - p2_z) / (p3_x - p2_x)

        # Hump A 堰顶 (P3-P5)
        mask_a_crest = (x >= p3_x) & (x < p5_x)
        z_bed[mask_a_crest] = p3_z  # p3_z 和 p5_z 相同

        # Hump A 下坡 (P5-P6)
        mask_a_down = (x >= p5_x) & (x < p6_x)
        z_bed[mask_a_down] = p5_z + (x[mask_a_down] - p5_x) * (p6_z - p5_z) / (p6_x - p5_x)

        # Hump B 上坡 (P7-P8)
        mask_b_up = (x >= p7_x) & (x < p8_x)
        z_bed[mask_b_up] = p7_z + (x[mask_b_up] - p7_x) * (p8_z - p7_z) / (p8_x - p7_x)

        # Hump B 堰顶 (P8-P9), P9在x=6.6
        p9_x_ocr = 6.60  # 根据OCR图3-26中的Point P9的x坐标
        mask_b_crest = (x >= p8_x) & (x < p9_x_ocr)
        z_bed[mask_b_crest] = p8_z

        # Hump B 下坡 (P9-P10)
        mask_b_down = (x >= p9_x_ocr) & (x <= p10_x)  # 注意这里用 <= 包含P10
        # 确保分母不为零
        if (p10_x - p9_x_ocr) > 1e-9:
            z_bed[mask_b_down] = p8_z + (x[mask_b_down] - p9_x_ocr) * (p10_z - p8_z) / (p10_x - p9_x_ocr)
        else:  # 如果p9_x_ocr 和 p10_x 几乎相同 (虽然在此例中不太可能)
            z_bed[mask_b_down] = p8_z  # 或 p10_z

        # 确保河道起点和终点的底高程是0 (在分段赋值之外的区域默认为0)
        # 上述逻辑已经覆盖了0-2.0 和 2.5-6.1 以及 7.0 之后的区域（因为z_bed初始化为0）
        return z_bed  # 返回计算的底高程
    elif test_case_name == "二维非对称局部溃坝":  # 如果是二维非对称局部溃坝算例
        print("  地形函数: 为 '二维非对称局部溃坝' 计算底高程 (平底，但有未溃坝段)。")  # 打印信息
        z_bed = np.zeros_like(x)  # 默认是平底，高程为0

        # 定义大坝和溃口参数 (与YAML中initial_conditions一致，但这里用于地形)
        dam_y_start = 100.0
        dam_y_end = 115.0
        breach_x_start = 95.0
        breach_x_end = 170.0

        # 未溃坝段的高程 (设定一个足够高的值，确保不被淹没)
        unbreached_dam_elevation = 50.0  # 例如15m，高于上游水位10m

        # 创建掩码来识别未溃坝段
        # 条件1: y坐标在大坝的范围内
        mask_y_in_dam_strip = (y >= dam_y_start) & (y < dam_y_end)  # 定义y坐标在大坝范围内的掩码

        # 条件2: x坐标在未溃口部分
        mask_x_unbreached_left = (x < breach_x_start)  # 定义x坐标在左侧未溃口部分的掩码
        mask_x_unbreached_right = (x >= breach_x_end)  # 定义x坐标在右侧未溃口部分的掩码

        # 组合掩码找到未溃坝段
        mask_unbreached_dam_left = mask_y_in_dam_strip & mask_x_unbreached_left  # 定义左侧未溃坝段的掩码
        mask_unbreached_dam_right = mask_y_in_dam_strip & mask_x_unbreached_right  # 定义右侧未溃坝段的掩码

        # 设置未溃坝段的高程
        z_bed[mask_unbreached_dam_left] = unbreached_dam_elevation  # 设置左侧未溃坝段的高程
        z_bed[mask_unbreached_dam_right] = unbreached_dam_elevation  # 设置右侧未溃坝段的高程

        return z_bed  # 返回计算的底高程
    elif test_case_name == "LabSymmetricDamBreak":  # 如果是实验室对称溃坝算例 # 中文注释：判断是否为实验室对称溃坝算例
        print("  地形函数: 为 'LabSymmetricDamBreak' 计算底高程 (平底，但有高坝体段)。")  # 中文注释：打印地形函数信息
        z_bed = np.zeros_like(x)  # 默认为平底，高程为0 # 中文注释：初始化底高程为0 (平底)

        # 定义大坝和溃口参数 (与 .poly 文件和config对应)
        dam_location_x = 1.0  # 大坝的x坐标 # 中文注释：大坝的x坐标
        breach_y_bottom = 0.8  # 溃口底部y坐标 # 中文注释：溃口底部y坐标
        breach_y_top = 1.2  # 溃口顶部y坐标 # 中文注释：溃口顶部y坐标

        # 残留坝体的高度 (文献中设为2m)
        unbreached_dam_elevation = 2.0  # 中文注释：未溃坝段的高程设为2m

        # 创建掩码来识别未溃坝段
        # 条件1: x坐标在大坝的位置 (允许一个小公差)
        # 注意：在.poly文件中，坝体是由线段定义的。
        # 在这里，我们是为 *所有* 网格节点计算高程。
        # 节点如果在代表未溃坝段的 *原始* .poly 线段附近，
        # 并且这些原始.poly线段的marker是3，那么这些节点在三角剖分后，
        # 如果它们所属的单元的重心或节点本身满足以下条件，则应被抬高。
        # 更简单的方法是，如果一个节点 (x,y) 的 x 约等于 dam_location_x，
        # 并且其 y 不在溃口范围内，则它是未溃坝部分。

        # 考虑一个小的厚度，因为网格节点不一定精确落在x=1.0上
        dam_thickness_for_eval = 0.05  # 例如，坝体在x方向影响范围为 +/- 0.025m (可调) # 中文注释：为高程函数评估设定一个虚拟的坝体厚度
        mask_x_on_dam_line = (x >= dam_location_x - dam_thickness_for_eval / 2) & \
                             (x <= dam_location_x + dam_thickness_for_eval / 2)  # 中文注释：定义x坐标在大坝线附近的掩码

        # 条件2: y坐标在未溃口部分
        mask_y_unbreached_bottom = (y < breach_y_bottom)  # 中文注释：定义y坐标在溃口下方的掩码
        mask_y_unbreached_top = (y >= breach_y_top)  # 中文注释：定义y坐标在溃口上方的掩码

        # 组合掩码找到未溃坝段
        mask_unbreached_dam_bottom_part = mask_x_on_dam_line & mask_y_unbreached_bottom  # 中文注释：定义底部未溃坝段的掩码
        mask_unbreached_dam_top_part = mask_x_on_dam_line & mask_y_unbreached_top  # 中文注释：定义顶部未溃坝段的掩码

        # 设置未溃坝段的高程
        z_bed[mask_unbreached_dam_bottom_part] = unbreached_dam_elevation  # 中文注释：设置底部未溃坝段的高程
        z_bed[mask_unbreached_dam_top_part] = unbreached_dam_elevation  # 中文注释：设置顶部未溃坝段的高程

        return z_bed  # 中文注释：返回计算的底高程
    elif test_case_name == "LShapedChannelDamBreak":  # 中文注释：L型弯曲河道溃坝试验
        print("  地形函数: 为 'LShapedChannelDamBreak' 返回平底 (0m)。")  # 中文注释：打印信息
        return np.zeros_like(x)  # 中文注释：返回全零底高程
    elif test_case_name == "StillWaterDryBedCProperty":  # 如果是静水干湿界面C属性测试
        print("  地形函数: 为 'StillWaterDryBedCProperty' 计算高斯钟形圆形底坎。")  # 打印提示信息

        x0 = 0.5  # 底坎中心x坐标 # 中文注释：底坎中心x坐标
        y0 = 0.5  # 底坎中心y坐标 # 中文注释：底坎中心y坐标
        h_mound = 0.22  # 底坎最大高度 # 中文注释：底坎最大高度 (可以根据文献图调整)
        sigma = 0.15  # 高斯函数的标准差，控制底坎的宽度/坡度 # 中文注释：高斯函数的标准差 (控制宽度)

        # 计算每个点到中心的距离的平方
        dist_sq = np.square(x - x0) + np.square(y - y0)  # 中文注释：计算到中心的距离平方

        # 计算高斯函数定义的底高程
        z_bed_gaussian = h_mound * np.exp(-dist_sq / (2 * np.square(sigma)))  # 中文注释：计算高斯函数值

        # 确保底高程非负 (虽然高斯函数本身非负，但以防万一)
        z_bed = np.maximum(0.0, z_bed_gaussian)  # 中文注释：确保底高程非负
        return z_bed
    elif test_case_name == "过驼峰溃坝":  # 修改: 使用中文算例名
        print("  地形函数: 为 '过驼峰溃坝' 计算驼峰底高程。")  # 打印信息
        # 驼峰参数 (来自OCR Page 31)
        # Hump 1 (1m高)
        cx1, cy1, h_peak1, k1 = 30.0, 6.0, 1.0, 0.125
        # Hump 2 (1m高)
        cx2, cy2, h_peak2, k2 = 30.0, 24.0, 1.0, 0.125
        # Hump 3 (3m高)
        cx3, cy3, h_peak3, k3 = 47.5, 15.0, 3.0, 0.3

        # 计算到各驼峰中心的距离
        dist1 = np.sqrt((x - cx1) ** 2 + (y - cy1) ** 2)  # 计算到驼峰1中心的距离
        dist2 = np.sqrt((x - cx2) ** 2 + (y - cy2) ** 2)  # 计算到驼峰2中心的距离
        dist3 = np.sqrt((x - cx3) ** 2 + (y - cy3) ** 2)  # 计算到驼峰3中心的距离

        # 计算各驼峰贡献的高程 (根据公式 H_peak - k * dist，且不小于0)
        z_hump1_contrib = np.maximum(0.0, h_peak1 - k1 * dist1)  # 计算驼峰1贡献的高程
        z_hump2_contrib = np.maximum(0.0, h_peak2 - k2 * dist2)  # 计算驼峰2贡献的高程
        z_hump3_contrib = np.maximum(0.0, h_peak3 - k3 * dist3)  # 计算驼峰3贡献的高程

        # 最终底高程是所有驼峰贡献的最大值 (因为公式是 max[0, H1_formula, H2_formula, H3_formula])
        # np.maximum.reduce 会对一个列表中的数组逐元素取最大值
        z_bed = np.maximum.reduce([z_hump1_contrib, z_hump2_contrib, z_hump3_contrib])  # 取各驼峰贡献的最大值
        # 确保整体不小于0 (虽然每个contrib已经保证了，但以防万一)
        z_bed = np.maximum(0.0, z_bed)  # 确保最终底高程不小于0
        return z_bed  # 返回计算的底高程
    elif test_case_name == "收缩扩散水槽溃坝":  # 算例名称
        print("  地形函数: 为 '收缩扩散水槽溃坝' 计算线性底坡，x=0处底高程为0.15m。")  # 打印信息

        # 参数
        x_datum = 0.0  # 参考点的x坐标 (水闸中心)
        z_datum = 0.15  # 参考点x_datum处的底高程 (m)
        bed_slope = 0.002  # 水槽的总体底坡 (正值表示向下游倾斜，即x增大z减小)

        # 计算底高程 z(x) = z_datum - bed_slope * (x - x_datum)
        z_bed = z_datum - bed_slope * (x - x_datum)  # 计算所有点的底高程

        # 调试打印，检查关键点的高程
        if x.size > 0:  # 检查x数组是否为空
            print("    地形调试打印 (基于线性坡面):")  # 打印地形调试信息

            x_upstream_entry = -8.5  # 上游入口x坐标
            idx_upstream_entry = np.argmin(np.abs(x - x_upstream_entry))  # 找到最接近上游入口的索引
            print(
                f"      x ≈ {x[idx_upstream_entry]:.3f}m (上游入口): z_bed={z_bed[idx_upstream_entry]:.4f}m (理论值: {z_datum - bed_slope * (x_upstream_entry - x_datum):.4f})")  # 打印上游入口高程信息

            idx_throat = np.argmin(np.abs(x - x_datum))  # 找到最接近水闸中心的索引
            print(
                f"      x ≈ {x[idx_throat]:.3f}m (水闸中心): z_bed={z_bed[idx_throat]:.4f}m (理论值: {z_datum:.4f})")  # 打印水闸中心高程信息

            x_downstream_exit = 12.7  # 下游出口x坐标
            idx_downstream_exit = np.argmin(np.abs(x - x_downstream_exit))  # 找到最接近下游出口的索引
            print(
                f"      x ≈ {x[idx_downstream_exit]:.3f}m (下游出口): z_bed={z_bed[idx_downstream_exit]:.4f}m (理论值: {z_datum - bed_slope * (x_downstream_exit - x_datum):.4f})")  # 打印下游出口高程信息

        return z_bed  # 返回计算的底高程数组
    elif test_case_name == "二维抛物盆地自由水面":  # 算例名称
        print("  地形函数: 为 '二维抛物盆地自由水面' (Thacker) 计算中心嵌入式抛物线水缸地形。")  # 打印信息

        # 抛物线形状参数 (与之前相同)
        a_param = 8025.5  # m
        h0_param = 10.0  # m

        # 新增参数：定义嵌入式水缸的属性
        # R_cylinder: 水缸在水平面上的影响半径。在这个半径之外，地形是平坦的。
        # 我们需要选择一个合理的 R_cylinder。
        # Thacker算例中，水体晃动的最大范围可以达到 a + sigma。
        # sigma = a/10 = 802.55m。所以 a + sigma ≈ 8828m。
        # 我们可以让水缸的半径略大于这个值，例如 R_cylinder = 9000m。
        # 或者，如果我们希望在计算域边界附近是平的，可以取更小的值，
        # 比如 R_cylinder = 7000m，这样从7000m到10000m就是平地。
        # 让我们先尝试一个相对较小的 R_cylinder，以便能清晰看到平坦边缘。
        R_cylinder = 7500.0  # m (示例值，可以调整)

        if a_param == 0:  # 防止除以零
            print("    错误: Thacker算例参数 'a' 为零，返回平底。")  # 打印错误信息
            return np.zeros_like(x)  # 返回全零数组

        # 计算在水缸边缘 r = R_cylinder 处的抛物线底高程
        # 这个值将作为水缸外部平坦区域的背景底高程
        z_background = h0_param * (R_cylinder ** 2) / (a_param ** 2)  # 计算背景高程
        print(
            f"    水缸参数: R_cylinder={R_cylinder:.1f}m, 计算得到的背景高程 z_background={z_background:.4f}m")  # 打印水缸参数

        # 计算每个点到原点(水缸中心)的距离的平方
        r_squared = np.square(x) + np.square(y)  # 计算到中心的距离平方

        # 计算基于纯抛物线公式的底高程 (对于所有点)
        z_parabolic_full = h0_param * r_squared / np.square(a_param)  # 计算纯抛物线高程

        # 根据点是否在水缸内，决定最终的底高程
        # 创建一个与 x 形状相同的数组，并用背景高程填充
        z_bed = np.full_like(x, z_background, dtype=float)  # 初始化z_bed为背景高程

        # 找到在水缸内部的点 (r < R_cylinder)
        mask_inside_cylinder = r_squared < (R_cylinder ** 2)  # 定义水缸内部的掩码

        # 对于水缸内部的点，使用抛物线公式计算的高程
        z_bed[mask_inside_cylinder] = z_parabolic_full[mask_inside_cylinder]  # 将水缸内部点的高程设为抛物线高程

        # 调试打印
        if x.size > 0:  # 检查x数组是否为空
            print(f"    Thacker参数: a={a_param}, h0={h0_param}")  # 打印参数信息
            idx_center = np.argmin(r_squared)  # 找到最接近中心的点
            print(
                f"      在 x≈{x[idx_center]:.1f}, y≈{y[idx_center]:.1f} (中心附近): z_parabolic={z_parabolic_full[idx_center]:.4f}, z_bed_final={z_bed[idx_center]:.4f} (理论值: 0.0)")  # 打印中心点高程

            # 检查一个在水缸边缘的点 (例如 x=R_cylinder, y=0)
            idx_on_edge = np.argmin(np.abs(x - R_cylinder) + np.abs(y - 0.0))  # 找到最接近(R_cylinder,0)的点
            if np.sqrt(r_squared[idx_on_edge]) >= R_cylinder - R_cylinder * 0.01 and \
                    np.sqrt(r_squared[idx_on_edge]) <= R_cylinder + R_cylinder * 0.01:  # 如果点确实在边缘附近
                print(
                    f"      在 x≈{x[idx_on_edge]:.1f}, y≈{y[idx_on_edge]:.1f} (水缸边缘 r≈R_cylinder): z_parabolic={z_parabolic_full[idx_on_edge]:.4f}, z_bed_final={z_bed[idx_on_edge]:.4f} (理论值: {z_background:.4f})")  # 打印水缸边缘点高程

            # 检查一个在水缸外部的点 (例如 x=R_cylinder + 100, y=0)
            x_outside_check = R_cylinder + 100.0  # 定义水缸外部检查点x坐标
            idx_outside = np.argmin(np.abs(x - x_outside_check) + np.abs(y - 0.0))  # 找到最接近该点的索引
            if np.sqrt(r_squared[idx_outside]) > R_cylinder:  # 如果点确实在水缸外
                print(
                    f"      在 x≈{x[idx_outside]:.1f}, y≈{y[idx_outside]:.1f} (水缸外部): z_parabolic={z_parabolic_full[idx_outside]:.4f}, z_bed_final={z_bed[idx_outside]:.4f} (理论值: {z_background:.4f})")  # 打印水缸外部点高程

            print(f"      计算得到的最终 z_bed 范围: min={np.min(z_bed):.4f}, max={np.max(z_bed):.4f}")  # 打印最终z_bed范围

        return z_bed  # 返回计算的底高程数组
    else:  # 其他未知情况
        print(f"  地形函数: 未识别的测试用例名称 '{test_case_name}' 或 'function' 方法用于默认平底，返回0。")  # 打印信息
        return np.zeros_like(x)  # 默认返回平底0


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


def save_cell_file(filepath, triangles, triangle_attributes): # 新增参数
    print(f"保存单元文件 (含区域属性) 到 {filepath}...")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 文件头现在需要指明有属性列
            # <#单元数> <每个单元节点数(3)> <#单元属性(1)>
            f.write(f"{len(triangles)} 3 1\n") # 指明有1个单元属性
            for i, tri_nodes in enumerate(triangles):
                attr = triangle_attributes[i] if triangle_attributes is not None and i < len(triangle_attributes) else 0.0 # 默认属性0
                # 写入: 单元ID 节点1 节点2 节点3 属性
                f.write(f"{i} {tri_nodes[0]} {tri_nodes[1]} {tri_nodes[2]} {attr:.1f}\n") # 将属性格式化为浮点数
        print("单元文件 (含区域属性) 保存成功.")
        return True
    except Exception as e:
        print(f"保存单元文件 {filepath} 时出错: {e}")
        return False


def save_edge_file(filepath, edges_data_list): # edges_data_list 是 (n1, n2, type_marker, original_seg_id) 的列表
    num_edges = len(edges_data_list)
    # 文件头现在可以简单地指明列数，或者C++端可以根据实际读取的列来判断
    # 这里我们约定，如果写入了 original_seg_id，C++就会读取它
    # has_marker 在这里表示 boundary_type_marker 是否 > 0
    # 我们需要一个新的概念，或者让C++总是尝试读取第五列
    # 为了简单，让 .edge 文件对所有边（包括内部边）都有5列：
    # edge_idx n1 n2 type_marker original_seg_id (内部边时 type_marker=0, original_seg_id=-1)
    print(f"保存边文件到 {filepath} (包含类型标记和原始Segment ID)...")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 文件头可以只写 num_edges，C++解析时判断每行的列数
            # 或者，文件头指示是否有原始ID列，例如 num_edges <has_type_marker> <has_original_id>
            # 我们采用固定5列的方式，所以文件头可以简单些：
            f.write(f"{num_edges} 1\n") # 1 表示 type_marker 列存在 (original_id 列也隐含存在)
            for i, edge_data in enumerate(edges_data_list):
                node1_id, node2_id, type_marker, original_seg_id = edge_data
                # 写入每行边数据：边文件行号、节点1 ID、节点2 ID、类型标记、原始Segment ID
                f.write(f"{i} {node1_id} {node2_id} {type_marker} {original_seg_id}\n") # <--- 修改此行
        print("边文件保存成功.")
        return True
    except Exception as e:
        print(f"保存边文件 {filepath} 时出错: {e}")
        return False


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


def visualize_mesh_3d(parsed_poly_data, generated_mesh_points_xy, generated_mesh_z_bed, generated_triangles):
    """可视化生成的带地形的三维网格。"""
    print("  正在生成三维网格可视化图...")  # 打印信息
    fig = plt.figure(figsize=(12, 9))  # 创建图形，可以调整大小
    ax = fig.add_subplot(111, projection='3d')  # 添加3D子图

    if generated_mesh_points_xy is not None and generated_mesh_z_bed is not None and generated_triangles is not None:
        # 组合X, Y, Z坐标
        x = generated_mesh_points_xy[:, 0]
        y = generated_mesh_points_xy[:, 1]
        z = generated_mesh_z_bed  # 这个应该是与 generated_mesh_points_xy 中的节点一一对应的高程值

        # 绘制三维三角网格表面
        # plot_trisurf 非常适合这种情况
        # alpha 用于设置透明度，以便能看到一些结构
        # cmap 用于设置颜色映射，可以根据高程着色
        ax.plot_trisurf(x, y, generated_triangles, z, cmap=plt.cm.viridis, edgecolor='none', alpha=0.8,
                        antialiased=True)  # 绘制三维三角网格面
        # 如果想要更清晰地看到网格线，可以单独绘制边
        # ax.plot_trisurf(x, y, generated_triangles, z, color='lightblue', edgecolor='k', lw=0.1, alpha=0.5)

        # 绘制原始定义的边界线段 (如果提供了.poly数据)
        # 我们需要将这些2D线段投影到3D地形上
        if parsed_poly_data and 'points' in parsed_poly_data and 'segments' in parsed_poly_data:
            poly_points_xy = parsed_poly_data['points']  # .poly 文件中的原始2D节点坐标
            poly_segments_indices = parsed_poly_data['segments']  # .poly 文件中的线段，由节点索引组成

            if len(poly_points_xy) > 0 and len(poly_segments_indices) > 0:
                print("    正在绘制原始.poly边界线段 (投影到地形)...")  # 打印信息
                # 为了将.poly的2D节点投影到3D，我们需要它们的高程。
                # 最简单的方法是找到这些.poly节点在生成的网格节点中的最近对应点，并使用其高程。
                # 或者，如果.poly节点直接就是生成的网格节点的一部分，可以直接使用其高程。
                # 假设 parsed_poly_data['points'] 的节点顺序和ID与生成的网格节点的前N个对应
                # (这取决于 parse_poly_file 和 triangle 的行为)
                # 一个更稳健的方法是，如果.poly节点是作为约束点输入给triangle的，
                # 那么它们应该会出现在 generated_mesh_points_xy 中。

                # 简化处理：我们假设 poly_points_xy 中的点是网格生成时的输入点。
                # 我们需要找到这些点在 generated_mesh_points_xy 中的索引，或者直接插值获取它们的高程。
                # 为了简单起见，如果 poly_points_xy 与 generated_mesh_points_xy 的前 N 个点相同，我们可以直接用高程。
                # 否则，需要插值或最近邻查找。

                # 这里我们先尝试直接绘制原始2D边界在某个平均Z平面上，或者您可以实现更复杂的投影。
                # 为了在3D中显示.poly边界，我们需要为.poly的节点找到Z值。
                # 假设 bed_elevation_func 或插值函数可以用于原始.poly节点

                # 遍历.poly的线段
                for seg_indices in poly_segments_indices:
                    node_idx1 = seg_indices[0]  # 0-based index into poly_points_xy
                    node_idx2 = seg_indices[1]  # 0-based index into poly_points_xy

                    if 0 <= node_idx1 < len(poly_points_xy) and 0 <= node_idx2 < len(poly_points_xy):
                        p1_xy = poly_points_xy[node_idx1]
                        p2_xy = poly_points_xy[node_idx2]

                        # 获取这两个点的高程
                        # 方案1: 如果这些点就是生成的网格节点的一部分
                        # (这需要确定它们在 generated_mesh_points_xy 中的索引)
                        # 方案2: 重新用高程函数/插值计算这些点的高程
                        z1 = bed_elevation_func(np.array([p1_xy[0]]), np.array([p1_xy[1]]))[
                            0] if ELEVATION_SOURCE_METHOD == "function" else \
                            interpolate_elevation(topo_x_global, topo_y_global, topo_z_global, np.array([p1_xy]),
                                                  INTERPOLATION_METHOD_global)[
                                0] if 'topo_x_global' in globals() else np.mean(generated_mesh_z_bed)  # 尝试获取高程
                        z2 = bed_elevation_func(np.array([p2_xy[0]]), np.array([p2_xy[1]]))[
                            0] if ELEVATION_SOURCE_METHOD == "function" else \
                            interpolate_elevation(topo_x_global, topo_y_global, topo_z_global, np.array([p2_xy]),
                                                  INTERPOLATION_METHOD_global)[
                                0] if 'topo_x_global' in globals() else np.mean(generated_mesh_z_bed)  # 尝试获取高程

                        # 为了让边界线更突出，可以稍微抬高一点Z值
                        offset_z = 0.05  # 向上偏移一点，避免被表面覆盖
                        ax.plot([p1_xy[0], p2_xy[0]], [p1_xy[1], p2_xy[1]], [z1 + offset_z, z2 + offset_z], color='r',
                                lw=2, label="原始.poly边界" if not ax.plot([], [], [], 'r-', label="原始.poly边界")[
                                0].get_label() else "")  # 绘制3D线段
    else:
        print("  警告: 缺少足够的网格数据 (节点坐标, Z坐标, 或单元连接关系) 来生成三维可视化。")  # 打印警告

    ax.set_xlabel("X 坐标 (m)")  # 设置X轴标签
    ax.set_ylabel("Y 坐标 (m)")  # 设置Y轴标签
    ax.set_zlabel("高程 Z (m)")  # 设置Z轴标签
    ax.set_title(f"生成的三维网格 (高程来源: {ELEVATION_SOURCE_METHOD})")  # 设置标题

    # 调整视角以便更好地观察
    ax.view_init(elev=30, azim=-60)  # 可以调整 elev (仰角) 和 azim (方位角)

    # 添加颜色条 (如果使用了cmap)
    # m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    # m.set_array(z) # z 是用于着色的高程数据
    # fig.colorbar(m, shrink=0.5, aspect=5, label='高程 (m)')

    # 确保图例只显示一次
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:  # 仅当有可显示的图例项时才显示图例
        ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()  # 自动调整子图参数，使其填充整个图像区域
    print("  三维可视化图已生成，准备显示...")  # 打印信息
    plt.show()  # 显示图形

topo_x_global, topo_y_global, topo_z_global = None, None, None
INTERPOLATION_METHOD_global = None

# --- 主程序 (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    # --- 步骤 1: 解析 .poly 文件 (路径来自配置) ---
    print("--- 步骤 1: 解析 .poly 文件 ---")
    poly_data = parse_poly_file('../' + POLY_FILE)  # 使用配置中的路径
    if poly_data is None: exit()

    # --- 新增：预处理原始.poly线段信息，以便后续匹配 ---
    # 这个列表将存储 (original_poly_id, node_idx1_0based, node_idx2_0based, type_marker_from_poly)
    # 注意：parse_poly_file 需要调整或你在这里重新处理 .poly 文件来获得这些信息
    # 以下是一个假设性的处理，你需要根据 parse_poly_file.py 的实际输出来调整
    # 或者直接修改 parse_poly_file.py 让它返回更丰富的信息。

    # 假设 poly_data['segments_raw_info'] 是一个列表，
    # 每个元素是 {'poly_id': 原始.poly线段ID, 'nodes': [原始节点ID1, 原始节点ID2], 'marker': 类型标记}
    # 并且 poly_data['node_index_base'] 是 .poly 文件中节点ID的起始值 (0或1)
    # poly_data['points'] 已经是0-based的节点坐标了

    original_poly_segments_for_mapping = []
    if 'segments_raw_info' in poly_data and 'node_index_base' in poly_data:  # 假设parse_poly_file提供了这些
        node_index_base = poly_data['node_index_base']
        for raw_seg_info in poly_data['segments_raw_info']:
            original_poly_id = raw_seg_info['poly_id']
            # 将.poly中的原始节点ID转换为0-based的索引
            # 假设 parse_poly_file 已经处理了这个转换，或者你需要在这里做
            # poly_data['segments'] 返回的是转换后的节点索引
            # 我们需要找到一种方式将原始.poly文件中的线段ID与triangle库使用的节点索引关联起来
            # 最简单的方法是修改 parse_poly_file.py，让它在返回 segments 和 segment_markers 的同时，
            # 也返回一个与 segments 对应的原始 .poly 线段ID列表。

            # 简化假设：假设 poly_data['segments'] 是 [[n1,n2], [n3,n4]...] (0-based)
            # poly_data['segment_markers'] 是 [m1, m2...]
            # 我们需要原始的 .poly 线段ID。
            # 如果 parse_poly_file 不提供，你需要在这里重新解析 .poly 的线段部分来获取原始ID
            # 或者，我们假设 parse_poly_file 返回的 poly_data 结构如下：
            # poly_data = {
            #    'points': np.array(...),
            #    'point_markers': np.array(...),
            #    'segments': np.array([[n1_0based, n2_0based], ...]), # triangle使用的节点索引
            #    'segment_markers': np.array([...]), # 最后一个数字 (类型标记)
            #    'original_segment_ids': np.array([...]) # 第一个数字 (原始.poly线段ID)
            # }
            pass  # 这个预处理部分依赖于 parse_poly_file.py 的输出

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
    if 'regions' in poly_data and len(poly_data['regions']) > 0:
        triangle_input['regions'] = poly_data['regions'].tolist()

    # --- 步骤 3: 调用 triangle.triangulate 生成网格 ---
    print("\n--- 步骤 3: 调用 triangle.triangulate 生成网格 ---")
    print(f"使用选项字符串: '{TRIANGLE_OPTS}'")
    try:
        mesh_data_dict = triangle.triangulate(triangle_input, TRIANGLE_OPTS)  # mesh_data_dict 在此定义
        print("网格生成成功。")
    except Exception as e:
        print(f"调用 triangle.triangulate 时出错: {e}")
        exit()

    # --- 步骤 4: 提取生成的网格数据 ---
    print("\n--- 步骤 4: 提取生成的网格数据 ---")

    generated_nodes_xy = mesh_data_dict.get('vertices')
    generated_node_markers = mesh_data_dict.get('vertex_markers')

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # 确保这两行在这里，从 mesh_data_dict 中获取数据并赋值给这两个变量
    generated_triangles = mesh_data_dict.get('triangles')
    generated_triangle_attributes = mesh_data_dict.get('triangle_attributes')
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    generated_edges = mesh_data_dict.get('edges')
    generated_edge_markers = mesh_data_dict.get('edge_markers')

    # --- 现在，下面的代码块（您之前粘贴的那个）就可以安全地使用 generated_triangle_attributes 了 ---
    if generated_triangles is None:
        print("错误: triangle 未返回单元数据 ('triangles')...")
        generated_triangles = np.array([], dtype=int).reshape(0, 3)
        generated_triangle_attributes = np.array([], dtype=float)  # 保持一致性

    if generated_triangle_attributes is not None:  # 此时 generated_triangle_attributes 应该已经被定义
        print(f"提取到 {len(generated_triangle_attributes)} 个单元的区域属性。")
        if generated_triangle_attributes.ndim > 1 and generated_triangle_attributes.shape[1] == 1:
            generated_triangle_attributes = generated_triangle_attributes.ravel()
            print(f"  区域属性已展平为一维数组，形状: {generated_triangle_attributes.shape}")
        # ... (其他对 generated_triangle_attributes 的处理) ...
    else:  # generated_triangle_attributes is None
        print("警告: triangle 未返回单元区域属性 ('triangle_attributes')...")
        if generated_triangles is not None and len(generated_triangles) > 0:
            print("  将为所有单元生成默认区域属性值 0.0。")
            generated_triangle_attributes = np.zeros(len(generated_triangles), dtype=float)
        else:
            generated_triangle_attributes = np.array([], dtype=float)
            print("  由于没有单元数据，单元区域属性也为空。")


    # --- 检查提取和处理后的变量 ---
    print(f"最终 generated_triangles: shape={generated_triangles.shape if generated_triangles is not None else 'None'}")
    print(
        f"最终 generated_triangle_attributes: shape={generated_triangle_attributes.shape if generated_triangle_attributes is not None else 'None'}")
    # ******** 在这里插入 `edges_to_write` 的构建逻辑 ********
    print("\n--- 步骤 4b: 准备写入.edge文件的数据 (含原始Segment ID) ---")  # 新增打印
    edges_to_write = []
    if generated_edges is not None:  # 仅当triangle生成了边数据时才执行
        # 创建一个从 (排序节点对) 到 original_poly_segment_id 的映射
        # 这里是关键，你需要确保 poly_data 提供了这种映射所需的信息
        poly_edge_to_original_id_map = {}
        # 假设 poly_data['parsed_poly_segments'] 是一个列表，每个元素是字典:
        # {'original_id': 原始.poly线段ID, 'nodes_0based': (排序后的0-based节点索引元组)}
        # 这个 'parsed_poly_segments' 需要由 parse_poly_file.py 提供，或者在这里根据 poly_data 构建
        # 例如，如果 parse_poly_file 返回 'original_segment_ids' 和 'segments' (0-based node pairs)
        # ******** 新增/修改代码开始 ********
        if 'original_segment_ids' in poly_data and 'segments' in poly_data and \
                len(poly_data['original_segment_ids']) == len(poly_data['segments']):
            print(f"  构建 poly_edge_to_original_id_map: 共 {len(poly_data['segments'])} 条原始.poly线段。")  # 调试打印
            for i_seg in range(len(poly_data['segments'])):
                original_id = int(poly_data['original_segment_ids'][i_seg])  # 确保是整数
                # poly_data['segments'][i_seg] 应该是 [node_idx1_0based, node_idx2_0based]
                nodes_0based_pair_from_poly = tuple(sorted(poly_data['segments'][i_seg]))
                poly_edge_to_original_id_map[nodes_0based_pair_from_poly] = original_id
                # 调试打印，看map的内容是否正确
                # if i_seg < 5 or i_seg > len(poly_data['segments']) - 5 : # 只打印少量样本
                #     print(f"    Map Entry: poly_edge={nodes_0based_pair_from_poly} -> original_id={original_id}")
        else:
            print(
                "警告: 步骤4b - poly_data 中缺少 'original_segment_ids' 或 'segments'，或长度不匹配。原始Segment ID映射可能不正确。")
            if 'original_segment_ids' not in poly_data: print("  'original_segment_ids' not in poly_data")
            if 'segments' not in poly_data: print("  'segments' not in poly_data")
            if 'original_segment_ids' in poly_data and 'segments' in poly_data:
                print(
                    f"  len(original_segment_ids)={len(poly_data['original_segment_ids'])}, len(segments)={len(poly_data['segments'])}")
        # ******** 新增/修改代码结束 ********

        for i, edge_nodes_pair_triangle in enumerate(
                generated_edges):  # edge_nodes_pair_triangle是triangle输出的0-based节点索引对
            node1_idx_triangle = int(edge_nodes_pair_triangle[0])
            node2_idx_triangle = int(edge_nodes_pair_triangle[1])

            type_marker_triangle = 0
            if generated_edge_markers is not None and i < len(generated_edge_markers):
                type_marker_triangle = int(generated_edge_markers[i].item())

            original_seg_id_to_write = -1
            # ******** 新增/修改代码开始 ********
            # 关键的匹配逻辑：
            # triangle 输出的 generated_edges 和 generated_edge_markers 是最终网格的边。
            # 如果一条 generated_edge 是边界边 (type_marker_triangle != 0)，
            # 我们需要找到它对应于输入给triangle的 segments 列表中的哪一个。
            # triangle 库的 'edge_markers' 通常会保留输入 segment 的 marker。
            # 重要的是，triangle 可能会细分输入的 segment。
            # 因此，一条原始的 .poly segment 可能对应多条 generated_edges。
            # 这些细分后的 generated_edges 都应该继承原始 .poly segment 的 original_id。

            if type_marker_triangle != 0:  # 只处理 triangle 标记为边界的边
                # 策略：遍历所有原始.poly线段，看这条triangle生成的边是否在其上，并且类型标记匹配。
                # 这需要几何判断，比较复杂。

                # 简化策略（假设triangle不合并原始边界段，只细分，且标记准确传递）：
                # 我们用 (排序后的triangle输出节点对) 作为键，直接在 poly_edge_to_original_id_map 中查找。
                # 这假设 triangle 输出的边界边端点与原始 .poly 线段定义的端点（转换后）完全一致。
                # 如果 triangle 细分了原始线段，这个直接查找会失败。

                key_nodes_from_triangle_output = tuple(sorted((node1_idx_triangle, node2_idx_triangle)))

                if key_nodes_from_triangle_output in poly_edge_to_original_id_map:
                    original_seg_id_to_write = poly_edge_to_original_id_map[key_nodes_from_triangle_output]
                    # 进一步验证：这条triangle边使用的类型标记是否与原始.poly线段的类型标记一致
                    # （这需要 poly_edge_to_original_id_map 同时存储原始类型标记，或者我们信任 generated_edge_markers）
                    # print(f"  匹配成功: triangle_edge={key_nodes_from_triangle_output}, marker={type_marker_triangle} -> original_id={original_seg_id_to_write}")
                else:
                    # 如果直接匹配失败，可能是因为细分。
                    # 这时，我们需要更复杂的逻辑：
                    # 检查 (node1_idx_triangle, node2_idx_triangle) 是否是 poly_edge_to_original_id_map 中
                    # 某个原始线段 (orig_n1, orig_n2) 的一部分。
                    # 并且，这条 generated_edge 的 type_marker_triangle 应该与那条原始线段的类型标记一致。

                    # 这是一个更鲁棒（但计算量稍大）的匹配方法：
                    # 遍历我们从 .poly 文件解析出来的所有原始线段信息
                    found_match_for_subsegment = False
                    if 'original_segment_ids' in poly_data and 'segments' in poly_data and 'segment_markers' in poly_data:
                        for k_poly_seg in range(len(poly_data['segments'])):
                            poly_original_id = int(poly_data['original_segment_ids'][k_poly_seg])
                            poly_type_marker = int(poly_data['segment_markers'][k_poly_seg])
                            poly_n1_0based, poly_n2_0based = poly_data['segments'][k_poly_seg]  # 这些是0-based

                            # 条件1: triangle输出的边的类型标记 与 这条原始.poly线段的类型标记 相同
                            if type_marker_triangle == poly_type_marker:
                                # 条件2: triangle输出的边的两个端点，都在这条原始.poly线段的两个端点所定义的直线上
                                # 并且，这两个端点位于原始线段的两个端点之间（或其中一个重合）
                                # (这是一个简化的共线性和平行性检查，实际可能需要更精确的几何判断)

                                # 获取原始.poly线段端点的坐标
                                p_poly_n1 = generated_nodes_xy[poly_n1_0based]
                                p_poly_n2 = generated_nodes_xy[poly_n2_0based]
                                # 获取triangle输出边的端点坐标
                                p_tri_n1 = generated_nodes_xy[node1_idx_triangle]
                                p_tri_n2 = generated_nodes_xy[node2_idx_triangle]

                                # 向量法判断共线性和是否在内部
                                vec_poly = p_poly_n2 - p_poly_n1
                                vec_tri1_from_poly1 = p_tri_n1 - p_poly_n1
                                vec_tri2_from_poly1 = p_tri_n2 - p_poly_n1

                                cross_prod1 = np.cross(vec_poly, vec_tri1_from_poly1)
                                cross_prod2 = np.cross(vec_poly, vec_tri2_from_poly1)

                                # 容差
                                tol = 1e-9 * np.linalg.norm(vec_poly) if np.linalg.norm(vec_poly) > 1e-9 else 1e-9

                                if abs(cross_prod1) < tol and abs(cross_prod2) < tol:  # 共线
                                    dot_tri1 = np.dot(vec_tri1_from_poly1, vec_poly)
                                    dot_tri2 = np.dot(vec_tri2_from_poly1, vec_poly)
                                    len_sq_poly = np.dot(vec_poly, vec_poly)

                                    # 检查投影是否在 [0, len_sq_poly] 之间
                                    on_segment1 = (0 - tol <= dot_tri1 <= len_sq_poly + tol)
                                    on_segment2 = (0 - tol <= dot_tri2 <= len_sq_poly + tol)

                                    if on_segment1 and on_segment2:
                                        original_seg_id_to_write = poly_original_id
                                        found_match_for_subsegment = True
                                        # print(f"  细分匹配: tri_edge=({node1_idx_triangle}-{node2_idx_triangle}), marker={type_marker_triangle} "
                                        #       f"属于原始poly_seg_id={original_seg_id_to_write} (nodes_poly={poly_n1_0based}-{poly_n2_0based}, marker_poly={poly_type_marker})")
                                        break  # 找到一个匹配的原始线段就够了

                    if not found_match_for_subsegment:
                        print(
                            f"警告: 步骤4b - 无法为triangle生成的边界边 ({node1_idx_triangle}-{node2_idx_triangle}, 类型标记={type_marker_triangle}) 找到对应的原始 .poly segment ID (尝试了直接匹配和细分检查)。将使用-1。")
            # ******** 新增/修改代码结束 ********
            edges_to_write.append((node1_idx_triangle, node2_idx_triangle, type_marker_triangle, original_seg_id_to_write))
            # ******** 新增/修改代码开始 ********
            # # 在循环内部添加打印，以便看到每条边被赋予的 original_seg_id
            # if type_marker_triangle != 0:  # 只看我们关心的边界边
            #     print(
            #         f"  DEBUG prepare_mesh (edges_to_write): tri_edge=({node1_idx_triangle}-{node2_idx_triangle}), type_marker={type_marker_triangle}, ASSIGNED original_seg_id={original_seg_id_to_write}")
            # # ******** 新增/修改代码结束 ********
    else:
        print("信息: triangle库未生成边数据 (`generated_edges` 为 None)。将不生成 .edge 文件。")
    # ******** `edges_to_write` 构建逻辑结束 ********
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
    mesh_z_bed = None  # 初始化
    if ELEVATION_SOURCE_METHOD == "function":
        # **确保 bed_elevation_func 函数已在此脚本中定义**
        mesh_z_bed = bed_elevation_func(generated_nodes_xy[:, 0], generated_nodes_xy[:, 1])
        print(f"已为 {len(mesh_z_bed)} 个节点通过函数 '{bed_elevation_func.__name__}' 计算底高程。")
    elif ELEVATION_SOURCE_METHOD == "interpolation":
        TOPOGRAPHY_FILE = file_paths.get('topography_file')  # 从配置获取地形文件路径
        INTERPOLATION_METHOD = mesh_gen_config.get('interpolation_method', 'kriging')  # 从配置获取插值方法
        # ******** 新增/修改代码开始 ********
        INTERPOLATION_METHOD_global = INTERPOLATION_METHOD  # 设置全局变量以便可视化函数使用
        # ******** 新增/修改代码结束 ********
        if not TOPOGRAPHY_FILE:
            print("错误: 高程来源为 'interpolation' 但未在 yaml 的 file_paths 中配置 topography_file。")
            exit()
        print(f"  使用插值方法: {INTERPOLATION_METHOD}, 地形文件: {TOPOGRAPHY_FILE}")
        topo_x, topo_y, topo_z = read_topography_csv('../' + TOPOGRAPHY_FILE)
        # ******** 新增/修改代码开始 ********
        topo_x_global, topo_y_global, topo_z_global = topo_x, topo_y, topo_z  # 设置全局变量
        # ******** 新增/修改代码结束 ********
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

    # --- 步骤 7: 保存最终模型输入文件 ---
    print("\n--- 步骤 7: 保存最终模型输入文件 ---")
    save_node_file_with_z('../' + NODE_FINAL_FILE, generated_nodes_xy, mesh_z_bed, final_node_markers)
    # 在调用 save_cell_file 之前，确保 generated_triangles 和 generated_triangle_attributes 都是有效的 NumPy 数组
    if generated_triangles is not None and generated_triangle_attributes is not None:
        # 确保长度匹配 (在正常情况下，如果两者都来自triangle且处理正确，应该匹配)
        if len(generated_triangles) == len(generated_triangle_attributes):
            save_cell_file('../' + CELL_FINAL_FILE, generated_triangles, generated_triangle_attributes)  # 传递属性
        else:
            print(
                f"错误: 单元数量 ({len(generated_triangles)}) 与单元属性数量 ({len(generated_triangle_attributes)}) 不匹配！无法保存带属性的单元文件。")
            # 可以选择保存不带属性的单元文件作为备用，或者直接报错退出
            # save_cell_file_without_attributes('../' + CELL_FINAL_FILE, generated_triangles) # 需要一个这样的函数
    elif generated_triangles is not None:
        print("警告: 单元属性数据无效，将尝试保存不含属性的单元文件。")
        # save_cell_file_without_attributes('../' + CELL_FINAL_FILE, generated_triangles) # (如果需要)
    else:
        print("错误: 没有有效的单元数据可供保存。")

    save_cell_file('../' + CELL_FINAL_FILE, generated_triangles, generated_triangle_attributes) # 传递属性

    # 修改 save_edge_file 的调用
    if generated_edges is not None and edges_to_write:  # 确保有边且edges_to_write已填充
        save_edge_file('../' + EDGE_FINAL_FILE, edges_to_write)  # <--- 修改调用
    elif generated_edges is not None and not edges_to_write:
        print("警告: `generated_edges` 存在但 `edges_to_write` 为空。可能是因为处理逻辑跳过了所有边。不保存.edge文件。")

    print("\n--- 步骤 8: 保存 VTK 可视化文件 ---")
    mesh_points_3d = np.hstack([generated_nodes_xy, mesh_z_bed.reshape(-1, 1)])
    save_vtk_for_visualization('../' + OUTPUT_VTK_VIS, mesh_points_3d, generated_triangles)  # 保存到模拟输出目录下的 vtk 文件

    print(f"\n网格数据准备流程完成。")
    # print(f"网格文件输出到: {output_dir_base}")
    print(f"VTK 可视化文件: {OUTPUT_VTK_VIS}")
    # # --- 步骤 9: (可选) 可视化 ---
    # print("\n--- 步骤 9: 可视化三维网格 ---")  # 修改打印信息
    # if generated_nodes_xy is not None and mesh_z_bed is not None and generated_triangles is not None:
    #     # ******** 新增/修改代码开始 ********
    #     visualize_mesh_3d(poly_data, generated_nodes_xy, mesh_z_bed, generated_triangles)  # 调用新的3D可视化函数
    #     # ******** 新增/修改代码结束 ********
    # else:
    #     print("  跳过可视化，因为缺少必要的网格数据。")  # 打印信息

    print("\n基于 triangle 库的网格数据准备流程完成。")
    print(f"最终节点文件: {NODE_FINAL_FILE}")  # 打印最终节点文件路径
    print(f"最终单元文件: {CELL_FINAL_FILE}")  # 打印最终单元文件路径
    if generated_edges is not None: print(f"最终边文件: {EDGE_FINAL_FILE}")  # 如果有边文件，打印其路径
    print(f"VTK可视化文件: {OUTPUT_VTK_VIS}")  # 打印VTK文件路径