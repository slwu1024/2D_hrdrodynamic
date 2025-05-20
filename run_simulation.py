# run_simulation.py
import numpy as np  # 导入 numpy
import os  # 导入 os 模块
import yaml  # 导入 yaml 模块
import sys  # 导入 sys 模块
import pandas as pd  # 导入 pandas 用于数据处理和CSV输出
import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图
import re
import time # 新增：导入time模块



print("\n--- Attempting to import hydro_model_cpp ---")  # 尝试导入hydro_model_cpp
try:  # 尝试
    import hydro_model_cpp  # 导入hydro_model_cpp
except ImportError as e:  # 捕获导入错误
    print(f"ERROR: Failed to import 'hydro_model_cpp'.")  # 打印错误信息
    print(f"ImportError message: {e}")  # 打印导入错误消息

    print("\n--- Searching for hydro_model_cpp.pyd in common build locations ---")  # 搜索hydro_model_cpp.pyd
    possible_locations = [  # 可能的位置列表
        os.path.join(os.getcwd(), '_skbuild'),  # 项目根目录下的 _skbuild
    ]  # 结束列表
    found_pyd_paths = []  # 初始化找到的pyd路径列表
    for loc_dir_name in possible_locations:  # 遍历可能的位置
        if os.path.exists(loc_dir_name):  # 如果路径存在
            for root, dirs, files in os.walk(loc_dir_name):  # 遍历目录
                for file in files:  # 遍历文件
                    if file.lower() == "hydro_model_cpp.pyd" or file.lower().startswith(
                            "hydro_model_cpp.") and file.lower().endswith((".pyd", ".so")):  # 如果文件名匹配
                        found_pyd_paths.append(os.path.join(root, file))  # 添加到找到的路径列表
    if found_pyd_paths:  # 如果找到pyd文件
        print("Found potential .pyd files at:")  # 打印找到的潜在pyd文件位置
        for p_path in found_pyd_paths:  # 遍历找到的路径
            print(f"  - {p_path}")  # 打印路径
    else:  # 如果未找到
        print("Could not automatically find 'hydro_model_cpp.pyd' in common _skbuild locations.")  # 打印未找到信息
print("-" * 30)  # 打印分隔线

try:  # 尝试
    import hydro_model_cpp  # 导入编译好的C++模块
except ImportError:  # 捕获导入错误
    print("错误: 未找到 C++ 核心模块 'hydro_model_cpp'。请确保已编译并放置在正确路径。")  # 打印错误信息
    sys.exit(1)  # 退出程序
try:  # 尝试
    import meshio  # 导入 meshio 用于VTK输出
except ImportError:  # 捕获导入错误
    print("警告: 未找到 'meshio' 库。VTK输出将不可用。请尝试 'pip install meshio'。")  # 打印警告信息
    meshio = None  # 设置为None，以便后续检查

NUMERICAL_EPSILON = 1e-9  # 定义一个数值比较用的小量


def load_config(config_filepath='config.yaml'):  # 加载配置文件函数
    """加载并返回 YAML 配置文件内容。"""
    try:  # 尝试
        with open(config_filepath, 'r', encoding='utf-8') as f:  # 打开文件
            config_data = yaml.safe_load(f)  # 加载yaml配置
        print(f"配置已从 {config_filepath} 加载。")  # 打印加载信息
        return config_data  # 返回配置数据
    except FileNotFoundError:  # 捕获文件未找到异常
        print(f"错误: 配置文件 '{config_filepath}' 未找到。")  # 打印错误信息
        sys.exit(1)  # 退出程序
    except yaml.YAMLError as e:  # 捕获YAML解析错误
        print(f"错误: 解析配置文件 '{config_filepath}' 失败: {e}")  # 打印错误信息
        sys.exit(1)  # 退出程序
    except Exception as e:  # 捕获其他异常
        print(f"加载配置文件时发生未知错误: {e}")  # 打印错误信息
        sys.exit(1)  # 退出程序


def get_parameters_from_config(config_data):  # 从配置数据获取参数函数
    """从加载的配置字典中提取并返回结构化的参数。"""
    params = {}  # 初始化参数字典
    # 文件路径
    fp_conf = config_data.get('file_paths', {})  # 获取文件路径配置
    params['node_file'] = fp_conf.get('node_file')  # 获取节点文件路径
    params['cell_file'] = fp_conf.get('cell_file')  # 获取单元文件路径
    params['edge_file'] = fp_conf.get('edge_file')  # 获取边文件路径
    params['output_directory'] = fp_conf.get('output_directory', 'output')  # 获取输出目录，默认为'output'

    # 模拟控制
    sc_conf = config_data.get('simulation_control', {})  # 获取模拟控制配置
    params['total_time'] = float(sc_conf.get('total_time', 10.0))  # 获取总模拟时长，转为浮点数
    params['output_dt'] = float(sc_conf.get('output_dt', 1.0))  # 获取输出时间间隔，转为浮点数
    params['cfl_number'] = float(sc_conf.get('cfl_number', 0.5))  # 获取CFL数，转为浮点数
    params['max_dt'] = float(sc_conf.get('max_dt', 0.1))  # 获取最大时间步长，转为浮点数

    # 物理参数
    pp_conf = config_data.get('physical_parameters', {})  # 获取物理参数配置
    params['gravity'] = float(pp_conf.get('gravity', 9.81))  # 获取重力加速度，转为浮点数
    params['min_depth'] = float(pp_conf.get('min_depth', 1e-6))  # 获取最小水深，转为浮点数

    # 数值方案
    ns_conf = config_data.get('numerical_schemes', {})  # 获取数值方案配置
    recon_str = ns_conf.get('reconstruction_scheme', 'FIRST_ORDER').upper()  # 获取重构方案字符串
    params['recon_scheme_cpp'] = getattr(hydro_model_cpp.ReconstructionScheme_cpp, recon_str,  # 获取C++重构方案枚举值
                                         hydro_model_cpp.ReconstructionScheme_cpp.FIRST_ORDER)  # 默认一阶
    riemann_str = ns_conf.get('riemann_solver', 'HLLC').upper()  # 获取黎曼求解器字符串
    params['riemann_solver_cpp'] = getattr(hydro_model_cpp.RiemannSolverType_cpp, riemann_str,  # 获取C++黎曼求解器枚举值
                                           hydro_model_cpp.RiemannSolverType_cpp.HLLC)  # 默认HLLC
    time_str = ns_conf.get('time_scheme', 'RK2_SSP').upper()  # 获取时间积分方案字符串
    params['time_scheme_cpp'] = getattr(hydro_model_cpp.TimeScheme_cpp, time_str,  # 获取C++时间积分方案枚举值
                                        hydro_model_cpp.TimeScheme_cpp.RK2_SSP)  # 默认RK2_SSP

    # 曼宁系数相关
    mp_conf = config_data.get('model_parameters', {})  # 获取模型参数配置
    params['manning_file'] = mp_conf.get('manning_file')  # 获取曼宁文件路径
    params['default_manning'] = float(mp_conf.get('manning_n_default', 0.025))  # 获取默认曼宁系数，转为浮点数

    # 初始条件
    ic_conf_from_yaml = config_data.get('initial_conditions', {})  # 从YAML中获取初始条件配置字典
    params['initial_conditions'] = ic_conf_from_yaml  # 新增: 将整个initial_conditions子字典存入params

    # 为了兼容旧的直接从params获取初始条件参数的代码，可以保留下面这些，
    # 但推荐后续都从 params['initial_conditions'] 中获取
    params['initial_condition_type'] = ic_conf_from_yaml.get('type', 'uniform_elevation')  # 获取初始条件类型
    params['initial_water_surface_elevation'] = float(
        ic_conf_from_yaml.get('water_surface_elevation', 0.0))  # 获取初始水位，转为浮点数
    params['initial_water_depth'] = float(ic_conf_from_yaml.get('water_depth', 0.1))  # 获取初始水深，转为浮点数
    params['initial_hu'] = float(ic_conf_from_yaml.get('hu', 0.0))  # 获取初始hu，转为浮点数
    params['initial_hv'] = float(ic_conf_from_yaml.get('hv', 0.0))  # 获取初始hv，转为浮点数

    if params['initial_condition_type'] == 'dam_break_custom':  # 如果是自定义溃坝
        params['dam_position_x'] = float(ic_conf_from_yaml.get('dam_position_x', 10.0))  # 获取坝位置x坐标
        # 注意：这里的 water_depth_left/right 是旧的参数名，新配置里已经没有了
        # 它们会被 prepare_initial_conditions 中更详细的 upstream/downstream 设置覆盖
        params['water_depth_left'] = float(ic_conf_from_yaml.get('water_depth_left', 1.0))  # 获取左侧水深 (兼容旧配置)
        params['water_depth_right'] = float(ic_conf_from_yaml.get('water_depth_right', 0.0))  # 获取右侧水深 (兼容旧配置)
    elif params['initial_condition_type'] == '2d_partial_dam_break':
        params['dam_y_start'] = float(ic_conf_from_yaml.get('dam_y_start', 0.0))  # 大坝区域起始Y
        params['dam_y_end'] = float(ic_conf_from_yaml.get('dam_y_end', 0.0))  # 大坝区域结束Y
        params['breach_x_start'] = float(ic_conf_from_yaml.get('breach_x_start', 0.0))  # 溃口起始X
        params['breach_x_end'] = float(ic_conf_from_yaml.get('breach_x_end', 0.0))  # 溃口结束X
        params['water_surface_elevation_upstream'] = float(
            ic_conf_from_yaml.get('water_surface_elevation_upstream', 0.0))  # 上游水位
        params['water_surface_elevation_downstream'] = float(
            ic_conf_from_yaml.get('water_surface_elevation_downstream', 0.0))  # 下游水位

    # 边界条件
    params['boundary_definitions_py'] = config_data.get('boundary_conditions', {}).get('definitions',
                                                                                       {})  # 获取Python边界定义
    params['boundary_timeseries_file'] = fp_conf.get('boundary_timeseries_file')  # 从配置中获取统一的边界时间序列文件路径

    # --- (新增) 读取剖面线定义 ---
    raw_profile_lines = config_data.get('profile_output_lines', [])  # 从config_data获取剖面线定义，默认为空列表
    params['profile_output_lines'] = []  # 初始化参数字典中的剖面线列表
    if isinstance(raw_profile_lines, list):  # 确保获取到的是列表
        for line_def in raw_profile_lines:  # 遍历原始剖面线定义
            if isinstance(line_def, dict) and \
                    'name' in line_def and \
                    'start_xy' in line_def and isinstance(line_def['start_xy'], list) and len(
                line_def['start_xy']) == 2 and \
                    'end_xy' in line_def and isinstance(line_def['end_xy'], list) and len(
                line_def['end_xy']) == 2:  # 检查必要字段和类型

                try:  # 尝试转换
                    start_xy = [float(line_def['start_xy'][0]), float(line_def['start_xy'][1])]  # 转换起点坐标
                    end_xy = [float(line_def['end_xy'][0]), float(line_def['end_xy'][1])]  # 转换终点坐标
                except ValueError:  # 捕获值错误
                    print(f"警告: 剖面线 '{line_def.get('name', '未命名')}' 的坐标无法转换为浮点数，已跳过。")  # 打印警告
                    continue  # 继续下一条剖面线

                buffer_width = float(line_def.get('buffer_width', 0.1))  # 获取缓冲宽度，默认为0.1
                is_enabled = line_def.get('enabled', True)  # 获取启用状态，默认为True

                sample_points_x_from_config = line_def.get('sample_points_x')  # 从配置中获取 sample_points_x
                sample_interval_from_config = line_def.get('sample_interval')  # 从配置中获取 sample_interval

                profile_definition_dict = {  # 创建剖面线定义字典
                    'name': str(line_def['name']),  # 剖面线名称
                    'start_xy': start_xy,  # 起点坐标
                    'end_xy': end_xy,  # 终点坐标
                    'buffer_width': buffer_width  # 缓冲宽度
                }

                if sample_points_x_from_config is not None:  # 如果配置了 sample_points_x
                    if isinstance(sample_points_x_from_config, list) and \
                            all(isinstance(pt, (int, float)) for pt in sample_points_x_from_config):  # 检查类型
                        profile_definition_dict['sample_points_x'] = [float(pt) for pt in
                                                                      sample_points_x_from_config]  # 添加到字典
                    else:  # 类型不正确
                        print(
                            f"警告: 剖面线 '{line_def['name']}' 的 'sample_points_x' 配置无效 (应为数值列表)，已忽略。")  # 打印警告

                if sample_interval_from_config is not None:  # 如果配置了 sample_interval
                    try:  # 尝试转换
                        profile_definition_dict['sample_interval'] = float(sample_interval_from_config)  # 添加到字典
                    except ValueError:  # 转换失败
                        print(
                            f"警告: 剖面线 '{line_def['name']}' 的 'sample_interval' 配置无效 (应为数值)，已忽略。")  # 打印警告

                if is_enabled:  # 如果启用
                    params['profile_output_lines'].append(profile_definition_dict)  # 添加完整的剖面线定义字典
                else:  # 如果未启用
                    print(f"  信息: 剖面线 '{line_def['name']}' 已禁用（在配置中设置 enabled: false），跳过。")  # 打印信息
            else:  # 如果定义无效
                print(f"警告: 无效的剖面线定义格式，已跳过: {line_def}")  # 打印警告
    elif raw_profile_lines is not None:  # 如果配置了但不是列表
        print(f"警告: 'profile_output_lines' 配置项不是一个列表，已忽略。实际类型: {type(raw_profile_lines)}")  # 打印警告

    return params  # 返回参数字典


def load_manning_values_from_file(manning_filepath, num_cells_expected, default_manning_val):  # 从文件加载曼宁值函数
    """从文件加载曼宁值。如果失败，则返回用默认值填充的列表。"""
    if manning_filepath and os.path.exists(manning_filepath):  # 如果文件路径有效且存在
        try:  # 尝试
            manning_values = np.loadtxt(manning_filepath, dtype=float)  # 加载曼宁值
            if manning_values.ndim == 0:  # 如果只有一个值
                manning_values = np.array([manning_values.item()])  # 转为数组
            if len(manning_values) == 1 and num_cells_expected > 1:  # 如果只有一个值但期望多个
                print(f"  曼宁文件 {manning_filepath} 只包含一个值，将用于所有 {num_cells_expected} 个单元。")  # 打印信息
                return np.full(num_cells_expected, manning_values[0], dtype=float).tolist()  # 返回填充数组
            elif len(manning_values) == num_cells_expected:  # 如果数量匹配
                return manning_values.tolist()  # 返回列表
            else:  # 数量不匹配
                print(
                    f"  警告: 曼宁文件 {manning_filepath} 中的值数量 ({len(manning_values)}) 与单元数 ({num_cells_expected}) 不符。将使用默认值。")  # 打印警告
        except Exception as e:  # 捕获异常
            print(f"  读取曼宁文件 {manning_filepath} 出错: {e}。将使用默认值。")  # 打印错误信息
    else:  # 文件不存在
        print(
            f"  曼宁文件 '{manning_filepath}' 未找到或未指定。将为所有单元使用默认曼宁系数 {default_manning_val}。")  # 打印信息
    return np.full(num_cells_expected, default_manning_val, dtype=float).tolist()  # 返回填充数组


def prepare_initial_conditions(params, num_cells_cpp, mesh_cpp_ptr_for_ic):
    ic_conf = params.get('initial_conditions', {})
    print(f"DEBUG_PREPARE_IC: ic_conf loaded = {ic_conf}")
    print(
        f"DEBUG_PREPARE_IC: initial_condition_type from ic_conf.get = {ic_conf.get('type', 'DEFAULT_TYPE_NOT_FOUND_IN_IC_CONF')}")

    h_initial = np.zeros(num_cells_cpp, dtype=float)
    default_hu = params.get('initial_hu', 0.0)
    default_hv = params.get('initial_hv', 0.0)
    hu_initial_val = float(ic_conf.get('hu', default_hu))
    hv_initial_val = float(ic_conf.get('hv', default_hv))
    initial_condition_type = ic_conf.get('type', 'uniform_elevation')

    if initial_condition_type == 'dam_break_custom':
        dam_pos_x = float(ic_conf.get('dam_position_x', 0.0))

        # --- 处理上游 ---
        upstream_type = ic_conf.get('upstream_setting_type', 'elevation').lower()
        upstream_value = float(ic_conf.get('upstream_setting_value', 0.0))

        # 为上游单元计算实际水深时可能用到的变量
        upstream_target_eta_for_calc = 0.0  # 如果是基于elevation
        upstream_direct_depth_for_calc = 0.0  # 如果是直接基于depth
        apply_direct_depth_upstream = False

        if upstream_type == 'elevation':
            upstream_target_eta_for_calc = upstream_value
            print(f"  上游设置 (高程): 目标水面高程 = {upstream_target_eta_for_calc:.3f} m")
        elif upstream_type == 'depth':
            # 如果用户希望用"depth"类型来定义一个平水面，他们仍需提供参考底高程
            if 'upstream_reference_bed_elevation' in ic_conf:
                ref_bed_elev = float(ic_conf.get('upstream_reference_bed_elevation', 0.0))
                upstream_target_eta_for_calc = ref_bed_elev + upstream_value  # 水深值 + 参考底高程 = 目标水位
                print(
                    f"  上游设置 (水深带参考点): 参考点底高程={ref_bed_elev:.3f}m 处水深={upstream_value:.3f}m => 计算目标水面高程={upstream_target_eta_for_calc:.3f}m")
            else:
                # 如果没有参考底高程，则认为 "depth" 是直接指定每个上游单元的水深
                upstream_direct_depth_for_calc = upstream_value
                apply_direct_depth_upstream = True
                print(f"  上游设置 (直接水深): 所有上游单元水深 = {upstream_direct_depth_for_calc:.3f} m")
        else:
            print(f"警告: 未知的上游设置类型 '{upstream_type}'。将使用默认水位0。")
            upstream_target_eta_for_calc = 0.0  # 默认水位为0

        # --- 处理下游 (保持之前的逻辑，下游 "depth" 通常是直接水深) ---
        downstream_type = ic_conf.get('downstream_setting_type', 'depth').lower()
        downstream_value = float(ic_conf.get('downstream_setting_value', 0.0))

        downstream_direct_depth_for_calc = 0.0
        downstream_target_eta_for_calc = 0.0
        apply_direct_depth_downstream = False

        if downstream_type == 'elevation':
            downstream_target_eta_for_calc = downstream_value
            print(f"  下游设置 (高程): 目标水面高程 = {downstream_target_eta_for_calc:.3f} m")
        elif downstream_type == 'depth':
            if 'downstream_reference_bed_elevation' in ic_conf:  # 如果也为下游提供了参考点
                ref_bed_elev_down = float(ic_conf.get('downstream_reference_bed_elevation', 0.0))
                downstream_target_eta_for_calc = ref_bed_elev_down + downstream_value
                print(
                    f"  下游设置 (水深带参考点): 参考点底高程={ref_bed_elev_down:.3f}m 处水深={downstream_value:.3f}m => 计算目标水面高程={downstream_target_eta_for_calc:.3f}m")
            else:  # 默认下游 "depth" 是直接水深
                downstream_direct_depth_for_calc = downstream_value
                apply_direct_depth_downstream = True
                print(f"  下游设置 (直接水深): 水深值 = {downstream_direct_depth_for_calc:.3f} m")
        else:
            print(f"警告: 未知的下游设置类型 '{downstream_type}'。将使用默认直接水深0。")
            downstream_direct_depth_for_calc = 0.0
            apply_direct_depth_downstream = True

        print(f"  自定义溃坝: 坝位置 x={dam_pos_x:.3f}")

        for i in range(num_cells_cpp):
            cell = mesh_cpp_ptr_for_ic.get_cell(i)
            cell_z_bed = cell.z_bed_centroid

            if cell.centroid[0] < dam_pos_x:  # 上游单元
                if apply_direct_depth_upstream:
                    h_initial[i] = max(0.0, upstream_direct_depth_for_calc)
                else:  # 基于计算的eta
                    h_initial[i] = max(0.0, upstream_target_eta_for_calc - cell_z_bed)
            else:  # 下游单元
                if apply_direct_depth_downstream:
                    h_initial[i] = max(0.0, downstream_direct_depth_for_calc)
                else:  # 基于计算的eta
                    h_initial[i] = max(0.0, downstream_target_eta_for_calc - cell_z_bed)

    elif initial_condition_type == 'uniform_elevation':
        uniform_eta = float(ic_conf.get('setting_value', ic_conf.get('water_surface_elevation', 0.0)))  # 尝试新旧参数名
        print(f"  统一水位设置: 水面高程 = {uniform_eta:.3f} m")
        for i in range(num_cells_cpp):
            cell = mesh_cpp_ptr_for_ic.get_cell(i)
            h_initial[i] = max(0.0, uniform_eta - cell.z_bed_centroid)

    elif initial_condition_type == 'uniform_depth':
        uniform_depth_val = float(ic_conf.get('setting_value', ic_conf.get('water_depth', 0.1)))  # 尝试新旧参数名
        h_initial.fill(max(0.0, uniform_depth_val))
        print(f"  统一水深设置: 水深 = {uniform_depth_val:.3f} m")

    elif initial_condition_type == 'custom_L_shaped_dam_break': # 新增对此特定类型的处理 # 中文注释：L型弯曲河道溃坝试验的初始条件
        dam_pos_x = float(ic_conf.get('dam_position_x', 1.0)) # 获取坝体X坐标 (根据你的.poly文件调整) # 中文注释：获取坝体X坐标
        reservoir_depth = float(ic_conf.get('reservoir_water_depth', 0.2)) # 获取上游水库水深 # 中文注释：获取上游水库水深
        downstream_depth = float(ic_conf.get('downstream_water_depth', 0.0)) # 获取下游河道水深 # 中文注释：获取下游河道水深

        print(f"  L型弯曲河道溃坝初始条件: 坝位置 x={dam_pos_x:.3f}, 上游水深={reservoir_depth:.3f}, 下游水深={downstream_depth:.3f}") # 中文注释：打印初始条件信息

        for i in range(num_cells_cpp): # 遍历所有单元 # 中文注释：遍历所有单元
            cell = mesh_cpp_ptr_for_ic.get_cell(i) # 获取当前单元 # 中文注释：获取当前单元
            if cell.centroid[0] < dam_pos_x:  # 如果单元在坝体上游（水库区域） # 中文注释：判断是否为上游水库区域
                h_initial[i] = max(0.0, reservoir_depth) # 设置为水库水深 # 中文注释：设置水库水深
            else:  # 下游河道区域 # 中文注释：否则为下游河道区域
                h_initial[i] = max(0.0, downstream_depth) # 设置为下游水深 # 中文注释：设置下游水深

    # ... (2d_partial_dam_break logic)
    elif initial_condition_type == '2d_partial_dam_break':
        dam_y_start = float(ic_conf.get('dam_y_start'))
        dam_y_end = float(ic_conf.get('dam_y_end'))
        breach_x_start = float(ic_conf.get('breach_x_start'))
        breach_x_end = float(ic_conf.get('breach_x_end'))

        # 假设此类型总是基于水位，使用 upstream_setting_value 和 downstream_setting_value
        # 并且假设配置文件中 upstream_setting_type 和 downstream_setting_type 都是 'elevation'
        eta_upstream_val = float(ic_conf.get('upstream_setting_value', 0.0))
        eta_downstream_val = float(ic_conf.get('downstream_setting_value', 0.0))
        print(
            f"  设置二维局部溃坝初始条件: dam_y=[{dam_y_start}-{dam_y_end}], breach_x=[{breach_x_start}-{breach_x_end}], eta_up={eta_upstream_val}, eta_down={eta_downstream_val}")

        for i in range(num_cells_cpp):
            cell = mesh_cpp_ptr_for_ic.get_cell(i)
            cx, cy = cell.centroid[0], cell.centroid[1]
            current_target_eta = 0.0
            if cy >= dam_y_end:
                current_target_eta = eta_upstream_val
            elif cy < dam_y_start:
                current_target_eta = eta_downstream_val
            else:
                if breach_x_start <= cx < breach_x_end:
                    current_target_eta = eta_downstream_val
                else:
                    current_target_eta = eta_upstream_val
            h_initial[i] = max(0.0, current_target_eta - cell.z_bed_centroid)
    elif initial_condition_type == 'custom_lab_symmetric_dam_break':  # 新增对实验室对称溃坝的处理 # 中文注释：判断初始条件类型是否为实验室对称溃坝
        dam_pos_x = float(ic_conf.get('dam_position_x', 1.0))  # 获取坝体x坐标 # 中文注释：获取配置文件中坝体的x坐标
        reservoir_depth = float(ic_conf.get('reservoir_water_depth', 0.6))  # 获取水库水深 # 中文注释：获取配置文件中水库的初始水深
        downstream_depth = float(ic_conf.get('downstream_water_depth', 0.0))  # 获取下游水深 # 中文注释：获取配置文件中下游的初始水深

        print(
            f"  实验室对称溃坝初始条件: 坝位置 x={dam_pos_x:.3f}, 上游水深={reservoir_depth:.3f}, 下游水深={downstream_depth:.3f}")  # 中文注释：打印初始条件信息

        for i in range(num_cells_cpp):  # 遍历所有单元 # 中文注释：遍历所有单元以设置初始水深
            cell = mesh_cpp_ptr_for_ic.get_cell(i)  # 获取当前单元对象 # 中文注释：获取当前单元对象
            # 水深是直接给定的，不依赖于底高程来计算初始水面（因为是平底假设下的水深）
            if cell.centroid[0] < dam_pos_x:  # 如果单元在坝体上游 # 中文注释：判断单元是否位于大坝上游
                h_initial[i] = max(0.0, reservoir_depth)  # 设置为水库水深 # 中文注释：将水深设置为水库初始水深
            else:  # 如果单元在坝体下游 # 中文注释：如果单元位于大坝下游
                h_initial[i] = max(0.0, downstream_depth)  # 设置为下游水深 # 中文注释：将水深设置为下游初始水深
    elif initial_condition_type == "custom_surface_function":  # 新增自定义水面函数类型
        print("  设置自定义函数生成的初始水面条件...")  # 打印信息

        surface_conf = ic_conf.get('surface_params', {})  # 获取surface_params子字典

        base_eta = float(surface_conf.get('base_elevation', 0.0))  # 获取基础水面高程
        slope_x = float(surface_conf.get('slope_x', 0.0))  # 获取x方向坡度
        slope_y = float(surface_conf.get('slope_y', 0.0))  # 获取y方向坡度

        print(f"    自定义水面参数: base_eta={base_eta:.3f}, slope_x={slope_x:.4f}, slope_y={slope_y:.4f}")  # 打印参数信息

        h_initial_np = np.zeros(num_cells_cpp, dtype=float)  # 初始化水深数组

        initial_hu_val = float(ic_conf.get('hu', 0.0))  # 获取初始hu值
        initial_hv_val = float(ic_conf.get('hv', 0.0))  # 获取初始hv值
        hu_initial_np = np.full(num_cells_cpp, initial_hu_val, dtype=float)  # 初始化hu数组
        hv_initial_np = np.full(num_cells_cpp, initial_hv_val, dtype=float)  # 初始化hv数组

        # --- 新增：创建一个数组来存储所有单元的底高程，用于后续统计 ---
        z_bed_all_cells_np = np.zeros(num_cells_cpp, dtype=float)  # 初始化所有单元底高程数组

        for i in range(num_cells_cpp):  # 遍历所有单元
            cell = mesh_cpp_ptr_for_ic.get_cell(i)  # 获取当前单元对象
            x_cell = cell.centroid[0]  # 获取单元形心x坐标
            y_cell = cell.centroid[1]  # 获取单元形心y坐标
            z_bed_cell = cell.z_bed_centroid  # 获取单元形心底高程
            z_bed_all_cells_np[i] = z_bed_cell  # --- 新增：存储底高程 ---

            eta_cell = base_eta + slope_x * x_cell + slope_y * y_cell  # 计算单元中心水面高程

            current_h = eta_cell - z_bed_cell  # 计算水深

            h_initial_np[i] = max(0.0, current_h)  # 确保水深非负

        # 调试打印
        min_depth_threshold_for_wet_count = params.get('min_depth', 1e-6)  # 获取用于统计湿单元的最小水深阈值
        wet_mask = h_initial_np > min_depth_threshold_for_wet_count  # 创建湿单元掩码
        num_wet_cells = np.count_nonzero(wet_mask)  # 计算湿单元数量
        print(f"    计算得到 {num_wet_cells} 个初始湿润单元 (h > {min_depth_threshold_for_wet_count:.1e})。")  # 打印湿单元数量

        if num_wet_cells > 0:  # 如果存在湿单元
            # 提取湿润单元的水深和对应的底高程
            h_wet_cells = h_initial_np[wet_mask]  # 获取湿单元的水深
            z_bed_wet_cells = z_bed_all_cells_np[wet_mask]  # 获取湿单元的底高程

            print(
                f"    初始水深范围 (湿润单元): min_h_wet={np.min(h_wet_cells):.4f}, max_h_wet={np.max(h_wet_cells):.4f}")  # 打印湿单元水深范围

            # 计算湿润单元的水面高程 eta = z_bed + h
            eta_wet_cells = z_bed_wet_cells + h_wet_cells  # 计算湿单元水面高程
            print(
                f"    对应水面高程范围 (湿润单元，基于形心): min_eta_wet≈{np.min(eta_wet_cells):.4f}, max_eta_wet≈{np.max(eta_wet_cells):.4f}")  # 打印湿单元水面高程范围
        else:  # 如果不存在湿单元
            print("    没有计算得到初始湿润单元。")  # 打印信息

        return np.column_stack((h_initial_np, hu_initial_np, hv_initial_np))  # 返回组合后的初始条件数组
    else:
        print(f"警告: 未知的初始条件类型 '{initial_condition_type}'。使用默认零水深。")

    hu_initial_np = np.full(num_cells_cpp, hu_initial_val, dtype=float)
    hv_initial_np = np.full(num_cells_cpp, hv_initial_val, dtype=float)
    return np.column_stack((h_initial, hu_initial_np, hv_initial_np))


def prepare_boundary_conditions_for_cpp(params):  # 准备C++边界条件函数
    """转换Python边界配置为C++期望的格式。"""
    bc_defs_cpp = {}  # 初始化C++边界定义字典
    # 1. 解析 boundary_definitions_py (这部分不变，因为它是 边界类型标记 -> C++边界类型枚举)
    for marker_str, py_def in params.get('boundary_definitions_py', {}).items():  # 遍历Python边界定义
        try:  # 尝试
            marker_int = int(marker_str)  # 转换标记为整数
            cpp_def = hydro_model_cpp.BoundaryDefinition_cpp()  # 创建C++边界定义对象
            type_str = py_def.get('type', 'WALL').upper()  # 获取类型字符串
            cpp_def.type = getattr(hydro_model_cpp.BoundaryType_cpp, type_str,  # 设置C++边界类型
                                   hydro_model_cpp.BoundaryType_cpp.WALL)
            bc_defs_cpp[marker_int] = cpp_def  # 添加到字典，键是边界类型标记
        except ValueError:  # 捕获值错误
            print(f"警告: 边界定义标记 '{marker_str}' 不是有效整数，已跳过。")  # 打印警告
        except AttributeError:  # 捕获属性错误
            type_str_for_error = py_def.get('type', 'UNKNOWN').upper()  # 获取用于错误信息的类型字符串
            print(f"警告: 边界类型 '{type_str_for_error}' (标记 {marker_str}) 无效，已设为WALL。")  # 打印警告
            cpp_def_fallback = hydro_model_cpp.BoundaryDefinition_cpp()  # 创建备用C++边界定义对象
            cpp_def_fallback.type = hydro_model_cpp.BoundaryType_cpp.WALL  # 设为墙体
            if marker_str.isdigit():  # 确保marker_str可以转为int
                bc_defs_cpp[int(marker_str)] = cpp_def_fallback  # 添加到字典
            else:  # marker_str无法转为int
                print(f"错误: 边界标记 '{marker_str}' 无法转换为整数，已忽略此回退定义。")  # 打印错误

    wl_ts_data_cpp = {}  # 初始化水位时间序列数据字典 (键将是 线段ID)
    discharge_ts_data_cpp = {}  # 初始化流量时间序列数据字典 (键将是 线段ID)

    unified_ts_file_path = params.get('boundary_timeseries_file')  # 获取统一的时间序列文件路径

    if unified_ts_file_path and os.path.exists(unified_ts_file_path):  # 如果文件路径有效且存在
        print(f"  正在从统一边界时间序列文件 '{unified_ts_file_path}' 加载数据 (基于线段ID)...")  # 打印加载信息
        try:  # 尝试读取和解析
            df_ts = pd.read_csv(unified_ts_file_path)  # 读取CSV文件
            if 'time' not in df_ts.columns:  # 检查是否有时间列
                print(f"警告: 统一边界时间序列文件 '{unified_ts_file_path}' 缺少 'time' 列。")  # 打印警告
            else:  # 如果有时间列
                time_col = df_ts['time'].values  # 获取时间列数据

                for col_name in df_ts.columns:  # 遍历所有列名
                    if col_name.lower() == 'time':  # 跳过时间列本身
                        continue  # 继续下一列

                    match = re.fullmatch(r"b(\d+)_(elev|flux)", col_name, re.IGNORECASE)  # 进行正则匹配

                    if match:  # 如果匹配成功
                        segment_id = int(match.group(1))  # <--- 提取的是线段ID
                        data_type_suffix = match.group(2).lower()  # 提取类型后缀 (elev 或 flux)

                        ts_points = []  # 初始化时间序列点列表
                        for t_val, data_val in zip(time_col, df_ts[col_name].values):  # 遍历时间和数据值
                            if pd.notna(t_val) and pd.notna(data_val):  # 如果时间和数据都不是NaN
                                pt = hydro_model_cpp.TimeseriesPoint_cpp()  # 创建C++时间序列点对象
                                pt.time = float(t_val)  # 设置时间
                                pt.value = float(data_val)  # 设置值
                                ts_points.append(pt)  # 添加到列表

                        if ts_points:  # 如果成功提取到时间序列点
                            if data_type_suffix == 'elev':  # 如果数据类型是水位
                                wl_ts_data_cpp[segment_id] = ts_points  # 使用线段ID作为键
                                print(f"    已为线段ID {segment_id} 加载水位时间序列 (elev)。")  # 打印加载信息
                            elif data_type_suffix == 'flux':  # 如果数据类型是流量
                                discharge_ts_data_cpp[segment_id] = ts_points  # 使用线段ID作为键
                                print(f"    已为线段ID {segment_id} 加载流量时间序列 (flux)。")  # 打印加载信息
        # ... (异常捕获不变) ...
        except ImportError:  # 捕获Pandas导入错误
            print(f"警告: pandas 未安装，无法解析统一边界时间序列CSV文件 '{unified_ts_file_path}'。")  # 打印警告
        except Exception as e:  # 捕获其他读取或解析错误
            print(f"处理统一边界时间序列文件 '{unified_ts_file_path}' 时出错: {e}")  # 打印错误信息
    elif unified_ts_file_path:  # 如果文件路径已配置但文件不存在
        print(f"警告: 统一边界时间序列文件 '{unified_ts_file_path}' 未找到。")  # 打印警告

    return bc_defs_cpp, wl_ts_data_cpp, discharge_ts_data_cpp  # 返回准备好的C++边界条件数据


def save_results_to_vtk(vtk_filepath, points_coords, cells_connectivity, cell_data_dict):  # 保存结果到VTK文件函数
    """使用 meshio 将结果保存为 VTK (.vtu) 文件。"""
    if not meshio:  # 如果 meshio 未导入
        print(f"  Meshio 未加载，无法保存VTK文件: {vtk_filepath}")  # 打印信息
        return  # 返回

    formatted_cell_data = {key: [value_array] for key, value_array in cell_data_dict.items()}  # 格式化单元数据

    try:  # 尝试
        meshio.write_points_cells(  # 调用meshio写入文件
            vtk_filepath,  # 文件路径
            points_coords,  # 节点坐标
            cells_connectivity,  # 单元连接关系
            cell_data=formatted_cell_data,  # 单元数据
            file_format="vtu"  # 文件格式
        )  # 结束调用
        print(f"    VTK 文件已保存: {vtk_filepath}")  # 打印保存信息
    except Exception as e:  # 捕获异常
        print(f"    保存VTK文件 {vtk_filepath} 时出错: {e}")  # 打印错误信息


# --- (新增) 获取剖面线上的单元ID和坐标（按剖面线方向排序） ---
def get_profile_cells(mesh_ptr, profile_start_xy, profile_end_xy, buffer_width=0.05, sample_points_x=None,
                      sample_interval=None):  # 获取剖面线上的单元
    """
    获取剖面线上的单元信息。
    如果提供了 sample_points_x，则尝试找到离这些x坐标最近的单元（在剖面线和缓冲区内）。
    如果提供了 sample_interval，则沿剖面线按间隔采样。
    否则，返回缓冲区内所有单元。
    返回一个元组列表: [(cell_id, distance_along_profile, cell_centroid_x, cell_centroid_y), ...]
    按 distance_along_profile 排序。
    """
    profile_cells_data = []  # 初始化剖面线单元数据列表
    p1 = np.array(profile_start_xy)  # 剖面线起点
    p2 = np.array(profile_end_xy)  # 剖面线终点
    profile_vec = p2 - p1  # 剖面线向量
    profile_length = np.linalg.norm(profile_vec)  # 剖面线长度
    if profile_length < NUMERICAL_EPSILON: return []  # 如果剖面线长度过小，返回空列表

    profile_unit_vec = profile_vec / profile_length  # 剖面线单位向量
    is_horizontal_profile = abs(profile_unit_vec[1]) < NUMERICAL_EPSILON  # 判断是否为近似水平剖面线 (y基本不变)

    num_cells = mesh_ptr.get_num_cells()  # 获取单元总数

    if sample_points_x and is_horizontal_profile:  # 如果提供了X坐标采样点且是水平剖面线
        print(f"  剖面线: 使用指定的 {len(sample_points_x)} 个X坐标进行采样。")  # 打印采样信息
        sampled_cell_ids_on_profile = set()  # 用于记录已采样的单元ID，避免重复

        # 对采样点排序，以确保沿剖面线方向
        sorted_sample_points_x = sorted(list(set(sample_points_x)))  # 去重并排序

        for target_x in sorted_sample_points_x:  # 遍历目标X坐标
            # 找到剖面线上x=target_x的点
            # 对于水平剖面线，profile_start_xy[0] + dist * profile_unit_vec[0] = target_x
            # dist = (target_x - profile_start_xy[0]) / profile_unit_vec[0] (假设 profile_unit_vec[0] != 0)
            if abs(profile_unit_vec[0]) < NUMERICAL_EPSILON: continue  # 垂直剖面线不适用此逻辑

            dist_along = (target_x - p1[0]) / profile_unit_vec[0] if profile_unit_vec[0] != 0 else 0  # 计算沿剖面线距离
            # 确保采样点在剖面线长度范围内
            if not (0 - NUMERICAL_EPSILON <= dist_along <= profile_length + NUMERICAL_EPSILON):
                # print(f"    跳过采样点 x={target_x:.2f}，因为它不在剖面线段内 (计算距离: {dist_along:.2f})。") # 调试信息
                continue

            target_point_on_line = p1 + dist_along * profile_unit_vec  # 计算剖面线上的目标点

            best_cell_id = -1  # 初始化最佳单元ID
            min_dist_sq_to_target = float('inf')  # 初始化到目标点的最小平方距离

            for i in range(num_cells):  # 遍历所有单元
                cell = mesh_ptr.get_cell(i)  # 获取当前单元对象
                cell_centroid = np.array(cell.centroid)  # 获取单元形心

                # 1. 检查单元是否在剖面线缓冲区内 (可选，但可以加速)
                p1_to_centroid_vec = cell_centroid - p1  # 计算P1到形心的向量
                distance_along_profile_cell = np.dot(p1_to_centroid_vec, profile_unit_vec)  # 计算投影长度
                projection_on_profile_vec = distance_along_profile_cell * profile_unit_vec  # 计算在剖面线上的投影向量
                perpendicular_vec = p1_to_centroid_vec - projection_on_profile_vec  # 计算垂直向量
                distance_to_profile_line = np.linalg.norm(perpendicular_vec)  # 计算到剖面线的垂直距离

                if not (
                        0 <= distance_along_profile_cell <= profile_length and distance_to_profile_line < buffer_width):  # 如果不在缓冲区内
                    continue  # 跳过此单元

                # 2. 计算单元形心到当前采样点的距离
                dist_sq = np.sum((cell_centroid - target_point_on_line) ** 2)  # 计算平方距离
                if dist_sq < min_dist_sq_to_target:  # 如果距离更小
                    min_dist_sq_to_target = dist_sq  # 更新最小平方距离
                    best_cell_id = cell.id  # 更新最佳单元ID

            if best_cell_id != -1 and best_cell_id not in sampled_cell_ids_on_profile:  # 如果找到最佳单元且未被采样
                cell_obj_found = mesh_ptr.get_cell(best_cell_id)  # 获取最佳单元对象
                profile_cells_data.append({  # 添加单元数据到列表
                    "id": best_cell_id,  # 单元ID
                    "dist": dist_along,  # 使用采样点沿剖面线的精确距离
                    "x": cell_obj_found.centroid[0],  # 实际单元形心x
                    "y": cell_obj_found.centroid[1],  # 实际单元形心y
                    "target_x_on_profile": target_x  # 记录这个单元是为哪个目标x点采样的
                })
                sampled_cell_ids_on_profile.add(best_cell_id)  # 记录已采样
            elif best_cell_id != -1 and best_cell_id in sampled_cell_ids_on_profile:  # 如果单元已被采样
                # print(f"    采样点 x={target_x:.2f} 找到的单元 {best_cell_id} 已被之前的采样点输出，跳过。") # 调试信息
                pass

    elif sample_interval and sample_interval > 0:  # 如果提供了采样间隔
        print(f"  剖面线: 按间隔 {sample_interval:.2f}m 进行采样。")  # 打印采样信息
        num_samples = int(profile_length / sample_interval) + 1  # 计算采样点数量
        sampled_cell_ids_on_profile = set()  # 用于记录已采样的单元ID

        for i_sample in range(num_samples):  # 遍历采样点
            dist_along = i_sample * sample_interval  # 计算当前采样点沿剖面线的距离
            if dist_along > profile_length: dist_along = profile_length  # 确保不超过剖面线长度
            target_point_on_line = p1 + dist_along * profile_unit_vec  # 计算剖面线上的目标点

            best_cell_id = -1  # 初始化最佳单元ID
            min_dist_sq_to_target = float('inf')  # 初始化到目标点的最小平方距离
            # ... (寻找最近单元的逻辑同上) ...
            for i in range(num_cells):  # 遍历所有单元
                cell = mesh_ptr.get_cell(i)  # 获取当前单元对象
                cell_centroid = np.array(cell.centroid)  # 获取单元形心
                # 检查是否在缓冲区内 (可选优化)
                p1_to_centroid_vec_c = cell_centroid - p1
                distance_along_profile_cell_c = np.dot(p1_to_centroid_vec_c, profile_unit_vec)
                if not (0 <= distance_along_profile_cell_c <= profile_length):  # 粗略检查
                    # 进一步检查到线段的垂直距离
                    projection_on_profile_vec_c = distance_along_profile_cell_c * profile_unit_vec
                    perpendicular_vec_c = p1_to_centroid_vec_c - projection_on_profile_vec_c
                    distance_to_profile_line_c = np.linalg.norm(perpendicular_vec_c)
                    if distance_to_profile_line_c >= buffer_width:
                        continue
                dist_sq = np.sum((cell_centroid - target_point_on_line) ** 2)  # 计算平方距离
                if dist_sq < min_dist_sq_to_target:  # 如果距离更小
                    min_dist_sq_to_target = dist_sq  # 更新最小平方距离
                    best_cell_id = cell.id  # 更新最佳单元ID

            if best_cell_id != -1 and best_cell_id not in sampled_cell_ids_on_profile:  # 如果找到最佳单元且未被采样
                cell_obj_found = mesh_ptr.get_cell(best_cell_id)  # 获取最佳单元对象
                profile_cells_data.append({  # 添加单元数据到列表
                    "id": best_cell_id,  # 单元ID
                    "dist": dist_along,  # 使用采样点沿剖面线的距离
                    "x": cell_obj_found.centroid[0],  # 单元形心x
                    "y": cell_obj_found.centroid[1]  # 单元形心y
                })
                sampled_cell_ids_on_profile.add(best_cell_id)  # 记录已采样
    else:  # 默认行为：获取缓冲区内所有单元
        print(f"  剖面线: 获取缓冲区内所有单元 (buffer_width={buffer_width:.2f}m)。")  # 打印信息
        for i in range(num_cells):  # 遍历所有单元
            cell = mesh_ptr.get_cell(i)  # 获取当前单元对象
            cell_centroid = np.array(cell.centroid)  # 获取单元形心
            p1_to_centroid_vec = cell_centroid - p1  # 计算P1到形心的向量
            distance_along_profile = np.dot(p1_to_centroid_vec, profile_unit_vec)  # 计算投影长度
            projection_on_profile_vec = distance_along_profile * profile_unit_vec  # 计算在剖面线上的投影向量
            perpendicular_vec = p1_to_centroid_vec - projection_on_profile_vec  # 计算垂直向量
            distance_to_profile_line = np.linalg.norm(perpendicular_vec)  # 计算到剖面线的垂直距离
            if 0 <= distance_along_profile <= profile_length and distance_to_profile_line < buffer_width:  # 如果满足筛选条件
                profile_cells_data.append({  # 添加单元数据到列表
                    "id": cell.id,  # 单元ID
                    "dist": distance_along_profile,  # 沿剖面线距离
                    "x": cell_centroid[0],  # 形心x坐标
                    "y": cell_centroid[1]  # 形心y坐标
                })

    profile_cells_data.sort(key=lambda item: item["dist"])  # 按距离排序
    # 如果是按X坐标采样，可能需要根据 item["target_x_on_profile"] 或 item["dist"] 再次确认顺序
    if sample_points_x and is_horizontal_profile:
        profile_cells_data.sort(key=lambda item: item.get("target_x_on_profile", item["dist"]))

    # 进一步去重逻辑：如果排序后发现相邻两个item的 "id" 相同，只保留一个
    # 这主要针对 sample_points_x 非常密集，导致多个x点落在同一单元的情况
    if (sample_points_x and is_horizontal_profile) or (sample_interval and sample_interval > 0):
        unique_profile_cells_data = []  # 初始化唯一剖面线单元数据列表
        last_added_cell_id = -1  # 初始化上一个添加的单元ID
        for item in profile_cells_data:  # 遍历排序后的数据
            if item["id"] != last_added_cell_id:  # 如果当前单元ID与上一个不同
                unique_profile_cells_data.append(item)  # 添加到唯一列表
                last_added_cell_id = item["id"]  # 更新上一个单元ID
        profile_cells_data = unique_profile_cells_data  # 更新为唯一列表

    return profile_cells_data  # 返回排序后的剖面线单元数据


# --- 主程序 ---
if __name__ == "__main__":  # 主程序入口
    # --- 新增：计时相关的初始化 ---
    overall_start_time_py = time.time()  # 记录整个脚本开始执行的时间
    last_vtk_save_time_py = overall_start_time_py  # 初始化上一次VTK保存的时间点
    # --- 计时初始化结束 ---
    config = load_config()  # 加载配置文件
    params = get_parameters_from_config(config)  # 从配置数据获取参数

    if not params.get('node_file') or not params.get('cell_file'):  # 检查节点和单元文件路径是否存在
        print("错误: 必须在 config.yaml 的 file_paths 中配置 node_file 和 cell_file。")  # 打印错误
        sys.exit(1)  # 退出程序

    model_core = hydro_model_cpp.HydroModelCore_cpp()  # 创建C++模型对象
    print("Python: C++ HydroModelCore_cpp object created.")  # 打印创建信息

    num_cells_for_manning = 0  # 初始化单元数
    try:  # 尝试
        with open(params['cell_file'], 'r') as f_cell:  # 打开单元文件
            header_cell = f_cell.readline().split()  # 读取头部
            num_cells_for_manning = int(header_cell[0])  # 获取单元数
    except Exception as e:  # 捕获异常
        print(f"错误: 无法从 {params['cell_file']} 读取单元数量以加载曼宁值: {e}")  # 打印错误
        sys.exit(1)  # 退出程序

    cell_manning_list = load_manning_values_from_file(  # 加载曼宁值
        params['manning_file'], num_cells_for_manning, params['default_manning']
    )  # 结束加载

    model_core.initialize_model_from_files(  # 调用C++模型初始化方法
        params['node_file'], params['cell_file'],
        params['edge_file'] if params['edge_file'] and os.path.exists(params['edge_file']) else "",  # 边文件路径 (如果存在)
        cell_manning_list,  # 曼宁值列表
        params['gravity'], params['min_depth'], params['cfl_number'],  # 模拟参数
        params['total_time'], params['output_dt'], params['max_dt'],  # 时间参数
        params['recon_scheme_cpp'], params['riemann_solver_cpp'], params['time_scheme_cpp']  # 数值方案
    )  # 结束初始化
    print("Python: C++ model initialized with mesh and parameters.")  # 打印初始化完成信息

    mesh_cpp_ptr = model_core.get_mesh_ptr()  # 获取指向C++ Mesh_cpp对象的指针
    num_cells_cpp_from_core = mesh_cpp_ptr.get_num_cells()  # 从C++核心获取单元数

    U_initial_np = prepare_initial_conditions(params, num_cells_cpp_from_core, mesh_cpp_ptr)  # 准备初始条件

    #打印结束
    model_core.set_initial_conditions_py(U_initial_np)  # 设置初始条件

    # ******** 新增：准备并设置边界条件 ********
    bc_defs_cpp, wl_ts_data_cpp, discharge_ts_data_cpp = prepare_boundary_conditions_for_cpp(params)  # 调用你已有的函数准备边界数据
    if not bc_defs_cpp:  # 检查边界定义是否为空
        print("警告: 未能从配置中解析出任何有效的边界条件定义。")  # 打印警告
    else:  # 如果边界定义不为空
        print(f"Python: 准备了 {len(bc_defs_cpp)} 个边界定义。")  # 打印准备的边界定义数量
        # 假设你的C++核心有一个名为 set_boundary_conditions_py 的方法
        # 你需要确认这个C++方法的具体名称和参数
        # model_core.set_boundary_conditions_py(bc_defs_cpp, wl_ts_data_cpp, discharge_ts_data_cpp) # 旧的调用
        model_core.setup_boundary_conditions_cpp(bc_defs_cpp, wl_ts_data_cpp, discharge_ts_data_cpp)  # 修改: 调用正确的C++方法名
        print("Python: 边界条件已传递给C++核心。")  # 打印边界条件传递信息
    # ******** 边界条件设置结束 ********


    # --- 定义剖面线并获取相关单元 (从配置中读取) ---
    profile_lines_definitions_from_config = params.get('profile_output_lines', [])  # 从参数字典获取剖面线定义

    profile_data_collectors = {}  # 初始化剖面线数据收集器字典
    if profile_lines_definitions_from_config:  # 如果配置中定义了剖面线
        print("\n剖面线输出已配置:")  # 打印信息
        for p_def in profile_lines_definitions_from_config:  # 遍历配置中的剖面线定义
            profile_name = p_def['name']  # 获取剖面线名称
            start_xy = p_def['start_xy']  # 获取起点坐标
            end_xy = p_def['end_xy']  # 获取终点坐标
            buffer_width_prof = p_def['buffer_width']  # 获取缓冲宽度
            sample_points_x_prof = p_def.get('sample_points_x')  # 获取X坐标采样点 (新增)
            sample_interval_prof = p_def.get('sample_interval')  # 获取采样间隔 (新增)

            print(
                f"  - 处理剖面线: '{profile_name}', 起点: {start_xy}, 终点: {end_xy}, 缓冲: {buffer_width_prof}")  # 打印剖面线信息
            if sample_points_x_prof: print(f"    采样X坐标点: {len(sample_points_x_prof)} 个")  # 打印X坐标采样点数量
            if sample_interval_prof: print(f"    采样间隔: {sample_interval_prof}")  # 打印采样间隔
            # ******** 增加调试打印 ********
            if sample_points_x_prof:
                print(f"    DEBUG: sample_points_x_prof for '{profile_name}': {sample_points_x_prof}")  # 调试打印
            if sample_interval_prof:
                print(f"    DEBUG: sample_interval_prof for '{profile_name}': {sample_interval_prof}")  # 调试打印
            # ******** 调试打印结束 ********

            mesh_ptr_for_profile = model_core.get_mesh_ptr()  # 获取网格指针 (C++对象)
            profile_cell_info = get_profile_cells(  # 调用修改后的函数
                mesh_ptr_for_profile, start_xy, end_xy,
                buffer_width_prof,
                sample_points_x=sample_points_x_prof,  # 传递采样点
                sample_interval=sample_interval_prof  # 传递采样间隔
            )
            profile_data_collectors[p_def['name']] = {  # 初始化当前剖面线的数据收集器
                "cell_ids": [info["id"] for info in profile_cell_info], # 存储单元ID
                "cell_distances": [info["dist"] for info in profile_cell_info], # 存储单元距离 (或者用 target_x)
                "cell_x_coords": [info["x"] for info in profile_cell_info], # 存储单元X坐标
                "cell_y_coords": [info["y"] for info in profile_cell_info], # 存储单元Y坐标
                "time_data": [], # 存储时间数据
                "eta_data": [],  # 水位
                "h_data": [],    # 水深
                "u_data": [],    # x方向流速
                "v_data": [],     # y方向流速
                "fr_data": []  # 新增：用于存储弗劳德数数据
            }  # 结束初始化
    else:  # 如果配置中没有定义剖面线
        print("\n未在配置文件中找到 'profile_output_lines' 或其为空，不进行剖面线输出。")  # 打印信息
    # --- 结束剖面线定义和处理 ---

    # --- 准备VTK输出所需的静态网格信息 ---
    points_for_vtk = np.zeros((mesh_cpp_ptr.get_num_nodes(), 3))  # 初始化VTK节点坐标数组
    for i in range(mesh_cpp_ptr.get_num_nodes()):  # 遍历节点
        node = mesh_cpp_ptr.get_node(i)  # 获取节点对象
        points_for_vtk[i, 0] = node.x  # 设置x坐标
        points_for_vtk[i, 1] = node.y  # 设置y坐标
        points_for_vtk[i, 2] = node.z_bed  # 将底高程设为Z坐标，以便在ParaView中查看地形
    cells_connectivity_for_vtk = []  # 初始化VTK单元连接关系列表
    for i in range(num_cells_cpp_from_core):  # 遍历单元
        cell = mesh_cpp_ptr.get_cell(i)  # 获取单元对象
        if len(cell.node_ids) == 3:  # 如果是三角形单元
            cells_connectivity_for_vtk.append(list(cell.node_ids))  # 添加节点ID列表
    cells_for_vtk = [("triangle", np.array(cells_connectivity_for_vtk, dtype=int))]  # 创建meshio单元格式

    vtk_output_dir = params['output_directory']  # 获取VTK输出目录
    os.makedirs(vtk_output_dir, exist_ok=True)  # 创建目录 (如果不存在)
    print(f"VTK files will be saved to: {os.path.abspath(vtk_output_dir)}")  # 打印VTK文件保存路径

    # --- 模拟循环与VTK输出 ---
    print("\nPython: Starting C++ simulation...")  # 打印开始模拟信息
    output_counter = 0  # 初始化输出计数器
    next_output_time = model_core.get_current_time()  # 初始化下一个输出时间
    simulation_active = True  # 初始化模拟活动标志
    while simulation_active:  # 当模拟活动时循环
        current_t_cpp = model_core.get_current_time()  # 获取当前C++时间
        if current_t_cpp >= next_output_time - NUMERICAL_EPSILON or output_counter == 0:  # 第一次也输出
            # --- 新增：计算并打印时间 ---
            current_wall_time_py = time.time()  # 获取当前墙上时间
            time_since_last_vtk_py = current_wall_time_py - last_vtk_save_time_py  # 计算自上次VTK保存以来的用时
            total_elapsed_time_py = current_wall_time_py - overall_start_time_py  # 计算总的脚本运行时间

            # 只有在第一次输出之后才打印单步用时，避免除零或不准确
            if output_counter > 0:  # 如果不是第一次输出
                print(
                    f"  Python: Output at t = {current_t_cpp:.3f} s (C++ step = {model_core.get_step_count()})")  # 打印输出信息
                print(
                    f"    Time for this output interval: {time_since_last_vtk_py:.2f} s (Python wall time)")  # 打印本次输出间隔用时
            else:  # 如果是第一次输出
                print(
                    f"  Python: Initial Output at t = {current_t_cpp:.3f} s (C++ step = {model_core.get_step_count()})")  # 打印初始输出信息
            print(f"    Total elapsed simulation time: {total_elapsed_time_py:.2f} s (Python wall time)")  # 打印总用时
            last_vtk_save_time_py = current_wall_time_py  # 更新上次VTK保存时间点
            # --- 时间打印结束 ---

            U_current_py = model_core.get_U_state_all_py()  # 获取当前守恒量
            eta_current_py = model_core.get_eta_previous_py()  # 获取当前水位
            h_current = U_current_py[:, 0]  # 获取水深
            hu_current = U_current_py[:, 1]  # 获取hu
            hv_current = U_current_py[:, 2]  # 获取hv

            u_current = np.divide(hu_current, h_current, out=np.zeros_like(hu_current),
                                  where=h_current > params['min_depth'] / 10.0)  # 计算u速度
            v_current = np.divide(hv_current, h_current, out=np.zeros_like(hv_current),
                                  where=h_current > params['min_depth'] / 10.0)  # 计算v速度
            velocity_magnitude = np.sqrt(u_current ** 2 + v_current ** 2)  # 计算流速大小
            # 计算弗劳德数
            # 避免除以零或在干单元计算 (h_current <= min_depth_for_fr_calc)
            min_depth_for_fr_calc = params.get('min_depth', 1e-6) / 10.0  # 使用一个比min_depth更小的值作为计算Fr的阈值
            # 或者直接用 params['min_depth']，但要确保分母不为零
            sqrt_gh = np.sqrt(params['gravity'] * h_current)  # 计算 sqrt(g*h)
            froude_number = np.divide(velocity_magnitude, sqrt_gh,
                                      out=np.zeros_like(velocity_magnitude),
                                      where=h_current > min_depth_for_fr_calc)  # 只在湿单元计算，干单元Fr为0
            # --- 弗劳德数计算结束 ---
            # --- (新增) 收集剖面线数据 ---
            for profile_name, collector in profile_data_collectors.items():  # 遍历剖面线数据收集器
                if not collector["time_data"] or abs(
                        collector["time_data"][-1] - current_t_cpp) > NUMERICAL_EPSILON / 10.0:
                    collector["time_data"].append(current_t_cpp)  # 添加最终时间
                    # current_profile_etas = eta_current_py[collector["cell_ids"]]  # 获取剖面线上单元的水位
                    # collector["time_data"].append(current_t_cpp)  # 添加当前时间
                    # collector["eta_data"].append(current_profile_etas.tolist())  # 添加水位数据
                    # --- 新的收集逻辑 ---
                    selected_cell_ids = collector["cell_ids"]  # 获取已选定的剖面线单元ID
                    collector["time_data"].append(current_t_cpp)  # 添加当前时间
                    collector["eta_data"].append(eta_current_py[selected_cell_ids].tolist())  # 添加水位数据
                    collector["h_data"].append(h_current[selected_cell_ids].tolist())  # 添加水深数据
                    collector["u_data"].append(u_current[selected_cell_ids].tolist())  # 添加u速度数据
                    collector["v_data"].append(v_current[selected_cell_ids].tolist())  # 添加v速度数据
                    collector["fr_data"].append(froude_number[selected_cell_ids].tolist())  # 新增：收集剖面线上的弗劳德数
            # --- 结束剖面线数据收集 ---


            u_current = np.divide(hu_current, h_current, out=np.zeros_like(hu_current),
                                  where=h_current > params['min_depth'] / 10.0)  # 计算u速度
            v_current = np.divide(hv_current, h_current, out=np.zeros_like(hv_current),
                                  where=h_current > params['min_depth'] / 10.0)  # 计算v速度
            velocity_magnitude = np.sqrt(u_current ** 2 + v_current ** 2)  # 计算流速大小

            cell_data_for_vtk = {  # 准备VTK单元数据
                "water_depth": h_current,  # 水深
                "eta": eta_current_py,  # 水位
                "velocity_u": u_current,  # u速度
                "velocity_v": v_current,  # v速度
                "velocity_magnitude": velocity_magnitude,  # 流速大小
                "froude_number": froude_number  # 新增：弗劳德数
            }  # 结束准备
            vtk_filepath = os.path.join(vtk_output_dir, f"results_t{output_counter:04d}.vtu")  # 构建VTK文件路径
            save_results_to_vtk(vtk_filepath, points_for_vtk, cells_for_vtk, cell_data_for_vtk)  # 保存结果到VTK文件

            output_counter += 1  # 增加输出计数器
            if current_t_cpp < params['total_time'] - NUMERICAL_EPSILON:  # 如果当前时间小于总时间
                next_output_time += params['output_dt']  # 更新下一个输出时间
                if next_output_time > params['total_time'] + NUMERICAL_EPSILON:  # 如果超过总时间
                    next_output_time = params['total_time']  # 设为总时间
            else:  # 如果已达到或超过总时间
                pass  # 不再增加 next_output_time

        simulation_active = model_core.advance_one_step()  # 执行一步C++模拟并更新活动标志
        if model_core.is_simulation_finished() and simulation_active:  # 如果C++认为结束了但Python循环还想继续
            simulation_active = False  # 强制Python循环结束

    # --- 确保在总时间点进行最后一次输出 ---
    current_t_cpp = model_core.get_current_time()  # 获取最终C++时间
    # --- 新增：计算并打印最后一次的时间 ---
    current_wall_time_py = time.time()  # 获取当前墙上时间
    time_since_last_vtk_py = current_wall_time_py - last_vtk_save_time_py  # 计算自上次VTK保存以来的用时
    total_elapsed_time_py = current_wall_time_py - overall_start_time_py  # 计算总的脚本运行时间
    print(
        f"  Python: Final Output at t = {current_t_cpp:.3f} s (C++ step = {model_core.get_step_count()})")  # 打印最终输出信息
    print(
        f"    Time for this final output interval: {time_since_last_vtk_py:.2f} s (Python wall time)")  # 打印本次输出间隔用时
    print(f"    Total elapsed simulation time (end): {total_elapsed_time_py:.2f} s (Python wall time)")  # 打印总用时
    # --- 时间打印结束 ---

    U_final_py = model_core.get_U_state_all_py()  # 获取最终守恒量
    eta_final_py = model_core.get_eta_previous_py()  # 获取最终水位

    # --- (新增) 收集最后时刻的剖面线数据 ---
    for profile_name, collector in profile_data_collectors.items():  # 遍历剖面线数据收集器
        if collector["cell_ids"]:  # 如果该剖面线有单元
            if not collector["time_data"] or abs(
                    collector["time_data"][-1] - current_t_cpp) > NUMERICAL_EPSILON / 10.0:  # 如果时间不重复
                final_profile_etas = eta_final_py[collector["cell_ids"]]  # 获取最终剖面线水位
                collector["time_data"].append(current_t_cpp)  # 添加最终时间
                collector["eta_data"].append(final_profile_etas.tolist())  # 添加最终水位数据
    # --- 结束最后时刻剖面线数据收集 ---

    h_final = U_final_py[:, 0]  # 获取最终水深
    hu_final = U_final_py[:, 1]  # 获取最终hu
    hv_final = U_final_py[:, 2]  # 获取最终hv
    u_final = np.divide(hu_final, h_final, out=np.zeros_like(hu_final),
                        where=h_final > params['min_depth'] / 10.0)  # 计算最终u速度
    v_final = np.divide(hv_final, h_final, out=np.zeros_like(hv_final),
                        where=h_final > params['min_depth'] / 10.0)  # 计算最终v速度
    velocity_magnitude_final = np.sqrt(u_final ** 2 + v_final ** 2)  # 计算最终流速大小
    # --- (在其后添加) ---
    sqrt_gh_final = np.sqrt(params['gravity'] * h_final)
    froude_number_final = np.divide(velocity_magnitude_final, sqrt_gh_final,
                                    out=np.zeros_like(velocity_magnitude_final),
                                    where=h_final > min_depth_for_fr_calc)  # 使用之前定义的阈值
    # --- 弗劳德数计算结束 ---

    final_cell_data_for_vtk = {  # 准备最终VTK单元数据
        "water_depth": h_final,  # 水深
        "eta": eta_final_py,  # 水位
        "velocity_u": u_final,  # u速度
        "velocity_v": v_final,  # v速度
        "velocity_magnitude": velocity_magnitude_final,  # 流速大小
        "froude_number": froude_number_final  # 新增：最终时刻的弗劳德数
    }  # 结束准备
    vtk_filepath_final = os.path.join(vtk_output_dir, f"results_t{output_counter:04d}_final.vtu")  # 构建最终VTK文件路径
    save_results_to_vtk(vtk_filepath_final, points_for_vtk, cells_for_vtk, final_cell_data_for_vtk)  # 保存最终结果到VTK文件

    print("Python: C++ simulation finished.")  # 打印模拟结束信息
    print(f"  Final time: {model_core.get_current_time():.3f} s")  # 打印最终时间
    print(f"  Total steps: {model_core.get_step_count()}")  # 打印总步数

    # --- (新增) 保存剖面线数据到CSV文件和绘图 ---
    if profile_data_collectors:  # 仅当有剖面线数据时才执行
        profile_output_dir = os.path.join(params['output_directory'], "profile_data")  # 定义剖面线数据输出目录
        os.makedirs(profile_output_dir, exist_ok=True)  # 创建目录
        print(f"\n保存剖面线数据到: {os.path.abspath(profile_output_dir)}")  # 打印保存路径信息

        for profile_name, collector in profile_data_collectors.items():  # 遍历剖面线数据收集器
            if not collector["cell_ids"] or not collector["time_data"]:  # 如果没有单元或没有时间数据
                print(f"  跳过剖面线 '{profile_name}'，因为它没有收集到单元或时间数据。")  # 打印跳过信息
                continue  # 继续下一个剖面线

            # --- 确定X轴标签和值 (距离或目标X坐标) ---
            x_axis_values = []  # 初始化X轴值列表
            x_axis_label = "Distance along profile (m)"  # 初始化X轴标签
            column_labels_suffix = []  # 初始化列名后缀列表

            # 尝试使用 target_x (如果通过 sample_points_x 采样)
            # 假设在 get_profile_cells 中，如果用了 sample_points_x,
            # collector["cell_distances"] 存储的是 target_x_on_profile 或者排序后的采样点X值
            # 并且 collector["cell_ids"] 的顺序与之对应
            # 我们需要一种方式从 collector 中获取原始的采样点X值作为绘图的X轴
            # 例如，如果 get_profile_cells 返回的 info 中有 'target_x_on_profile'
            # 那么在初始化 collector 时可以存储一个 target_x_values 列表

            # 为了简单起见，我们优先使用 cell_distances (它可能是实际沿线距离或目标X值)
            # 您可以在 get_profile_cells 返回和 collector 初始化时更明确地处理这一点
            if collector.get("cell_distances"):  # 如果有cell_distances
                x_axis_values = collector["cell_distances"]  # 使用cell_distances作为X轴值
                # 生成列标签时，也用这个距离
                column_labels_suffix = [f"dist{dist:.2f}_id{cell_id}"
                                        for cell_id, dist in
                                        zip(collector["cell_ids"], collector["cell_distances"])]  # 定义列名后缀
            elif collector.get("cell_x_coords"):  # 如果没有cell_distances但有cell_x_coords (备用)
                x_axis_values = collector["cell_x_coords"]  # 使用cell_x_coords作为X轴值
                x_axis_label = "X-coordinate (m)"  # 更新X轴标签
                column_labels_suffix = [f"x{x_coord:.2f}_id{cell_id}"
                                        for cell_id, x_coord in
                                        zip(collector["cell_ids"], collector["cell_x_coords"])]  # 定义列名后缀
            else:  # 如果都没有
                print(f"  警告: 剖面线 '{profile_name}' 缺少距离或X坐标信息，无法生成有意义的列标签和绘图X轴。")  # 打印警告
                continue  # 跳过此剖面线

            df_columns_base = ["time"] + column_labels_suffix  # 定义DataFrame的基础列名

            data_types_to_process = {  # 定义要处理的数据类型及其在收集器中的键名和绘图标签
                "eta": {"key": "eta_data", "label": "Water Surface Elevation (eta) [m]", "csv_suffix": "eta"},
                "depth": {"key": "h_data", "label": "Water Depth (h) [m]", "csv_suffix": "depth"},
                "u_velocity": {"key": "u_data", "label": "Velocity u (m/s)", "csv_suffix": "u_vel"},
                "v_velocity": {"key": "v_data", "label": "Velocity v (m/s)", "csv_suffix": "v_vel"},
                "froude": {"key": "fr_data", "label": "Froude Number (-)", "csv_suffix": "froude"}  # 新增：弗劳德数
            }

            for data_name, data_info in data_types_to_process.items():  # 遍历要处理的数据类型
                collector_key = data_info["key"]  # 获取收集器中的键名
                plot_label_y = data_info["label"]  # 获取绘图Y轴标签
                csv_suffix = data_info["csv_suffix"]  # 获取CSV文件后缀

                if collector_key not in collector or not collector[collector_key]:  # 如果数据不存在或为空
                    print(f"  剖面线 '{profile_name}' 的 '{data_name}' 数据为空，跳过。")  # 打印跳过信息
                    continue  # 继续下一个数据类型

                data_for_df = []  # 初始化DataFrame数据列表
                raw_data_list = collector[collector_key]  # 获取原始数据列表

                for t_idx, time_val in enumerate(collector["time_data"]):  # 遍历时间数据
                    if t_idx < len(raw_data_list) and len(raw_data_list[t_idx]) == len(collector["cell_ids"]):  # 如果长度一致
                        row_data = [time_val] + raw_data_list[t_idx]  # 构建行数据
                        data_for_df.append(row_data)  # 添加到列表
                    else:  # 如果长度不一致
                        print(
                            f"警告: 时间 {time_val:.3f}s 的剖面线 '{profile_name}' 的 '{data_name}' 数据长度不匹配 ({len(raw_data_list[t_idx])} vs {len(collector['cell_ids'])})，已跳过此行。")  # 打印警告

                if not data_for_df:  # 如果没有有效数据行
                    print(f"  剖面线 '{profile_name}' 的 '{data_name}' 没有有效数据行可供保存或绘图。")  # 打印信息
                    continue  # 继续下一个数据类型

                df_profile_data = pd.DataFrame(data_for_df, columns=df_columns_base)  # 创建DataFrame

                # --- 保存到CSV ---
                csv_filename = f"profile_{profile_name}_{csv_suffix}.csv"  # 构建CSV文件名
                csv_filepath = os.path.join(profile_output_dir, csv_filename)  # 构建CSV文件路径
                try:  # 尝试保存CSV
                    df_profile_data.to_csv(csv_filepath, index=False, float_format='%.6f')  # 保存到CSV
                    print(f"  剖面线 '{profile_name}' 的 '{data_name}' 数据已保存到: {csv_filepath}")  # 打印保存信息
                except Exception as e_csv:  # 捕获保存CSV异常
                    print(f"  错误: 保存剖面线 '{profile_name}' 的 '{data_name}' 数据到CSV时出错: {e_csv}")  # 打印错误信息

                # --- 绘图：绘制特定时间点的空间分布图 (类似算例图) ---
                # (例如，绘制第一个、中间和最后一个时间点，或者您可以配置特定时间点)
                if df_profile_data.shape[0] > 0 and df_profile_data.shape[1] > 1 and len(x_axis_values) == (
                        df_profile_data.shape[1] - 1):  # 如果数据有效
                    time_indices_to_plot = []  # 初始化要绘制的时间索引列表
                    if df_profile_data.shape[0] == 1:  # 如果只有一行数据
                        time_indices_to_plot.append(0)  # 只绘制第一行
                    elif df_profile_data.shape[0] > 1:  # 如果有多行数据
                        time_indices_to_plot.append(0)  # 第一个时间点
                        if df_profile_data.shape[0] > 2: time_indices_to_plot.append(
                            df_profile_data.shape[0] // 2)  # 中间时间点
                        time_indices_to_plot.append(df_profile_data.shape[0] - 1)  # 最后一个时间点
                    time_indices_to_plot = sorted(list(set(time_indices_to_plot)))  # 去重并排序

                    plt.figure(figsize=(12, 7))  # 创建图形
                    for t_idx in time_indices_to_plot:  # 遍历要绘制的时间索引
                        time_value = df_profile_data.iloc[t_idx, 0]  # 获取时间值
                        values_at_time = df_profile_data.iloc[t_idx, 1:].values  # 获取该时间点的数据值 (排除时间列)
                        plt.plot(x_axis_values, values_at_time, marker='o', markersize=3, linestyle='-',
                                 label=f"t = {time_value:.2f} s")  # 绘制折线图

                    plt.xlabel(x_axis_label)  # 设置X轴标签
                    plt.ylabel(plot_label_y)  # 设置Y轴标签
                    plt.title(f"{data_name.capitalize()} along Profile: {profile_name}")  # 设置标题
                    plt.legend()  # 显示图例
                    plt.grid(True, linestyle='--', alpha=0.7)  # 显示网格
                    plot_filename = f"profile_{profile_name}_{csv_suffix}_spatial.png"  # 构建图片文件名
                    plot_filepath = os.path.join(profile_output_dir, plot_filename)  # 构建图片文件路径
                    try:  # 尝试保存图片
                        plt.savefig(plot_filepath)  # 保存图片
                        print(
                            f"  剖面线 '{profile_name}' 的 '{data_name}' 空间分布图已保存到: {plot_filepath}")  # 打印保存信息
                    except Exception as e_plot_spatial:  # 捕获保存图片异常
                        print(
                            f"  错误: 保存剖面线 '{profile_name}' 的 '{data_name}' 空间分布图时出错: {e_plot_spatial}")  # 打印错误信息
                    plt.close()  # 关闭图形

                    # --- 绘图：时空等值线图 (如果数据点足够多) ---
                    if df_profile_data.shape[0] > 1 and len(x_axis_values) > 1:  # 如果时间和空间点都大于1
                        plot_X_contour = np.array(x_axis_values)  # X轴数据 (空间)
                        plot_Y_contour = df_profile_data['time'].to_numpy()  # Y轴数据 (时间)
                        plot_Z_contour = df_profile_data.iloc[:, 1:].to_numpy()  # Z轴数据 (值)

                        # 确保维度匹配
                        if plot_X_contour.ndim == 1 and plot_Y_contour.ndim == 1 and \
                                plot_Z_contour.shape[0] == len(plot_Y_contour) and \
                                plot_Z_contour.shape[1] == len(plot_X_contour):  # 如果维度匹配

                            plt.figure(figsize=(12, 7))  # 创建图形
                            # 确定合适的等值线级别数
                            num_levels = min(30, max(5, int(np.nanmax(plot_Z_contour) - np.nanmin(
                                plot_Z_contour)) * 2) if not np.all(np.isnan(plot_Z_contour)) else 10)

                            try:  # 尝试绘制等值线图
                                contour_filled = plt.contourf(plot_X_contour, plot_Y_contour, plot_Z_contour,
                                                              levels=num_levels, cmap="viridis")  # 绘制填充等值线图
                                plt.colorbar(contour_filled, label=plot_label_y)  # 添加颜色条
                                plt.xlabel(x_axis_label)  # 设置X轴标签
                                plt.ylabel("Time (s)")  # 设置Y轴标签
                                plt.title(f"{data_name.capitalize()} Spacetime Contour: {profile_name}")  # 设置标题
                                contour_plot_filename = f"profile_{profile_name}_{csv_suffix}_spacetime_contour.png"  # 构建图片文件名
                                contour_plot_filepath = os.path.join(profile_output_dir,
                                                                     contour_plot_filename)  # 构建图片文件路径
                                plt.savefig(contour_plot_filepath)  # 保存图片
                                print(
                                    f"  剖面线 '{profile_name}' 的 '{data_name}' 时空等值线图已保存到: {contour_plot_filepath}")  # 打印保存信息
                            except Exception as e_contour:  # 捕获绘制异常
                                print(
                                    f"  警告: 绘制剖面线 '{profile_name}' 的 '{data_name}' 时空等值线图时出错: {e_contour}")  # 打印警告
                            plt.close()  # 关闭图形
                        else:  # 如果维度不匹配
                            print(
                                f"  跳过绘制剖面线 '{profile_name}' 的 '{data_name}' 时空等值线图，因为数据维度不匹配。")  # 打印跳过信息
                            print(
                                f"    X_shape: {plot_X_contour.shape}, Y_shape: {plot_Y_contour.shape}, Z_shape: {plot_Z_contour.shape}")  # 打印维度信息


    else:  # 如果没有剖面线数据收集器
        print("\n没有配置或有效的剖面线数据收集器，不进行剖面线数据保存或绘图。")  # 打印信息