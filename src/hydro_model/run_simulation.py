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

    # --- (在这里添加或修改) 读取内部流量线定义 ---
    params['internal_flow_lines'] = config_data.get('internal_flow_lines', []) # 从config_data中获取'internal_flow_lines'，如果不存在则返回空列表
    if not isinstance(params['internal_flow_lines'], list): # 检查获取到的是否为列表
        print(f"警告: 'internal_flow_lines' 配置项不是一个列表，已忽略。实际类型: {type(params['internal_flow_lines'])}") # 如果不是列表，打印警告
        params['internal_flow_lines'] = [] # 将其重置为空列表
    else: # 如果是列表
        valid_flow_lines = [] # 初始化一个用于存储有效流量线定义的列表
        for line_def in params['internal_flow_lines']: # 遍历从配置文件中读取的每个流量线定义
            if isinstance(line_def, dict) and \
               'name' in line_def and \
               'poly_node_ids' in line_def and isinstance(line_def['poly_node_ids'], list) and \
               'direction' in line_def and isinstance(line_def['direction'], list) and len(line_def['direction']) == 2: # 检查定义是否为字典，且包含必要的键和正确的数据类型
                try:
                    # 确保 poly_node_ids 是整数列表
                    poly_ids_int = [int(pid) for pid in line_def['poly_node_ids']] # 将poly_node_ids中的每个元素转换为整数
                    # 确保 direction 是浮点数列表
                    direction_float = [float(d_val) for d_val in line_def['direction']] # 将direction中的每个元素转换为浮点数
                    line_def['poly_node_ids'] = poly_ids_int # 更新line_def中的poly_node_ids
                    line_def['direction'] = direction_float # 更新line_def中的direction
                    valid_flow_lines.append(line_def) # 将验证和转换后的流量线定义添加到有效列表中
                except ValueError: # 如果在转换过程中发生值错误 (例如，无法将字符串转换为整数或浮点数)
                    print(f"警告: 内部流量线 '{line_def.get('name', '未命名')}' 的 poly_node_ids 或 direction 包含无法转换的数值，已跳过。") # 打印警告
            else: # 如果流量线定义的格式无效
                print(f"警告: 无效的内部流量线定义格式，已跳过: {line_def}") # 打印警告
        params['internal_flow_lines'] = valid_flow_lines # 用验证后的列表更新params中的'internal_flow_lines'
    # --- 读取内部流量线定义结束 ---

    # --- 新增：读取内部点源定义 ---
    params['internal_point_sources'] = config_data.get('internal_point_sources',
                                                       [])  # 从config_data获取internal_point_sources配置，默认为空列表
    if not isinstance(params['internal_point_sources'], list):  # 检查获取到的是否为列表
        print(
            f"警告: 'internal_point_sources' 配置项不是一个列表，已忽略。实际类型: {type(params['internal_point_sources'])}")  # 打印警告
        params['internal_point_sources'] = []  # 将其重置为空列表
    else:  # 如果是列表
        valid_point_sources = []  # 初始化有效点源列表
        for ps_def in params['internal_point_sources']:  # 遍历每个点源定义
            if isinstance(ps_def, dict) and \
                    'name' in ps_def and \
                    'coordinates' in ps_def and isinstance(ps_def['coordinates'], list) and len(
                ps_def['coordinates']) == 2:  # 检查定义是否为字典且包含必要字段和类型
                try:
                    coords = [float(ps_def['coordinates'][0]), float(ps_def['coordinates'][1])]  # 转换坐标为浮点数
                    ps_def['coordinates'] = coords  # 更新定义中的坐标
                    # timeseries_column 是可选的，所以这里不做强制检查，C++端会处理
                    valid_point_sources.append(ps_def)  # 添加到有效列表
                except ValueError:  # 捕获转换错误
                    print(
                        f"警告: 内部点源 '{ps_def.get('name', '未命名')}' 的 coordinates 包含无法转换的数值，已跳过。")  # 打印警告
            else:  # 如果格式无效
                print(f"警告: 无效的内部点源定义格式，已跳过: {ps_def}")  # 打印警告
        params['internal_point_sources'] = valid_point_sources  # 更新为有效列表
    # --- 读取内部点源定义结束 ---

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


def prepare_initial_conditions(params, num_cells_cpp, mesh_cpp_ptr_for_ic,
                               parsed_poly_data=None):  # parsed_poly_data可能不再直接需要，除非某些IC类型仍依赖它
    ic_conf_main = params.get('initial_conditions', {})
    print(f"DEBUG_PREPARE_IC: Top-level ic_conf loaded = {ic_conf_main}")

    global_default_hu = float(ic_conf_main.get('hu', 0.0))
    global_default_hv = float(ic_conf_main.get('hv', 0.0))

    h_initial = np.zeros(num_cells_cpp, dtype=float)
    hu_initial_np = np.full(num_cells_cpp, global_default_hu, dtype=float)
    hv_initial_np = np.full(num_cells_cpp, global_default_hv, dtype=float)

    print("  应用初始条件 (基于单元区域属性和配置规则)...")

    rules_list = ic_conf_main.get('rules', [])
    final_default_rule = ic_conf_main.get('default_if_no_match',
                                          {'type': 'uniform_depth', 'setting_value': 0.0,
                                           'hu': global_default_hu, 'hv': global_default_hv})

    if not rules_list:
        print("    警告: 'initial_conditions.rules' 列表为空或未定义。所有单元将使用 'default_if_no_match'。")
    else:
        for i, rule in enumerate(rules_list):
            print(
                f"    已定义规则 {i}: 属性={rule.get('region_poly_attribute')}, 类型='{rule.get('type')}', 参数={ {k: v for k, v in rule.items() if k not in ['region_poly_attribute', 'type']} }")

    print(
        f"    最终默认设置 (若无规则匹配): 类型='{final_default_rule['type']}', 值/参数='{ {k: v for k, v in final_default_rule.items() if k != 'type'} }'")

    for i in range(num_cells_cpp):
        cell = mesh_cpp_ptr_for_ic.get_cell(i)  # 假设返回一个可访问成员的对象
        cell_attr = mesh_cpp_ptr_for_ic.get_cell_region_attribute(i)

        applied_rule = None
        for rule in rules_list:
            rule_attr = rule.get('region_poly_attribute')
            if rule_attr is not None and abs(cell_attr - float(rule_attr)) < 1e-3:
                applied_rule = rule
                break

        if applied_rule is None:
            applied_rule = final_default_rule

        # --- 从 applied_rule 中提取参数 ---
        # 注意: 每个类型可能需要不同的参数，这里只是示例
        # 优先从 applied_rule 获取 hu, hv，如果规则中没有，则使用全局默认
        current_hu = float(applied_rule.get('hu', global_default_hu))
        current_hv = float(applied_rule.get('hv', global_default_hv))

        ic_type_from_rule = applied_rule.get('type')
        # print(f"DEBUG_PREPARE_IC: Cell {i}, Attr {cell_attr}, RuleType '{ic_type_from_rule}'") # 详细调试

        h_val_cell = 0.0  # 在每个类型内部计算

        # --- 在这里嵌入您所有的初始条件类型判断逻辑 ---
        # --- 您需要将 applied_rule 作为这些逻辑的参数来源 ---

        if ic_type_from_rule == 'uniform_elevation':
            setting_value = applied_rule.get('setting_value')
            if setting_value is not None:
                wse = float(setting_value)
                h_val_cell = max(0.0, wse - cell.z_bed_centroid)
            else:
                print(f"警告: 单元 {i} 规则类型 'uniform_elevation' 缺少 'setting_value'。应用最终默认。")
                # 应用最终默认的水深计算逻辑
                if final_default_rule.get('type') == 'uniform_depth':
                    h_val_cell = max(0.0, float(final_default_rule.get('setting_value', 0.0)))
                # (可以添加更多对final_default_rule类型的处理)


        elif ic_type_from_rule == 'uniform_depth':
            setting_value = applied_rule.get('setting_value')
            if setting_value is not None:
                depth = float(setting_value)
                h_val_cell = max(0.0, depth)
            else:
                print(f"警告: 单元 {i} 规则类型 'uniform_depth' 缺少 'setting_value'。应用最终默认。")
                if final_default_rule.get('type') == 'uniform_depth':
                    h_val_cell = max(0.0, float(final_default_rule.get('setting_value', 0.0)))


        elif ic_type_from_rule == 'linear_wse_slope':
            try:
                up_wse = float(applied_rule.get('upstream_wse'))
                down_wse = float(applied_rule.get('downstream_wse'))
                start_coord_val = float(applied_rule.get('river_start_coord'))
                end_coord_val = float(applied_rule.get('river_end_coord'))
                axis_str = applied_rule.get('coord_axis_for_slope', 'x')
                axis_idx = 0 if axis_str == 'x' else 1

                total_len_coord = end_coord_val - start_coord_val
                if abs(total_len_coord) < 1e-6:
                    target_wse = (up_wse + down_wse) / 2.0
                else:
                    current_coord_val = cell.centroid[axis_idx]
                    ratio = (current_coord_val - start_coord_val) / total_len_coord
                    if ratio < 0:
                        target_wse = up_wse
                    elif ratio > 1:
                        target_wse = down_wse
                    else:
                        target_wse = up_wse + ratio * (down_wse - up_wse)
                h_val_cell = max(0.0, target_wse - cell.z_bed_centroid)
            except Exception as e:
                print(f"警告: 单元 {i} 规则类型 'linear_wse_slope' 参数配置错误: {e}。应用最终默认。")
                if final_default_rule.get('type') == 'uniform_depth':
                    h_val_cell = max(0.0, float(final_default_rule.get('setting_value', 0.0)))


        elif ic_type_from_rule == 'dam_break_custom':
            # 从 applied_rule 获取参数
            dam_pos_x = float(applied_rule.get('dam_position_x', 0.0))
            upstream_type = applied_rule.get('upstream_setting_type', 'elevation').lower()
            upstream_value = float(applied_rule.get('upstream_setting_value', 0.0))
            # ... (获取所有 dam_break_custom 需要的参数，如 upstream_reference_bed_elevation 等)
            # ... (然后是您已有的计算 h_val_cell 的逻辑)
            # (为简洁，这里省略了完整的 dam_break_custom 内部计算逻辑，您需要从原函数复制并调整参数来源)
            # 例如:
            apply_direct_depth_upstream = False
            upstream_target_eta_for_calc = 0.0
            upstream_direct_depth_for_calc = 0.0
            if upstream_type == 'elevation':
                upstream_target_eta_for_calc = upstream_value
            elif upstream_type == 'depth':
                if 'upstream_reference_bed_elevation' in applied_rule:
                    upstream_target_eta_for_calc = float(
                        applied_rule.get('upstream_reference_bed_elevation')) + upstream_value
                else:
                    upstream_direct_depth_for_calc = upstream_value
                    apply_direct_depth_upstream = True
            # ... (类似地处理下游 downstream_... ) ...
            downstream_type = applied_rule.get('downstream_setting_type', 'depth').lower()
            downstream_value = float(applied_rule.get('downstream_setting_value', 0.0))
            apply_direct_depth_downstream = False
            downstream_target_eta_for_calc = 0.0
            downstream_direct_depth_for_calc = 0.0
            if downstream_type == 'elevation':
                downstream_target_eta_for_calc = downstream_value
            elif downstream_type == 'depth':
                if 'downstream_reference_bed_elevation' in applied_rule:
                    downstream_target_eta_for_calc = float(
                        applied_rule.get('downstream_reference_bed_elevation')) + downstream_value
                else:
                    downstream_direct_depth_for_calc = downstream_value
                    apply_direct_depth_downstream = True

            if cell.centroid[0] < dam_pos_x:  # 上游
                if apply_direct_depth_upstream:
                    h_val_cell = max(0.0, upstream_direct_depth_for_calc)
                else:
                    h_val_cell = max(0.0, upstream_target_eta_for_calc - cell.z_bed_centroid)
            else:  # 下游
                if apply_direct_depth_downstream:
                    h_val_cell = max(0.0, downstream_direct_depth_for_calc)
                else:
                    h_val_cell = max(0.0, downstream_target_eta_for_calc - cell.z_bed_centroid)


        # --- 在此 elif 块中添加您其他的 initial_condition_type ---
        # 例如: 'custom_L_shaped_dam_break', '2d_partial_dam_break', 等等
        # 确保每个类型的逻辑都从 applied_rule 中获取其参数

        elif ic_type_from_rule == 'custom_surface_function':
            surface_params = applied_rule.get('surface_params', {})
            base_eta = float(surface_params.get('base_elevation', 0.0))
            slope_x = float(surface_params.get('slope_x', 0.0))
            slope_y = float(surface_params.get('slope_y', 0.0))
            eta_cell = base_eta + slope_x * cell.centroid[0] + slope_y * cell.centroid[1]
            h_val_cell = max(0.0, eta_cell - cell.z_bed_centroid)

        else:  # 如果类型未被以上任何 if/elif 处理
            print(f"警告: 单元 {i} (属性 {cell_attr:.1f}) 的规则类型 '{ic_type_from_rule}' 未被实现。应用最终默认。")
            # 应用最终默认的水深计算逻辑
            if final_default_rule.get('type') == 'uniform_depth':
                h_val_cell = max(0.0, float(final_default_rule.get('setting_value', 0.0)))
            elif final_default_rule.get('type') == 'uniform_elevation':
                h_val_cell = max(0.0, float(final_default_rule.get('setting_value', 0.0)) - cell.z_bed_centroid)
            # ... (可以添加更多对final_default_rule类型的处理)
            else:
                h_val_cell = 0.0  # 绝对后备

        h_initial[i] = h_val_cell
        hu_initial_np[i] = current_hu  # 使用从规则或全局默认获取的hu
        hv_initial_np[i] = current_hv  # 使用从规则或全局默认获取的hv

    num_dry_cells = np.sum(h_initial < params.get('min_depth', 1e-6))  # min_depth 可能更合适
    print(f"  初始条件设置完毕。基于规则，计算得到 {num_dry_cells} / {num_cells_cpp} 个干单元或水深极浅单元。")

    return np.column_stack((h_initial, hu_initial_np, hv_initial_np))


def prepare_boundary_conditions_for_cpp(params):
    # ... (调试打印 hydro_model_cpp.BoundaryType_cpp 成员的代码可以保留或删除) ...

    bc_defs_cpp = {}
    py_def_dict_top = params.get('boundary_definitions_py', {})

    for marker_str, py_def_item in py_def_dict_top.items():
        try:
            marker_int = int(marker_str)
            cpp_def = hydro_model_cpp.BoundaryDefinition_cpp()

            # 获取原始字符串
            type_str_raw = py_def_item.get('type', 'WALL')

            # ***** 关键修复：显式转换为 str 类型 *****
            type_str_from_config = str(type_str_raw)

            print(
                f"DEBUG_BC_PREP: Marker {marker_str}, Type from config (raw): '{type_str_raw}' (type: {type(type_str_raw)}), Converted to str: '{type_str_from_config}' (type: {type(type_str_from_config)})")  # 详细调试

            if type_str_from_config == "WALL":
                cpp_def.type = hydro_model_cpp.BoundaryType_cpp.WALL
            elif type_str_from_config == "WATERLEVEL":
                cpp_def.type = hydro_model_cpp.BoundaryType_cpp.WATERLEVEL
            elif type_str_from_config == "TOTAL_DISCHARGE":
                cpp_def.type = hydro_model_cpp.BoundaryType_cpp.TOTAL_DISCHARGE
            elif type_str_from_config == "FREE_OUTFLOW":
                cpp_def.type = hydro_model_cpp.BoundaryType_cpp.FREE_OUTFLOW
            else:
                print(
                    f"警告: 边界类型 '{type_str_from_config}' (标记 {marker_str}) 在config.yaml中无效或未在Python端处理，将设为WALL。")
                cpp_def.type = hydro_model_cpp.BoundaryType_cpp.WALL

            # 处理 flow_target_direction (这部分逻辑不变)
            if 'flow_target_direction' in py_def_item:
                direction = py_def_item['flow_target_direction']
                if isinstance(direction, list) and len(direction) == 2:
                    try:
                        cpp_def.flow_direction_hint_x = float(direction[0])
                        cpp_def.flow_direction_hint_y = float(direction[1])
                        cpp_def.has_flow_direction_hint = True
                    except ValueError:
                        print(f"警告: 边界标记 {marker_int} 的 flow_target_direction 坐标无法转换为浮点数。")
                        cpp_def.has_flow_direction_hint = False
                else:
                    print(f"警告: 边界标记 {marker_int} 的 flow_target_direction 格式不正确，应为 [dx, dy]。")
                    cpp_def.has_flow_direction_hint = False
            else:
                cpp_def.has_flow_direction_hint = False

            bc_defs_cpp[marker_int] = cpp_def
        except ValueError:
            print(f"警告: 边界定义标记 '{marker_str}' 不是有效整数，已跳过。")

    # ... (wl_ts_data_cpp, discharge_ts_data_cpp 的逻辑不变) ...
    wl_ts_data_cpp = {}
    discharge_ts_data_cpp = {}
    unified_ts_file_path = params.get('boundary_timeseries_file')

    if unified_ts_file_path and os.path.exists(unified_ts_file_path):
        print(f"  正在从统一边界时间序列文件 '{unified_ts_file_path}' 加载数据 (基于线段ID)...")
        try:
            df_ts = pd.read_csv(unified_ts_file_path)
            if 'time' not in df_ts.columns:
                print(f"警告: 统一边界时间序列文件 '{unified_ts_file_path}' 缺少 'time' 列。")
            else:
                time_col = df_ts['time'].values
                for col_name in df_ts.columns:
                    if col_name.lower() == 'time':
                        continue
                    match = re.fullmatch(r"b(\d+)_(elev|flux)", col_name, re.IGNORECASE)
                    if match:
                        segment_id = int(match.group(1))
                        data_type_suffix = match.group(2).lower()
                        ts_points = []
                        for t_val, data_val in zip(time_col, df_ts[col_name].values):
                            if pd.notna(t_val) and pd.notna(data_val):
                                pt = hydro_model_cpp.TimeseriesPoint_cpp()
                                pt.time = float(t_val)
                                pt.value = float(data_val)
                                ts_points.append(pt)
                        if ts_points:
                            if data_type_suffix == 'elev':
                                wl_ts_data_cpp[segment_id] = ts_points
                                print(f"    已为原始线段ID {segment_id} 加载水位时间序列 (elev)。")
                            elif data_type_suffix == 'flux':
                                discharge_ts_data_cpp[segment_id] = ts_points
                                print(f"    已为原始线段ID {segment_id} 加载流量时间序列 (flux)。")
        except ImportError:
            print(f"警告: pandas 未安装，无法解析统一边界时间序列CSV文件 '{unified_ts_file_path}'。")
        except Exception as e:
            print(f"处理统一边界时间序列文件 '{unified_ts_file_path}' 时出错: {e}")
    elif unified_ts_file_path:
        print(f"警告: 统一边界时间序列文件 '{unified_ts_file_path}' 未找到。")

    return bc_defs_cpp, wl_ts_data_cpp, discharge_ts_data_cpp


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

    # ******** 内部流量源项设置 ********
    # 从config中获取内部流量线定义 (假设你会在config中添加类似以下的结构)
    # internal_flow_lines:
    #   - name: "inflow_segment"
    #     poly_node_ids: [5, 6] # .poly 文件中定义的节点 ID
    #     direction: [1.0, 0.0]

    df_ts_all = None  # 初始化为None
    if params.get('boundary_timeseries_file') and os.path.exists(params['boundary_timeseries_file']):
        try:
            df_ts_all = pd.read_csv(params['boundary_timeseries_file'])
        except Exception as e_csv_main:
            print(
                f"Python: CRITICAL ERROR - Could not read the main timeseries file: {params['boundary_timeseries_file']}. Error: {e_csv_main}")
            df_ts_all = None  # 确保出错时为None

    if df_ts_all is None:
        print(
            "Python: WARNING - Main timeseries data CSV could not be loaded. Internal flow sources requiring timeseries will not be set up or will use Q=0.")

    internal_flow_config_list_py = params.get('internal_flow_lines', [])
    if internal_flow_config_list_py:
        print(f"Python: Processing {len(internal_flow_config_list_py)} internal flow line definitions...")
        for flow_def in internal_flow_config_list_py:
            line_name = flow_def.get('name')
            poly_ids = flow_def.get('poly_node_ids')
            direction = flow_def.get('direction')

            if not all([line_name, poly_ids, direction]):
                print(f"  Skipping incomplete internal_flow_line definition: {flow_def}")
                continue

            q_timeseries_for_cpp = []
            if df_ts_all is not None and line_name in df_ts_all.columns:
                time_col_from_csv = df_ts_all['time'].values
                q_values_from_csv = df_ts_all[line_name].values
                for t_val, q_val in zip(time_col_from_csv, q_values_from_csv):
                    if pd.notna(t_val) and pd.notna(q_val):
                        ts_point = hydro_model_cpp.TimeseriesPoint_cpp(float(t_val), float(q_val))
                        q_timeseries_for_cpp.append(ts_point)

                if not q_timeseries_for_cpp:
                    print(
                        f"  Warning: No valid (non-NaN) data points found for timeseries column '{line_name}' for internal flow line '{line_name}'. Source will effectively be Q=0.")
            else:
                if df_ts_all is None:
                    print(
                        f"  Warning: Main timeseries CSV not loaded. Cannot get timeseries for internal flow line '{line_name}'. Source will effectively be Q=0.")
                else:
                    print(
                        f"  Warning: Timeseries column '{line_name}' (from internal_flow_line name) not found in {params['boundary_timeseries_file']}. Source will effectively be Q=0.")
                # 即使找不到时程，也传递一个空vector，C++端会处理（或警告流量为0）

            # 确保方向是浮点数列表/元组
            try:
                cpp_direction = [float(direction[0]), float(direction[1])]
            except (TypeError, IndexError, ValueError) as e_dir:
                print(
                    f"  ERROR: Invalid 'direction' format for internal flow line '{line_name}': {direction}. Error: {e_dir}. Using [0,0].")
                cpp_direction = [0.0, 0.0]

            print(
                f"  Python: Calling C++ setup_internal_flow_source for '{line_name}' with {len(q_timeseries_for_cpp)} points and direction {cpp_direction}.")
            model_core.setup_internal_flow_source(
                line_name,
                poly_ids,
                q_timeseries_for_cpp,  # 即使为空也传递
                cpp_direction
            )
    else:
        print("Python: No internal_flow_lines configured.")
    # ******** 内部流量源项设置结束 ********

    # ******** 新增：内部点源设置 ********
    internal_point_source_config_list_py = params.get('internal_point_sources', [])  # 获取点源配置列表
    if internal_point_source_config_list_py:  # 如果存在点源配置
        print(
            f"Python: Processing {len(internal_point_source_config_list_py)} internal point source definitions...")  # 打印处理信息
        for ps_def in internal_point_source_config_list_py:  # 遍历每个点源定义
            ps_name = ps_def.get('name')  # 获取点源名称
            coordinates = ps_def.get('coordinates')  # 获取点源坐标
            timeseries_col_name = ps_def.get('timeseries_column')  # 获取时程列名 (可选)

            if not ps_name or not coordinates:  # 如果名称或坐标缺失
                print(f"  Skipping incomplete internal_point_source definition: {ps_def}")  # 打印跳过信息
                continue  # 继续下一个定义

            q_ps_timeseries_for_cpp = []  # 初始化点源流量时程列表
            if timeseries_col_name and df_ts_all is not None and timeseries_col_name in df_ts_all.columns:  # 如果配置了列名且CSV已加载且列存在
                time_col_from_csv_ps = df_ts_all['time'].values  # 获取时间列
                q_values_from_csv_ps = df_ts_all[timeseries_col_name].values  # 获取流量列
                for t_val, q_val in zip(time_col_from_csv_ps, q_values_from_csv_ps):  # 遍历时程数据
                    if pd.notna(t_val) and pd.notna(q_val):  # 如果时间和流量值都有效
                        ts_point = hydro_model_cpp.TimeseriesPoint_cpp(float(t_val), float(q_val))  # 创建C++时程点对象
                        q_ps_timeseries_for_cpp.append(ts_point)  # 添加到列表

                if not q_ps_timeseries_for_cpp:  # 如果没有有效的时程点
                    print(
                        f"  Warning: No valid (non-NaN) data points found for timeseries column '{timeseries_col_name}' for point source '{ps_name}'. Source will effectively be Q=0.")  # 打印警告
            elif timeseries_col_name:  # 如果配置了列名但上述条件不满足
                if df_ts_all is None:  # 如果CSV未加载
                    print(
                        f"  Warning: Main timeseries CSV not loaded. Cannot get timeseries for point source '{ps_name}' using column '{timeseries_col_name}'. Source will effectively be Q=0.")  # 打印警告
                else:  # CSV已加载但列不存在
                    print(
                        f"  Warning: Timeseries column '{timeseries_col_name}' for point source '{ps_name}' not found in {params['boundary_timeseries_file']}. Source will effectively be Q=0.")  # 打印警告
            else:  # 如果没有配置时程列名
                print(
                    f"  Info: No 'timeseries_column' specified for point source '{ps_name}'. Source will be Q=0 unless C++ has a default.")  # 打印信息

            # 调用C++方法设置点源
            # 假设 model_core 有一个名为 setup_internal_point_source 的方法
            # 它接收: name (string), coordinates (list/array of 2 floats), q_timeseries (vector of TimeseriesPoint_cpp)
            print(
                f"  Python: Calling C++ setup_internal_point_source for '{ps_name}' at coords {coordinates} with {len(q_ps_timeseries_for_cpp)} timeseries points.")  # 打印调用信息
            model_core.setup_internal_point_source_cpp(  # 调用C++方法
                ps_name,
                coordinates,  # 传递 Python 列表 [x, y]
                q_ps_timeseries_for_cpp
            )
    else:  # 如果没有点源配置
        print("Python: No internal_point_sources configured.")  # 打印信息
    # ******** 内部点源设置结束 ********

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