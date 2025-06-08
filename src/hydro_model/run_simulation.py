# run_simulation.py
import numpy as np  # 导入 numpy
import os  # 导入 os 模块
import yaml  # 导入 yaml 模块
import sys  # 导入 sys 模块
import pandas as pd  # 导入 pandas 用于数据处理和CSV输出
import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图
import importlib.resources
import re
import time # 新增：导入time模块



try:
    from . import hydro_model_cpp  # <--- 使用相对导入
except ImportError:
    # 如果直接运行脚本，相对导入会失败，尝试绝对导入
    try:
        from hydro_model import hydro_model_cpp
    except ImportError:
        print("错误: 无法导入 'hydro_model_cpp' 模块。")
        sys.exit(1)


try:  # 尝试
    import meshio  # 导入 meshio 用于VTK输出
except ImportError:  # 捕获导入错误
    print("警告: 未找到 'meshio' 库。VTK输出将不可用。请尝试 'pip install meshio'。")  # 打印警告信息
    meshio = None  # 设置为None，以便后续检查

NUMERICAL_EPSILON = 1e-9  # 定义一个数值比较用的小量


def load_config(config_filename='config.yaml'):  # 函数名可以更通用
    """
    加载并返回 YAML 配置文件内容。
    使用 importlib.resources 来安全地从包内部加载数据文件。
    """
    try:
        # 这是查找与 'hydro_model' 包一起安装的数据文件的标准方法
        # 它返回一个可以安全操作的路径对象
        # 即使包安装在 .zip 文件中，这个方法也能工作
        config_file_path_obj = importlib.resources.files('hydro_model').joinpath(config_filename)

        # 'as_file' 提供一个临时的、真实的 filesystem 路径，在 with 块内可用
        with importlib.resources.as_file(config_file_path_obj) as config_filepath:
            with open(config_filepath, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            print(f"配置已从包内文件 {config_filepath} 加载。")
            return config_data

    except FileNotFoundError:  # 如果在包内找不到文件
        print(f"错误: 配置文件 '{config_filename}' 未在 'hydro_model' 包中找到。")
        sys.exit(1)
    except yaml.YAMLError as e:  # 捕获YAML解析错误
        print(f"错误: 解析配置文件 '{config_filename}' 失败: {e}")
        sys.exit(1)
    except Exception as e:  # 捕获其他异常
        print(f"加载配置文件时发生未知错误: {e}")
        sys.exit(1)


def get_parameters_from_config(config_data):
    """从加载的配置字典中提取并返回结构化的参数。"""
    params = {} # 初始化参数字典
    # 文件路径(使用绝对路径拼接)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    fp_conf = config_data.get('file_paths', {})
    # 将所有相对路径转换为绝对路径
    params['node_file'] = os.path.join(project_root, fp_conf.get('node_file'))
    params['cell_file'] = os.path.join(project_root, fp_conf.get('cell_file'))
    params['edge_file'] = os.path.join(project_root, fp_conf.get('edge_file'))
    params['output_directory'] = os.path.join(project_root, fp_conf.get('output_directory', 'output'))
    params['boundary_timeseries_file'] = os.path.join(project_root, fp_conf.get('boundary_timeseries_file'))

    # 模拟控制
    sc_conf = config_data.get('simulation_control', {}) # 获取模拟控制配置
    params['total_time'] = float(sc_conf.get('total_time', 10.0)) # 总模拟时长
    params['output_dt'] = float(sc_conf.get('output_dt', 1.0)) # 输出时间间隔
    params['cfl_number'] = float(sc_conf.get('cfl_number', 0.5)) # CFL数
    params['max_dt'] = float(sc_conf.get('max_dt', 0.1)) # 最大时间步长
    params['use_gpu'] = bool(sc_conf.get('use_gpu', False)) # 是否使用GPU
    params['gpu_modules_to_enable'] = sc_conf.get('gpu_modules', []) # GPU启用的模块
    if not isinstance(params['gpu_modules_to_enable'], list): # 确保是列表
        print(
            f"警告: config.yaml 中的 'gpu_modules' 不是一个列表，将被视为空列表。得到类型: {type(params['gpu_modules_to_enable'])}")
        params['gpu_modules_to_enable'] = []

    # 物理参数
    pp_conf = config_data.get('physical_parameters', {}) # 获取物理参数配置
    params['gravity'] = float(pp_conf.get('gravity', 9.81)) # 重力加速度
    params['min_depth'] = float(pp_conf.get('min_depth', 1e-6)) # 最小水深

    # 数值方案
    ns_conf = config_data.get('numerical_schemes', {}) # 获取数值方案配置
    recon_str = ns_conf.get('reconstruction_scheme', 'FIRST_ORDER').upper() # 重构方案字符串
    params['recon_scheme_cpp'] = getattr(hydro_model_cpp.ReconstructionScheme_cpp, recon_str,
                                         hydro_model_cpp.ReconstructionScheme_cpp.FIRST_ORDER) # C++重构方案枚举
    riemann_str = ns_conf.get('riemann_solver', 'HLLC').upper() # 黎曼求解器字符串
    params['riemann_solver_cpp'] = getattr(hydro_model_cpp.RiemannSolverType_cpp, riemann_str,
                                           hydro_model_cpp.RiemannSolverType_cpp.HLLC) # C++黎曼求解器枚举
    time_str = ns_conf.get('time_scheme', 'RK2_SSP').upper() # 时间积分方案字符串
    params['time_scheme_cpp'] = getattr(hydro_model_cpp.TimeScheme_cpp, time_str,
                                        hydro_model_cpp.TimeScheme_cpp.RK2_SSP) # C++时间积分方案枚举

    # 曼宁系数相关
    mp_conf = config_data.get('model_parameters', {}) # 获取模型参数配置
    params['manning_file'] = os.path.join(project_root, mp_conf.get('manning_file')) # 曼宁文件路径
    params['default_manning'] = float(mp_conf.get('manning_n_default', 0.025)) # 默认曼宁系数

    # 初始条件
    ic_conf_from_yaml = config_data.get('initial_conditions', {}) # 获取初始条件配置
    params['initial_conditions'] = ic_conf_from_yaml # 存储整个初始条件子字典
    # (以下是为了兼容旧代码，推荐后续从 params['initial_conditions'] 获取)
    params['initial_condition_type'] = ic_conf_from_yaml.get('type', 'uniform_elevation')
    params['initial_water_surface_elevation'] = float(
        ic_conf_from_yaml.get('water_surface_elevation', 0.0))
    params['initial_water_depth'] = float(ic_conf_from_yaml.get('water_depth', 0.1))
    params['initial_hu'] = float(ic_conf_from_yaml.get('hu', 0.0))
    params['initial_hv'] = float(ic_conf_from_yaml.get('hv', 0.0))
    if params['initial_condition_type'] == 'dam_break_custom':
        params['dam_position_x'] = float(ic_conf_from_yaml.get('dam_position_x', 10.0))
        params['water_depth_left'] = float(ic_conf_from_yaml.get('water_depth_left', 1.0))
        params['water_depth_right'] = float(ic_conf_from_yaml.get('water_depth_right', 0.0))
    elif params['initial_condition_type'] == '2d_partial_dam_break':
        params['dam_y_start'] = float(ic_conf_from_yaml.get('dam_y_start', 0.0))
        params['dam_y_end'] = float(ic_conf_from_yaml.get('dam_y_end', 0.0))
        params['breach_x_start'] = float(ic_conf_from_yaml.get('breach_x_start', 0.0))
        params['breach_x_end'] = float(ic_conf_from_yaml.get('breach_x_end', 0.0))
        params['water_surface_elevation_upstream'] = float(
            ic_conf_from_yaml.get('water_surface_elevation_upstream', 0.0))
        params['water_surface_elevation_downstream'] = float(
            ic_conf_from_yaml.get('water_surface_elevation_downstream', 0.0))

    # 边界条件定义
    bc_definitions_conf = config_data.get('boundary_conditions', {}) # 获取边界条件配置
    params['boundary_definitions_py'] = bc_definitions_conf.get('definitions', {}) # 获取Python边界定义

    # 剖面线定义
    raw_profile_lines = config_data.get('profile_output_lines', []) # 获取剖面线定义
    params['profile_output_lines'] = []
    if isinstance(raw_profile_lines, list):
        for line_def in raw_profile_lines:
            if isinstance(line_def, dict) and \
                    'name' in line_def and \
                    'start_xy' in line_def and isinstance(line_def['start_xy'], list) and len(
                line_def['start_xy']) == 2 and \
                    'end_xy' in line_def and isinstance(line_def['end_xy'], list) and len(
                line_def['end_xy']) == 2:
                try:
                    start_xy = [float(line_def['start_xy'][0]), float(line_def['start_xy'][1])]
                    end_xy = [float(line_def['end_xy'][0]), float(line_def['end_xy'][1])]
                except ValueError:
                    print(f"警告: 剖面线 '{line_def.get('name', '未命名')}' 的坐标无法转换为浮点数，已跳过。")
                    continue
                buffer_width = float(line_def.get('buffer_width', 0.1))
                is_enabled = line_def.get('enabled', True)
                sample_points_x_from_config = line_def.get('sample_points_x')
                sample_interval_from_config = line_def.get('sample_interval')
                profile_definition_dict = {
                    'name': str(line_def['name']),
                    'start_xy': start_xy,
                    'end_xy': end_xy,
                    'buffer_width': buffer_width
                }
                if sample_points_x_from_config is not None:
                    if isinstance(sample_points_x_from_config, list) and \
                            all(isinstance(pt, (int, float)) for pt in sample_points_x_from_config):
                        profile_definition_dict['sample_points_x'] = [float(pt) for pt in
                                                                      sample_points_x_from_config]
                    else:
                        print(
                            f"警告: 剖面线 '{line_def['name']}' 的 'sample_points_x' 配置无效 (应为数值列表)，已忽略。")
                if sample_interval_from_config is not None:
                    try:
                        profile_definition_dict['sample_interval'] = float(sample_interval_from_config)
                    except ValueError:
                        print(
                            f"警告: 剖面线 '{line_def['name']}' 的 'sample_interval' 配置无效 (应为数值)，已忽略。")
                if is_enabled:
                    params['profile_output_lines'].append(profile_definition_dict)
                else:
                    print(f"  信息: 剖面线 '{line_def['name']}' 已禁用（在配置中设置 enabled: false），跳过。")
            else:
                print(f"警告: 无效的剖面线定义格式，已跳过: {line_def}")
    elif raw_profile_lines is not None:
        print(f"警告: 'profile_output_lines' 配置项不是一个列表，已忽略。实际类型: {type(raw_profile_lines)}")

    # 内部流量线定义
    params['internal_flow_lines'] = config_data.get('internal_flow_lines', [])
    if not isinstance(params['internal_flow_lines'], list):
        print(f"警告: 'internal_flow_lines' 配置项不是一个列表，已忽略。实际类型: {type(params['internal_flow_lines'])}")
        params['internal_flow_lines'] = []
    else:
        valid_flow_lines = []
        for line_def in params['internal_flow_lines']:
            if isinstance(line_def, dict) and \
               'name' in line_def and \
               'poly_node_ids' in line_def and isinstance(line_def['poly_node_ids'], list) and \
               'direction' in line_def and isinstance(line_def['direction'], list) and len(line_def['direction']) == 2:
                try:
                    poly_ids_int = [int(pid) for pid in line_def['poly_node_ids']]
                    direction_float = [float(d_val) for d_val in line_def['direction']]
                    line_def['poly_node_ids'] = poly_ids_int
                    line_def['direction'] = direction_float
                    valid_flow_lines.append(line_def)
                except ValueError:
                    print(f"警告: 内部流量线 '{line_def.get('name', '未命名')}' 的 poly_node_ids 或 direction 包含无法转换的数值，已跳过。")
            else:
                print(f"警告: 无效的内部流量线定义格式，已跳过: {line_def}")
        params['internal_flow_lines'] = valid_flow_lines

    # 内部点源定义
    params['internal_point_sources'] = config_data.get('internal_point_sources', [])
    if not isinstance(params['internal_point_sources'], list):
        print(
            f"警告: 'internal_point_sources' 配置项不是一个列表，已忽略。实际类型: {type(params['internal_point_sources'])}")
        params['internal_point_sources'] = []
    else:
        valid_point_sources = []
        for ps_def in params['internal_point_sources']:
            if isinstance(ps_def, dict) and \
                    'name' in ps_def and \
                    'coordinates' in ps_def and isinstance(ps_def['coordinates'], list) and len(
                ps_def['coordinates']) == 2:
                try:
                    coords = [float(ps_def['coordinates'][0]), float(ps_def['coordinates'][1])]
                    ps_def['coordinates'] = coords
                    valid_point_sources.append(ps_def)
                except ValueError:
                    print(
                        f"警告: 内部点源 '{ps_def.get('name', '未命名')}' 的 coordinates 包含无法转换的数值，已跳过。")
            else:
                print(f"警告: 无效的内部点源定义格式，已跳过: {ps_def}")
        params['internal_point_sources'] = valid_point_sources

    return params # 返回参数字典




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


def prepare_initial_conditions(params, num_cells_cpp, mesh_cpp_ptr_for_ic):  # 移除了不再使用的 parsed_poly_data
    ic_conf_main = params.get('initial_conditions', {})  # 获取顶层 initial_conditions 字典
    print(f"DEBUG_PREPARE_IC: Top-level ic_conf loaded = {ic_conf_main}")

    # 尝试从顶层获取全局默认的 hu, hv
    global_default_hu = float(ic_conf_main.get('hu', 0.0))
    global_default_hv = float(ic_conf_main.get('hv', 0.0))

    h_initial = np.zeros(num_cells_cpp, dtype=float)  # 初始化水深数组
    hu_initial_np = np.full(num_cells_cpp, global_default_hu, dtype=float)  # 初始化 hu 数组
    hv_initial_np = np.full(num_cells_cpp, global_default_hv, dtype=float)  # 初始化 hv 数组

    print("  应用初始条件...")

    rules_list_from_config = ic_conf_main.get('rules')  # 尝试获取 rules 列表
    default_rule_from_config = ic_conf_main.get('default_if_no_match')  # 尝试获取 default_if_no_match

    # 确定最终的默认规则 (final_default_rule)
    if default_rule_from_config:
        final_default_rule = default_rule_from_config
        print(f"    将使用配置文件中的 'default_if_no_match' 作为最终默认: {final_default_rule}")
    else:  # 如果配置文件没有 default_if_no_match，则使用硬编码的后备默认
        final_default_rule = {'type': 'uniform_depth', 'setting_value': 0.0,
                              'hu': global_default_hu, 'hv': global_default_hv}
        print(f"    警告: 'default_if_no_match' 未在配置文件中定义。使用硬编码的后备默认 (干底): {final_default_rule}")

    # 确定要应用的规则列表 (active_rules_list)
    active_rules_list = []
    if rules_list_from_config:  # 如果配置文件中有 'rules'
        active_rules_list = rules_list_from_config
        print(f"    将处理 {len(active_rules_list)} 条来自 'initial_conditions.rules' 的规则。")
        for i_rule, rule_cfg in enumerate(active_rules_list):
            print(f"      Rule {i_rule}: {rule_cfg}")
    elif default_rule_from_config:  # 如果没有 'rules' 但有 'default_if_no_match'
        # 将 default_if_no_match 视为唯一规则应用于所有单元（如果它不包含 region_poly_attribute）
        # 或者它将作为下面循环中匹配失败时的后备
        print(f"    没有 'initial_conditions.rules'。'default_if_no_match' ({final_default_rule}) 将作为主要或后备规则。")
        # 在这种情况下，下面的循环如果找不到匹配的区域属性，就会退到 final_default_rule
        # 如果 final_default_rule 自身没有 region_poly_attribute，它将普适于那些没有被其他规则覆盖的单元
    else:  # 如果既没有 'rules' 也没有 'default_if_no_match'，尝试使用顶层简单配置
        print("    没有 'initial_conditions.rules' 或 'default_if_no_match'。尝试顶层简单配置...")
        simple_ic_type = ic_conf_main.get('type')
        simple_setting_value = None
        # 为简单类型提取参数
        if simple_ic_type == 'uniform_depth':
            simple_setting_value = ic_conf_main.get('water_depth')
        elif simple_ic_type == 'uniform_elevation':
            simple_setting_value = ic_conf_main.get('water_surface_elevation')
        elif simple_ic_type == 'dam_break_custom':  # 支持 dam_break_custom 作为顶层简单配置
            # 将整个 ic_conf_main 视为规则字典
            # 但确保 'type' 键存在，并且其他 dam_break_custom 需要的键也存在于 ic_conf_main 中
            if all(key in ic_conf_main for key in
                   ['dam_position_x', 'upstream_setting_type', 'upstream_setting_value', 'downstream_setting_type',
                    'downstream_setting_value']):
                # final_default_rule 会被构造成这个顶层配置
                # active_rules_list 将包含这个构造出的规则
                pass  # 下面的 final_default_rule 构建会处理
            else:
                simple_ic_type = None  # 标记为无效的简单配置

        if simple_ic_type:
            constructed_rule = {'type': simple_ic_type, 'hu': global_default_hu, 'hv': global_default_hv}
            if simple_setting_value is not None:  # 对于 uniform_depth/elevation
                constructed_rule['setting_value'] = float(simple_setting_value)
            elif simple_ic_type == 'dam_break_custom':  # 对于顶层dam_break
                for key in ['dam_position_x', 'upstream_setting_type', 'upstream_setting_value',
                            'upstream_reference_bed_elevation', 'downstream_setting_type',
                            'downstream_setting_value', 'downstream_reference_bed_elevation']:
                    if key in ic_conf_main:
                        # 尝试转换为 float，如果适用
                        try:
                            constructed_rule[key] = float(ic_conf_main[key]) if isinstance(ic_conf_main[key],
                                                                                           (int, float,
                                                                                            str)) and key.endswith(
                                ('_value', '_x', '_elevation')) else ic_conf_main[key]
                        except ValueError:
                            constructed_rule[key] = ic_conf_main[key]  # 保留原始字符串如果不能转为float

            final_default_rule = constructed_rule  # 这个简单配置成为默认
            active_rules_list = [final_default_rule]  # 并且是唯一要处理的规则
            print(f"    使用顶层简单配置作为唯一规则: {final_default_rule}")
        else:
            # 如果连简单配置都没有或无效，则 active_rules_list 为空，所有单元都将使用硬编码的 final_default_rule
            print(f"    警告: 未找到有效的顶层简单初始条件配置。所有单元将使用最终默认规则: {final_default_rule}")

    for i in range(num_cells_cpp):  # 遍历所有单元
        cell = mesh_cpp_ptr_for_ic.get_cell(i)  # 获取单元对象
        cell_attr_val = mesh_cpp_ptr_for_ic.get_cell_region_attribute(i)  # 获取单元的区域属性

        applied_rule_for_cell = None  # 初始化当前单元应用的规则

        # 尝试从 active_rules_list 中通过 region_poly_attribute 匹配规则
        if active_rules_list:
            for rule_item in active_rules_list:
                rule_region_attr_str = rule_item.get('region_poly_attribute')
                if rule_region_attr_str is not None:  # 如果规则指定了区域属性
                    try:
                        if abs(cell_attr_val - float(rule_region_attr_str)) < 1e-3:
                            applied_rule_for_cell = rule_item
                            break
                    except (ValueError, TypeError):
                        print(f"警告: 规则中的 region_poly_attribute '{rule_region_attr_str}' 无法转换为浮点数。")
                elif len(
                        active_rules_list) == 1 and rule_item == final_default_rule:  # 如果是唯一的 "普适" 规则 (例如由简单配置或无rules的default_if_no_match转化而来)
                    applied_rule_for_cell = rule_item
                    break

        if applied_rule_for_cell is None:  # 如果没有通过区域属性匹配上，或者 active_rules_list 为空
            applied_rule_for_cell = final_default_rule  # 则使用最终的默认规则

        # 从选定的规则中获取参数
        current_hu_val = float(applied_rule_for_cell.get('hu', global_default_hu))
        current_hv_val = float(applied_rule_for_cell.get('hv', global_default_hv))
        ic_type_from_rule_val = applied_rule_for_cell.get('type')

        h_val_cell_calc = 0.0  # 初始化当前单元计算得到的水深

        # --- 根据规则类型计算水深 ---
        if ic_type_from_rule_val == 'uniform_elevation':
            setting_val = applied_rule_for_cell.get('setting_value')
            if setting_val is not None:
                wse = float(setting_val)
                h_val_cell_calc = max(0.0, wse - cell.z_bed_centroid)
            else:
                print(f"警告: 单元 {i} 规则类型 'uniform_elevation' 缺少 'setting_value'。水深设为0。")

        elif ic_type_from_rule_val == 'uniform_depth':
            setting_val = applied_rule_for_cell.get('setting_value')
            if setting_val is not None:
                depth = float(setting_val)
                h_val_cell_calc = max(0.0, depth)
            else:
                print(f"警告: 单元 {i} 规则类型 'uniform_depth' 缺少 'setting_value'。水深设为0。")

        elif ic_type_from_rule_val == 'linear_wse_slope':
            try:
                up_wse = float(applied_rule_for_cell.get('upstream_wse'))
                down_wse = float(applied_rule_for_cell.get('downstream_wse'))
                start_coord_val = float(applied_rule_for_cell.get('river_start_coord'))
                end_coord_val = float(applied_rule_for_cell.get('river_end_coord'))
                axis_str = applied_rule_for_cell.get('coord_axis_for_slope', 'x').lower()
                axis_idx = 0 if axis_str == 'x' else 1

                total_len_coord = end_coord_val - start_coord_val
                if abs(total_len_coord) < 1e-6:  # 避免除以零
                    target_wse = (up_wse + down_wse) / 2.0
                else:
                    current_coord_val_cell = cell.centroid[axis_idx]
                    ratio = (current_coord_val_cell - start_coord_val) / total_len_coord
                    # 线性插值，并处理超出范围的情况
                    if ratio <= 0:
                        target_wse = up_wse
                    elif ratio >= 1:
                        target_wse = down_wse
                    else:
                        target_wse = up_wse + ratio * (down_wse - up_wse)
                h_val_cell_calc = max(0.0, target_wse - cell.z_bed_centroid)
            except Exception as e_slope:  # 捕获参数缺失或类型错误
                print(f"警告: 单元 {i} 应用 'linear_wse_slope' 规则时出错: {e_slope}。水深设为0。")

        elif ic_type_from_rule_val == 'dam_break_custom':
            try:
                dam_pos_x_val = float(applied_rule_for_cell.get('dam_position_x'))

                up_type = applied_rule_for_cell.get('upstream_setting_type', 'elevation').lower()
                up_val = float(applied_rule_for_cell.get('upstream_setting_value'))
                up_ref_bed_str = applied_rule_for_cell.get('upstream_reference_bed_elevation')
                up_ref_bed = float(up_ref_bed_str) if up_ref_bed_str is not None else None

                down_type = applied_rule_for_cell.get('downstream_setting_type', 'depth').lower()
                down_val = float(applied_rule_for_cell.get('downstream_setting_value'))
                down_ref_bed_str = applied_rule_for_cell.get('downstream_reference_bed_elevation')
                down_ref_bed = float(down_ref_bed_str) if down_ref_bed_str is not None else None

                # 计算上游水深
                if cell.centroid[0] < dam_pos_x_val:
                    if up_type == 'elevation':
                        h_val_cell_calc = max(0.0, up_val - cell.z_bed_centroid)
                    elif up_type == 'depth':
                        if up_ref_bed is not None:  # 基于参考底高程的水深
                            h_val_cell_calc = max(0.0, (up_ref_bed + up_val) - cell.z_bed_centroid)
                        else:  # 直接水深
                            h_val_cell_calc = max(0.0, up_val)
                # 计算下游水深
                else:
                    if down_type == 'elevation':
                        h_val_cell_calc = max(0.0, down_val - cell.z_bed_centroid)
                    elif down_type == 'depth':
                        if down_ref_bed is not None:  # 基于参考底高程的水深
                            h_val_cell_calc = max(0.0, (down_ref_bed + down_val) - cell.z_bed_centroid)
                        else:  # 直接水深
                            h_val_cell_calc = max(0.0, down_val)
            except Exception as e_dam:  # 捕获参数缺失或类型错误
                print(f"警告: 单元 {i} 应用 'dam_break_custom' 规则时出错: {e_dam}。水深设为0。")

        # ... (可以继续添加其他 elif ic_type_from_rule_val == '...' )

        else:  # 如果规则类型未被以上任何 if/elif 处理
            print(f"警告: 单元 {i} (属性 {cell_attr_val:.1f}) 的规则类型 '{ic_type_from_rule_val}' 未被实现。水深设为0。")
            h_val_cell_calc = 0.0  # 绝对后备

        h_initial[i] = h_val_cell_calc  # 设置初始水深
        hu_initial_np[i] = current_hu_val  # 设置初始hu
        hv_initial_np[i] = current_hv_val  # 设置初始hv

    num_dry_cells_calc = np.sum(h_initial < params.get('min_depth', 1e-6))
    print(f"  初始条件设置完毕。基于规则，计算得到 {num_dry_cells_calc} / {num_cells_cpp} 个干单元或水深极浅单元。")

    return np.column_stack((h_initial, hu_initial_np, hv_initial_np))


def prepare_boundary_conditions_for_cpp(params):

    bc_defs_cpp = {}
    py_def_dict_top = params.get('boundary_definitions_py', {})

    for marker_str, py_def_item in py_def_dict_top.items():
        try:
            marker_int = int(marker_str)
            cpp_def = hydro_model_cpp.BoundaryDefinition_cpp()

            # 获取原始字符串
            type_str_raw = py_def_item.get('type', 'WALL')

            # ***** 关键修复：显式转换为 str 类型 *****
            type_str_from_config = str(type_str_raw).upper() # 转换为大写

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
    while simulation_active:
        current_t_cpp = model_core.get_current_time()
        if current_t_cpp >= next_output_time - NUMERICAL_EPSILON or output_counter == 0:
            current_wall_time_py = time.time()
            time_since_last_vtk_py = current_wall_time_py - last_vtk_save_time_py
            total_elapsed_time_py = current_wall_time_py - overall_start_time_py

            if output_counter > 0:
                print(f"  Python: Output at t = {current_t_cpp:.3f} s (C++ step = {model_core.get_step_count()})")
                print(f"    Time for this output interval: {time_since_last_vtk_py:.2f} s (Python wall time)")
            else:
                print(
                    f"  Python: Initial Output at t = {current_t_cpp:.3f} s (C++ step = {model_core.get_step_count()})")
            print(f"    Total elapsed simulation time: {total_elapsed_time_py:.2f} s (Python wall time)")
            last_vtk_save_time_py = current_wall_time_py

            U_current_py = model_core.get_U_state_all_py()
            eta_current_py = model_core.get_eta_previous_py()
            h_current = U_current_py[:, 0]
            hu_current = U_current_py[:, 1]
            hv_current = U_current_py[:, 2]

            u_current = np.divide(hu_current, h_current, out=np.zeros_like(hu_current),
                                  where=h_current > params['min_depth'] / 10.0)
            v_current = np.divide(hv_current, h_current, out=np.zeros_like(hv_current),
                                  where=h_current > params['min_depth'] / 10.0)
            velocity_magnitude = np.sqrt(u_current ** 2 + v_current ** 2)
            min_depth_for_fr_calc = params.get('min_depth', 1e-6) / 10.0
            sqrt_gh = np.sqrt(params['gravity'] * h_current)
            froude_number = np.divide(velocity_magnitude, sqrt_gh,
                                      out=np.zeros_like(velocity_magnitude),
                                      where=h_current > min_depth_for_fr_calc)

            # --- (修改后) 收集剖面线数据 ---
            for profile_name, collector in profile_data_collectors.items():
                selected_cell_ids = collector["cell_ids"]
                # 只有当剖面线有单元时才记录数据
                if selected_cell_ids:
                    # 避免重复记录完全相同的时间点的数据 (可以根据需要调整此逻辑)
                    if not collector["time_data"] or abs(
                            collector["time_data"][-1] - current_t_cpp) > NUMERICAL_EPSILON / 100.0:
                        collector["time_data"].append(current_t_cpp)
                        collector["eta_data"].append(eta_current_py[selected_cell_ids].tolist())
                        collector["h_data"].append(h_current[selected_cell_ids].tolist())
                        collector["u_data"].append(u_current[selected_cell_ids].tolist())
                        collector["v_data"].append(v_current[selected_cell_ids].tolist())
                        collector["fr_data"].append(froude_number[selected_cell_ids].tolist())
                # 如果 selected_cell_ids 为空，则不为该剖面线记录任何数据

            cell_data_for_vtk = {
                "water_depth": h_current, "eta": eta_current_py,
                "velocity_u": u_current, "velocity_v": v_current,
                "velocity_magnitude": velocity_magnitude, "froude_number": froude_number
            }
            vtk_filepath = os.path.join(vtk_output_dir, f"results_t{output_counter:04d}.vtu")
            save_results_to_vtk(vtk_filepath, points_for_vtk, cells_for_vtk, cell_data_for_vtk)

            output_counter += 1
            if current_t_cpp < params['total_time'] - NUMERICAL_EPSILON:
                next_output_time += params['output_dt']
                if next_output_time > params['total_time'] + NUMERICAL_EPSILON:
                    next_output_time = params['total_time']
            else:
                pass

        simulation_active = model_core.advance_one_step()
        if model_core.is_simulation_finished() and simulation_active:
            simulation_active = False

        # --- 确保在总时间点进行最后一次输出 ---
    current_t_cpp = model_core.get_current_time()
    current_wall_time_py = time.time()
    time_since_last_vtk_py = current_wall_time_py - last_vtk_save_time_py
    total_elapsed_time_py = current_wall_time_py - overall_start_time_py
    print(f"  Python: Final Output at t = {current_t_cpp:.3f} s (C++ step = {model_core.get_step_count()})")
    print(f"    Time for this final output interval: {time_since_last_vtk_py:.2f} s (Python wall time)")
    print(f"    Total elapsed simulation time (end): {total_elapsed_time_py:.2f} s (Python wall time)")

    U_final_py = model_core.get_U_state_all_py()
    eta_final_py = model_core.get_eta_previous_py()

    # --- (修改后) 收集最后时刻的剖面线数据 ---
    for profile_name, collector in profile_data_collectors.items():
        selected_cell_ids = collector["cell_ids"]
        if selected_cell_ids:  # 仅当剖面线实际捕获到单元时才添加数据
            # 避免重复记录完全相同的时间点的数据
            if not collector["time_data"] or abs(
                    collector["time_data"][-1] - current_t_cpp) > NUMERICAL_EPSILON / 100.0:
                collector["time_data"].append(current_t_cpp)
                # 需要从 U_final_py 和 eta_final_py 重新计算最终的 h, u, v, fr
                h_final_prof = U_final_py[selected_cell_ids, 0]
                hu_final_prof = U_final_py[selected_cell_ids, 1]
                hv_final_prof = U_final_py[selected_cell_ids, 2]
                u_final_prof = np.divide(hu_final_prof, h_final_prof, out=np.zeros_like(hu_final_prof),
                                         where=h_final_prof > params['min_depth'] / 10.0)
                v_final_prof = np.divide(hv_final_prof, h_final_prof, out=np.zeros_like(hv_final_prof),
                                         where=h_final_prof > params['min_depth'] / 10.0)
                vel_mag_final_prof = np.sqrt(u_final_prof ** 2 + v_final_prof ** 2)
                sqrt_gh_final_prof = np.sqrt(params['gravity'] * h_final_prof)
                fr_final_prof = np.divide(vel_mag_final_prof, sqrt_gh_final_prof,
                                          out=np.zeros_like(vel_mag_final_prof),
                                          where=h_final_prof > min_depth_for_fr_calc)

                collector["eta_data"].append(eta_final_py[selected_cell_ids].tolist())
                collector["h_data"].append(h_final_prof.tolist())
                collector["u_data"].append(u_final_prof.tolist())
                collector["v_data"].append(v_final_prof.tolist())
                collector["fr_data"].append(fr_final_prof.tolist())

    # ... (VTK保存和结束语不变) ...
    h_final = U_final_py[:, 0]
    hu_final = U_final_py[:, 1]
    hv_final = U_final_py[:, 2]
    u_final = np.divide(hu_final, h_final, out=np.zeros_like(hu_final),
                        where=h_final > params['min_depth'] / 10.0)
    v_final = np.divide(hv_final, h_final, out=np.zeros_like(hv_final),
                        where=h_final > params['min_depth'] / 10.0)
    velocity_magnitude_final = np.sqrt(u_final ** 2 + v_final ** 2)
    sqrt_gh_final = np.sqrt(params['gravity'] * h_final)
    froude_number_final = np.divide(velocity_magnitude_final, sqrt_gh_final,
                                    out=np.zeros_like(velocity_magnitude_final),
                                    where=h_final > min_depth_for_fr_calc)

    final_cell_data_for_vtk = {
        "water_depth": h_final, "eta": eta_final_py,
        "velocity_u": u_final, "velocity_v": v_final,
        "velocity_magnitude": velocity_magnitude_final, "froude_number": froude_number_final
    }
    vtk_filepath_final = os.path.join(vtk_output_dir, f"results_t{output_counter:04d}_final.vtu")
    save_results_to_vtk(vtk_filepath_final, points_for_vtk, cells_for_vtk, final_cell_data_for_vtk)

    print("Python: C++ simulation finished.")
    print(f"  Final time: {model_core.get_current_time():.3f} s")
    print(f"  Total steps: {model_core.get_step_count()}")

    # --- (修改后) 保存剖面线数据到CSV文件和绘图 ---
    if profile_data_collectors:
        profile_output_dir = os.path.join(params['output_directory'], "profile_data")
        os.makedirs(profile_output_dir, exist_ok=True)
        print(f"\n保存剖面线数据到: {os.path.abspath(profile_output_dir)}")

        for profile_name, collector in profile_data_collectors.items():
            if not collector["cell_ids"] or not collector["time_data"]:
                print(f"  跳过剖面线 '{profile_name}'，因为它没有收集到单元或有效的时间数据。")
                continue

            # 检查所有数据列表的长度是否与time_data一致
            # 确保所有数据列表长度与 time_data 一致
            num_time_points = len(collector["time_data"])
            consistent_data = True
            for data_key in ["eta_data", "h_data", "u_data", "v_data", "fr_data"]:
                if len(collector[data_key]) != num_time_points:
                    print(
                        f"  警告: 剖面线 '{profile_name}' 的 '{data_key}' 数据点数量 ({len(collector[data_key])}) 与时间点数量 ({num_time_points}) 不一致。跳过此剖面线的数据处理。")
                    consistent_data = False
                    break
            if not consistent_data:
                continue

            x_axis_values = []
            x_axis_label = "Distance along profile (m)"
            column_labels_suffix = []

            if collector.get("cell_distances"):
                x_axis_values = collector["cell_distances"]
                column_labels_suffix = [f"dist{dist:.2f}_id{cell_id}"
                                        for cell_id, dist in
                                        zip(collector["cell_ids"], collector["cell_distances"])]
            elif collector.get("cell_x_coords"):
                x_axis_values = collector["cell_x_coords"]
                x_axis_label = "X-coordinate (m)"
                column_labels_suffix = [f"x{x_coord:.2f}_id{cell_id}"
                                        for cell_id, x_coord in
                                        zip(collector["cell_ids"], collector["cell_x_coords"])]
            else:
                print(f"  警告: 剖面线 '{profile_name}' 缺少距离或X坐标信息，无法生成有意义的列标签和绘图X轴。")
                continue

            df_columns_base = ["time"] + column_labels_suffix

            data_types_to_process = {
                "eta": {"key": "eta_data", "label": "Water Surface Elevation (eta) [m]", "csv_suffix": "eta"},
                "depth": {"key": "h_data", "label": "Water Depth (h) [m]", "csv_suffix": "depth"},
                "u_velocity": {"key": "u_data", "label": "Velocity u (m/s)", "csv_suffix": "u_vel"},
                "v_velocity": {"key": "v_data", "label": "Velocity v (m/s)", "csv_suffix": "v_vel"},
                "froude": {"key": "fr_data", "label": "Froude Number (-)", "csv_suffix": "froude"}
            }

            for data_name, data_info in data_types_to_process.items():
                collector_key = data_info["key"]
                plot_label_y = data_info["label"]
                csv_suffix = data_info["csv_suffix"]

                if collector_key not in collector or not collector[collector_key]:
                    print(f"  剖面线 '{profile_name}' 的 '{data_name}' 数据为空，跳过。")
                    continue

                data_for_df = []
                raw_data_list = collector[collector_key]

                for t_idx, time_val in enumerate(collector["time_data"]):
                    # raw_data_list[t_idx] 应该是一个包含该时间点所有剖面单元值的列表
                    if t_idx < len(raw_data_list) and isinstance(raw_data_list[t_idx], list) and len(
                            raw_data_list[t_idx]) == len(collector["cell_ids"]):
                        row_data = [time_val] + raw_data_list[t_idx]
                        data_for_df.append(row_data)
                    else:
                        # 理论上，如果前面数据收集是正确的，这里不应该发生
                        print(
                            f"严重警告: 时间 {time_val:.3f}s 的剖面线 '{profile_name}' 的 '{data_name}' 数据格式或长度不匹配，请检查收集逻辑。数据行跳过。 "
                            f"Expected len: {len(collector['cell_ids'])}, Got len: {len(raw_data_list[t_idx]) if t_idx < len(raw_data_list) else 'N/A'}")

                if not data_for_df:
                    print(f"  剖面线 '{profile_name}' 的 '{data_name}' 没有有效数据行可供保存或绘图。")
                    continue

                df_profile_data = pd.DataFrame(data_for_df, columns=df_columns_base)

                csv_filename = f"profile_{profile_name}_{csv_suffix}.csv"
                csv_filepath = os.path.join(profile_output_dir, csv_filename)
                try:
                    df_profile_data.to_csv(csv_filepath, index=False, float_format='%.6f')
                    print(f"  剖面线 '{profile_name}' 的 '{data_name}' 数据已保存到: {csv_filepath}")
                except Exception as e_csv:
                    print(f"  错误: 保存剖面线 '{profile_name}' 的 '{data_name}' 数据到CSV时出错: {e_csv}")

                if df_profile_data.shape[0] > 0 and df_profile_data.shape[1] > 1 and len(x_axis_values) == (
                        df_profile_data.shape[1] - 1):
                    time_indices_to_plot = []
                    if df_profile_data.shape[0] == 1:
                        time_indices_to_plot.append(0)
                    elif df_profile_data.shape[0] > 1:
                        time_indices_to_plot.append(0)
                        if df_profile_data.shape[0] > 2: time_indices_to_plot.append(df_profile_data.shape[0] // 2)
                        time_indices_to_plot.append(df_profile_data.shape[0] - 1)
                    time_indices_to_plot = sorted(list(set(time_indices_to_plot)))

                    plt.figure(figsize=(12, 7))
                    for t_idx_plot in time_indices_to_plot:
                        time_value_plot = df_profile_data.iloc[t_idx_plot, 0]
                        values_at_time_plot = df_profile_data.iloc[t_idx_plot, 1:].values
                        plt.plot(x_axis_values, values_at_time_plot, marker='o', markersize=3, linestyle='-',
                                 label=f"t = {time_value_plot:.2f} s")

                    plt.xlabel(x_axis_label)
                    plt.ylabel(plot_label_y)
                    plt.title(f"{data_name.capitalize()} along Profile: {profile_name}")
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plot_filename = f"profile_{profile_name}_{csv_suffix}_spatial.png"
                    plot_filepath = os.path.join(profile_output_dir, plot_filename)
                    try:
                        plt.savefig(plot_filepath)
                        print(f"  剖面线 '{profile_name}' 的 '{data_name}' 空间分布图已保存到: {plot_filepath}")
                    except Exception as e_plot_spatial:
                        print(
                            f"  错误: 保存剖面线 '{profile_name}' 的 '{data_name}' 空间分布图时出错: {e_plot_spatial}")
                    plt.close()

                    if df_profile_data.shape[0] > 1 and len(x_axis_values) > 1:
                        plot_X_contour = np.array(x_axis_values)
                        plot_Y_contour = df_profile_data['time'].to_numpy()
                        plot_Z_contour = df_profile_data.iloc[:, 1:].to_numpy()

                        if plot_X_contour.ndim == 1 and plot_Y_contour.ndim == 1 and \
                                plot_Z_contour.shape[0] == len(plot_Y_contour) and \
                                plot_Z_contour.shape[1] == len(plot_X_contour):

                            plt.figure(figsize=(12, 7))
                            num_levels = min(30, max(5, int(np.nanmax(plot_Z_contour) - np.nanmin(
                                plot_Z_contour)) * 2) if not np.all(np.isnan(plot_Z_contour)) else 10)

                            try:
                                contour_filled = plt.contourf(plot_X_contour, plot_Y_contour, plot_Z_contour,
                                                              levels=num_levels, cmap="viridis")
                                plt.colorbar(contour_filled, label=plot_label_y)
                                plt.xlabel(x_axis_label)
                                plt.ylabel("Time (s)")
                                plt.title(f"{data_name.capitalize()} Spacetime Contour: {profile_name}")
                                contour_plot_filename = f"profile_{profile_name}_{csv_suffix}_spacetime_contour.png"
                                contour_plot_filepath = os.path.join(profile_output_dir, contour_plot_filename)
                                plt.savefig(contour_plot_filepath)
                                print(
                                    f"  剖面线 '{profile_name}' 的 '{data_name}' 时空等值线图已保存到: {contour_plot_filepath}")
                            except Exception as e_contour:
                                print(
                                    f"  警告: 绘制剖面线 '{profile_name}' 的 '{data_name}' 时空等值线图时出错: {e_contour}")
                            plt.close()
                        else:
                            print(
                                f"  跳过绘制剖面线 '{profile_name}' 的 '{data_name}' 时空等值线图，因为数据维度不匹配。")
                            print(
                                f"    X_shape: {plot_X_contour.shape}, Y_shape: {plot_Y_contour.shape}, Z_shape: {plot_Z_contour.shape}")
    else:
        print("\n没有配置或有效的剖面线数据收集器，不进行剖面线数据保存或绘图。")