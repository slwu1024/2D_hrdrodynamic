# run_simulation.py
import numpy as np
import os
import yaml
import sys # 用于检查Python版本或退出

try:
    import hydro_model_cpp # 导入编译好的C++模块
except ImportError:
    print("错误: 未找到 C++ 核心模块 'hydro_model_cpp'。请确保已编译并放置在正确路径。")
    sys.exit(1) # 退出程序
try:
    import meshio # 导入 meshio 用于VTK输出
except ImportError:
    print("警告: 未找到 'meshio' 库。VTK输出将不可用。请尝试 'pip install meshio'。")
    meshio = None # 设置为None，以便后续检查

# --- 全局epsilon，用于浮点比较 ---
NUMERICAL_EPSILON = 1e-9 # 定义一个数值比较用的小量

def load_config(config_filepath='config.yaml'): # 加载配置文件函数
    """加载并返回 YAML 配置文件内容。"""
    try:
        with open(config_filepath, 'r', encoding='utf-8') as f: # 打开文件
            config_data = yaml.safe_load(f) # 加载yaml配置
        print(f"配置已从 {config_filepath} 加载。") # 打印加载信息
        return config_data # 返回配置数据
    except FileNotFoundError: # 捕获文件未找到异常
        print(f"错误: 配置文件 '{config_filepath}' 未找到。") # 打印错误信息
        sys.exit(1) # 退出程序
    except yaml.YAMLError as e: # 捕获YAML解析错误
        print(f"错误: 解析配置文件 '{config_filepath}' 失败: {e}") # 打印错误信息
        sys.exit(1) # 退出程序
    except Exception as e: # 捕获其他异常
        print(f"加载配置文件时发生未知错误: {e}") # 打印错误信息
        sys.exit(1) # 退出程序

def get_parameters_from_config(config_data): # 从配置数据获取参数函数
    """从加载的配置字典中提取并返回结构化的参数。"""
    params = {} # 初始化参数字典
    # 文件路径
    fp_conf = config_data.get('file_paths', {}) # 获取文件路径配置
    params['node_file'] = fp_conf.get('node_file') # 获取节点文件路径
    params['cell_file'] = fp_conf.get('cell_file') # 获取单元文件路径
    params['edge_file'] = fp_conf.get('edge_file') # 获取边文件路径
    params['output_directory'] = fp_conf.get('output_directory', 'output') # 获取输出目录，默认为'output'

    # 模拟控制
    sc_conf = config_data.get('simulation_control', {}) # 获取模拟控制配置
    params['total_time'] = float(sc_conf.get('total_time', 10.0)) # 获取总模拟时长，转为浮点数
    params['output_dt'] = float(sc_conf.get('output_dt', 1.0)) # 获取输出时间间隔，转为浮点数
    params['cfl_number'] = float(sc_conf.get('cfl_number', 0.5)) # 获取CFL数，转为浮点数
    params['max_dt'] = float(sc_conf.get('max_dt', 0.1)) # 获取最大时间步长，转为浮点数

    # 物理参数
    pp_conf = config_data.get('physical_parameters', {}) # 获取物理参数配置
    params['gravity'] = float(pp_conf.get('gravity', 9.81)) # 获取重力加速度，转为浮点数
    params['min_depth'] = float(pp_conf.get('min_depth', 1e-6)) # 获取最小水深，转为浮点数

    # 数值方案
    ns_conf = config_data.get('numerical_schemes', {}) # 获取数值方案配置
    recon_str = ns_conf.get('reconstruction_scheme', 'FIRST_ORDER').upper() # 获取重构方案字符串
    params['recon_scheme_cpp'] = getattr(hydro_model_cpp.ReconstructionScheme_cpp, recon_str, # 获取C++重构方案枚举值
                                         hydro_model_cpp.ReconstructionScheme_cpp.FIRST_ORDER)
    riemann_str = ns_conf.get('riemann_solver', 'HLLC').upper() # 获取黎曼求解器字符串
    params['riemann_solver_cpp'] = getattr(hydro_model_cpp.RiemannSolverType_cpp, riemann_str, # 获取C++黎曼求解器枚举值
                                           hydro_model_cpp.RiemannSolverType_cpp.HLLC)
    time_str = ns_conf.get('time_scheme', 'RK2_SSP').upper() # config.yaml中是time_scheme # 获取时间积分方案字符串
    params['time_scheme_cpp'] = getattr(hydro_model_cpp.TimeScheme_cpp, time_str, # 获取C++时间积分方案枚举值
                                        hydro_model_cpp.TimeScheme_cpp.RK2_SSP) # 默认RK2_SSP

    # 曼宁系数相关
    mp_conf = config_data.get('model_parameters', {}) # 获取模型参数配置
    params['manning_file'] = mp_conf.get('manning_file') # 获取曼宁文件路径
    params['default_manning'] = float(mp_conf.get('manning_n_default', 0.025)) # 获取默认曼宁系数，转为浮点数

    # 初始条件
    ic_conf = config_data.get('initial_conditions', {}) # 获取初始条件配置
    params['initial_condition_type'] = ic_conf.get('type', 'uniform_elevation') # 获取初始条件类型
    params['initial_water_surface_elevation'] = float(ic_conf.get('water_surface_elevation', 0.0)) # 获取初始水位，转为浮点数
    params['initial_water_depth'] = float(ic_conf.get('water_depth', 0.1)) # 获取初始水深，转为浮点数
    params['initial_hu'] = float(ic_conf.get('hu', 0.0)) # 获取初始hu，转为浮点数
    params['initial_hv'] = float(ic_conf.get('hv', 0.0)) # 获取初始hv，转为浮点数

    # 边界条件
    params['boundary_definitions_py'] = config_data.get('boundary_conditions', {}).get('definitions', {}) # 获取Python边界定义
    params['boundary_timeseries_elevation_file'] = fp_conf.get('boundary_timeseries_elevation_file') # 获取水位时间序列文件路径
    params['boundary_timeseries_discharge_file'] = fp_conf.get('boundary_timeseries_discharge_file') # 获取流量时间序列文件路径

    return params # 返回参数字典

def load_manning_values_from_file(manning_filepath, num_cells_expected, default_manning_val): # 从文件加载曼宁值函数
    """从文件加载曼宁值。如果失败，则返回用默认值填充的列表。"""
    if manning_filepath and os.path.exists(manning_filepath): # 如果文件路径有效且存在
        try:
            manning_values = np.loadtxt(manning_filepath, dtype=float) # 加载曼宁值
            if manning_values.ndim == 0: # 如果只有一个值
                manning_values = np.array([manning_values.item()]) # 转为数组
            if len(manning_values) == 1 and num_cells_expected > 1: # 如果只有一个值但期望多个
                 print(f"  曼宁文件 {manning_filepath} 只包含一个值，将用于所有 {num_cells_expected} 个单元。") # 打印信息
                 return np.full(num_cells_expected, manning_values[0], dtype=float).tolist() # 返回填充数组
            elif len(manning_values) == num_cells_expected: # 如果数量匹配
                return manning_values.tolist() # 返回列表
            else: # 数量不匹配
                print(f"  警告: 曼宁文件 {manning_filepath} 中的值数量 ({len(manning_values)}) 与单元数 ({num_cells_expected}) 不符。将使用默认值。") # 打印警告
        except Exception as e: # 捕获异常
            print(f"  读取曼宁文件 {manning_filepath} 出错: {e}。将使用默认值。") # 打印错误信息
    else: # 文件不存在
        print(f"  曼宁文件 '{manning_filepath}' 未找到或未指定。将为所有单元使用默认曼宁系数 {default_manning_val}。") # 打印信息
    return np.full(num_cells_expected, default_manning_val, dtype=float).tolist() # 返回填充数组

def prepare_initial_conditions(params, num_cells_cpp, mesh_cpp_ptr_for_ic): # 准备初始条件函数
    """根据配置准备初始守恒量 U_initial_np。"""
    h_initial = np.zeros(num_cells_cpp, dtype=float) # 初始化水深数组
    if params['initial_condition_type'] == 'uniform_elevation': # 如果是均匀水位
        eta_initial = params['initial_water_surface_elevation'] # 获取初始水位
        for i in range(num_cells_cpp): # 遍历单元
            cell = mesh_cpp_ptr_for_ic.get_cell(i) # 获取单元对象
            h_initial[i] = max(0.0, eta_initial - cell.z_bed_centroid) # 计算水深
    elif params['initial_condition_type'] == 'uniform_depth': # 如果是均匀水深
        h_initial.fill(params['initial_water_depth']) # 填充水深
    else: # 其他类型
        print(f"警告: 未知的初始条件类型 '{params['initial_condition_type']}'。使用默认零水深。") # 打印警告

    hu_initial = np.full(num_cells_cpp, params['initial_hu'], dtype=float) # 初始化hu数组
    hv_initial = np.full(num_cells_cpp, params['initial_hv'], dtype=float) # 初始化hv数组
    return np.column_stack((h_initial, hu_initial, hv_initial)) # 返回组合后的NumPy数组

def prepare_boundary_conditions_for_cpp(params): # 准备C++边界条件函数
    """转换Python边界配置为C++期望的格式。"""
    bc_defs_cpp = {} # 初始化C++边界定义字典
    for marker_str, py_def in params.get('boundary_definitions_py', {}).items(): # 遍历Python边界定义
        try:
            marker_int = int(marker_str) # 转换标记为整数
            cpp_def = hydro_model_cpp.BoundaryDefinition_cpp() # 创建C++边界定义对象
            type_str = py_def.get('type', 'WALL').upper() # 获取类型字符串
            cpp_def.type = getattr(hydro_model_cpp.BoundaryType_cpp, type_str, hydro_model_cpp.BoundaryType_cpp.WALL) # 设置C++边界类型
            bc_defs_cpp[marker_int] = cpp_def # 添加到字典
        except ValueError: # 捕获值错误
            print(f"警告: 边界定义标记 '{marker_str}' 不是有效整数，已跳过。") # 打印警告
        except AttributeError: # 捕获属性错误
            type_str_for_error = py_def.get('type', 'UNKNOWN').upper() # 获取用于错误信息的类型字符串
            print(f"警告: 边界类型 '{type_str_for_error}' (标记 {marker_str}) 无效，已设为WALL。") # 打印警告
            cpp_def_fallback = hydro_model_cpp.BoundaryDefinition_cpp() # 创建备用C++边界定义对象
            cpp_def_fallback.type = hydro_model_cpp.BoundaryType_cpp.WALL # 设为墙体
            bc_defs_cpp[marker_int] = cpp_def_fallback # 添加到字典

    # 加载时间序列数据 (水位)
    wl_ts_data_cpp = {} # 初始化水位时间序列数据字典
    py_wl_ts_path = params.get('boundary_timeseries_elevation_file') # 获取水位时间序列文件路径
    if py_wl_ts_path and os.path.exists(py_wl_ts_path): # 如果路径有效且文件存在
        try:
            import pandas as pd # 导入pandas
            df_wl = pd.read_csv(py_wl_ts_path) # 读取CSV
            if 'time' not in df_wl.columns: # 检查是否有时间列
                print(f"警告: 水位时间序列文件 '{py_wl_ts_path}' 缺少 'time' 列。") # 打印警告
            else: # 如果有时间列
                time_col = df_wl['time'].values # 获取时间列
                for col_name in df_wl.columns: # 遍历列名
                    if col_name.startswith('elev_'): # 如果是高程列
                        try:
                            marker = int(col_name.split('_')[-1]) # 获取标记
                            ts_points = [] # 初始化时间序列点列表
                            for t_val, data_val in zip(time_col, df_wl[col_name].values): # 遍历时间和值
                                if pd.notna(t_val) and pd.notna(data_val): # 检查是否为NaN
                                    pt = hydro_model_cpp.TimeseriesPoint_cpp() # 创建时间序列点对象
                                    pt.time = float(t_val) # 设置时间
                                    pt.value = float(data_val) # 设置值
                                    ts_points.append(pt) # 添加到列表
                            if ts_points: wl_ts_data_cpp[marker] = ts_points # 如果有数据则添加到字典
                        except ValueError: # 捕获值错误
                            print(f"警告: 无法从水位时间序列列名 '{col_name}' 解析标记。") # 打印警告
        except ImportError: # 捕获导入错误
            print("警告: pandas 未安装，无法解析水位时间序列CSV文件。") # 打印警告
        except Exception as e: # 捕获其他异常
            print(f"处理水位时间序列文件 {py_wl_ts_path} 时出错: {e}") # 打印错误

    # 加载时间序列数据 (流量)
    discharge_ts_data_cpp = {} # 初始化流量时间序列数据字典
    py_q_ts_path = params.get('boundary_timeseries_discharge_file') # 获取流量时间序列文件路径
    if py_q_ts_path and os.path.exists(py_q_ts_path): # 如果路径有效且文件存在
        try:
            import pandas as pd # 导入pandas
            df_q = pd.read_csv(py_q_ts_path) # 读取CSV
            if 'time' not in df_q.columns: # 检查是否有时间列
                print(f"警告: 流量时间序列文件 '{py_q_ts_path}' 缺少 'time' 列。") # 打印警告
            else: # 如果有时间列
                time_col = df_q['time'].values # 获取时间列
                for col_name in df_q.columns: # 遍历列名
                    if col_name.startswith('flux_'): # 如果是流量列
                        try:
                            marker = int(col_name.split('_')[-1]) # 获取标记
                            ts_points = [] # 初始化时间序列点列表
                            for t_val, data_val in zip(time_col, df_q[col_name].values): # 遍历时间和值
                                if pd.notna(t_val) and pd.notna(data_val): # 检查是否为NaN
                                    pt = hydro_model_cpp.TimeseriesPoint_cpp() # 创建时间序列点对象
                                    pt.time = float(t_val) # 设置时间
                                    pt.value = float(data_val) # 设置值
                                    ts_points.append(pt) # 添加到列表
                            if ts_points: discharge_ts_data_cpp[marker] = ts_points # 如果有数据则添加到字典
                        except ValueError: # 捕获值错误
                            print(f"警告: 无法从流量时间序列列名 '{col_name}' 解析标记。") # 打印警告
        except ImportError: # 捕获导入错误
            print("警告: pandas 未安装，无法解析流量时间序列CSV文件。") # 打印警告
        except Exception as e: # 捕获其他异常
            print(f"处理流量时间序列文件 {py_q_ts_path} 时出错: {e}") # 打印错误

    return bc_defs_cpp, wl_ts_data_cpp, discharge_ts_data_cpp # 返回C++边界条件数据

def save_results_to_vtk(vtk_filepath, points_coords, cells_connectivity, cell_data_dict): # 保存结果到VTK文件函数
    """使用 meshio 将结果保存为 VTK (.vtu) 文件。"""
    if not meshio: # 如果 meshio 未导入
        print(f"  Meshio 未加载，无法保存VTK文件: {vtk_filepath}") # 打印信息
        return # 返回

    # meshio 期望 cell_data 的值是列表的列表，即使只有一个数据集
    formatted_cell_data = {key: [value_array] for key, value_array in cell_data_dict.items()} # 格式化单元数据

    try:
        meshio.write_points_cells( # 调用meshio写入文件
            vtk_filepath, # 文件路径
            points_coords, # 节点坐标
            cells_connectivity, # 单元连接关系
            cell_data=formatted_cell_data, # 单元数据
            file_format="vtu" # 文件格式
        )
        print(f"    VTK 文件已保存: {vtk_filepath}") # 打印保存信息
    except Exception as e: # 捕获异常
        print(f"    保存VTK文件 {vtk_filepath} 时出错: {e}") # 打印错误信息

# --- 主程序 ---
if __name__ == "__main__": # 主程序入口
    config = load_config() # 加载配置文件
    params = get_parameters_from_config(config) # 从配置数据获取参数

    if not params.get('node_file') or not params.get('cell_file'): # 检查节点和单元文件路径是否存在
        print("错误: 必须在 config.yaml 的 file_paths 中配置 node_file 和 cell_file。") # 打印错误
        sys.exit(1) # 退出程序

    model_core = hydro_model_cpp.HydroModelCore_cpp() # 创建C++模型对象
    print("Python: C++ HydroModelCore_cpp object created.") # 打印创建信息

    # 为了加载曼宁值，我们需要知道单元数量。
    num_cells_for_manning = 0 # 初始化单元数
    try:
        with open(params['cell_file'], 'r') as f_cell: # 打开单元文件
            header_cell = f_cell.readline().split() # 读取头部
            num_cells_for_manning = int(header_cell[0]) # 获取单元数
    except Exception as e: # 捕获异常
        print(f"错误: 无法从 {params['cell_file']} 读取单元数量以加载曼宁值: {e}") # 打印错误
        sys.exit(1) # 退出程序

    cell_manning_list = load_manning_values_from_file( # 加载曼宁值
        params['manning_file'], num_cells_for_manning, params['default_manning']
    )

    model_core.initialize_model_from_files( # 调用C++模型初始化方法
        params['node_file'], params['cell_file'],
        params['edge_file'] if params['edge_file'] and os.path.exists(params['edge_file']) else "", # 边文件路径 (如果存在)
        cell_manning_list, # 曼宁值列表
        params['gravity'], params['min_depth'], params['cfl_number'], # 模拟参数
        params['total_time'], params['output_dt'], params['max_dt'], # 时间参数
        params['recon_scheme_cpp'], params['riemann_solver_cpp'], params['time_scheme_cpp'] # 数值方案
    )
    print("Python: C++ model initialized with mesh and parameters.") # 打印初始化完成信息

    mesh_cpp_ptr = model_core.get_mesh_ptr() # 获取指向C++ Mesh_cpp对象的指针
    num_cells_cpp_from_core = mesh_cpp_ptr.get_num_cells() # 从C++核心获取单元数

    U_initial_np = prepare_initial_conditions(params, num_cells_cpp_from_core, mesh_cpp_ptr) # 准备初始条件
    model_core.set_initial_conditions_py(U_initial_np) # 设置初始条件
    print(f"Python: Initial conditions set for {num_cells_cpp_from_core} cells.") # 打印设置完成信息

    bc_defs_cpp, wl_ts_data_cpp, discharge_ts_data_cpp = prepare_boundary_conditions_for_cpp(params) # 准备C++边界条件
    model_core.setup_boundary_conditions_cpp(bc_defs_cpp, wl_ts_data_cpp, discharge_ts_data_cpp) # 设置边界条件
    print("Python: Boundary conditions set.") # 打印设置完成信息

    print("\nPython: Starting C++ simulation...") # 打印开始模拟信息
    output_counter = 0 # 初始化输出计数器
    next_output_time = model_core.get_current_time() # 初始化下一个输出时间

    # --- 准备VTK输出所需的静态网格信息 ---
    points_for_vtk = np.zeros((mesh_cpp_ptr.get_num_nodes(), 3)) # 初始化VTK节点坐标数组
    for i in range(mesh_cpp_ptr.get_num_nodes()): # 遍历节点
        node = mesh_cpp_ptr.get_node(i) # 获取节点对象
        points_for_vtk[i, 0] = node.x # 设置x坐标
        points_for_vtk[i, 1] = node.y # 设置y坐标
        points_for_vtk[i, 2] = node.z_bed # 将底高程设为Z坐标，以便在ParaView中查看地形
        # points_for_vtk[i, 2] = 0.0 # 或者对于纯2D可视化，设为0

    cells_connectivity_for_vtk = [] # 初始化VTK单元连接关系列表
    for i in range(num_cells_cpp_from_core): # 遍历单元
        cell = mesh_cpp_ptr.get_cell(i) # 获取单元对象
        if len(cell.node_ids) == 3: # 如果是三角形单元
            cells_connectivity_for_vtk.append(list(cell.node_ids)) # 添加节点ID列表
    # meshio期望的单元格式是 [("triangle", np.array(connectivity_list))]
    cells_for_vtk = [("triangle", np.array(cells_connectivity_for_vtk, dtype=int))] # 创建meshio单元格式

    # 创建输出目录 (使用config中定义的output_directory)
    vtk_output_dir = params['output_directory'] # 获取VTK输出目录
    os.makedirs(vtk_output_dir, exist_ok=True) # 创建目录 (如果不存在)
    print(f"VTK files will be saved to: {os.path.abspath(vtk_output_dir)}") # 打印VTK文件保存路径

    # --- 模拟循环与VTK输出 ---
    simulation_active = True # 初始化模拟活动标志
    while simulation_active: # 当模拟活动时循环
        current_t_cpp = model_core.get_current_time() # 获取当前C++时间
        # 检查是否到达输出时间点
        if current_t_cpp >= next_output_time - NUMERICAL_EPSILON or output_counter == 0 : # 第一次也输出
            print(f"  Python: Output at t = {current_t_cpp:.3f} s, step = {model_core.get_step_count()}") # 打印输出信息
            U_current_py = model_core.get_U_state_all_py() # 获取当前守恒量
            eta_current_py = model_core.get_eta_previous_py() # 获取当前水位

            h_current = U_current_py[:, 0] # 获取水深
            hu_current = U_current_py[:, 1] # 获取hu
            hv_current = U_current_py[:, 2] # 获取hv
            # 避免除零来计算速度
            u_current = np.divide(hu_current, h_current, out=np.zeros_like(hu_current), where=h_current > params['min_depth'] / 10.0) # 计算u速度
            v_current = np.divide(hv_current, h_current, out=np.zeros_like(hv_current), where=h_current > params['min_depth'] / 10.0) # 计算v速度
            velocity_magnitude = np.sqrt(u_current**2 + v_current**2) # 计算流速大小

            cell_data_for_vtk = { # 准备VTK单元数据
                "water_depth": h_current, # 水深
                "eta": eta_current_py, # 水位
                "velocity_u": u_current, # u速度
                "velocity_v": v_current, # v速度
                "velocity_magnitude": velocity_magnitude # 流速大小
                # 可以添加 "hu": hu_current, "hv": hv_current 如果需要
            }
            vtk_filepath = os.path.join(vtk_output_dir, f"results_t{output_counter:04d}.vtu") # 构建VTK文件路径
            save_results_to_vtk(vtk_filepath, points_for_vtk, cells_for_vtk, cell_data_for_vtk) # 保存结果到VTK文件

            output_counter += 1 # 增加输出计数器
            # 只有当模拟还在进行时才更新下一个输出时间
            if current_t_cpp < params['total_time'] - NUMERICAL_EPSILON: # 如果当前时间小于总时间
                next_output_time += params['output_dt'] # 更新下一个输出时间
                # 确保下一个输出时间不会超过总时间太多 (以总时间为上限)
                if next_output_time > params['total_time'] + NUMERICAL_EPSILON: # 如果超过总时间
                     next_output_time = params['total_time'] # 设为总时间
            else: # 如果已达到或超过总时间
                 # 如果这是在总时间点或之后，我们可能只需要最后一次输出
                 pass # 不再增加 next_output_time

        simulation_active = model_core.advance_one_step() # 执行一步C++模拟并更新活动标志
        if model_core.is_simulation_finished() and simulation_active: # 如果C++认为结束了但Python循环还想继续（通常不会）
            simulation_active = False # 强制Python循环结束
            # 可能需要在循环外再做一次最终输出，以确保总时间点的数据被精确捕获

    # --- 确保在总时间点进行最后一次输出 (如果模拟恰好在输出点之间结束) ---
    current_t_cpp = model_core.get_current_time() # 获取最终C++时间
    # 检查最后一次输出的时间是否足够接近总时间
    # output_counter > 0 确保至少有过一次常规输出
    needs_final_explicit_output = True # 默认需要最终输出
    if output_counter > 0: # 如果至少有过一次输出
        last_vtk_filename = os.path.join(vtk_output_dir, f"results_t{output_counter-1:04d}.vtu") # 获取上一个VTK文件名
        # 可以检查上一个输出的时间是否非常接近当前总时间
        # 这里的逻辑是，如果 advance_one_step 返回 false，说明模拟结束了。
        # 我们应该在 advance_one_step 返回 false 后，获取并保存最后的状态。
        # 前面的循环在 advance_one_step 返回 false 后会退出。
        # 所以这里获取的状态就是最终状态。

    print(f"  Python: Final Output at t = {current_t_cpp:.3f} s, step = {model_core.get_step_count()}") # 打印最终输出信息
    U_final_py = model_core.get_U_state_all_py() # 获取最终守恒量
    eta_final_py = model_core.get_eta_previous_py() # 获取最终水位
    h_final = U_final_py[:, 0] # 获取最终水深
    hu_final = U_final_py[:, 1] # 获取最终hu
    hv_final = U_final_py[:, 2] # 获取最终hv
    u_final = np.divide(hu_final, h_final, out=np.zeros_like(hu_final), where=h_final > params['min_depth'] / 10.0) # 计算最终u速度
    v_final = np.divide(hv_final, h_final, out=np.zeros_like(hv_final), where=h_final > params['min_depth'] / 10.0) # 计算最终v速度
    velocity_magnitude_final = np.sqrt(u_final**2 + v_final**2) # 计算最终流速大小

    final_cell_data_for_vtk = { # 准备最终VTK单元数据
        "water_depth": h_final, # 水深
        "eta": eta_final_py, # 水位
        "velocity_u": u_final, # u速度
        "velocity_v": v_final, # v速度
        "velocity_magnitude": velocity_magnitude_final # 流速大小
    }
    vtk_filepath_final = os.path.join(vtk_output_dir, f"results_t{output_counter:04d}_final.vtu") # 构建最终VTK文件路径
    save_results_to_vtk(vtk_filepath_final, points_for_vtk, cells_for_vtk, final_cell_data_for_vtk) # 保存最终结果到VTK文件

    print("Python: C++ simulation finished.") # 打印模拟结束信息
    print(f"  Final time: {model_core.get_current_time():.3f} s") # 打印最终时间
    print(f"  Total steps: {model_core.get_step_count()}") # 打印总步数