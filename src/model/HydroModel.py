# src/model/HydroModel.py
import numpy as np
import time as pytime
import os
import traceback
import pprint
import pandas as pd  # 用于读取CSV

# 导入C++模块
try:
    import hydro_model_cpp  # 导入编译好的C++模块
except ImportError:
    print("错误: 未找到 C++ 核心模块 'hydro_model_cpp'。请确保已编译并正确放置。")  # 打印错误
    exit()  # 退出

# 从项目中导入Python工具
from .MeshData import Mesh  # 假设Python MeshData仍然用于网格的初始表示和结果保存时的几何信息
# Python VFR不再需要为核心模型，但可能用于初始条件准备（如果C++不完全接管）
# from .WettingDrying import VFRCalculator as PyVFRCalculator
try:
    # 尝试从 src.initialization 导入正确的函数名
    from src.initialization import load_mesh_data_structure_cpp
except ImportError as e:
    print(f"错误：无法导入必要的模块。请检查文件路径和类定义是否正确。")
    print(f"详细错误: {e}")
    # 打印当前工作目录和sys.path，帮助调试路径问题
    import os
    import sys
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python搜索路径 (sys.path): {sys.path}")
    exit()


class HydroModel:  # 定义水动力模型类
    def __init__(self, mesh_py: Mesh, parameters: dict):  # 初始化方法，mesh_py是Python的Mesh对象
        self.mesh_py = mesh_py  # 存储Python的Mesh对象，主要用于结果保存时的几何信息
        self.parameters = parameters  # 存储完整的参数字典
        self.min_depth = parameters.get('physical_parameters', {}).get('min_depth', 1e-6)  # 获取最小水深 (Python端也存一份，可能有用)
        self.total_time = parameters.get('simulation_control', {}).get('total_time', 10.0)  # 获取总模拟时长
        self.output_dt = parameters.get('simulation_control', {}).get('output_dt', 1.0)  # 获取输出时间间隔

        # --- 1. 创建和加载C++网格 ---
        print("  Python: 开始加载网格数据到C++核心...")  # 打印信息
        num_cells_py = len(self.mesh_py.cells)  # 获取Python端单元数量
        manning_default = parameters.get('model_parameters', {}).get('manning_n_default', 0.025)  # 获取默认曼宁值
        cell_manning_values_np = np.full(num_cells_py, manning_default, dtype=np.float64)  # 创建曼宁值数组

        mesh_cpp_obj = load_mesh_data_structure_cpp(  # 调用函数加载C++网格
            parameters['file_paths']['node_file'],  # 节点文件路径
            parameters['file_paths']['cell_file'],  # 单元文件路径
            parameters['file_paths']['edge_file'],  # 边文件路径
            cell_manning_values_np  # 单元曼宁值
        )  # 结束调用
        if mesh_cpp_obj is None:  # 如果加载失败
            raise RuntimeError("Python: Failed to load mesh data into C++ core.")  # 抛出运行时错误
        print("  Python: C++网格对象创建并加载完毕。")  # 打印信息

        # --- 2. 实例化C++水动力模型核心 ---
        self.cpp_hydro_model_core = hydro_model_cpp.HydroModelCore_cpp(mesh_cpp_obj)  # 创建C++水动力模型核心对象
        print("  Python: C++ HydroModelCore_cpp 实例化完毕。")  # 打印信息

        # --- 3. 传递参数给C++核心 ---
        physical_params = parameters.get('physical_parameters', {})  # 获取物理参数
        sim_control = parameters.get('simulation_control', {})  # 获取模拟控制参数
        numerical_schemes_config = parameters.get('numerical_schemes', {})  # 获取数值方案配置

        self.cpp_hydro_model_core.set_simulation_parameters(  # 调用C++方法设置模拟参数
            physical_params.get('gravity', 9.81),  # 重力加速度
            physical_params.get('min_depth', 1e-6),  # 最小水深
            sim_control.get('cfl_number', 0.5),  # CFL数
            sim_control.get('total_time', 10.0),  # 总模拟时长
            sim_control.get('output_dt', 1.0),  # 输出时间间隔
            sim_control.get('max_dt', sim_control.get('output_dt', 1.0))  # 最大时间步长
        )
        print("  Python: 模拟参数已传递给C++核心。")  # 打印信息

        recon_scheme_str = numerical_schemes_config.get('reconstruction_scheme', "FIRST_ORDER").upper()  # 获取重构方案字符串
        recon_scheme_cpp = hydro_model_cpp.ReconstructionScheme_cpp.FIRST_ORDER  # 默认重构方案
        if recon_scheme_str == "SECOND_ORDER_LIMITED":  # 如果是二阶限制
            recon_scheme_cpp = hydro_model_cpp.ReconstructionScheme_cpp.SECOND_ORDER_LIMITED  # 设置为二阶限制方案

        riemann_solver_str = numerical_schemes_config.get('riemann_solver', "HLLC").upper()  # 获取黎曼求解器字符串
        riemann_solver_cpp = hydro_model_cpp.RiemannSolverType_cpp.HLLC  # 默认黎曼求解器

        time_scheme_str = numerical_schemes_config.get('time_scheme', "FORWARD_EULER").upper()  # 获取时间积分方案字符串
        time_scheme_cpp = hydro_model_cpp.TimeScheme_cpp.FORWARD_EULER  # 默认时间积分方案
        if time_scheme_str == "RK2_SSP":  # 如果是RK2_SSP
            time_scheme_cpp = hydro_model_cpp.TimeScheme_cpp.RK2_SSP  # 设置为RK2_SSP方案

        self.cpp_hydro_model_core.set_numerical_schemes(recon_scheme_cpp, riemann_solver_cpp,
                                                        time_scheme_cpp)  # 调用C++方法设置数值方案
        print(
            f"  Python: 数值方案 (Recon: {recon_scheme_str}, Riemann: {riemann_solver_str}, Time: {time_scheme_str}) 已传递给C++核心。")  # 打印信息

        # --- 4. 准备并传递初始条件 (只传递U，eta由C++计算) ---
        self.U = np.zeros((len(self.mesh_py.cells), 3), dtype=np.float64)  # 初始化Python端的U数组 (主要用于初始计算)
        initial_conditions_config = parameters.get('initial_conditions', {})  # 获取初始条件配置
        self._initialize_conserved_variables_for_cpp(initial_conditions_config)  # 调用辅助函数计算初始U

        U_initial_list = self.U.tolist()  # 将初始U转换为list of lists
        self.cpp_hydro_model_core.set_initial_conditions_cpp(U_initial_list)  # 调用C++方法设置初始U
        print("  Python: 初始守恒量U已传递给C++核心 (初始eta将在C++内部计算)。")  # 打印信息

        # Python端的eta_previous现在将从C++获取，在保存结果时或按需获取
        self.eta_previous = np.array(self.cpp_hydro_model_core.get_eta_previous_internal_copy())  # 获取C++计算的初始eta

        # --- 5. 准备并传递边界条件 ---
        bc_config_py = parameters.get('boundary_conditions', {})  # 获取Python边界条件配置
        file_paths_config = parameters.get('file_paths', {})  # 获取文件路径配置

        cpp_bc_definitions = {}  # 初始化C++边界定义字典
        for marker_str, definition_py in bc_config_py.items():  # 遍历Python配置
            # (转换逻辑同上一回答中的建议)
            if marker_str == 'default':
                marker_int = 0  # 特殊处理default标记
            else:
                try:
                    marker_int = int(marker_str)  # 转换标记为整数
                except ValueError:
                    print(f"警告: 无法将边界标记 '{marker_str}' 转换为整数。跳过。"); continue  # 打印警告并继续

            bc_def_cpp = hydro_model_cpp.BoundaryDefinition_cpp()  # 创建C++边界定义对象
            type_str = definition_py.get('type', 'WALL').upper()  # 获取类型字符串并转大写
            if type_str == "WALL":
                bc_def_cpp.type = hydro_model_cpp.BoundaryType_cpp.WALL  # 设置类型
            elif type_str == "WATERLEVEL_TIMESERIES":
                bc_def_cpp.type = hydro_model_cpp.BoundaryType_cpp.WATERLEVEL_TIMESERIES  # 设置类型
            elif type_str == "TOTAL_DISCHARGE_TIMESERIES":
                bc_def_cpp.type = hydro_model_cpp.BoundaryType_cpp.TOTAL_DISCHARGE_TIMESERIES  # 设置类型
            elif type_str == "FREE_OUTFLOW":
                bc_def_cpp.type = hydro_model_cpp.BoundaryType_cpp.FREE_OUTFLOW  # 设置类型
            else:
                print(
                    f"警告: 边界标记 {marker_int} 的未知类型 '{type_str}'。默认为 WALL。"); bc_def_cpp.type = hydro_model_cpp.BoundaryType_cpp.WALL  # 打印警告并默认为墙体
            cpp_bc_definitions[marker_int] = bc_def_cpp  # 添加到字典

        cpp_elev_ts_data = self._prepare_timeseries_for_cpp(  # 调用辅助函数准备水位时间序列数据
            file_paths_config.get('boundary_timeseries_elevation_file'), "elev_"  # 文件路径和前缀
        )
        cpp_discharge_ts_data = self._prepare_timeseries_for_cpp(  # 调用辅助函数准备流量时间序列数据
            file_paths_config.get('boundary_timeseries_discharge_file'), "flux_"  # 文件路径和前缀
        )

        self.cpp_hydro_model_core.setup_boundary_conditions_cpp(  # 调用C++方法设置边界条件
            cpp_bc_definitions, cpp_elev_ts_data, cpp_discharge_ts_data  # 传入参数
        )
        print("  Python: 边界条件配置已传递给C++核心。")  # 打印信息
        print("水动力模型 (Python控制器 -> C++核心) 初始化完成。")  # 打印完成信息

    def _initialize_conserved_variables_for_cpp(self, initial_conditions: dict):  # 初始化守恒量以供C++使用 (辅助方法)
        """仅计算初始U，基于Python的Mesh对象获取z_bed_centroid。"""
        initial_condition_type = initial_conditions.get("type", "uniform_elevation")  # 获取初始条件类型
        print(f"  Python: 计算初始守恒量U，类型: {initial_condition_type}")  # 打印信息

        if initial_condition_type == "uniform_elevation":  # 如果是均匀水位
            eta0 = initial_conditions.get('water_surface_elevation', None)  # 获取初始水位
            if eta0 is None: eta0 = -float('inf')  # 如果未指定则设为极小值
            for i, cell_py in enumerate(self.mesh_py.cells):  # 遍历Python网格单元
                actual_h = max(0.0, eta0 - cell_py.z_bed_centroid)  # 计算实际水深
                if actual_h < self.min_depth: actual_h = 0.0  # 应用最小水深阈值
                self.U[i, 0] = actual_h  # 设置水深
                self.U[i, 1:] = 0.0  # 设置动量为0
        elif initial_condition_type == "uniform_depth":  # 如果是均匀水深
            h0 = initial_conditions.get('water_depth', 0.0)  # 获取初始水深
            if h0 < self.min_depth: h0 = 0.0  # 应用最小水深阈值
            for i in range(len(self.mesh_py.cells)):  # 遍历单元索引
                self.U[i, 0] = h0  # 设置水深
                self.U[i, 1:] = 0.0  # 设置动量为0
        else:  # 其他未知类型
            raise ValueError(f"未知的初始条件类型: {initial_condition_type}")  # 抛出值错误

    def _prepare_timeseries_for_cpp(self, filepath: str | None, expected_prefix: str) -> dict:  # 准备时间序列数据以供C++使用 (辅助方法)
        """加载CSV并转换为C++期望的格式。"""
        cpp_ts_data = {}  # 初始化C++时间序列数据字典
        if filepath and os.path.exists(filepath):  # 如果文件路径有效且存在
            try:  # 尝试读取CSV
                df_py = pd.read_csv(filepath)  # 读取CSV
                if 'time' not in df_py.columns:  # 检查是否有time列
                    print(f"警告: 时间序列文件 '{filepath}' 缺少 'time' 列。")  # 打印警告
                    return cpp_ts_data  # 返回空字典

                # 确保时间是单调递增的
                if not pd.Series(df_py['time']).is_monotonic_increasing:  # 如果时间非单调递增
                    print(f"警告: 时间序列文件 '{filepath}' 中的 'time' 列未排序。将尝试排序。")  # 打印警告
                    df_py = df_py.sort_values(by='time').reset_index(drop=True)  # 排序并重置索引

                for col_name_py in df_py.columns:  # 遍历列
                    if col_name_py.startswith(expected_prefix):  # 如果列名以前缀开头
                        try:  # 尝试提取标记
                            marker_py = int(col_name_py.split('_')[-1])  # 提取标记
                            points_for_marker_py = []  # 初始化点列表
                            for time_idx, time_val_py in enumerate(df_py['time']):  # 遍历时间
                                data_val_py = df_py[col_name_py].iloc[time_idx]  # 获取数据值
                                if not np.isnan(data_val_py):  # 如果数据非NaN
                                    points_for_marker_py.append(
                                        hydro_model_cpp.TimeseriesPoint_cpp(float(time_val_py),
                                                                                      float(data_val_py))  # 创建并添加点
                                    )
                            if points_for_marker_py:  # 如果列表非空
                                cpp_ts_data[marker_py] = points_for_marker_py  # 添加到字典
                        except ValueError:  # 捕获值错误
                            print(f"警告: 无法从列名 {col_name_py} (文件: {filepath}) 解析时间序列标记。")  # 打印警告
            except Exception as e_csv:  # 捕获读取CSV错误
                print(f"警告: 读取时间序列文件 '{filepath}' 失败: {e_csv}")  # 打印警告
        return cpp_ts_data  # 返回准备好的时间序列数据

    def run_simulation(self):  # 运行模拟 (Python控制循环版本)
        output_dir = self.parameters.get('file_paths', {}).get("output_directory", "simulation_output")  # 获取输出目录
        if not os.path.exists(output_dir): os.makedirs(output_dir); print(f"已创建输出目录: {output_dir}")  # 创建目录

        # 使用从C++核心获取的参数打印信息
        sim_total_time_cpp = self.cpp_hydro_model_core.get_total_time()  # 获取C++总模拟时长
        sim_output_dt_cpp = self.cpp_hydro_model_core.get_output_dt()  # 获取C++输出间隔
        print(
            f"\n开始模拟 (C++核心驱动)，总时长: {sim_total_time_cpp:.2f} s, 输出间隔: {sim_output_dt_cpp:.2f} s")  # 打印开始信息

        # 保存初始状态 (从C++核心获取)
        self.U = np.array(self.cpp_hydro_model_core.get_U_state_all_internal_copy())  # 获取初始U状态
        self.eta_previous = np.array(self.cpp_hydro_model_core.get_eta_previous_internal_copy())  # 获取初始eta状态
        self._save_results(0.0, 0, output_dir)  # 保存初始结果

        sim_start_wall_time = pytime.time()  # 记录模拟开始墙上时间
        next_output_time = 0.0  # 初始化下一输出时间
        if sim_output_dt_cpp > 1e-9 and sim_output_dt_cpp < sim_total_time_cpp:  # 如果设置了有效输出间隔且小于总时间
            next_output_time = sim_output_dt_cpp  # 设置下一输出时间
        elif sim_output_dt_cpp > 1e-9 and sim_output_dt_cpp >= sim_total_time_cpp:  # 如果输出间隔大于等于总时间
            next_output_time = sim_total_time_cpp  # 下一输出时间即为总时间
        else:  # output_dt <= 0, only output at end
            next_output_time = sim_total_time_cpp + 1.0  # 确保只在最后触发 (比总时间大一点)

        while not self.cpp_hydro_model_core.is_simulation_finished():  # 当C++模拟未结束时循环
            can_continue = self.cpp_hydro_model_core.advance_one_step()  # 调用C++执行一步
            if not can_continue:  # 如果不能继续
                break  # 跳出循环

            current_time_cpp = self.cpp_hydro_model_core.get_current_time()  # 获取C++当前时间
            step_count_cpp = self.cpp_hydro_model_core.get_step_count()  # 获取C++步数
            last_dt_cpp = self.cpp_hydro_model_core.get_last_dt()  # 获取C++上一时间步长

            if current_time_cpp >= next_output_time - 1e-9 or self.cpp_hydro_model_core.is_simulation_finished():  # 如果达到输出时间或模拟结束
                self.U = np.array(self.cpp_hydro_model_core.get_U_state_all_internal_copy())  # 获取U状态
                self.eta_previous = np.array(self.cpp_hydro_model_core.get_eta_previous_internal_copy())  # 获取eta状态

                wall_time_elapsed = pytime.time() - sim_start_wall_time  # 计算已过墙上时间
                print(  # 打印进度
                    f"时间: {current_time_cpp:>{8}.{3}f} s | 步数: {step_count_cpp:>{6}} | dt: {last_dt_cpp:{8}.{2}e} s | 已耗时: {wall_time_elapsed:{6}.{1}f} s"
                )
                self._save_results(current_time_cpp, step_count_cpp, output_dir)  # 保存结果

                if self.cpp_hydro_model_core.is_simulation_finished(): break  # 如果模拟结束则跳出

                if sim_output_dt_cpp > 1e-9:  # 如果设置了有效输出间隔
                    current_output_num = int(round(current_time_cpp / sim_output_dt_cpp + 1e-6))  # 计算当前输出周期数
                    next_output_time = min(sim_total_time_cpp, (current_output_num + 1) * sim_output_dt_cpp)  # 计算下一输出时间
                    if next_output_time - current_time_cpp < 1e-6 and next_output_time < sim_total_time_cpp:  # 如果下一输出时间过近
                        next_output_time = min(sim_total_time_cpp,
                                               (current_output_num + 2) * sim_output_dt_cpp)  # 调整到更远的输出时间
                # else: next_output_time 保持不变 (在循环开始前已设为 total_time + 1.0)

        sim_end_wall_time = pytime.time()  # 记录模拟结束墙上时间
        final_time_cpp = self.cpp_hydro_model_core.get_current_time()  # 获取最终C++时间
        final_steps_cpp = self.cpp_hydro_model_core.get_step_count()  # 获取最终C++步数
        print(f"\n模拟完成于时间 {final_time_cpp:.3f} s.")  # 打印结束信息
        print(f"总墙上计算时间: {sim_end_wall_time - sim_start_wall_time:.2f} s.")  # 打印总耗时
        print(f"总计算步数: {final_steps_cpp}")  # 打印总步数

    def _save_results(self, time_val, step_num, output_dir):  # 保存结果 (与您之前的版本基本一致，确保使用self.mesh_py)
        try:  # 尝试导入meshio
            import meshio  # 导入meshio
        except ImportError:  # 捕获导入错误
            print("警告: 未找到 'meshio'。无法保存 VTK 文件。")  # 打印警告
            return  # 返回

        print(f"    保存时间 {time_val:.3f} s (步数 {step_num}) 的结果...")  # 打印保存信息
        try:  # 尝试保存
            # 使用 self.mesh_py (Python的Mesh对象) 获取几何信息
            points_3d = np.array([[node.x, node.y, node.z_bed] for node in self.mesh_py.nodes],
                                 dtype=np.float64)  # 创建节点三维坐标数组
            triangles = np.array([[node.id for node in cell.nodes] for cell in self.mesh_py.cells],
                                 dtype=int)  # 创建单元节点连接数组

            # 使用 self.U 和 self.eta_previous (已从C++获取并更新)
            h_vals = self.U[:, 0].copy()  # 复制当前水深数组
            hu_vals = self.U[:, 1].copy()  # 复制当前x方向动量数组
            hv_vals = self.U[:, 2].copy()  # 复制当前y方向动量数组

            u_vals = np.zeros_like(h_vals)  # 初始化u速度数组
            v_vals = np.zeros_like(h_vals)  # 初始化v速度数组
            non_dry_mask = h_vals > self.min_depth  # 创建湿单元掩码
            if np.any(non_dry_mask):  # 如果存在湿单元
                h_div = h_vals[non_dry_mask] + 1e-12  # 安全水深
                u_vals[non_dry_mask] = hu_vals[non_dry_mask] / h_div  # 计算u
                v_vals[non_dry_mask] = hv_vals[non_dry_mask] / h_div  # 计算v

            eta_vals_calc = self.eta_previous.copy()  # 复制水位数组
            z_bed_cell = np.array([cell.z_bed_centroid for cell in self.mesh_py.cells], dtype=np.float64)  # 获取单元形心底高程

            # 获取糙率: 假设self.parameters中存储了用于初始化的糙率值或配置
            # 如果C++核心修改了糙率，则需要一个getter从C++获取
            # 这里简单地使用初始配置的默认值，并假设它在Python端可用
            manning_n_default_for_output = self.parameters.get('model_parameters', {}).get('manning_n_default',
                                                                                           0.025)  # 获取默认曼宁值
            manning_n_cell = np.full(len(self.mesh_py.cells), manning_n_default_for_output, dtype=np.float64)  # 创建曼宁值数组
            # 如果您的C++ Mesh_cpp对象存储了每个单元的糙率，并且Python的self.mesh_py也同步了，可以这样：
            # manning_n_cell = np.array([cell.manning_n for cell in self.mesh_py.cells], dtype=np.float64)

            point_data_dict = {  # 创建节点数据字典
                'bed_elevation_node': np.array([node.z_bed for node in self.mesh_py.nodes], dtype=np.float64)  # 节点底高程
            }

            cell_data_dict = {  # 创建单元数据字典
                'water_depth': [h_vals],  # 水深
                'velocity_u': [u_vals],  # u速度
                'velocity_v': [v_vals],  # v速度
                'water_surface': [eta_vals_calc],  # 水位
                'bed_elevation_cell': [z_bed_cell],  # 单元底高程
                'manning_n': [manning_n_cell]  # 曼宁系数
            }

            cells_info = [("triangle", triangles)]  # 定义单元信息

            mesh_to_write = meshio.Mesh(points_3d, cells_info, point_data=point_data_dict,
                                        cell_data=cell_data_dict)  # 创建meshio对象
            filename = os.path.join(output_dir, f"result_{step_num:06d}.vtk")  # 构建文件名
            mesh_to_write.write(filename, file_format="vtk", binary=True)  # 写入文件
            print(f"    结果已保存到: {filename}")  # 打印保存路径
        except Exception as e:  # 捕获保存错误
            print(f"错误: 保存结果到 VTK 时出错于时间 {time_val:.3f} s: {e}")  # 打印错误信息
            traceback.print_exc()  # 打印详细堆栈