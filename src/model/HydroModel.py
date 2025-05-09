# src/model/HydroModel.py
import numpy as np
import time as pytime
import os # 导入os模块
import traceback # 确保导入 traceback
import pprint # 导入 pprint

from .MeshData import Mesh, Cell, HalfEdge, Node # 从 .MeshData 导入 Mesh, Cell, HalfEdge, Node 类
from .TimeIntegrator import TimeIntegrator, TimeIntegrationSchemes # 从 .TimeIntegrator 导入 TimeIntegrator, TimeIntegrationSchemes 类
from .FluxCalculator import FluxCalculator, RiemannSolvers # 从 .FluxCalculator 导入 FluxCalculator, RiemannSolvers 类
from .Reconstruction import Reconstruction, ReconstructionSchemes # 从 .Reconstruction 导入 Reconstruction, ReconstructionSchemes 类
from .SourceTerms import SourceTermCalculator # 从 .SourceTerms 导入 SourceTermCalculator 类
from .BoundaryConditions import BoundaryConditionHandler # 导入边界处理器
from .WettingDrying import VFRCalculator # **** 添加这一行导入 ****


class HydroModel: # 定义水动力模型类
    def __init__(self, mesh: Mesh, parameters: dict): # 初始化方法
        self.mesh = mesh # 网格对象
        self.parameters = parameters # 存储完整的参数字典

        # --- 从 physical_parameters 加载重力加速度 ---
        physical_params = parameters.get('physical_parameters', {}) # 获取物理参数子字典，如果不存在则为空字典
        self.gravity = physical_params.get('gravity', 9.81) # 从子字典获取重力加速度

        # --- 统一定义和获取最小水深 (从 physical_parameters 加载) ---
        self.min_depth = physical_params.get('min_depth', 1e-6) # **从 physical_params 获取 min_depth**
        print(f"  使用统一最小水深阈值: {self.min_depth}") # 打印使用的值


        # --- 其他初始化代码 (加载数值方案、模型参数等) ---
        numerical_schemes = parameters.get('numerical_schemes', {}) # 获取数值方案子字典
        model_params = parameters.get('model_parameters', {}) # 获取模型参数子字典

        self.num_vars = 3 # 变量数
        self.U = np.zeros((len(self.mesh.cells), self.num_vars), dtype=np.float64) # 初始化守恒量数组
        self.eta_previous = np.zeros(len(self.mesh.cells), dtype=np.float64) # 新增：存储上一步的水位
        self.vfr_calculator = VFRCalculator(min_depth=self.min_depth) # 初始化VFR计算器
        # --- 初始化顺序调整：先初始化U，再计算初始eta ---
        initial_conditions = parameters.get('initial_conditions', {})
        # --- 添加打印语句 ---
        print("[*] HydroModel.__init__: 接收到的 initial_conditions 内容:")
        pprint.pprint(initial_conditions)
        print("-" * 50)
        # --- 打印结束 ---
        self._initialize_conserved_variables(initial_conditions) # **先计算初始 U**
        self._initialize_initial_eta() # **再根据初始 U 计算初始 eta**

        # --- 初始化糙率 ---
        default_n = model_params.get('manning_n_default', 0.025) # 获取默认曼宁系数
        self._initialize_manning(default_n) # 初始化曼宁系数


        # --- *** 实例化数值方法组件 (传入统一的 self.min_depth) *** ---
        # 1. 重构器 (从 numerical_schemes 读取)
        reconstruction_scheme_str = numerical_schemes.get('reconstruction_scheme',
                                                          "FIRST_ORDER").upper() # **从 numerical_schemes 获取**
        # ... (try-except 不变) ...
        recon_enum = ReconstructionSchemes.FIRST_ORDER # 默认值
        try: # 尝试获取枚举值
            recon_enum = ReconstructionSchemes[reconstruction_scheme_str] # 获取重构方案枚举
        except KeyError: # 捕获键错误
            print(f"警告: 无效的重构方案 '{reconstruction_scheme_str}'. 使用 FIRST_ORDER.") # 打印警告
        self.reconstructor = Reconstruction( # 创建实例
            scheme=recon_enum, mesh=self.mesh, gravity=self.gravity, # 传入方案、网格、重力
            min_depth=self.min_depth # **传入统一值**
        ) # 结束重构器实例化

        # 2. 通量计算器 (从 numerical_schemes 读取)
        riemann_solver_str = numerical_schemes.get('riemann_solver', "HLLC").upper() # **从 numerical_schemes 获取**
        # ... (try-except 不变) ...
        solver_enum = RiemannSolvers.HLLC # 默认值
        try: # 尝试获取枚举值
            solver_enum = RiemannSolvers[riemann_solver_str] # 获取黎曼求解器枚举
        except KeyError: # 捕获键错误
            print(f"警告: 无效的黎曼求解器 '{riemann_solver_str}'. 使用 HLLC.") # 打印警告
        self.flux_calculator = FluxCalculator( # 创建实例
            solver_type=solver_enum, gravity=self.gravity, # 传入类型、重力
            min_depth=self.min_depth # **传入统一值**
        ) # 结束通量计算器实例化

        # 3. 源项计算器 (传入统一值)
        self.source_term_calculator = SourceTermCalculator( # 创建实例
            gravity=self.gravity, # 传入重力
            min_depth=self.min_depth # **传入统一值 (修改FluxCalculator构造函数)**
        ) # 结束源项计算器实例化

        # 4. 边界条件处理器 (从 boundary_conditions 和 file_paths 读取)
        bc_config = parameters.get('boundary_conditions', {}) # 获取边界配置
        file_paths_config = parameters.get('file_paths', {}) # 获取文件路径配置
        # 注意：你的yaml示例中，时间序列文件路径在 file_paths 下，但上个回答假设它们直接在 parameters 下
        elev_file_path = file_paths_config.get('boundary_timeseries_elevation_file') # **从 file_paths_config 获取**
        discharge_file_path = file_paths_config.get('boundary_timeseries_discharge_file') # **从 file_paths_config 获取**

        self.boundary_handler = BoundaryConditionHandler( # 创建实例
            mesh=self.mesh, flux_calculator=self.flux_calculator, reconstructor=self.reconstructor, # 传入网格、通量计算器、重构器
            gravity=self.gravity, # 传入重力
            min_depth=self.min_depth, # **传入统一值**
            bc_definitions=bc_config, # 传入边界定义
            elev_timeseries_filepath=elev_file_path, # 传入水位时间序列文件路径
            discharge_timeseries_filepath=discharge_file_path # 传入流量时间序列文件路径
        ) # 结束边界条件处理器实例化

        # 5. 时间积分器 (从 numerical_schemes 读取)
        time_scheme_str = numerical_schemes.get('time_scheme', "FORWARD_EULER").upper() # **从 numerical_schemes 获取**
        # ... (try-except 不变) ...
        time_enum = TimeIntegrationSchemes.FORWARD_EULER # 默认值
        try: # 尝试获取枚举值
            time_enum = TimeIntegrationSchemes[time_scheme_str] # 获取时间积分方案枚举
        except KeyError: # 捕获键错误
            print(f"警告: 无效的时间积分方案 '{time_scheme_str}'. 使用 FORWARD_EULER.") # 打印警告
        self.time_integrator = TimeIntegrator( # 创建实例
            scheme=time_enum, rhs_function=self._calculate_rhs_explicit_part, # 传入方案、显式RHS函数
            friction_function=self._apply_friction_semi_implicit, num_vars=self.num_vars # 传入摩擦函数、变量数
        ) # 结束时间积分器实例化
        # 6. VFR 计算器 (如果需要，在这里初始化)
        # from .WettingDrying import VFRCalculator # 导入
        # self.vfr_calculator = VFRCalculator(min_depth=self.min_depth) # 初始化并传入统一值

        # --- 模拟控制参数 (从 simulation_control 读取) ---
        sim_control = parameters.get('simulation_control', {}) # 获取模拟控制子字典
        self.cfl_number = sim_control.get('cfl_number', 0.5) # CFL数
        self.total_time = sim_control.get('total_time', 10.0) # 总时间
        self.output_dt = sim_control.get('output_dt', 1.0) # 输出间隔
        self.max_dt = sim_control.get('max_dt', self.output_dt) # 最大步长
        # 如果 output_dt 小于 0 或未设置，可能表示只输出最后结果
        if self.output_dt <= 0: # 如果输出间隔小于等于0
            print("信息: output_dt <= 0，模型将只在模拟结束时输出结果。") # 打印信息
            self.output_dt = self.total_time # 将输出间隔设为总时间

        print(f"水动力模型初始化完成:") # 打印完成信息
        # ... (其他打印信息，例如打印加载的方案) ...
        print(f"  使用的重构方案: {self.reconstructor.scheme.value}") # 打印重构方案
        print(f"  使用的黎曼求解器: {self.flux_calculator.solver_type.value}") # 打印黎曼求解器
        print(f"  使用的时间积分方案: {self.time_integrator.scheme.value}") # 打印时间积分方案
        print(f"  配置的边界条件标记: {list(bc_config.keys()) if bc_config else '无'}") # 打印边界条件标记
        print(f"  统一最小水深阈值: {self.min_depth}") # 打印最小水深阈值

    # --- 重命名初始化函数，使其更清晰 ---
    def _initialize_initial_eta(self): # 计算初始水位
        """根据初始的守恒量 self.U 计算初始水位 self.eta_previous。"""
        print("  计算初始水位 eta...") # 打印计算初始水位信息
        if not hasattr(self, 'vfr_calculator'): # 确保 vfr_calculator 已初始化
            print("错误: VFRCalculator 未初始化，无法计算初始 eta。") # 打印错误
            # 或者在这里初始化: self.vfr_calculator = VFRCalculator(min_depth=self.min_depth)
            return # 返回

        for i, cell in enumerate(self.mesh.cells): # 遍历所有单元
            if self.U[i, 0] >= self.min_depth / 10: # 如果初始水深大于某个小阈值（这里取min_depth/10）
                try: # 尝试计算
                    nodes_with_z = sorted([(node, node.z_bed) for node in cell.nodes], key=lambda item: item[1]) # 按底高程对节点排序
                    cell_nodes_sorted = [item[0] for item in nodes_with_z] # 获取排序后的节点对象列表
                    b_sorted = [item[1] for item in nodes_with_z] # 获取排序后的底高程列表
                    self.eta_previous[i] = self.vfr_calculator.get_eta_from_h( # 调用VFR计算eta
                        self.U[i, 0], b_sorted, cell_nodes_sorted, cell.area, # 传入水深、排序高程、排序节点、面积
                        eta_previous_guess=None, # 第一次没有猜测值
                        cell_id_for_debug=f"{i}_init" # 传入调试ID
                    ) # 结束VFR调用
                except Exception as e: # 捕获计算异常
                    print(f"错误: 初始化单元 {i} 的 eta 时出错: {e}") # 打印错误信息
                    traceback.print_exc() # 打印详细错误堆栈
                    b_sorted_dry = sorted([n.z_bed for n in cell.nodes]) if cell.nodes else [-float('inf')] # 获取排序后的底高程（容错）
                    self.eta_previous[i] = b_sorted_dry[0] # 将eta设为最低点高程
            else: # 如果单元初始为干
                b_sorted_dry = sorted([n.z_bed for n in cell.nodes]) if cell.nodes else [-float('inf')] # 获取排序后的底高程（容错）
                self.eta_previous[i] = b_sorted_dry[0] # 将eta设为最低点高程
    def _initialize_conserved_variables_and_eta(self, initial_conditions: dict): # 初始化守恒量和水位
        # ... (你的代码实现) ...
        initial_condition_type = initial_conditions.get("type", "uniform_elevation") # 获取初始条件类型，默认为均匀水位
        # 根据类型计算 self.U
        if initial_condition_type == "uniform_elevation": # 如果是均匀水位
            eta0 = initial_conditions.get('water_surface_elevation', None) # 获取指定的水面高程
            if eta0 is None: eta0 = -float('inf') # 如果未指定，设为一个极小值（干底）
            for i, cell in enumerate(self.mesh.cells): # 遍历所有单元
                actual_h = max(0.0, eta0 - cell.z_bed_centroid) # 计算形心处的水深
                # 初始化时直接用 min_depth 判断初始水深是否有效
                if actual_h < self.min_depth: actual_h = 0.0 # 如果小于最小水深，设为0
                self.U[i, 0] = actual_h # 设置水深守恒量
                self.U[i, 1:] = 0.0 # 设置动量守恒量为0
        elif initial_condition_type == "uniform_depth": # 如果是均匀水深
            h0 = initial_conditions.get('water_depth', 0.0) # 获取指定的水深
            # 初始化时直接用 min_depth 判断
            if h0 < self.min_depth: h0 = 0.0 # 如果小于最小水深，设为0
            for i, cell in enumerate(self.mesh.cells): # 遍历所有单元
                self.U[i, 0] = h0 # 设置水深守恒量
                self.U[i, 1:] = 0.0 # 设置动量守恒量为0
        else: # 如果是未知的初始条件类型
            raise ValueError(f"未知的初始条件类型: {initial_condition_type}") # 抛出值错误

        # --- 计算初始 eta ---
        print("  计算初始水位 eta...") # 打印计算初始水位信息
        self.eta_previous = np.zeros(len(self.mesh.cells), dtype=np.float64) # 初始化eta数组
        for i, cell in enumerate(self.mesh.cells): # 遍历单元
            if self.U[i, 0] >= self.min_depth / 10: # 如果单元初始为湿 (用更小阈值触发计算)
                try: # 添加try-except保证健壮性
                    nodes_with_z = sorted([(node, node.z_bed) for node in cell.nodes], # 按高程排序节点
                                          key=lambda item: item[1]) # 按列表第二项（高程）排序
                    cell_nodes_sorted = [item[0] for item in nodes_with_z] # 获取排序后节点列表
                    b_sorted = [item[1] for item in nodes_with_z] # 获取排序后高程列表
                    # 第一次计算，没有 eta_previous_guess
                    self.eta_previous[i] = self.vfr_calculator.get_eta_from_h( # 调用VFR计算eta
                        self.U[i, 0], b_sorted, cell_nodes_sorted, cell.area, # 传入参数
                        eta_previous_guess=None, # 初始猜测为None
                        cell_id_for_debug=f"{i}_init" # 调试ID
                    ) # 结束VFR调用
                except Exception as e: # 捕获计算错误
                    print(f"错误: 初始化单元 {i} 的 eta 时出错: {e}") # 打印错误
                    b_sorted_dry = sorted([n.z_bed for n in cell.nodes]) if cell.nodes else [-float('inf')] # 容错处理
                    self.eta_previous[i] = b_sorted_dry[0] # 设为最低点

            else: # 如果单元初始为干
                b_sorted_dry = sorted([n.z_bed for n in cell.nodes]) if cell.nodes else [-float('inf')] # 获取排序高程
                self.eta_previous[i] = b_sorted_dry[0] # 初始eta设为最低点

    # --- _initialize_conserved_variables 和其他方法保持不变 ---
    def _initialize_manning(self, n_value_or_config): # 初始化曼宁糙率
        # ... (代码不变) ...
        num_cells = len(self.mesh.cells) # 获取单元数量
        if isinstance(n_value_or_config, (float, int)): # 如果输入是浮点数或整数
            self.manning_n_values = np.full(num_cells, float(n_value_or_config), dtype=np.float64) # 创建均匀糙率数组
            print(f"  使用均匀曼宁糙率 n: {n_value_or_config}") # 打印信息
        else: # 如果输入类型不支持
            print(f"警告: 不支持的曼宁糙率配置 '{n_value_or_config}'. 使用默认值 0.025。") # 打印警告
            self.manning_n_values = np.full(num_cells, 0.025, dtype=np.float64) # 使用默认值创建数组

    def _initialize_conserved_variables(self, initial_conditions: dict): # 初始化守恒量（此方法在_initialize_conserved_variables_and_eta中被调用）
        initial_condition_type = initial_conditions.get("type", "uniform_elevation") # 从子字典获取类型
        print(f"  设置初始条件，类型: {initial_condition_type}") # 打印类型

        if initial_condition_type == "uniform_elevation": # 如果是均匀水位
            # --- 访问前再次打印确认 ---
            print(f"[*] HydroModel: 尝试获取 'water_surface_elevation' from: {initial_conditions}")
            # ---
            eta0 = initial_conditions.get('water_surface_elevation', None) # 从子字典获取水位
            if eta0 is None: # 如果未指定
                print("警告: 初始条件类型为 'uniform_elevation' 但未指定 'water_surface_elevation'。假定为干底。") # 打印警告
                eta0 = -float('inf') # 设为极小值
            else:
                 print(f"[*] HydroModel: 成功获取 water_surface_elevation = {eta0}") # 打印成功获取的值

            for i, cell in enumerate(self.mesh.cells): # 遍历单元
                # 确保 z_bed_centroid 存在且有效
                if hasattr(cell, 'z_bed_centroid'):
                     actual_h = max(0.0, eta0 - cell.z_bed_centroid) # 计算水深
                else:
                     print(f"警告: 单元 {i} 缺少 z_bed_centroid 属性，无法计算水深。设为0。")
                     actual_h = 0.0 # 或者使用其他默认值

                if actual_h < self.min_depth: actual_h = 0.0 # 应用统一阈值
                self.U[i, 0] = actual_h # 设置水深
                self.U[i, 1:] = 0.0 # 设置动量为0
            # 在循环外打印一次信息
            if eta0 != -float('inf'):
                 print(f"  从水面高程 {eta0} 设置初始条件。") # 打印信息
            # else: # 如果 eta0 仍然是 -inf，则不需要这行打印

        elif initial_condition_type == "uniform_depth": # 如果是均匀水深
            h0 = initial_conditions.get('water_depth', 0.0) # 从子字典获取水深
            if h0 < self.min_depth: h0 = 0.0 # 应用统一阈值
            for i, cell in enumerate(self.mesh.cells): # 遍历单元
                self.U[i, 0] = h0 # 设置水深
                self.U[i, 1:] = 0.0 # 设置动量为0
            print(f"  使用均匀水深 {h0} 设置初始条件。") # 打印信息

        # elif initial_condition_type == "from_restart_file": # 如果是从重启文件 (可选扩展)
        # restart_file = initial_conditions.get('restart_file_path') # 获取重启文件路径
        # if restart_file and os.path.exists(restart_file): # 如果路径有效且文件存在
        # print(f"  从重启文件 {restart_file} 加载初始条件...") # 打印加载信息
        # success = self._load_state_from_restart(restart_file) # 加载状态 (需要实现 _load_state_from_restart)
        # if not success: # 如果加载失败
        # print("错误: 加载重启文件失败。程序终止。") # 打印错误
        # exit() # 退出
        # else: # 如果路径无效或文件不存在
        # print(f"错误: 初始条件类型为 'from_restart_file' 但重启文件 '{restart_file}' 无效。") # 打印错误
        # exit() # 退出
        else: # 其他未知类型
            raise ValueError(f"未知的初始条件类型: {initial_condition_type}") # 抛出错误


    def _conserved_to_primitive(self, U: np.ndarray, cell_id: int) -> np.ndarray: # 守恒量转原始量
        """Converts conserved variables [h, hu, hv] to primitive [h, u, v]. Handles dry cells."""
        h = U[0] # 获取水深 h
        # 使用 min_depth 进行计算时的判断
        if h < self.min_depth: # 如果水深小于最小阈值
            return np.array([h, 0.0, 0.0], dtype=np.float64) # 返回 h 和 零速度
        else: # 否则
            # 避免除零，可以使用 min_depth 或一个更小的数值 epsilon
            h_div = max(h, 1e-12) # 使用一个小的 epsilon 避免数值问题
            u = U[1] / h_div # 计算 u
            v = U[2] / h_div # 计算 v
            return np.array([h, u, v], dtype=np.float64) # 返回 h, u, v

    def _calculate_rhs_explicit_part(self, U_state: np.ndarray, # 计算显式右端项 (Well-Balanced 版本)
                                     time_current: float) -> np.ndarray: # 返回显式RHS数组
        """
        计算由于对流通量引起的 dU/dt (Well-Balanced 版本)。
        底坡源项的影响通过静水重构包含在通量计算中。
        """
        RHS_explicit = np.zeros_like(U_state) # 初始化显式RHS数组
        num_cells = len(self.mesh.cells) # 获取单元数量

        # --- 1. (高阶) 准备重构 ---
        gradients_available = False # 标记梯度是否可用
        if self.reconstructor.scheme != ReconstructionSchemes.FIRST_ORDER: # 如果不是一阶
            try: # 尝试准备重构
                # *** 注意：为实现精确的井平衡重构，建议修改Reconstruction类 ***
                # *** 使其能计算并提供 grad_h 和 grad_b (或 grad_eta) ***
                # *** 当前简化处理：reconstructor 只准备了 W=[h,u,v] 的梯度 ***
                self.reconstructor.prepare_for_step(U_state) # 调用重构器的准备步骤 (计算梯度和限制器)
                gradients_available = True # 标记梯度可用
            except Exception as e: # 捕获错误
                print(f"错误: 在重构准备阶段出错于时间 {time_current:.3f}: {e}") # 打印错误
                traceback.print_exc() # 打印堆栈
                raise # 重新抛出异常，终止程序

        # --- 2. 计算通量 (遍历物理边) ---
        edge_flux_contributions = np.zeros_like(U_state) # 初始化通量贡献数组

        for he in self.mesh.half_edges: # 遍历所有半边
            is_boundary_edge = (he.twin is None) # 判断是否为边界边
            # 处理内部边时，为避免重复计算，只处理 he.id < he.twin.id 的情况
            is_internal_edge_to_process = (not is_boundary_edge) and (he.id < he.twin.id) # 判断是否是需要处理的内部边

            if not (is_boundary_edge or is_internal_edge_to_process): # 如果既不是边界边，也不是需要处理的内部边
                continue # 跳过当前半边，继续下一条

            cell_L_obj = he.cell # 获取左侧单元对象
            cell_L_id = cell_L_obj.id # 获取左侧单元ID

            try: # 包裹通量计算过程，以便捕获可能的错误
                if is_internal_edge_to_process: # --- 处理内部边 ---
                    cell_R_obj = he.twin.cell # 获取右侧单元对象
                    cell_R_id = cell_R_obj.id # 获取右侧单元ID

                    # --- a. 获取界面原始重构值 W_L, W_R ---
                    # 从重构器获取界面左右两侧的原始变量状态 [h, u, v]
                    W_L_interface, W_R_interface = \
                        self.reconstructor.get_reconstructed_interface_states(
                            U_state, cell_L_obj, cell_R_obj, he, is_boundary=False # 传入状态、左右单元、半边、非边界标记
                        ) # 结束获取重构状态

                    # --- b. 获取界面处的底高程 z_face ---
                    #    使用边端点平均值作为界面底高程是比较稳妥的方式
                    if he.origin and he.end_node: # 检查半边的起点和终点是否存在
                        z_face = (he.origin.z_bed + he.end_node.z_bed) / 2.0 # 计算界面处底高程平均值
                    else: # 如果边的几何信息不完整（理论上不应发生），使用备选方法
                        z_face = (cell_L_obj.z_bed_centroid + cell_R_obj.z_bed_centroid) / 2.0 # 使用邻近单元形心高程的平均值
                        # 可以考虑打印警告信息
                        # print(f"警告: 内部边 {he.id} 缺少端点信息，使用单元形心平均估算 z_face。")

                    # --- c. 获取界面处重构的水面高程 eta_L, eta_R ---
                    #    方法1：使用重构的h和界面z_face (简化，非严格井平衡)
                    #    eta_L_at_face = W_L_interface[0] + z_face
                    #    eta_R_at_face = W_R_interface[0] + z_face

                    #    方法2：使用单元中心eta和梯度外插 (更接近Audusse，但需要eta梯度)
                    #    计算单元中心的水位 eta = h + z_bed_centroid
                    eta_L_center = U_state[cell_L_id, 0] + cell_L_obj.z_bed_centroid # 左单元中心水位
                    eta_R_center = U_state[cell_R_id, 0] + cell_R_obj.z_bed_centroid # 右单元中心水位

                    eta_L_at_face = eta_L_center # 默认使用中心值 (对应一阶精度)
                    eta_R_at_face = eta_R_center # 默认使用中心值 (对应一阶精度)

                    if gradients_available: # 如果是高阶方法且梯度已计算
                        # ** 理想情况：使用水位 eta 的梯度进行外插 **
                        # 获取水深梯度 grad_h
                        grad_h_L = self.reconstructor.gradients[cell_L_id, 0, :] # 获取左单元h梯度 [grad_x, grad_y]
                        grad_h_R = self.reconstructor.gradients[cell_R_id, 0, :] # 获取右单元h梯度 [grad_x, grad_y]
                        # 获取底高程梯度 grad_b (假设 Cell 对象存储了 b_slope_x, b_slope_y)
                        grad_b_L = np.array([cell_L_obj.b_slope_x, cell_L_obj.b_slope_y]) # 左单元底坡梯度
                        grad_b_R = np.array([cell_R_obj.b_slope_x, cell_R_obj.b_slope_y]) # 右单元底坡梯度
                        # 计算水位梯度 grad_eta = grad_h + grad_b
                        grad_eta_L = grad_h_L + grad_b_L # 计算左单元水位梯度
                        grad_eta_R = grad_h_R + grad_b_R # 计算右单元水位梯度

                        # 计算从单元中心到界面中点的向量
                        vec_L_to_face = np.array(he.mid_point) - np.array(cell_L_obj.centroid) # 左单元中心到界面中点向量
                        vec_R_to_face = np.array(he.mid_point) - np.array(cell_R_obj.centroid) # 右单元中心到界面中点向量

                        # 计算水位的外插修正量 delta_eta = grad_eta * r
                        delta_eta_L = np.dot(grad_eta_L, vec_L_to_face) # 计算左侧水位修正量
                        delta_eta_R = np.dot(grad_eta_R, vec_R_to_face) # 计算右侧水位修正量

                        # 计算界面处的水位
                        eta_L_at_face = eta_L_center + delta_eta_L # 外插得到界面左侧水位
                        eta_R_at_face = eta_R_center + delta_eta_R # 外插得到界面右侧水位

                        # ** 注意：这里需要确保 Reconstruction 类能正确提供或计算 grad_h, **
                        # ** 并且 Cell 类存储了精确的 b_slope_x, b_slope_y **

                    # --- d. 计算静水重构水深 h_L_star, h_R_star ---
                    # h* = max(0, eta_face - z_face)
                    h_L_star = max(0.0, eta_L_at_face - z_face) # 静水重构左水深
                    h_R_star = max(0.0, eta_R_at_face - z_face) # 静水重构右水深

                    # --- e. 构造用于通量计算的状态 W_L_flux, W_R_flux ---
                    #    使用静水重构水深 h* 和原始重构速度 u, v
                    #    W_L_interface 包含了重构后的 uL, vL
                    W_L_flux = np.array([h_L_star, W_L_interface[1], W_L_interface[2]]) # 左侧通量计算状态
                    #    W_R_interface 包含了重构后的 uR, vR
                    W_R_flux = np.array([h_R_star, W_R_interface[1], W_R_interface[2]]) # 右侧通量计算状态

                    # --- f. 计算 HLLC 数值通量 ---
                    # 使用修正后的状态 W_L_flux, W_R_flux 调用通量计算器
                    numerical_flux = self.flux_calculator.calculate_flux(
                        W_L_flux, W_R_flux, he.normal # 传入重构状态和法向量
                    ) # 结束通量计算

                    # --- g. 累加通量贡献到左右单元的 RHS ---
                    flux_term = numerical_flux * he.length # 通量乘以边长得到总通量值
                    # 根据有限体积法，通量贡献 = - (通量 * 边长) / 单元面积
                    if cell_L_obj.area > 1e-12: # 避免除以零面积
                        edge_flux_contributions[cell_L_id, :] -= flux_term / cell_L_obj.area # 左单元流出（减号）
                    if cell_R_obj.area > 1e-12: # 避免除以零面积
                        edge_flux_contributions[cell_R_id, :] += flux_term / cell_R_obj.area # 右单元流入（加号）

                elif is_boundary_edge: # --- 处理边界边 ---
                    # 调用边界条件处理器计算边界通量
                    # 边界处理函数内部应负责获取合适的左右状态（内部状态和虚拟外部状态）并计算通量
                    boundary_flux = self.boundary_handler.calculate_boundary_flux(
                        cell_L_obj, he, U_state, time_current # 传入左单元对象、半边、当前状态、当前时间
                    ) # 结束边界通量计算
                    flux_term = boundary_flux * he.length # 边界通量乘以边长
                    if cell_L_obj.area > 1e-12: # 避免除以零面积
                        edge_flux_contributions[cell_L_id, :] -= flux_term / cell_L_obj.area # 边界流出（减号）

            except Exception as e: # 捕获通量计算中的任何异常
                # 构造详细的错误信息
                edge_info = f"内部边 {he.id} (单元 {cell_L_id}-{cell_R_id})" if is_internal_edge_to_process else f"边界边 {he.id} (单元 {cell_L_id}, 标记 {he.boundary_marker})"
                print(f"错误: 在 {edge_info} 计算通量时出错于时间 {time_current:.3f}: {e}") # 打印错误信息
                traceback.print_exc() # 打印完整的错误堆栈信息
                raise # 重新抛出异常，通常应终止模拟

        # --- 步骤 3: 不再需要显式添加底坡源项 ---
        # S_bed_all = self.source_term_calculator.calculate_bed_slope_term_all_cells(...)
        # RHS_explicit = edge_flux_contributions + S_bed_all # 旧的方式

        # 井平衡格式下，显式 RHS 只包含通量梯度项
        RHS_explicit = edge_flux_contributions # 新的方式：只包含通量贡献

        # **注意：摩阻项仍然由时间积分器处理**

        return RHS_explicit # 返回计算得到的显式右端项

    def _apply_friction_semi_implicit(self, U_input_state, U_coeffs_state, dt_friction): # 应用半隐式摩擦
        """Applies semi-implicit friction to U_input_state."""
        # 调用源项计算器的摩擦函数
        return self.source_term_calculator.apply_friction_semi_implicit( # 返回摩擦计算结果
            U_input_state, # 需要施加摩擦的状态
            U_coeffs_state, # 用于计算摩擦系数的状态
            dt_friction, # 时间步长
            self.manning_n_values, # 每个单元的糙率
        ) # 结束摩擦计算调用

    def _calculate_dt(self): # 计算时间步长 dt
        min_dt_inv_term = 0.0 # 初始化时间步长倒数相关项的最大值（找最大值，其倒数是最小步长）
        for i, cell in enumerate(self.mesh.cells): # 遍历所有单元
            h = self.U[i, 0] # 获取当前单元的水深 h
            # --- 使用统一的 self.min_depth ---
            if h < self.min_depth: continue # **修改这里** 如果单元是干的，跳过

            # 使用 h + epsilon 避免除零
            h_div = h + 1e-12 # 用于除法的安全水深
            u = self.U[i, 1] / h_div # 计算 u 速度
            v = self.U[i, 2] / h_div # 计算 v 速度
            c = np.sqrt(self.gravity * h) # 计算波速 c (此时 h 保证 > min_depth)

            sum_lambda_L_over_area = 0 # 初始化当前单元的 (特征速度*边长/面积) 的总和
            for he in cell.half_edges_list: # 遍历单元的所有半边
                if he.length > 1e-9: # 如果边长有效
                    un = u * he.normal[0] + v * he.normal[1] # 计算法向速度 un
                    lambda_max_edge = abs(un) + c # 计算该边上的最大特征速度 |un|+c
                    sum_lambda_L_over_area += lambda_max_edge * he.length # 累加 (|un|+c)*L

            if cell.area > 1e-12 and sum_lambda_L_over_area > 1e-9: # 如果单元面积和累加值有效
                dt_inv_term_cell = sum_lambda_L_over_area / cell.area # 计算该单元的时间步长倒数项 sum((|un|+c)*L)/Area
                if dt_inv_term_cell > min_dt_inv_term: # 如果当前单元的项更大
                    min_dt_inv_term = dt_inv_term_cell # 更新全局最大值

        if min_dt_inv_term < 1e-9: # 如果全局最大值非常小（例如全干或静水）
            return self.max_dt # 返回允许的最大时间步长

        # 根据CFL数计算时间步长 dt = CFL / max(sum((|un|+c)*L)/Area)
        calculated_dt = self.cfl_number / min_dt_inv_term # 根据CFL数计算时间步长
        return min(calculated_dt, self.max_dt) # 返回计算得到的dt和最大允许dt中的较小者

    # run_simulation 和 _save_results 方法基本保持不变
    def run_simulation(self): # 运行模拟 (修正后版本)
        current_time = 0.0 # 初始化当前时间
        next_output_time = 0.0 # 初始化下一输出时间
        step_count = 0 # 初始化步数

        # --- 输出目录检查 (只需一次) ---
        output_dir = self.parameters.get('file_paths', {}).get("output_directory", "simulation_output") # 从file_paths获取输出目录
        if not os.path.exists(output_dir): # 如果不存在
            os.makedirs(output_dir) # 创建目录
            print(f"已创建输出目录: {output_dir}") # 打印信息

        print(f"\n开始模拟，总时长: {self.total_time} s, 输出间隔: {self.output_dt} s") # 打印开始信息
        self._save_results(current_time, step_count, output_dir) # 保存初始状态

        sim_start_wall_time = pytime.time() # 记录模拟开始的墙上时间

        while current_time < self.total_time: # 模拟主循环，直到达到总时间
            # --- 1. 计算时间步长 ---
            dt = self._calculate_dt() # 根据当前状态计算CFL时间步长

            # --- 2. 精确控制步长，确保输出时间和总时间点 ---
            # (保持你之前的 dt 控制逻辑)
            dt_original = dt # 记录原始计算的dt
            if current_time + dt > self.total_time: dt = self.total_time - current_time # 如果下一步将超过总时间，则调整dt恰好到达总时间
            time_to_next_output = next_output_time - current_time # 计算距离下一个输出时间的剩余时间
            # 注意: 如果 output_dt <= 0 (仅输出最后结果)，此条件不触发
            # 如果设置了输出间隔，并且距离下次输出时间很近
            if self.output_dt > 1e-9 and time_to_next_output > 1e-9:
                # 如果当前计算的dt几乎等于或大于到下次输出的时间，则调整dt恰好到达输出时间点
                if dt >= time_to_next_output * (1.0 - 1e-5): dt = time_to_next_output

            # 处理dt过小的情况
            if dt < 1e-12: # 如果计算出的dt非常小
                if np.isclose(current_time, self.total_time): break # 如果已经接近总时间，则结束循环
                # 尝试恢复，使用原始计算的dt或一个最小值，但不能超过max_dt
                dt = min(max(dt_original, 1e-9), self.max_dt) # 尝试使用原始dt或max_dt，但保证一个最小值
                if current_time + dt > self.total_time: dt = self.total_time - current_time # 再次检查是否超总时

                if dt < 1e-12: # 如果恢复后dt仍然过小
                    print(f"错误: 无法在时间 {current_time:.3f} 获得有效的时间步长。模拟终止。") # 打印错误信息
                    break # 终止模拟循环

            # --- 3. 时间积分 ---
            try: # 使用try-except包裹积分步骤以捕获运行时错误
                U_new = self.time_integrator.step(self.U, dt, current_time) # 调用时间积分器执行一步积分，得到下一时刻的状态 U_new
            except Exception as e: # 捕获积分过程中可能发生的异常
                print(f"错误: 时间积分失败在 t={current_time:.3f} (dt={dt:.2e}): {e}") # 打印错误信息
                traceback.print_exc() # 打印详细的错误堆栈
                break # 终止模拟循环

            # --- 4. 干单元处理 (作用于 U_new) ---
            # 使用统一的 self.min_depth 对新计算的状态进行干湿判断和处理
            dry_mask = U_new[:, 0] < self.min_depth # 找到水深小于阈值的干单元掩码
            U_new[dry_mask, 0] = 0.0 # 将干单元的水深设为0
            U_new[dry_mask, 1:] = 0.0 # 将干单元的动量设为0

            # --- 5. 更新模型状态 ---
            eta_current_step = self.eta_previous.copy() # **保存当前时间步开始时的eta，用于本次迭代的VFR猜测**
            self.U = U_new # **更新模型的守恒量状态为新计算的值**

            # --- 6. (可选但推荐) 计算并存储新的 eta 值供下一步使用 ---
            # 这一步是为了给下一个时间步的 VFR 计算提供更好的初始猜测值 eta_previous_guess
            eta_new = np.zeros_like(self.eta_previous) # 初始化存储新 eta 值的数组
            for i, cell in enumerate(self.mesh.cells): # 遍历所有单元
                if self.U[i, 0] >= self.min_depth / 10: # 如果单元当前是湿的 (用稍小的阈值触发计算)
                    try: # 包裹VFR计算以捕获可能的错误
                        # 准备 VFR 计算所需的输入
                        nodes_with_z = sorted([(node, node.z_bed) for node in cell.nodes], # 按高程排序节点
                                              key=lambda item: item[1]) # 按节点高程排序
                        cell_nodes_sorted = [item[0] for item in nodes_with_z] # 获取排序后节点对象列表
                        b_sorted = [item[1] for item in nodes_with_z] # 获取排序后高程列表
                        # 调用 VFR 计算器，使用上一步的 eta (即 eta_current_step) 作为初始猜测
                        eta_new[i] = self.vfr_calculator.get_eta_from_h( # 计算新的eta
                            self.U[i, 0], b_sorted, cell_nodes_sorted, cell.area, # 传入当前水深、排序高程、排序节点、面积
                            eta_previous_guess=eta_current_step[i], # **传入上一步的eta作为猜测**
                            cell_id_for_debug=f"{i}_t{current_time + dt:.2f}" # 传入调试ID，标记为下一个时间点
                        ) # 结束VFR计算
                    except Exception as e: # 捕获VFR计算中的错误
                        print(f"错误: 计算单元 {i} 在时间 {current_time + dt:.3f} 的 eta 时出错: {e}") # 打印错误信息
                        b_sorted_dry = sorted([n.z_bed for n in cell.nodes]) if cell.nodes else [-float('inf')] # 容错：获取排序高程
                        eta_new[i] = b_sorted_dry[0] # 将eta设为最低点高程
                else: # 如果单元当前是干的
                    b_sorted_dry = sorted([n.z_bed for n in cell.nodes]) if cell.nodes else [-float('inf')] # 获取排序高程
                    eta_new[i] = b_sorted_dry[0] # 新的eta也设为最低点高程
            self.eta_previous = eta_new # **更新存储的 eta_previous 以备下一步迭代使用**

            # --- 7. 更新时间和步数 ---
            current_time += dt # 更新当前模拟时间
            step_count += 1 # 更新步数计数器

            # --- 8. 输出结果 ---
            # 使用 next_output_time 进行判断，允许小的浮点误差 (1e-9)
            if current_time >= next_output_time - 1e-9 or np.isclose(current_time, self.total_time): # 如果达到或超过下一个输出时间点，或者接近总时间
                wall_time_elapsed = pytime.time() - sim_start_wall_time # 计算从模拟开始到现在经过的墙上时间
                print( # 打印进度信息
                    f"时间: {current_time:>{8}.{3}f} s | 步数: {step_count:>{6}} | dt: {dt:{8}.{2}e} s | 已耗时: {wall_time_elapsed:{6}.{1}f} s"
                ) # 结束打印
                self._save_results(current_time, step_count, output_dir) # 调用保存结果函数

                if np.isclose(current_time, self.total_time): break # 如果已经达到总时间，结束模拟循环

                # 计算下一个输出时间点 (保持你之前的逻辑)
                if self.output_dt > 1e-9: # 仅当设置了有效输出间隔时计算
                    # 找到当前是第几个输出周期
                    current_output_num = int(round(current_time / self.output_dt + 1e-6))
                    # 计算下一个输出时间，但不超过总时间
                    next_output_time = min(self.total_time, (current_output_num + 1) * self.output_dt)
                    # 防止因为dt恰好等于输出间隔导致死循环或重复输出
                    if next_output_time - current_time < 1e-6 and next_output_time < self.total_time:
                        next_output_time = min(self.total_time, (current_output_num + 2) * self.output_dt)
                else: # 如果只输出最后结果 (output_dt <= 0)
                    next_output_time = self.total_time + 1.0 # 设为一个永远不会达到的时间，确保只在最后输出

        # --- 模拟结束 ---
        sim_end_wall_time = pytime.time() # 记录模拟结束的墙上时间
        print(f"\n模拟完成于时间 {current_time:.3f} s.") # 打印结束信息
        print(f"总墙上计算时间: {sim_end_wall_time - sim_start_wall_time:.2f} s.") # 打印总耗时
        print(f"总计算步数: {step_count}") # 打印总步数

    def _save_results(self, time_val, step_num, output_dir): # 保存结果
        # ... (try import meshio, VFRCalculator 不变) ...
        try: # 尝试导入需要的库
            import meshio # 导入 meshio 用于写入 VTK 文件
            # from .WettingDrying import VFRCalculator # 如果计算eta则需要，但现在 HydroModel 自身持有实例
        except ImportError: # 如果导入失败
            print("警告: 未找到 'meshio'。无法保存 VTK 文件。") # 打印警告
            return # 直接返回，不保存

        print(f"    保存时间 {time_val:.3f} s (步数 {step_num}) 的结果...") # 打印保存信息
        try: # 使用 try-except 包裹保存过程，捕获可能的错误
            # 准备节点坐标 (x, y, z_bed) - 注意 z 使用底高程
            points_3d = np.array([[node.x, node.y, node.z_bed] for node in self.mesh.nodes], dtype=np.float64) # 创建节点三维坐标数组
            # 准备单元拓扑 (构成三角形的节点ID列表)
            triangles = np.array([[node.id for node in cell.nodes] for cell in self.mesh.cells], dtype=int) # 创建单元节点连接数组

            # --- 提取或计算要保存的物理量 ---
            h_vals = self.U[:, 0].copy() # 复制当前水深数组 h
            hu_vals = self.U[:, 1].copy() # 复制当前x方向动量数组 hu
            hv_vals = self.U[:, 2].copy() # 复制当前y方向动量数组 hv

            # 计算原始流速 u, v
            u_vals = np.zeros_like(h_vals) # 初始化 u 速度数组
            v_vals = np.zeros_like(h_vals) # 初始化 v 速度数组
            # 找到水深大于最小阈值的湿单元
            non_dry_mask = h_vals > self.min_depth # 使用统一 min_depth 创建湿单元掩码
            # 计算安全水深用于除法，避免除零
            h_div = h_vals[non_dry_mask] + 1e-12 # 添加小量 epsilon
            # 仅为湿单元计算速度
            u_vals[non_dry_mask] = hu_vals[non_dry_mask] / h_div # 计算 u = hu / h
            v_vals[non_dry_mask] = hv_vals[non_dry_mask] / h_div # 计算 v = hv / h

            # 计算水面高程 eta
            # 使用 self.eta_previous，这是在当前 U 更新后计算的最新 eta 值
            eta_vals_calc = self.eta_previous.copy() # 复制最新的水面高程数组

            # 获取单元形心处的底高程 z_bed_centroid
            z_bed_cell = np.array([cell.z_bed_centroid for cell in self.mesh.cells], dtype=np.float64) # 创建单元形心底高程数组

            # 获取每个单元的曼宁糙率 n
            manning_n_cell = self.manning_n_values # 使用模型中存储的曼宁系数数组

            # --- 准备节点数据 ---
            point_data = {
                'bed_elevation_node': np.array([node.z_bed for node in self.mesh.nodes], dtype=np.float64)
            }



            # --- 构建符合 meshio 期望的 cell_data ---
            cell_data_formatted = {
                'water_depth': [h_vals.copy()],  # 列表包含一个数组，对应 "triangle" 块
                'velocity_u': [u_vals.copy()],  # 同上
                'velocity_v': [v_vals.copy()],  # 同上
                'water_surface': [eta_vals_calc.copy()],  # 同上
                'bed_elevation_cell': [z_bed_cell.copy()],  # 同上
                'manning_n': [manning_n_cell.copy()]  # 同上
            }

            # --- 准备单元拓扑信息 ---
            # 使用元组列表形式的 cells，这与 meshio 内部处理更一致
            cells_list_of_tuples = [("triangle", triangles)]

            # --- 创建 meshio.Mesh 对象 ---
            mesh_to_write = meshio.Mesh(
                points_3d,
                cells_list_of_tuples,  # 传递元组列表
                point_data=point_data,  # point_data 格式之前是正确的
                cell_data=cell_data_formatted  # 使用新格式的 cell_data
            )

            # --- 写入 VTK 文件 ---
            filename = os.path.join(output_dir, f"result_{step_num:06d}.vtk")
            mesh_to_write.write(filename, file_format="vtk", binary=True)
        except Exception as e:
            print(f"错误: 保存结果到 VTK 时出错于时间 {time_val:.3f} s: {e}")
            if isinstance(e, KeyError):
                print(f"  KeyError detail: {e.args}")
            traceback.print_exc()