# src/model/BoundaryConditions.py
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from .MeshData import Mesh, Cell, HalfEdge, Node
from .FluxCalculator import FluxCalculator
from .Reconstruction import Reconstruction, ReconstructionSchemes


class BoundaryConditionHandler:
    def __init__(self, mesh: Mesh, flux_calculator: FluxCalculator,
                 reconstructor: Reconstruction, gravity: float,
                 min_depth: float,  # 使用统一的 min_depth
                 bc_definitions: dict,
                 elev_timeseries_filepath: str | None = None,
                 discharge_timeseries_filepath: str | None = None):
        """
        初始化边界条件处理器。

        Args:
            # ... (其他参数不变) ...
            bc_definitions: 从config加载的边界定义字典 {marker: {type: ...}, ...}。
            elev_timeseries_filepath: 水位时间序列CSV文件路径 (来自 config['file_paths']['boundary_timeseries_elevation_file'])。
            discharge_timeseries_filepath: 总流量时间序列CSV文件路径 (来自 config['file_paths']['boundary_timeseries_discharge_file'])。
        """
        self.mesh = mesh  # 网格对象
        self.flux_calculator = flux_calculator  # 通量计算器
        self.reconstructor = reconstructor  # 重构器
        self.g = gravity  # 重力加速度
        self.min_depth = min_depth  # 统一的最小水深
        self.bc_definitions = bc_definitions  # 边界定义字典

        self.elev_timeseries_df = None  # 初始化水位时间序列 DataFrame
        self.discharge_timeseries_df = None  # 初始化流量时间序列 DataFrame

        self.marker_to_edges = defaultdict(list)  # 存储每个标记对应的半边列表
        self.marker_total_lengths = defaultdict(float)  # 存储每个标记对应的总边界长度

        print("  初始化边界条件处理器...")  # 打印初始化信息
        self._preprocess_boundaries()  # 预处理边界边信息

        # --- 加载时间序列文件 ---
        print("    加载边界时间序列文件...")  # 打印加载信息
        if elev_timeseries_filepath and os.path.exists(elev_timeseries_filepath):  # 如果水位文件路径有效且存在
            print(f"      加载水位数据: {elev_timeseries_filepath}")  # 打印加载文件信息
            self.elev_timeseries_df = self._load_and_validate_timeseries_csv(elev_timeseries_filepath,
                                                                             'elev')  # 加载并验证水位CSV
            if self.elev_timeseries_df is None:  # 如果加载失败
                print(f"      警告: 加载水位时间序列失败。水位边界可能无法工作。")  # 打印警告
        else:  # 如果路径无效或文件不存在
            if any('waterlevel' in definition.get('type', '').lower() for definition in
                   bc_definitions.values()):  # 检查是否有水位边界被定义
                print(
                    f"      警告: 水位时间序列文件 '{elev_timeseries_filepath}' 未找到或未指定，但定义了水位边界。")  # 打印警告

        if discharge_timeseries_filepath and os.path.exists(discharge_timeseries_filepath):  # 如果流量文件路径有效且存在
            print(f"      加载流量数据: {discharge_timeseries_filepath}")  # 打印加载文件信息
            self.discharge_timeseries_df = self._load_and_validate_timeseries_csv(discharge_timeseries_filepath,
                                                                                  'flux')  # 加载并验证流量CSV
            if self.discharge_timeseries_df is None:  # 如果加载失败
                print(f"      警告: 加载流量时间序列失败。流量边界可能无法工作。")  # 打印警告
        else:  # 如果路径无效或文件不存在
            if any('discharge' in definition.get('type', '').lower() for definition in
                   bc_definitions.values()):  # 检查是否有流量边界被定义
                print(
                    f"      警告: 流量时间序列文件 '{discharge_timeseries_filepath}' 未找到或未指定，但定义了流量边界。")  # 打印警告

        print("  边界条件处理器初始化完成。")  # 打印初始化完成信息

    def _preprocess_boundaries(self):  # 预处理边界边信息
        print("    预处理边界边...")  # 打印信息
        for he in self.mesh.half_edges:  # 遍历所有半边
            if he.twin is None:  # 如果是边界边
                marker = he.boundary_marker  # 获取边界标记
                self.marker_to_edges[marker].append(he)  # 添加到对应标记的列表中
                self.marker_total_lengths[marker] += he.length  # 累加该标记的总长度
        for marker, edges in self.marker_to_edges.items():  # 打印每个标记的信息
            print(
                f"      标记 {marker}: 找到 {len(edges)} 条边界边, 总长度 {self.marker_total_lengths[marker]:.2f}")  # 打印统计

    def _load_and_validate_timeseries_csv(self, filepath: str,
                                          expected_prefix: str) -> pd.DataFrame | None:  # 加载并验证时间序列CSV文件
        """加载并验证CSV, 列名应为 time 和 prefix_marker"""
        try:
            df = pd.read_csv(filepath)  # 读取CSV文件
            if 'time' not in df.columns:  # 检查是否存在 'time' 列
                print(f"      错误: 时间序列文件 '{filepath}' 缺少 'time' 列。")  # 打印错误
                return None  # 返回None
            if not df['time'].is_monotonic_increasing:  # 检查 'time' 列是否单调递增
                print(f"      错误: 时间序列文件 '{filepath}' 中的 'time' 列未按升序排序。")  # 打印错误
                # df = df.sort_values(by='time').reset_index(drop=True) # 可以选择排序而不是报错
                # print(f"      警告: 时间序列文件 '{filepath}' 中的 'time' 列未排序，已自动排序。")
                return None  # 或者直接认为文件格式错误

            marker_cols_found = {}  # 用于存储找到的标记列 {marker: col_name}
            valid_cols_to_keep = ['time']  # 需要保留的有效列名列表，'time'列总是保留

            for col in df.columns:  # 遍历所有列
                if col.startswith(f"{expected_prefix}_"):  # 如果列名以前缀开头
                    try:
                        marker = int(col.split('_')[-1])  # 尝试提取标记号
                        # 可以在这里检查 marker 是否在 bc_definitions 中实际定义了对应类型的边界 (可选)
                        # if marker in self.bc_definitions and expected_prefix in self.bc_definitions[marker].get('type',''):
                        marker_cols_found[marker] = col  # 存储找到的标记和列名
                        valid_cols_to_keep.append(col)  # 添加到保留列表
                        # else:
                        #     print(f"      信息: 文件 '{filepath}' 中的列 '{col}' 对应的标记 {marker} 未在配置中定义为 '{expected_prefix}' 类型边界，将忽略此列。")
                    except (ValueError, IndexError):  # 如果提取标记失败
                        print(f"      警告: 无法从文件 '{filepath}' 的列名 '{col}' 中解析标记号。跳过此列。")  # 打印警告

            if len(valid_cols_to_keep) <= 1:  # 如果只找到了 'time' 列，没有找到有效的数据列
                print(f"      错误: 在文件 '{filepath}' 中未找到前缀为 '{expected_prefix}_' 的有效数据列。")  # 打印错误
                return None  # 返回None

            df_filtered = df[valid_cols_to_keep].copy()  # 选择有效列创建新的DataFrame
            df_filtered.set_index('time', inplace=True)  # 将 'time' 列设为索引
            print(f"      成功验证并加载了标记 {list(marker_cols_found.keys())} 的 '{expected_prefix}' 数据。")  # 打印成功信息
            return df_filtered  # 返回处理后的DataFrame

        except FileNotFoundError:  # 捕获文件未找到错误
            print(f"      错误: 时间序列文件 '{filepath}' 未找到。")
            return None
        except Exception as e:  # 捕获其他读取或处理错误
            print(f"      错误: 读取或处理时间序列文件 '{filepath}' 时出错: {e}")  # 打印错误信息
            return None  # 返回None

    def _get_timeseries_value(self, marker, time_current, boundary_type: str) -> float | None:  # 获取时间序列数据值
        """
        根据边界类型和标记，从对应的 DataFrame 获取插值。
        """
        df_to_use = None  # 要使用的DataFrame
        col_name_prefix = None  # 列名前缀

        # 根据边界类型选择 DataFrame 和列名前缀
        if 'waterlevel' in boundary_type:  # 如果是水位边界
            df_to_use = self.elev_timeseries_df  # 使用水位DataFrame
            col_name_prefix = 'elev'  # 列名前缀是elev
        elif 'discharge' in boundary_type:  # 如果是流量边界
            df_to_use = self.discharge_timeseries_df  # 使用流量DataFrame
            col_name_prefix = 'flux'  # 列名前缀是 'flux' (对应CSV列 flux_marker)
        else:  # 其他不支持的边界类型
            print(f"错误: 无法为未知的边界类型 '{boundary_type}' 获取时间序列值。")  # 打印错误
            return None  # 返回None

        if df_to_use is None:  # 如果对应的DataFrame未加载 (文件不存在或加载失败)
            # 此处不打印警告，因为在初始化时已经打印过
            # print(f"调试: 类型 '{boundary_type}' 的时间序列 DataFrame 未加载。")
            return None  # 返回None

        col_name = f"{col_name_prefix}_{marker}"  # 构建完整的列名，例如 'elev_1' 或 'flux_2'

        if col_name not in df_to_use.columns:  # 如果该列名在DataFrame中不存在
            # 可能的原因：CSV文件中缺少这一列，或者列名格式不符
            # print(f"调试: 列 '{col_name}' 在为类型 '{boundary_type}' 加载的时间序列数据中未找到。")
            return None  # 返回None

        try:
            # 使用 numpy.interp 进行线性插值
            times = df_to_use.index.values  # 获取时间索引 (numpy 数组)
            values = df_to_use[col_name].values  # 获取对应列的值 (numpy 数组)

            # 处理插值中的 NaN 值
            valid_mask = ~np.isnan(values)  # 创建一个掩码，标记非NaN值的位置
            if not np.any(valid_mask):  # 如果所有值都是 NaN
                print(f"警告: 时间序列标记 {marker} 类型 '{boundary_type}' 的所有值均为 NaN。")  # 打印警告
                return None  # 无法插值，返回None
            if len(times[valid_mask]) < 2:  # 如果有效数据点少于2个
                print(f"警告: 时间序列标记 {marker} 类型 '{boundary_type}' 的有效数据点不足2个，无法进行插值。")  # 打印警告
                if len(times[valid_mask]) == 1: return values[valid_mask][0]  # 如果只有一个点，返回该点的值
                return None  # 否则返回None

            # 执行线性插值
            # np.interp(查询点, x坐标, y坐标)
            # 注意：np.interp 在查询点超出x坐标范围时，默认返回端点值 (无外插)
            interpolated_value = np.interp(time_current, times[valid_mask], values[valid_mask])

            # 理论上 np.interp 对有效输入不会返回 NaN，但以防万一
            if np.isnan(interpolated_value):  # 检查插值结果是否为 NaN
                print(
                    f"警告: 对标记 {marker} 类型 '{boundary_type}' 在时间 {time_current} 的插值结果为 NaN。")  # 打印警告
                return None  # 返回None

            return float(interpolated_value)  # 返回插值结果 (转换为 float)

        except Exception as e:  # 捕获插值过程中可能发生的错误
            print(
                f"错误: 对标记 {marker} 类型 '{boundary_type}' 在时间 {time_current} 进行时间序列插值时出错: {e}")  # 打印错误信息
            return None  # 返回None

    def calculate_boundary_flux(self, cell_L: Cell, he: HalfEdge, U_state_all: np.ndarray,  # 注意：传入完整的U_state
                                time_current: float) -> np.ndarray:  # 计算边界通量
        marker = he.boundary_marker  # 获取半边的边界标记
        # 获取边界定义，如果特定标记没有定义，则尝试获取 'default' 定义
        bc_def = self.bc_definitions.get(marker, self.bc_definitions.get('default'))

        if bc_def is None:  # 如果连默认定义都没有
            print(f"警告: 边界标记 {marker} 未定义边界条件，也无默认设置。假定为墙体。")  # 打印警告
            bc_type = 'wall'  # 默认为墙体
        else:  # 如果有定义
            bc_type = bc_def.get('type', 'wall').lower()  # 获取边界类型，默认为 'wall'，并转为小写

        # 获取内部单元的守恒量 U_L
        U_L = U_state_all[cell_L.id, :]  # 从完整状态数组中提取左单元的守恒量

        # --- 高阶重构可能需要邻居信息，即使在边界 ---
        # 因此，我们将内部原始变量的获取推迟到具体处理函数中，
        # 并将 U_L (守恒量) 传递给需要它的重构步骤。

        # --- 根据类型调用处理函数 ---
        if bc_type == 'wall':  # 如果是墙体边界
            # 墙体边界通常只需要内部状态来计算压力项
            W_L_internal = self._conserved_to_primitive(U_L)  # 获取内部原始变量
            return self._handle_wall_boundary(W_L_internal, he)  # 处理墙体边界

        elif bc_type == 'waterlevel':  # 如果是水位时间序列边界
            target_eta = self._get_timeseries_value(marker, time_current, bc_type)  # 获取目标水位值
            if target_eta is None:  # 如果获取失败 (例如文件问题或时间超出范围)
                print(f"警告: 无法获取水位边界条件标记 {marker} 的值。假定为墙体。")  # 打印警告
                W_L_internal = self._conserved_to_primitive(U_L)  # 获取内部原始变量
                return self._handle_wall_boundary(W_L_internal, he)  # 退化为墙体
            # 水位边界处理函数内部会进行重构（如果需要）
            return self._handle_waterlevel_boundary(U_L, he, target_eta)  # 处理水位边界 (传入 U_L)

        elif bc_type == 'total_discharge':  # 如果是总流量时间序列边界
            target_Q_total = self._get_timeseries_value(marker, time_current, bc_type)  # 获取目标总流量值
            if target_Q_total is None:  # 如果获取失败
                print(f"警告: 无法获取总流量边界条件标记 {marker} 的值。假定为墙体。")  # 打印警告
                W_L_internal = self._conserved_to_primitive(U_L)  # 获取内部原始变量
                return self._handle_wall_boundary(W_L_internal, he)  # 退化为墙体
            # 流量边界处理函数内部会进行重构（如果需要）
            return self._handle_total_discharge_boundary(U_L, he, target_Q_total, marker)  # 处理流量边界 (传入 U_L)

        elif bc_type == 'free_outflow':  # 如果是自由出流边界
            # 自由出流通常假设边界外状态与内部相同
            # 处理函数内部会进行重构（如果需要）
            return self._handle_free_outflow_boundary(U_L, he)  # 处理自由出流边界 (传入 U_L)

        else:  # 其他未知的边界类型
            print(f"警告: 未知的边界条件类型 '{bc_type}' (标记 {marker})。假定为墙体。")  # 打印警告
            W_L_internal = self._conserved_to_primitive(U_L)  # 获取内部原始变量
            return self._handle_wall_boundary(W_L_internal, he)  # 默认为墙体

    def _conserved_to_primitive(self, U_cell: np.ndarray) -> np.ndarray:  # 守恒量转原始量 (辅助函数)
        """将单个单元的守恒量 U=[h, hu, hv] 转换为原始变量 W=[h, u, v]。"""
        h = U_cell[0]  # 获取水深 h
        # 使用统一的 min_depth
        if h < self.min_depth:  # 如果水深小于阈值
            return np.array([h, 0.0, 0.0], dtype=np.float64)  # 返回水深h和零速度
        else:  # 否则
            u = U_cell[1] / h  # 计算u
            v = U_cell[2] / h  # 计算v
            return np.array([h, u, v], dtype=np.float64)  # 返回h, u, v

    # --- 具体的边界处理函数 ---
    # _handle_wall_boundary (不变)
    def _handle_wall_boundary(self, W_L, he):  # 处理墙体边界 (输入为内部原始变量)
        hL, uL, vL = W_L[0], W_L[1], W_L[2]  # 分解内部原始变量
        nx, ny = he.normal  # 获取边界法向量 (指向外部)

        # 对于墙体，法向通量为零，切向动量通量可以认为是零（无滑移）或反射（自由滑移）
        # 这里采用反射边界条件（自由滑移）：法向速度反号，切向速度不变
        unL = uL * nx + vL * ny  # 计算内部法向速度
        utL = -uL * ny + vL * nx  # 计算内部切向速度

        # 构造虚拟单元状态 W_ghost
        un_ghost = -unL  # 法向速度反号
        ut_ghost = utL  # 切向速度不变
        u_ghost = un_ghost * nx - ut_ghost * ny  # 转换回x方向速度
        v_ghost = un_ghost * ny + ut_ghost * nx  # 转换回y方向速度
        # 虚拟单元水深通常设为内部水深
        h_ghost = hL
        W_ghost = np.array([h_ghost, u_ghost, v_ghost])  # 构造虚拟单元原始变量

        # --- 计算通量 ---
        # 注意：墙体边界的精确物理解释是法向速度为0。
        # 直接使用黎曼求解器计算 W_L 和 W_ghost 之间的通量是常用的数值处理方法。
        # 它能保证压力项的正确传递。

        # 获取界面上的左侧状态 W_L_for_flux (可能需要重构)
        # **注意：墙体边界通常不进行复杂重构，直接用 W_L 也可以，或者由重构器处理**
        # **为保持一致性，我们调用重构器获取界面值，但重构器应能处理边界情况**
        # **需要传递 U_L 给重构器**
        # W_L_for_flux, _ = self.reconstructor.get_reconstructed_interface_states(
        #     U_L, # 传递守恒量给重构器 (需要修改接口或在calculate_boundary_flux中获取U_L)
        #     self.mesh.cells[he.cell.id], None, he, is_boundary=True
        # )
        # **简化处理：墙体边界直接用内部值计算通量** (如果不用重构器)
        W_L_for_flux = W_L  # 直接使用内部原始变量

        flux = self.flux_calculator.calculate_flux(W_L_for_flux, W_ghost, he.normal)  # 使用黎曼求解器计算通量

        # 理论上，对于理想墙体，数值通量应为 [0, P*nx, P*ny]，其中P是压力项 0.5*g*h^2
        # HLLC求解器在 un=0 时应该能近似这个结果。
        # 可以验证一下：
        # pressure_term = 0.5 * self.g * hL**2 # 计算压力项
        # expected_flux = np.array([0.0, pressure_term * nx, pressure_term * ny]) # 预期的物理通量
        # print("Wall Flux Calculated:", flux)
        # print("Wall Flux Expected:", expected_flux)

        return flux  # 返回计算得到的数值通量

    def _handle_waterlevel_boundary(self, U_L, he, target_eta):  # 处理水位边界 (输入为内部守恒量)
        W_L_internal = self._conserved_to_primitive(U_L)  # 获取内部原始变量
        hL, uL, vL = W_L_internal[0], W_L_internal[1], W_L_internal[2]  # 分解内部原始变量
        nx, ny = he.normal  # 获取法向量

        # --- 估算边界底高程 b_bnd ---
        # 使用边的两个顶点的平均高程是较好的方法
        if he.origin and he.end_node:  # 如果边的起点和终点都存在
            b_bnd = (he.origin.z_bed + he.end_node.z_bed) / 2.0  # 取两端点底高程的平均值
        elif he.origin:  # 如果只有起点存在 (理论上对于组成单元的边应该总有终点)
            b_bnd = he.origin.z_bed  # 使用起点高程
        elif he.end_node:  # 如果只有终点存在
            b_bnd = he.end_node.z_bed  # 使用终点高程
        else:  # 如果起点终点都没有 (不太可能发生)
            b_bnd = he.cell.z_bed_centroid  # 最后备选：使用内部单元形心底高程
            print(f"警告: 无法精确确定边界边 {he.id} 的底高程。使用单元形心高程 {b_bnd:.2f}。")  # 打印警告

        h_bnd = max(0.0, target_eta - b_bnd)  # 计算边界处的水深 (目标水位 - 边界底高程)

        if h_bnd < self.min_depth:  # 如果边界处水深小于阈值 (干或接近干)
            # print(f"调试: 水位边界 {he.boundary_marker} 在边 {he.id} 处为干 (h_bnd={h_bnd:.2e})。视为墙体。")
            return self._handle_wall_boundary(W_L_internal, he)  # 按墙体处理
        else:  # 如果边界是湿的
            # --- 确定边界速度 (虚拟单元状态) ---
            # 对于缓流边界（常见假设），可以使用特征线法估计边界法向速度
            unL = uL * nx + vL * ny  # 内部法向速度
            utL = -uL * ny + vL * nx  # 内部切向速度
            cL = np.sqrt(self.g * hL) if hL > self.min_depth else 0  # 内部波速
            c_bnd = np.sqrt(self.g * h_bnd)  # 边界波速

            # 根据特征线理论 C+ = u + 2c, C- = u - 2c
            # 假设外流边界，内部信息沿 C+ 特征线传出 (u_bnd + 2*c_bnd = u_L + 2*c_L) => u_bnd = u_L + 2*(c_L - c_bnd)
            # 假设内流边界，外部信息沿 C- 特征线传入 (u_bnd - 2*c_bnd = u_R - 2*c_R)，u_R未知，但通常假设为外部提供的值或根据能量守恒等确定
            # 一种常用的简化处理（尤其当内外流向不确定时）：
            # 假设边界法向速度 un_bnd 由内部状态决定（对于出流），或由外部条件决定（对于入流）
            # 如果是亚临界流 (Fr < 1)，信息双向传播，需要更仔细处理。
            # Toro 书中和许多文献常用的一种方法是基于黎曼不变量：
            # 对于亚临界入流（水流进入计算域）: u_bnd - 2*c_bnd = u_far_field - 2*c_far_field (远场已知)
            # 对于亚临界出流（水流离开计算域）: u_bnd + 2*c_bnd = u_L + 2*c_L => u_bnd = u_L + 2*c_L - 2*c_bnd

            # 这里我们采用一个较通用的近似（假设缓流出流为主，或指定水位意味着外部影响足够强）：
            # 使用 u_bnd = u_L + 2*c_L - 2*c_bnd (出流特征线 C+) 可能导致入流计算不准
            # 使用 Riemann 求解器本身可以处理亚临界情况，关键是构造 W_ghost
            # 我们可以直接设定 W_ghost 的状态：
            W_ghost = np.array([h_bnd, 0.0, 0.0])  # 初始化虚拟单元状态
            # 如果是出流 (unL > 0)，虚拟单元的速度可能设为内部速度 un_ghost=unL, ut_ghost=utL
            # 如果是入流 (unL <= 0)，虚拟单元的速度可能需要根据外部情况设定，比如给定速度，或者用黎曼不变量 C-

            # 更稳妥的方式是：设置 W_ghost = [h_bnd, u_L, v_L]，让黎曼求解器处理界面。
            # 这相当于假设边界外的速度与内部相同，但水深由指定水位决定。
            W_ghost = np.array([h_bnd, uL, vL])

            # --- 获取界面左侧状态 W_L_for_flux ---
            # 使用重构器获取界面状态
            W_L_for_flux, _ = self.reconstructor.get_reconstructed_interface_states(
                U_L,  # 传递内部守恒量
                he.cell, None, he, is_boundary=True  # 告知是边界
            )
            # W_L_for_flux = W_L_internal # 如果用一阶或简化

            # --- 计算通量 ---
            flux = self.flux_calculator.calculate_flux(W_L_for_flux, W_ghost, he.normal)  # 计算数值通量
            return flux  # 返回通量

    def _handle_total_discharge_boundary(self, U_L, he, Q_total_target, marker):  # 处理总流量边界 (输入为内部守恒量)
        W_L_internal = self._conserved_to_primitive(U_L)  # 获取内部原始变量
        hL, uL, vL = W_L_internal[0], W_L_internal[1], W_L_internal[2]  # 分解内部原始变量
        nx, ny = he.normal  # 获取法向量

        # 获取该标记对应的总边界长度
        total_length_marker = self.marker_total_lengths.get(marker, 0.0)  # 从预处理结果中获取

        if total_length_marker < 1e-9:  # 如果总长度过小
            print(f"警告: 流量边界标记 {marker} 的总长度为零。视为墙体。")  # 打印警告
            return self._handle_wall_boundary(W_L_internal, he)  # 按墙体处理

        # --- 分配总流量到当前边，得到法向单宽流量 qn_bnd ---
        # 简单均匀分配 (假设单位宽度上的流量相同)
        qn_bnd = Q_total_target / total_length_marker  # 计算法向单宽流量 (m^2/s)
        # 注意：Q_total_target > 0 通常表示入流 (与法向量同向)， Q < 0 表示出流

        # --- 确定边界水深 h_bnd ---
        # **方法1：假设等于内部水深 (你之前的做法)**
        # h_bnd = hL
        # **方法2：根据流动状态估算 (例如，假设临界流)**
        #    适用于已知 qn_bnd 的情况，特别是入流边界。
        if qn_bnd > 0:  # 如果是入流
            # 计算临界水深 h_c = (q_n^2 / g)^(1/3)
            # 注意 qn_bnd 是法向单宽流量
            h_critical = (qn_bnd ** 2 / self.g) ** (1 / 3) if self.g > 1e-9 else 0.0  # 计算临界水深
            # 可以使用临界水深作为边界水深，或者取它和内部水深的某种组合
            h_bnd = max(h_critical, self.min_depth)  # 将边界水深设为临界水深（保证非负）
            # print(f"Debug: Inflow Discharge BC {marker}, qn={qn_bnd:.3f}, hc={h_critical:.3f}")
        else:  # 如果是出流 (qn_bnd <= 0)
            # 出流边界的水深通常由内部决定，设为内部水深是合理的
            h_bnd = hL
            # print(f"Debug: Outflow Discharge BC {marker}, qn={qn_bnd:.3f}, h_bnd=hL={hL:.3f}")

        # 确保边界水深不小于阈值，否则无法计算速度
        if h_bnd < self.min_depth:  # 如果计算出的边界水深过小
            # print(f"警告: 流量边界 {marker} 在边 {he.id} 处计算的水深过小 (h_bnd={h_bnd:.2e})。视为墙体。")
            # 或者是内部水深 hL 过小导致
            # print(f"警告: 流量边界 {marker} 在边 {he.id} 处内部水深过小 (hL={hL:.2e})。视为墙体。")
            return self._handle_wall_boundary(W_L_internal, he)  # 按墙体处理

        # --- 构造虚拟单元状态 W_ghost ---
        un_bnd = qn_bnd / h_bnd  # 根据 qn_bnd 和 h_bnd 计算法向速度
        # 切向速度通常设为与内部相同 (滑移边界)
        ut_bnd = -uL * ny + vL * nx  # 内部切向速度

        # 转换回 x, y 速度
        u_ghost = un_bnd * nx - ut_bnd * ny  # 虚拟单元 x 速度
        v_ghost = un_bnd * ny + ut_bnd * nx  # 虚拟单元 y 速度
        W_ghost = np.array([h_bnd, u_ghost, v_ghost])  # 构造虚拟单元状态

        # --- 获取界面左侧状态 W_L_for_flux ---
        W_L_for_flux, _ = self.reconstructor.get_reconstructed_interface_states(
            U_L, he.cell, None, he, is_boundary=True  # 使用重构器
        )
        # W_L_for_flux = W_L_internal # 如果用一阶

        # --- 计算通量 ---
        flux = self.flux_calculator.calculate_flux(W_L_for_flux, W_ghost, he.normal)  # 计算数值通量
        return flux  # 返回通量

    def _handle_free_outflow_boundary(self, U_L, he):  # 处理自由出流边界 (输入为内部守恒量)
        W_L_internal = self._conserved_to_primitive(U_L)  # 获取内部原始变量

        # 自由出流通常假设边界外的状态与边界内的状态相同 (零梯度外插)
        W_ghost = W_L_internal  # 将虚拟单元状态设为内部状态

        # --- 获取界面左侧状态 W_L_for_flux ---
        W_L_for_flux, _ = self.reconstructor.get_reconstructed_interface_states(
            U_L, he.cell, None, he, is_boundary=True  # 使用重构器
        )
        # W_L_for_flux = W_L_internal # 如果用一阶

        # --- 计算通量 ---
        # 此时计算的是 W_L_for_flux 和 W_ghost(=W_L_internal) 之间的通量
        flux = self.flux_calculator.calculate_flux(W_L_for_flux, W_ghost, he.normal)  # 计算数值通量
        return flux  # 返回通量

    # ... (可能需要添加 _calculate_total_conveyance 等辅助方法) ...