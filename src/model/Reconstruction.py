# src/model/Reconstruction.py
import numpy as np
from enum import Enum
# Adjust import path based on your project structure
from .MeshData import Mesh, Cell, HalfEdge, Node


class ReconstructionSchemes(Enum):
    """变量重构方案枚举"""
    FIRST_ORDER = "first_order"  # 一阶 (分片常数)
    # Placeholder for a specific second-order scheme
    SECOND_ORDER_LIMITED = "second_order_limited"  # 代表一个带限制器的二阶方法


class Reconstruction:
    """执行变量重构以获取单元界面值的类。"""

    def __init__(self, scheme: ReconstructionSchemes, mesh: Mesh, gravity: float, min_depth: float):
        """
        初始化重构器。

        Args:
            scheme (ReconstructionSchemes): 使用的重构方案。
            mesh (Mesh): 网格对象，用于访问几何和拓扑信息。
            gravity (float): 重力加速度 (用于计算原始变量)。
            min_depth (float): 计算中使用的最小水深 (例如，用于干单元判断和避免除零)。
        """
        if not isinstance(scheme, ReconstructionSchemes):
            raise TypeError(f"scheme必须是 ReconstructionSchemes 枚举成员, 得到 {type(scheme)}")

        self.scheme = scheme
        self.mesh = mesh
        self.g = gravity
        self.min_depth = min_depth

        # --- 用于高阶方法的存储 ---
        # 存储每个单元、每个原始变量(h, u, v)的梯度 [grad_x, grad_y]
        self.gradients = None  # Shape: (num_cells, 3, 2)
        # 存储每个单元、每个变量的限制器 phi (0 到 1)
        self.limiters_phi = None  # Shape: (num_cells, 3)

        print(f"  Reconstructor initialized with scheme: {self.scheme.value}")

    # --- Primitive Variable Conversion ---
    # (These might be better placed in a shared utility module or within HydroModel if used elsewhere)
    def _conserved_to_primitive(self, U: np.ndarray) -> np.ndarray:
        """将单个单元的守恒量 U=[h, hu, hv] 转换为原始变量 W=[h, u, v]。"""
        h = U[0] # 获取水深 h
        if h < self.min_depth: # 使用统一阈值判断干单元
            return np.array([h, 0.0, 0.0], dtype=np.float64) # 返回 h 和 零速度
        else: # 否则
            # 避免除零，可以使用 min_depth 或一个更小的数值 epsilon
            h_for_division = max(h, 1e-12) # 使用一个小的 epsilon 避免数值问题
            u = U[1] / h_for_division # 计算 u
            v = U[2] / h_for_division # 计算 v
            return np.array([h, u, v], dtype=np.float64) # 返回 h, u, v

    def _primitive_to_conserved(self, W: np.ndarray) -> np.ndarray:
        """将单个单元的原始变量 W=[h, u, v] 转换为守恒量 U=[h, hu, hv]。"""
        h, u, v = W[0], W[1], W[2]
        # 确保 h 不小于0
        h = max(0.0, h)
        # 如果 h 非常小，动量也应为0
        if h < self.min_depth:
            hu = 0.0
            hv = 0.0
        else:
            hu = h * u
            hv = h * v
        return np.array([h, hu, hv], dtype=np.float64)

    def _get_primitive_state_for_cell(self, cell_id: int, U_state: np.ndarray) -> np.ndarray:
        """获取指定单元的原始变量 W=[h, u, v]"""
        if cell_id < 0 or cell_id >= U_state.shape[0]:
            raise IndexError(f"Cell ID {cell_id} is out of bounds for U_state.")
        return self._conserved_to_primitive(U_state[cell_id, :])

    # --- Gradient and Limiter Calculation (for Higher Order) ---

    def prepare_for_step(self, U_state: np.ndarray):
        """
        (仅高阶) 在每个时间步开始时计算梯度并应用限制器。
        结果存储在 self.gradients 中。
        """
        if self.scheme == ReconstructionSchemes.FIRST_ORDER:
            self.gradients = None  # 一阶不需要梯度
            return

        print("    Calculating gradients and limiters (Second Order)...")
        num_cells = len(self.mesh.cells)
        num_vars = 3  # h, u, v (原始变量)

        # 1. 计算所有单元的原始变量状态
        W_state = np.array([self._get_primitive_state_for_cell(i, U_state) for i in range(num_cells)])

        # 2. 计算无限制的梯度 (Green-Gauss)
        unlimited_gradients = self._calculate_gradients_green_gauss(W_state)

        # 3. 计算限制器 phi
        self.limiters_phi = self._calculate_barth_jespersen_limiters(W_state, unlimited_gradients)

        # 4. 应用限制器得到最终梯度
        # limited_gradient[cell, var, dim] = phi[cell, var] * unlimited_gradient[cell, var, dim]
        # 使用 broadcasting
        self.gradients = self.limiters_phi[:, :, np.newaxis] * unlimited_gradients

        print("    Gradients and limiters prepared.")

    def _calculate_gradients_green_gauss(self, W_state: np.ndarray) -> np.ndarray:
        """使用格林-高斯定理计算每个单元原始变量的梯度。"""
        num_cells = W_state.shape[0]
        num_vars = W_state.shape[1]
        gradients = np.zeros((num_cells, num_vars, 2), dtype=np.float64)  # grad[:,:,0] is d/dx, grad[:,:,1] is d/dy

        for i, cell in enumerate(self.mesh.cells):
            W_i = W_state[i, :]  # 当前单元原始变量 W_i=[h,u,v]
            grad_W_i = np.zeros((num_vars, 2))  # 初始化梯度 [var, dim]

            if cell.area < 1e-12:  # 跳过面积过小的单元
                continue

            for he in cell.half_edges_list:  # 遍历单元的边
                normal = he.normal  # 外法向量 [nx, ny]
                length = he.length  # 边长

                W_neighbor = W_i  # 默认使用内部值 (用于边界或错误情况)
                if he.twin:  # 如果是内部边
                    neighbor_cell = he.twin.cell
                    W_neighbor = W_state[neighbor_cell.id, :]  # 获取邻居单元原始变量
                else:
                    # 边界处理：最简单的方法是假设边界外状态与内部相同
                    # (这对应于边界上的零梯度假设，对梯度计算本身影响较小)
                    W_neighbor = W_i

                    # 界面值近似为两侧单元平均值
                W_face = 0.5 * (W_i + W_neighbor)

                # 累加 Green-Gauss 贡献
                grad_W_i[:, 0] += W_face * normal[0] * length  # sum( W_f * nx * L )
                grad_W_i[:, 1] += W_face * normal[1] * length  # sum( W_f * ny * L )

            gradients[i, :, :] = grad_W_i / cell.area  # 除以面积得到梯度

        return gradients

    def _calculate_barth_jespersen_limiters(self, W_state: np.ndarray, unlimited_gradients: np.ndarray) -> np.ndarray:
        """计算 Barth-Jespersen 限制器 phi (0 到 1)。"""
        num_cells = W_state.shape[0]
        num_vars = W_state.shape[1]
        phi = np.ones((num_cells, num_vars), dtype=np.float64)  # 初始化限制器为 1 (无限制)

        for i, cell in enumerate(self.mesh.cells):
            W_i = W_state[i, :]  # 当前单元中心值

            # 找到邻居单元中心值的最大最小值
            W_max_neighbors = np.copy(W_i)  # 初始化为当前单元值
            W_min_neighbors = np.copy(W_i)  # 初始化为当前单元值
            has_neighbors = False  # 标记是否有邻居

            neighbor_cells = []  # 获取邻居单元
            for he in cell.half_edges_list:  # 遍历半边
                if he.twin:  # 如果是内部边
                    neighbor_cells.append(he.twin.cell)  # 添加邻居单元
                    has_neighbors = True  # 标记有邻居

            if not has_neighbors: continue  # 如果没有邻居，无法限制，phi=1

            for neighbor_cell in neighbor_cells:  # 遍历邻居
                W_neighbor = W_state[neighbor_cell.id, :]  # 获取邻居值
                W_max_neighbors = np.maximum(W_max_neighbors, W_neighbor)  # 更新最大值
                W_min_neighbors = np.minimum(W_min_neighbors, W_neighbor)  # 更新最小值

            # 检查每个顶点处的重构值是否超出了邻域的范围
            grad_W_i = unlimited_gradients[i, :, :]  # 获取当前单元的无限制梯度
            cell_centroid = np.array(cell.centroid)  # 当前单元形心

            min_phi_cell = np.ones(num_vars)  # 当前单元所有顶点的最小phi (对每个变量独立)

            for node in cell.nodes:  # 遍历单元的顶点
                vec_to_vertex = np.array([node.x, node.y]) - cell_centroid  # 形心到顶点的向量

                # 计算顶点处的无限制重构值
                delta_W_vertex = np.dot(grad_W_i, vec_to_vertex)  # (num_vars,) = (num_vars, 2) dot (2,)
                W_vertex_unlimited = W_i + delta_W_vertex  # 计算顶点重构值

                # 对每个变量检查是否超限
                for var_idx in range(num_vars):  # 遍历 h, u, v
                    W_diff = W_vertex_unlimited[var_idx] - W_i[var_idx]  # 重构值与中心值的差

                    phi_k = 1.0  # 初始化当前顶点、当前变量的phi为1

                    if W_diff > 1e-9:  # 如果重构值大于中心值 (检查是否超过最大值)
                        max_allowed_diff = W_max_neighbors[var_idx] - W_i[var_idx]  # 允许的最大差值
                        # 避免除以零或极小的正数
                        if W_diff > max_allowed_diff + 1e-9:  # 允许一点点超出
                            phi_k = max(0.0, max_allowed_diff / W_diff if abs(W_diff) > 1e-9 else 0.0)  # 计算限制因子
                    elif W_diff < -1e-9:  # 如果重构值小于中心值 (检查是否低于最小值)
                        min_allowed_diff = W_min_neighbors[var_idx] - W_i[var_idx]  # 允许的最小差值 (负数)
                        # 避免除以零或极小的负数
                        if W_diff < min_allowed_diff - 1e-9:
                            phi_k = max(0.0, min_allowed_diff / W_diff if abs(W_diff) > 1e-9 else 0.0)  # 计算限制因子
                    # 如果 W_diff 接近0，phi_k 保持为 1.0

                    min_phi_cell[var_idx] = min(min_phi_cell[var_idx], phi_k)  # 更新该变量在此单元的最小phi

            phi[i, :] = min_phi_cell  # 将计算得到的最小phi赋给该单元的所有变量

        return phi  # 返回限制器因子数组

    # --- Main Reconstruction Method ---

    def get_reconstructed_interface_states(self, U_state: np.ndarray,
                                           cell_L: Cell, cell_R: Cell | None,
                                           he: HalfEdge,
                                           is_boundary: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """
        获取界面左右两侧重构后的原始变量状态 W = [h, u, v]。
        """

        # --- 获取左侧状态 W_L ---
        W_L_interface = self._get_primitive_state_for_cell(cell_L.id, U_state)  # 先获取中心值

        if self.scheme != ReconstructionSchemes.FIRST_ORDER:  # 如果是高阶
            if self.gradients is None:
                raise RuntimeError("梯度未计算 (高阶)。请先调用 prepare_for_step()")

            grad_W_L = self.gradients[cell_L.id, :, :]  # 获取(限制后)梯度
            vec_L_to_face = np.array(he.mid_point) - np.array(cell_L.centroid)  # 位移向量
            delta_W_L = np.dot(grad_W_L, vec_L_to_face)  # 计算差值
            W_L_interface += delta_W_L  # 加上梯度贡献

            # 确保物理性
            W_L_interface[0] = max(0.0, W_L_interface[0])  # 水深 >= 0
            if W_L_interface[0] < self.min_depth:  # 水深过小
                W_L_interface[1:] = 0.0  # 速度 = 0

        # --- 获取右侧状态 W_R ---
        W_R_interface = None  # 初始化为 None
        if not is_boundary and cell_R:  # 如果是内部边
            W_R_interface = self._get_primitive_state_for_cell(cell_R.id, U_state)  # 获取中心值

            if self.scheme != ReconstructionSchemes.FIRST_ORDER:  # 如果是高阶
                grad_W_R = self.gradients[cell_R.id, :, :]  # 获取梯度
                vec_R_to_face = np.array(he.mid_point) - np.array(cell_R.centroid)  # 位移向量
                delta_W_R = np.dot(grad_W_R, vec_R_to_face)  # 计算差值
                W_R_interface += delta_W_R  # 加上梯度贡献

                # 确保物理性
                W_R_interface[0] = max(0.0, W_R_interface[0])  # 水深 >= 0
                if W_R_interface[0] < self.min_depth:  # 水深过小
                    W_R_interface[1:] = 0.0  # 速度 = 0
        # 对于边界，W_R_interface 保持为 None，由 BoundaryConditionHandler 处理

        return W_L_interface, W_R_interface  # 返回左右状态