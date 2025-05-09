# src/model/WettingDrying.py (或类似名称)
import numpy as np


class VFRCalculator:  # 体积自由面关系计算器类
    # 修改 __init__ 方法的参数列表，添加 relative_h_tolerance
    def __init__(self, min_depth=1e-6, min_eta_change_iter=1e-6, max_vfr_iters=20,
                 relative_h_tolerance=1e-4): # **添加 relative_h_tolerance 参数**
        """
        初始化VFR计算器。
        Args:
            min_depth (float): 计算中使用的最小水深阈值。
            min_eta_change_iter (float): VFR迭代中eta变化的绝对收敛阈值。
            max_vfr_iters (int): VFR迭代的最大次数。
            relative_h_tolerance (float): VFR迭代中计算水深与目标水深相对误差的收敛阈值。
        """
        self.min_depth = min_depth # 统一的最小水深阈值
        self.min_eta_change_iter = min_eta_change_iter # eta迭代绝对收敛阈值
        self.max_vfr_iters = max_vfr_iters # 最大迭代次数
        # 现在可以正确地使用传入的参数了
        self.relative_h_tolerance = relative_h_tolerance # **这行现在可以正确工作了**
        self.epsilon = 1e-12 # 用于避免除零的小量

    def _linear_interpolate(self, p1_coords, p1_z, p2_coords, p2_z, target_z):  # 线性插值辅助函数
        """在线段 p1-p2 上查找高程为 target_z 的点的 (x, y) 坐标。"""
        x1, y1 = p1_coords  # 点1坐标
        x2, y2 = p2_coords  # 点2坐标
        # 避免除零
        delta_z = p2_z - p1_z  # 高程差
        if abs(delta_z) < self.epsilon:  # 如果高程几乎相等
            # 如果 target_z 也接近，理论上边上所有点都符合，返回中点
            if abs(target_z - p1_z) < self.epsilon:
                return ((x1 + x2) / 2, (y1 + y2) / 2)
            else:  # 目标高程不在此范围，无法插值（理论上不应发生）
                # 返回一个端点或引发错误，这里返回p1
                # print(f"警告：线性插值目标 {target_z} 不在范围 [{p1_z}, {p2_z}] 内")
                return (x1, y1)  # 或返回None更好？

        # 计算插值比例 t
        t = (target_z - p1_z) / delta_z  # 插值比例
        # 限制 t 在 [0, 1] 之间，确保是内插
        t = np.clip(t, 0.0, 1.0)

        x_int = x1 + t * (x2 - x1)  # 计算插值点 x 坐标
        y_int = y1 + t * (y2 - y1)  # 计算插值点 y 坐标
        return (x_int, y_int)  # 返回插值点坐标

    def _polygon_area(self, vertices):  # 计算多边形面积辅助函数 (坐标列表)
        """使用 Shoelace 公式计算任意简单多边形的面积。"""
        n = len(vertices)  # 顶点数量
        if n < 3:  # 点或线段没有面积
            return 0.0
        area = 0.0  # 初始化面积
        for i in range(n):  # 遍历顶点
            j = (i + 1) % n  # 下一个顶点索引
            area += vertices[i][0] * vertices[j][1]  # 累加 x_i * y_{i+1}
            area -= vertices[j][0] * vertices[i][1]  # 减去 x_{i+1} * y_i
        return abs(area) / 2.0  # 返回面积绝对值的一半
    # --- _calculate_wet_surface_area 方法 (使用上一轮回答中改进的版本) ---
    def _calculate_wet_surface_area(self, eta, b_sorted, cell_nodes, cell_total_area):  # 计算给定eta下的水面面积 Aw (精确几何版)
        """
        根据水位eta和排序后的顶点底高程b_sorted计算三角形单元的水面面积 Aw。
        使用精确几何计算。
        Args:
            eta (float): 当前水面高程。
            b_sorted (list/array): 已排序的顶点底高程 [b1, b2, b3]。
            cell_nodes (list): 单元的 Node 对象列表 [node1, node2, node3]，顺序需与 b_sorted 对应。
            cell_total_area (float): 单元的总面积。
        Returns:
            float: 水面面积 Aw。
        """
        b1, b2, b3 = b_sorted[0], b_sorted[1], b_sorted[2]  # 获取排序后的顶点高程
        v = cell_nodes  # 假设 cell_nodes 顺序与 b_sorted 对应

        v_coords = [(node.x, node.y) for node in v]  # 获取顶点坐标列表 [(x1,y1), (x2,y2), (x3,y3)]
        v_z = [node.z_bed for node in v]  # 获取顶点高程列表 [z1, z2, z3] (应等于 b_sorted)

        if not np.allclose(v_z, b_sorted):  # 添加检查确保顺序一致
            print(
                f"警告: _calculate_wet_surface_area 中 cell_nodes 的高程顺序与 b_sorted 不一致！结果可能错误。 v_z={v_z}, b_sorted={b_sorted}")
            # 可以尝试根据 b_sorted 重新排序 v 和 v_coords，但这比较复杂且低效
            # 更好的方式是保证调用者传入正确的顺序

        # --- 分情况计算水面面积 Aw ---
        if eta <= b1 + self.epsilon:  # 情况1: 全干
            return 0.0  # 面积为0

        elif eta >= b3 - self.epsilon:  # 情况2: 全湿
            return cell_total_area  # 面积等于总面积

        elif eta <= b2 + self.epsilon:  # 情况3: 部分淹没，水面低于b2 (只有顶点v[0]被淹)
            # 水线交点 P12 在边 v[0]-v[1] 上, P13 在边 v[0]-v[2] 上
            p12_coords = self._linear_interpolate(v_coords[0], b1, v_coords[1], b2, eta)  # 计算交点P12坐标
            p13_coords = self._linear_interpolate(v_coords[0], b1, v_coords[2], b3, eta)  # 计算交点P13坐标
            # 湿区是小三角形 v[0]-P12-P13
            wet_vertices = [v_coords[0], p12_coords, p13_coords]  # 湿区顶点列表
            Aw = self._polygon_area(wet_vertices)  # 计算湿区面积
            return Aw  # 返回面积

        else:  # 情况4: 部分淹没，水面介于b2和b3之间 (顶点v[0]和v[1]被淹)
            # 水线交点 P13 在边 v[0]-v[2] 上, P23 在边 v[1]-v[2] 上
            p13_coords = self._linear_interpolate(v_coords[0], b1, v_coords[2], b3, eta)  # 计算交点P13坐标
            p23_coords = self._linear_interpolate(v_coords[1], b2, v_coords[2], b3, eta)  # 计算交点P23坐标
            # 湿区是四边形 v[0]-v[1]-P23-P13
            wet_vertices = [v_coords[0], v_coords[1], p23_coords, p13_coords]  # 湿区顶点列表
            Aw = self._polygon_area(wet_vertices)  # 计算湿区面积
            return Aw  # 返回面积



    def get_h_from_eta(self, eta, b_sorted, cell_total_area, cell_id_for_debug=""):
        """
        根据水面高程 (eta) 计算单元平均水深 (h_avg)。
        (此实现基于OCR识别的简化公式(2-56)。)
        """
        # ... (代码同上一个回答中的 get_h_from_eta) ...
        b1, b2, b3 = b_sorted[0], b_sorted[1], b_sorted[2]  # 获取排序后的顶点高程
        if eta <= b1: return 0.0  # 全干
        epsilon = 1e-12  # 避免除零

        if eta <= b2:  # 部分淹没1
            denominator = 2 * (b2 - b1 + epsilon) * (b3 - b1 + epsilon)
            if abs(denominator) < epsilon: return max(0.0, eta - b1)
            h_avg = (eta - b1) ** 3 / denominator
            return max(0.0, h_avg)
        elif eta <= b3:  # 部分淹没2
            term_numerator = eta ** 2 + eta * b3 - 3 * eta * b1 - b1 * b3 + b1 * b2 + b1 ** 2
            denominator = 3 * (b3 - b1 + epsilon)
            if abs(denominator) < epsilon: return max(0.0, eta - (b1 + b2) / 2.0)
            h_avg = term_numerator / denominator
            return max(0.0, h_avg)
        else:  # 全淹没
            h_avg = eta - (b1 + b2 + b3) / 3.0
            return max(0.0, h_avg)


    # --- get_eta_from_h 方法 (包含牛顿法和初始猜测逻辑) ---
    def get_eta_from_h(self, h_avg, b_sorted, cell_nodes, cell_total_area,
                       eta_previous_guess=None, cell_id_for_debug=""): # 添加 eta_previous_guess 参数
        """
        通过牛顿法迭代求解 H(eta) = h_avg 来计算水面高程 (eta)。
        Args:
            h_avg (float): 单元平均水深。
            b_sorted (list/array): 已排序的顶点底高程 [b1, b2, b3]。
            cell_nodes (list): 单元的 Node 对象列表，顺序需与 b_sorted 对应。
            cell_total_area (float): 单元的总面积。
            eta_previous_guess (float | None): 上一个时间步的eta值，作为初始猜测。
            cell_id_for_debug (str): 用于调试输出的单元ID。
        Returns:
            float: 计算得到的水面高程 eta。
        """
        b1, b2, b3 = b_sorted[0], b_sorted[1], b_sorted[2] # 获取排序顶点高程

        # --- 干判断 ---
        if h_avg < self.min_depth / 10: # 如果目标平均水深非常小
            return b1 # 返回最低点高程

        # --- 设置初始猜测值 eta_k ---
        if eta_previous_guess is not None and eta_previous_guess >= b1 - self.epsilon: # 如果提供了上一步的值且合理
            eta_k = eta_previous_guess # 使用上一步的值
        else: # 否则，使用基于当前信息的默认猜测
            # eta_k = b1 + h_avg # 策略1: 最低点 + 平均水深
            eta_k = (b1 + b2 + b3) / 3.0 + h_avg # 策略2: 形心高程 + 平均水深 (可能更鲁棒)
            eta_k = max(b1, eta_k) # 确保初始猜测不低于最低点

        # --- 牛顿法迭代 ---
        f_k = 1.0 # 初始化一个非零值，以便进入循环
        eta_k_next = eta_k # 初始化下一个迭代值

        for iter_count in range(self.max_vfr_iters): # 开始迭代
            # 1. 计算当前eta_k对应的平均水深 h_calc_k
            h_calc_k = self.get_h_from_eta(eta_k, b_sorted, cell_total_area, cell_id_for_debug) # 调用h(eta)函数

            # 2. 计算目标函数值 f(eta_k) = h_calc_k - h_avg
            f_k = h_calc_k - h_avg # 计算当前迭代的误差

            # 3. 检查收敛性 (结合绝对和相对误差)
            abs_err_converged = abs(f_k) < self.min_depth * self.relative_h_tolerance # 绝对误差是否满足
            rel_err_converged = abs(f_k / (h_avg + self.epsilon)) < self.relative_h_tolerance # 相对误差是否满足
            if abs_err_converged or rel_err_converged: # 如果任一满足
                break # 达到收敛，跳出循环

            # 4. 计算导数 df/d(eta) = Aw(eta) / cell_total_area
            # **需要确保 _calculate_wet_surface_area 和 get_h_from_eta 使用的节点顺序一致**
            Aw_k = self._calculate_wet_surface_area(eta_k, b_sorted, cell_nodes, cell_total_area) # 计算水面面积
            df_deta_k = Aw_k / cell_total_area if cell_total_area > self.epsilon else self.epsilon # 计算导数

            # 5. 处理导数过小的情况
            if abs(df_deta_k) < self.epsilon: # 如果导数接近零
                # print(f"Debug (get_eta_from_h, cell {cell_id_for_debug}): Derivative near zero. eta_k={eta_k:.4f}, Aw_k={Aw_k:.4e}, f_k={f_k:.4e}")
                # 导数过小，牛顿法失效。可以停止迭代，或采取小步长试探。
                # 这里选择停止迭代，并接受当前的eta_k。或者可以尝试一个小的固定步长调整：
                eta_k_next = eta_k - np.sign(f_k) * self.min_eta_change_iter * 10 # 尝试小步长调整
                # 限制调整后的值
                eta_k_next = max(b1, eta_k_next)
                # 如果小步长调整后变化也很小，也认为收敛
                if abs(eta_k_next - eta_k) < self.min_eta_change_iter:
                    eta_k = eta_k_next
                    break
            else: # 正常牛顿步
                # 6. 计算牛顿步长 delta_eta = f_k / df_deta_k
                delta_eta = f_k / df_deta_k # 计算修正量

                # 7. 限制步长 (增加鲁棒性)
                max_delta_eta = (b3 - b1 + 1.0) * 0.5 # 允许的最大步长，例如高程范围的一半加1
                delta_eta = np.clip(delta_eta, -max_delta_eta, max_delta_eta) # 限制步长防止跳跃过大

                # 8. 更新 eta_k_next = eta_k - delta_eta
                eta_k_next = eta_k - delta_eta # 牛顿法更新

            # 9. 限制 eta_k_next 的物理下界
            eta_k_next = max(b1, eta_k_next) # 确保水位不低于最低点高程

            # 10. 检查eta变化量是否小于阈值 (也可作为收敛条件)
            if abs(eta_k_next - eta_k) < self.min_eta_change_iter:
                eta_k = eta_k_next # 更新eta
                break # 变化足够小，认为收敛

            eta_k = eta_k_next # 更新eta_k，进行下一次迭代

        # --- 迭代结束后的处理 ---
        # 检查最终的收敛情况，如果未达到容差，打印警告
        if iter_count == self.max_vfr_iters - 1 and not (abs_err_converged or rel_err_converged):
            print(f"警告 (get_eta_from_h, cell {cell_id_for_debug}): VFR牛顿法未能在 {self.max_vfr_iters} 次内收敛。目标h_avg={h_avg:.4e}, 最终eta={eta_k:.4f}, 计算得h={self.get_h_from_eta(eta_k, b_sorted, cell_total_area):.4e}, f_k={f_k:.4e}")

        return max(b1, eta_k) # 返回最终计算得到的eta，并确保其不小于最低点高程