# src/model/SourceTerms.py
import numpy as np
from .MeshData import Mesh, Cell, HalfEdge, Node
from .Reconstruction import Reconstruction  # 导入 Reconstruction 用于类型提示 (如果底坡需要)


class SourceTermCalculator:
    """计算和处理模型源项（底坡、摩擦）的类。"""

    def __init__(self, gravity: float, min_depth: float):  # 修改参数名为 min_depth
        """
        初始化源项计算器。

        Args:
            gravity (float): 重力加速度 (m/s^2)。
            min_depth (float): 用于摩擦计算的最小水深阈值 (m)。
        """
        self.g = gravity
        self.min_depth = min_depth  # 存储统一的最小水深
        print(f"  源项计算器已初始化 (g={gravity}, min_depth={min_depth:.2e})")  # 打印初始化信息




    def apply_friction_semi_implicit(self,  # 应用半隐式摩阻
                                     U_input: np.ndarray,
                                     U_coeffs: np.ndarray,
                                     dt: float,
                                     manning_n_values: np.ndarray,
                                     ) -> np.ndarray:
        """
        对输入的守恒量状态 U_input 应用半隐式摩擦。
        摩擦系数 τ 的计算依赖于 U_coeffs 的状态。

        Args:
            U_input (np.ndarray): 需要施加摩擦的状态 [h, hu, hv], shape=(num_cells, 3)。
            U_coeffs (np.ndarray): 用于计算摩擦系数 τ 的状态 [h, hu, hv], shape=(num_cells, 3)。
            dt (float): 时间步长 (s)。
            manning_n_values (np.ndarray): 每个单元的曼宁糙率系数, shape=(num_cells,)。

        Returns:
            np.ndarray: 应用摩擦后的状态 [h, hu, hv], shape=(num_cells, 3)。
        """

        h_in = U_input[:, 0]  # 输入状态的水深 h
        hu_in = U_input[:, 1]  # 输入状态的 x 动量 hu
        hv_in = U_input[:, 2]  # 输入状态的 y 动量 hv

        # --- 从 U_coeffs 计算摩擦系数 τ ---
        # 使用统一的 min_depth 作为计算 τ 时的最小水深限制
        h_coeff = np.maximum(U_coeffs[:, 0], self.min_depth)  # 用于计算 τ 的水深 (限制最小深度)

        # 速度为0时，摩擦为0，避免除以过小的h
        u_coeff = np.zeros_like(h_coeff)  # 初始化u速度为0
        v_coeff = np.zeros_like(h_coeff)  # 初始化v速度为0
        # 使用统一的 min_depth 判断湿单元
        wet_mask_coeff = h_coeff > self.min_depth  # 找到用于计算系数的湿单元掩码

        # 仅在湿单元计算速度
        u_coeff[wet_mask_coeff] = U_coeffs[wet_mask_coeff, 1] / h_coeff[wet_mask_coeff]  # 计算u
        v_coeff[wet_mask_coeff] = U_coeffs[wet_mask_coeff, 2] / h_coeff[wet_mask_coeff]  # 计算v

        speed_coeff = np.sqrt(u_coeff ** 2 + v_coeff ** 2)  # 计算流速大小 V = sqrt(u^2 + v^2)

        # --- 计算摩擦项 τ = -g * n^2 * |V| / h^(4/3) ---
        tau = np.zeros_like(h_coeff)  # 初始化tau数组为零
        epsilon_speed = 1e-6  # 避免速度为零时产生 NaN 或不必要的计算的小速度阈值

        # 只在满足条件的湿单元计算 tau (水深 > min_depth 且速度 > epsilon_speed)
        calc_tau_mask = wet_mask_coeff & (speed_coeff > epsilon_speed)  # 同时满足水深和速度条件的掩码

        if np.any(calc_tau_mask):  # 如果存在需要计算tau的单元
            h_pow_4_3 = h_coeff[calc_tau_mask] ** (4.0 / 3.0)  # 计算 h^(4/3)
            # 确保分母不为零
            h_pow_4_3[h_pow_4_3 < 1e-12] = 1e-12  # 防止除零

            tau[calc_tau_mask] = -self.g * (manning_n_values[calc_tau_mask] ** 2) * \
                                 speed_coeff[calc_tau_mask] / h_pow_4_3  # 计算 tau

        # --- 应用半隐式公式 U_out = U_in / (1 - dt * tau) ---
        denominator = 1.0 - dt * tau  # 计算分母 (理论上 >= 1 因为 tau <= 0)
        # 防止分母过小或为负 (虽然理论上不应发生，但数值上可能)
        denominator = np.maximum(denominator, 1e-6)  # 限制分母的最小值

        # 对于输入状态 U_input，计算其对应的速度
        u_in = np.zeros_like(h_in)  # 初始化输入u
        v_in = np.zeros_like(h_in)  # 初始化输入v
        # 使用统一 min_depth 判断
        wet_mask_input = h_in > self.min_depth  # 输入状态的湿单元掩码
        # 仅为湿单元计算输入速度
        # 同样使用 h_in + epsilon 避免除零
        h_in_div = h_in[wet_mask_input] + 1e-12
        u_in[wet_mask_input] = hu_in[wet_mask_input] / h_in_div  # 计算输入u
        v_in[wet_mask_input] = hv_in[wet_mask_input] / h_in_div  # 计算输入v

        # 应用公式更新速度
        u_out = u_in / denominator  # 计算摩擦后的u
        v_out = v_in / denominator  # 计算摩擦后的v

        # --- 构造输出的守恒量 ---
        U_output = np.zeros_like(U_input)  # 初始化输出数组
        U_output[:, 0] = h_in  # 水深在摩擦步骤中不变
        # 对于干单元，u_out/v_out 应为0 (因为u_in/v_in为0或tau为0)
        U_output[:, 1] = u_out * h_in  # 更新 x 动量 hu
        U_output[:, 2] = v_out * h_in  # 更新 y 动量 hv

        # 对于干单元或速度很小的单元，其 tau 为 0，分母为 1，速度不变，结果也正确。

        return U_output  # 返回应用摩擦后的状态