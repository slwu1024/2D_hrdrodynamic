# src/model/TimeIntegrator.py
import numpy as np
from enum import Enum


class TimeIntegrationSchemes(Enum):
    """时间积分方案枚举"""
    FORWARD_EULER = "forward_euler"
    RK2_SSP = "rk2_ssp"
    # Future: RK3_SSP = "rk3_ssp"


class TimeIntegrator:
    """执行时间积分步骤的类。"""

    def __init__(self, scheme: TimeIntegrationSchemes,
                 rhs_function: callable,
                 friction_function: callable,
                 num_vars: int):
        """
        初始化时间积分器。

        Args:
            scheme (TimeIntegrationSchemes): 使用的时间积分方案。
            rhs_function (callable): 计算显式部分 (对流+底坡) RHS 的函数。
                                     签名应为: rhs_function(U_state, time) -> RHS_array
            friction_function (callable): 应用半隐式摩擦的函数。
                                          签名应为: friction_function(U_input, U_coeffs, dt) -> U_output
            num_vars (int): 守恒变量的数量。
        """
        if not isinstance(scheme, TimeIntegrationSchemes):  # 检查方案类型
            raise TypeError(f"scheme must be a TimeIntegrationSchemes enum member, got {type(scheme)}")  # 抛出类型错误
        self.scheme = scheme
        self._calculate_rhs_explicit = rhs_function
        self._apply_friction = friction_function
        self.num_vars = num_vars
        print(f"  TimeIntegrator initialized with scheme: {self.scheme.value}")  # 打印初始化信息

    def step(self, U_current: np.ndarray, dt: float, time_current: float) -> np.ndarray:
        """
        执行一个时间积分步骤。

        Args:
            U_current (np.ndarray): 当前时刻的守恒量数组, shape=(num_cells, num_vars)。
            dt (float): 当前时间步长。
            time_current (float): 当前模拟时间。

        Returns:
            np.ndarray: 下一时刻的守恒量数组, shape=(num_cells, num_vars)。
        """

        if self.scheme == TimeIntegrationSchemes.FORWARD_EULER:  # --- 前向欧拉法 ---
            # 1. 计算显式 RHS (对流 + 底坡)
            RHS_expl = self._calculate_rhs_explicit(U_current, time_current)  # 计算显式RHS

            # 2. 显式更新步骤 (得到不含摩擦的下一步状态)
            U_next_expl = U_current + dt * RHS_expl  # 计算不含摩擦的下一步状态

            # 3. 应用半隐式摩擦 (使用 U_current 计算系数 τ)
            U_next = self._apply_friction(U_next_expl, U_current, dt)  # 应用摩擦

            return U_next  # 返回最终状态

        elif self.scheme == TimeIntegrationSchemes.RK2_SSP:  # --- 二阶 SSP 龙格-库塔法 ---
            # 采用算子分裂: 先完成显式 RK 步，然后应用摩擦算子

            # --- 显式 RK 步骤 ---
            # Stage 1
            RHS1 = self._calculate_rhs_explicit(U_current, time_current)  # 计算第一阶段RHS
            U_s1 = U_current + dt * RHS1  # 计算第一阶段中间状态 U(1)

            # Stage 2
            RHS2 = self._calculate_rhs_explicit(U_s1, time_current + dt)  # 计算第二阶段RHS (基于U(1))
            # RK 组合得到最终的、仅包含显式项贡献的状态
            U_rk_result_explicit_only = 0.5 * U_current + 0.5 * (U_s1 + dt * RHS2)  # 计算RK组合结果

            # --- 摩擦步骤 (算子分裂) ---
            # 对显式 RK 的结果应用半隐式摩擦
            # 使用时间步开始时的状态 U_current 来计算摩擦系数 τ，以符合 Lie splitting
            U_next = self._apply_friction(U_rk_result_explicit_only, U_current, dt)  # 应用摩擦

            return U_next  # 返回最终状态

        else:  # 其他未实现的方案
            raise NotImplementedError(f"Time integration scheme '{self.scheme.value}' is not implemented.")  # 抛出未实现错误