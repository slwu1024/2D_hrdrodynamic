# src/model/FluxCalculator.py
import numpy as np  # 导入NumPy库
from enum import Enum  # 导入枚举类


class RiemannSolvers(Enum):  # 定义黎曼求解器的枚举类（目前只有HLLC）
    HLLC = "hllc"  # HLLC求解器
    # Add Roe, AUSM, etc. in the future


class FluxCalculator:  # 数值通量计算器类
    def __init__(self, solver_type, gravity, min_depth: float): # 修改参数名为 min_depth
        self.min_h_solver = None
        self.solver_type = solver_type  # 求解器类型
        self.g = gravity  # 重力加速度
        self.min_depth = min_depth  # 存储统一的最小水深 (替换 min_h_solver)

    def calculate_flux(self, vars_L, vars_R, normal_vector_L_to_R):  # 计算数值通量
        """
        Calculates the numerical flux across an interface.
        vars_L, vars_R: Reconstructed primitive or conserved variables at the left/right of the interface.
                        Assuming [h, u, v] (primitive) or [h, hu, hv] (conserved) AFTER rotation to normal-tangential if needed.
                        Or, the HLLC solver here handles the rotation.
        normal_vector_L_to_R: Numpy array [nx, ny]
        Returns: numerical flux vector [Fh, Fhu, Fhv] in the original Cartesian coordinates,
                 acting from L to R (i.e., in the direction of normal_vector_L_to_R).
        """
        if self.solver_type == RiemannSolvers.HLLC:  # 如果是HLLC求解器
            return self._hllc_solver(vars_L, vars_R, normal_vector_L_to_R)  # 调用HLLC求解器
        else:  # 如果是不支持的求解器
            raise NotImplementedError(f"Solver {self.solver_type} not implemented.")  # 抛出未实现错误

    def _hllc_solver(self, W_L, W_R, normal_vec):
        """HLLC approximate Riemann solver for SWE."""

        # --- 步骤 0: 预处理和旋转 ---
        nx, ny = normal_vec[0], normal_vec[1]  # 法向量分量

        hL, uL, vL = W_L[0], W_L[1], W_L[2]  # 左侧原始变量
        hR, uR, vR = W_R[0], W_R[1], W_R[2]  # 右侧原始变量

        dryL = (hL < self.min_depth)  # 判断左侧是否为干
        dryR = (hR < self.min_depth)  # 判断右侧是否为干

        if dryL and dryR:  # 如果两侧都干
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)  # 无通量，直接返回

        # 如果单侧为干，在旋转前先处理（或者在旋转后处理 un, ut）
        # 暂时先旋转，然后在 un/ut/cL/cR 计算时考虑干状态

        # 将速度旋转到法向 (un) 和切向 (ut)
        unL = uL * nx + vL * ny  # 左侧法向速度
        utL = -uL * ny + vL * nx  # 左侧切向速度 (法向 (nx,ny), 切向 (-ny,nx))
        unR = uR * nx + vR * ny  # 右侧法向速度
        utR = -uR * ny + vR * nx  # 右侧切向速度

        cL, cR = 0.0, 0.0  # 初始化波速
        if dryL:
            hL = 0.0
            cL = 0.0
            if not dryR:  # 如果右侧湿
                unL = -unR
                utL = utR
                cL = np.sqrt(self.g * hL)  # 这还是0
                cR = np.sqrt(self.g * hR)  # 只有右侧波速
            # 如果两侧都干，已经在前面返回
        elif dryR:
            hR = 0.0
            cR = 0.0
            # 左侧湿
            unR = -unL
            utR = utL
            cL = np.sqrt(self.g * hL)  # 只有左侧波速
            cR = np.sqrt(self.g * hR)  # 这还是0
        else:  # 两侧都湿
            cL = np.sqrt(self.g * hL)
            cR = np.sqrt(self.g * hR)

        # --- 步骤 1: 计算波速 sL, sR ---
        # 使用 Einfeldt 波速 + 干底修正

        # 首先计算 Roe 平均值 (即使在一侧为干时也要处理分母)
        sqrt_hL = np.sqrt(hL) if hL > 0 else 0.0  # 计算左侧水深平方根
        sqrt_hR = np.sqrt(hR) if hR > 0 else 0.0  # 计算右侧水深平方根
        sqrt_sum = sqrt_hL + sqrt_hR  # 计算平方根之和

        # 防止除零，如果两侧水深都接近零（理论上应该在 dryL and dryR 时返回了）
        if sqrt_sum < 1e-9:
            # 这种情况意味着 hL 和 hR 都非常小，Roe 平均无意义
            # 可以退化为简单波速或基于非干侧状态（如果只有一侧干）
            un_roe = 0.5 * (unL + unR)  # 简单平均速度
            h_roe = 0.5 * (hL + hR)  # 简单平均水深 (如果两侧干则为0)
            # 或者是 un_roe = unL if dryR else unR (用湿侧状态)
        else:
            un_roe = (sqrt_hL * unL + sqrt_hR * unR) / sqrt_sum  # 计算Roe平均法向速度
            # 不需要 h_roe 来算 c_roe，可以直接用 c_roe = sqrt(g * h_roe)
            # 但更常见的 Einfeldt 是用 un_roe +/- c_roe
            h_roe = sqrt_hL * sqrt_hR  # (另一种Roe平均水深h_roe = sqrt(hL*hR) 可能更好?)
            # 或者 h_roe = 0.5*(hL+hR) is standard
            # 为了计算 c_roe，还是用 h_roe = 0.5*(hL+hR)
            h_roe = 0.5 * (hL + hR)  # 标准Roe平均水深

        c_roe = np.sqrt(self.g * h_roe) if h_roe > 0 else 0.0  # 计算Roe平均波速

        # Einfeldt/Davis 估算
        sL_wet = un_roe - c_roe  # Roe左波速（湿）
        sR_wet = un_roe + c_roe  # Roe右波速（湿）
        sL_simple = unL - cL  # 简单左波速（若左湿）
        sR_simple = unR + cR  # 简单右波速（若右湿）

        sL_davis = min(sL_simple, sL_wet)  # Davis左波速
        sR_davis = max(sR_simple, sR_wet)  # Davis右波速

        # 应用干底修正
        if dryL:  # 如果左侧为干
            # sL = unR - 2 * cR  # Toro's vacuum generation wave speed
            # 检查Ying&Wang公式: sL = unR - 2*cR (将V替换为un) - 看起来一致
            sL = unR - 2 * cR  # 使用右侧信息估计左波速
        else:
            sL = sL_davis  # 否则使用Davis估算

        if dryR:  # 如果右侧为干
            # sR = unL + 2 * cL # Toro's vacuum generation wave speed
            sR = unL + 2 * cL  # 使用左侧信息估计右波速
        else:
            sR = sR_davis  # 否则使用Davis估算

        # 确保 sL <= sR (物理上和对于后续计算是必要的)
        # 虽然理论上 Davis 估算能保证, 但数值上可能出现问题, 尤其干湿边界
        # 简单的处理：如果 sL > sR，可能是状态有问题或者数值误差
        if sL > sR:
            # 交换，或者退化到简单的 HLL (用 0.5*(F_L+F_R) + 0.5*(sR+sL)/(sR-sL)*(U_R-U_L)?)
            # 或者使用更鲁棒的平均值作为速度？
            # print(f"Warning: HLLC wave speeds crossed sL={sL:.2f}, sR={sR:.2f}. Swapping or handling.")
            # sL, sR = min(sL, sR), max(sL, sR) # 简单交换可能掩盖问题
            # 一个常见处理方法是退化为 HLL 或者强制一个最小速度差
            if abs(sL - sR) < 1e-6:  # 如果速度几乎相等
                s_star = 0.5 * (sL + sR)  # 可以将中间波速设为平均值
            else:  # 尝试保持顺序
                sL = min(sL, sR - 1e-6)  # 强制sL略小于sR
                sR = max(sR, sL + 1e-6)  # 强制sR略大于sL

        # --- 步骤 1 完成 ---

        # --- 步骤 2: 计算中间波速 s_star ---
        PL = 0.5 * self.g * hL ** 2  # 计算左侧压力项
        PR = 0.5 * self.g * hR ** 2  # 计算右侧压力项

        # 论文/Toro 公式 (需要 careful 避免分母为0)
        den_s_star = hL * (sL - unL) - hR * (sR - unR)  # 计算分母
        if abs(den_s_star) < 1e-9:  # 如果分母接近零
            # 这种情况可能发生在特征速度等于流速时，或两侧状态很接近
            s_star = 0.5 * (unL + unR)  # 近似为平均速度或 0? 设为平均速度更合理
            # 或者 s_star = u_roe # 设为Roe平均速度
            # 这里先用简单平均值
            s_star = un_roe
        else:
            s_star = (PR - PL + hL * unL * (sL - unL) - hR * unR * (sR - unR)) / den_s_star

        # --- 步骤 3: 根据区域计算通量 F_hllc_nt (在法向-切向坐标系) ---
        # 定义左右通量 (法向-切向)
        FL_nt = np.array([hL * unL,
                          hL * unL ** 2 + PL,
                          hL * unL * utL], dtype=np.float64)
        FR_nt = np.array([hR * unR,
                          hR * unR ** 2 + PR,
                          hR * unR * utR], dtype=np.float64)

        if sL >= 0:  # Region L
            F_hllc_nt = FL_nt  # 通量等于左侧通量
        elif sR <= 0:  # Region R
            F_hllc_nt = FR_nt  # 通量等于右侧通量
        else:  # Star Region (*L or *R)
            # 定义左右守恒量 (法向-切向)
            UL_nt = np.array([hL, hL * unL, hL * utL], dtype=np.float64)
            UR_nt = np.array([hR, hR * unR, hR * utR], dtype=np.float64)

            if s_star >= 0:  # Region *L (interface to the right of contact)
                # Calculate U_starL
                h_starL = hL * (sL - unL) / (sL - s_star + 1e-9)  # 左星区水深
                U_starL_nt = np.array([h_starL,
                                       h_starL * s_star,  # 法向动量 hu_n* = h* * s*
                                       h_starL * utL  # 切向动量 hu_t* = h* * u_tL (切向速度不变)
                                       ], dtype=np.float64)
                # Calculate F_starL using HLL relationship F*L = FL + sL(U*L - UL)
                F_hllc_nt = FL_nt + sL * (U_starL_nt - UL_nt)  # 计算左星区通量

            else:  # Region *R (s_star < 0) (interface to the left of contact)
                # Calculate U_starR
                h_starR = hR * (sR - unR) / (sR - s_star + 1e-9)  # 右星区水深
                U_starR_nt = np.array([h_starR,
                                       h_starR * s_star,  # 法向动量 hu_n* = h* * s*
                                       h_starR * utR  # 切向动量 hu_t* = h* * u_tR (切向速度不变)
                                       ], dtype=np.float64)
                # Calculate F_starR using HLL relationship F*R = FR + sR(U*R - UR)
                F_hllc_nt = FR_nt + sR * (U_starR_nt - UR_nt)  # 计算右星区通量

        # --- 步骤 3 完成 ---

        # --- 步骤 4: 将 F_hllc_nt 旋转回笛卡尔坐标系 F_hllc_cartesian ---
        # F_hllc_nt = [Fh_n, Fun_n, Fut_n] (下标n表示normal-tangential frame)
        # F_cartesian = [Fh_c, Fhu_c, Fhv_c]
        Fh_n, Fun_n, Fut_n = F_hllc_nt[0], F_hllc_nt[1], F_hllc_nt[2]  # 获取法向切向通量分量

        # Rotation: (assuming normal_vec=[nx, ny], tangential_vec=[-ny, nx])
        # Cartesian_hu = NormalFlux_un * nx + TangentialFlux_ut * (-ny)
        # Cartesian_hv = NormalFlux_un * ny + TangentialFlux_ut * nx
        Fh_c = Fh_n  # 质量通量是标量，不用旋转
        Fhu_c = Fun_n * nx - Fut_n * ny  # x方向动量通量
        Fhv_c = Fun_n * ny + Fut_n * nx  # y方向动量通量

        F_hllc_cartesian = np.array([Fh_c, Fhu_c, Fhv_c], dtype=np.float64)  # 最终笛卡尔通量

        # --- 步骤 4 完成 ---

        return F_hllc_cartesian  # 返回最终结果