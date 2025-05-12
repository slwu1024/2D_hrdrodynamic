// src_cpp/src/FluxCalculator_cpp.cpp
#include "FluxCalculator_cpp.h" // 包含对应的头文件
#include <iostream> // 包含输入输出流

namespace HydroCore { // 定义HydroCore命名空间

FluxCalculator_cpp::FluxCalculator_cpp(double gravity, double min_depth_param, RiemannSolverType_cpp solver_type) // 构造函数实现
    : g(gravity), min_depth(min_depth_param), solver_type_internal(solver_type) { // 初始化列表
    // std::cout << "C++ FluxCalculator_cpp initialized (g=" << g << ", min_depth=" << min_depth << ")" << std::endl; // 打印初始化信息（可选）
} // 结束构造函数

std::array<double, 3> FluxCalculator_cpp::calculate_hllc_flux( // HLLC通量计算方法实现
    const PrimitiveVars_cpp& W_L_in, // 左侧原始变量
    const PrimitiveVars_cpp& W_R_in, // 右侧原始变量
    const std::array<double, 2>& normal_vec // 法向量
) { // 开始方法体
    // --- 步骤 0: 预处理和旋转 ---
    double nx = normal_vec[0]; // 法向量x分量
    double ny = normal_vec[1]; // 法向量y分量

    PrimitiveVars_cpp W_L = W_L_in; // 复制左侧状态（允许修改，例如干单元处理）
    PrimitiveVars_cpp W_R = W_R_in; // 复制右侧状态

    bool dryL = (W_L.h < min_depth); // 判断左侧是否为干
    bool dryR = (W_R.h < min_depth); // 判断右侧是否为干

    if (dryL && dryR) { // 如果两侧都干
        return {0.0, 0.0, 0.0}; // 无通量，直接返回
    }

    double unL, utL, unR, utR; // 声明法向和切向速度
    double cL, cR; // 声明波速

    // 将速度旋转到法向 (un) 和切向 (ut)
    unL = W_L.u * nx + W_L.v * ny; // 左侧法向速度
    utL = -W_L.u * ny + W_L.v * nx; // 左侧切向速度
    unR = W_R.u * nx + W_R.v * ny; // 右侧法向速度
    utR = -W_R.u * ny + W_R.v * nx; // 右侧切向速度

    if (dryL) { // 如果左侧为干
        W_L.h = 0.0; // 水深设为0
        cL = 0.0;    // 波速设为0
        if (!dryR) { // 如果右侧湿
            // 根据反射或某种假设设定左侧速度
            unL = -unR; // 法向速度反向
            utL = utR;  // 切向速度相同
            // cL 仍然是 0
            cR = std::sqrt(g * W_R.h); // 计算右侧波速
        } else { // (dryR is true) 两侧都干，已在前面处理
            cR = 0.0; // 右侧波速也为0
        }
    } else if (dryR) { // 如果右侧为干 (左侧必湿)
        W_R.h = 0.0; // 水深设为0
        cR = 0.0;    // 波速设为0
        unR = -unL; // 法向速度反向
        utR = utL;  // 切向速度相同
        cL = std::sqrt(g * W_L.h); // 计算左侧波速
        // cR 仍然是 0
    } else { // 两侧都湿
        cL = std::sqrt(g * W_L.h); // 计算左侧波速
        cR = std::sqrt(g * W_R.h); // 计算右侧波速
    }

    // --- 步骤 1: 计算波速 sL, sR (Einfeldt/Davis + 干底修正) ---
    double sqrt_hL = (W_L.h > 0) ? std::sqrt(W_L.h) : 0.0; // 计算左侧水深平方根
    double sqrt_hR = (W_R.h > 0) ? std::sqrt(W_R.h) : 0.0; // 计算右侧水深平方根
    double sqrt_sum = sqrt_hL + sqrt_hR; // 计算平方根之和

    double un_roe, h_roe_for_c, c_roe; // 声明Roe平均值
    if (sqrt_sum < 1e-9) { // 如果平方根之和过小
        un_roe = 0.5 * (unL + unR); // Roe平均法向速度用简单平均
        h_roe_for_c = 0.5 * (W_L.h + W_R.h); // Roe平均水深用简单平均
    } else { // 否则正常计算
        un_roe = (sqrt_hL * unL + sqrt_hR * unR) / sqrt_sum; // 计算Roe平均法向速度
        h_roe_for_c = 0.5 * (W_L.h + W_R.h); // 标准Roe平均水深 h_hat = 0.5*(hL+hR)
    }
    c_roe = (h_roe_for_c > 0) ? std::sqrt(g * h_roe_for_c) : 0.0; // 计算Roe平均波速

    double sL_wet = un_roe - c_roe; // Roe左波速（湿）
    double sR_wet = un_roe + c_roe; // Roe右波速（湿）
    double sL_simple = unL - cL; // 简单左波速
    double sR_simple = unR + cR; // 简单右波速

    double sL_davis = std::min(sL_simple, sL_wet); // Davis左波速
    double sR_davis = std::max(sR_simple, sR_wet); // Davis右波速

    double sL, sR; // 声明最终波速
    if (dryL) { // 如果左侧为干
        sL = unR - 2 * cR; // Toro真空波速
    } else { // 否则
        sL = sL_davis; // 使用Davis估算
    }
    if (dryR) { // 如果右侧为干
        sR = unL + 2 * cL; // Toro真空波速
    } else { // 否则
        sR = sR_davis; // 使用Davis估算
    }

    if (sL > sR - 1e-9) { // 如果波速交叉或非常接近 (允许一点数值误差)
        // 这种情况下，HLLC退化为HLL，或者可能意味着一个接触间断
        // 简单的处理是强制它们分开一点，或者采用更鲁棒的平均值
        // std::cerr << "Warning: HLLC wave speeds crossed or too close sL=" << sL << ", sR=" << sR << ". Adjusting." << std::endl; // 打印警告
        double s_avg = 0.5 * (sL + sR); // 计算平均波速
        sL = s_avg - 1e-6; // 强制分开
        sR = s_avg + 1e-6; // 强制分开
        if (sL > sR) { // 极端情况的最后防线
             sL = std::min(sL_simple,sR_simple) - 1e-6; // 使用简单波速再调整
             sR = std::max(sL_simple,sR_simple) + 1e-6; // 使用简单波速再调整
        }
    }


    // --- 步骤 2: 计算中间波速 s_star ---
    double PL = 0.5 * g * W_L.h * W_L.h; // 计算左侧压力
    double PR = 0.5 * g * W_R.h * W_R.h; // 计算右侧压力

    double den_s_star = W_L.h * (sL - unL) - W_R.h * (sR - unR); // 计算s_star分母
    double s_star; // 声明中间波速
    double epsilon_den = 1e-9; // 分母的容差

    if (std::abs(den_s_star) < epsilon_den) { // 如果分母接近零
        s_star = un_roe; // 近似为Roe平均速度
    } else { // 否则
        s_star = (PR - PL + W_L.h * unL * (sL - unL) - W_R.h * unR * (sR - unR)) / den_s_star; // 计算中间波速
    }


    // --- 步骤 3: 根据区域计算通量 F_hllc_nt (在法向-切向坐标系) ---
    std::array<double, 3> FL_nt = {W_L.h * unL, W_L.h * unL * unL + PL, W_L.h * unL * utL}; // 左侧法向-切向通量
    std::array<double, 3> FR_nt = {W_R.h * unR, W_R.h * unR * unR + PR, W_R.h * unR * utR}; // 右侧法向-切向通量
    std::array<double, 3> F_hllc_nt; // 声明HLLC法向-切向通量

    if (sL >= 0) { // 区域 L
        F_hllc_nt = FL_nt; // 通量等于左侧通量
    } else if (sR <= 0) { // 区域 R
        F_hllc_nt = FR_nt; // 通量等于右侧通量
    } else { // 星区 (*L or *R)
        std::array<double, 3> UL_nt = {W_L.h, W_L.h * unL, W_L.h * utL}; // 左侧法向-切向守恒量
        std::array<double, 3> UR_nt = {W_R.h, W_R.h * unR, W_R.h * utR}; // 右侧法向-切向守恒量

        if (s_star >= 0) { // 区域 *L
            double h_starL_num = W_L.h * (sL - unL); // 左星区水深分子
            double h_starL_den = sL - s_star; // 左星区水深分母
            double h_starL = (std::abs(h_starL_den) < epsilon_den) ? W_L.h : h_starL_num / h_starL_den; // 计算左星区水深（处理分母为0）
            h_starL = std::max(0.0, h_starL); // 保证水深非负

            std::array<double, 3> U_starL_nt = {h_starL, h_starL * s_star, h_starL * utL}; // 左星区守恒量
            F_hllc_nt[0] = FL_nt[0] + sL * (U_starL_nt[0] - UL_nt[0]); // 计算通量
            F_hllc_nt[1] = FL_nt[1] + sL * (U_starL_nt[1] - UL_nt[1]); // 计算通量
            F_hllc_nt[2] = FL_nt[2] + sL * (U_starL_nt[2] - UL_nt[2]); // 计算通量
        } else { // 区域 *R
            double h_starR_num = W_R.h * (sR - unR); // 右星区水深分子
            double h_starR_den = sR - s_star; // 右星区水深分母
            double h_starR = (std::abs(h_starR_den) < epsilon_den) ? W_R.h : h_starR_num / h_starR_den; // 计算右星区水深（处理分母为0）
            h_starR = std::max(0.0, h_starR); // 保证水深非负

            std::array<double, 3> U_starR_nt = {h_starR, h_starR * s_star, h_starR * utR}; // 右星区守恒量
            F_hllc_nt[0] = FR_nt[0] + sR * (U_starR_nt[0] - UR_nt[0]); // 计算通量
            F_hllc_nt[1] = FR_nt[1] + sR * (U_starR_nt[1] - UR_nt[1]); // 计算通量
            F_hllc_nt[2] = FR_nt[2] + sR * (U_starR_nt[2] - UR_nt[2]); // 计算通量
        }
    }

    // --- 步骤 4: 将 F_hllc_nt 旋转回笛卡尔坐标系 F_hllc_cartesian ---
    double Fh_n = F_hllc_nt[0]; // 法向质量通量
    double Fun_n = F_hllc_nt[1]; // 法向动量通量的法向分量
    double Fut_n = F_hllc_nt[2]; // 法向动量通量的切向分量

    std::array<double, 3> F_hllc_cartesian; // 声明笛卡尔坐标系通量
    F_hllc_cartesian[0] = Fh_n; // 质量通量是标量
    F_hllc_cartesian[1] = Fun_n * nx - Fut_n * ny; // x方向动量通量
    F_hllc_cartesian[2] = Fun_n * ny + Fut_n * nx; // y方向动量通量

    return F_hllc_cartesian; // 返回最终结果
} // 结束方法体

} // namespace HydroCore