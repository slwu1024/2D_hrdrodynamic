// src_cpp/src/TimeIntegrator_cpp.cpp
#include "TimeIntegrator_cpp.h" // 包含对应的头文件
#include <stdexcept> // 包含标准异常
#include <iostream> // 包含输入输出流 (用于调试)

namespace HydroCore { // 定义HydroCore命名空间

TimeIntegrator_cpp::TimeIntegrator_cpp(TimeScheme_cpp scheme, // 构造函数实现
                                       RHSFunction rhs_func,
                                       FrictionFunction friction_func,
                                       int num_vars_per_cell)
    : scheme_internal(scheme), calculate_rhs_explicit_part(rhs_func), // 初始化列表
      apply_friction_semi_implicit(friction_func), num_vars(num_vars_per_cell) {
    if (!calculate_rhs_explicit_part || !apply_friction_semi_implicit) { // 检查函数是否有效
        throw std::invalid_argument("RHSFunction and FrictionFunction must be valid callables."); // 抛出无效参数异常
    }
    // std::cout << "C++ TimeIntegrator_cpp initialized with scheme: " << static_cast<int>(scheme_internal) << std::endl; // 打印初始化信息
} // 结束构造函数

StateVector TimeIntegrator_cpp::step(const StateVector& U_current, // 执行一步时间积分的方法实现
                                   double dt,
                                   double time_current) const {
    if (U_current.empty()) { // 如果当前状态为空
        return {}; // 返回空状态
    }
    size_t num_cells = U_current.size(); // 获取单元数量
    StateVector U_next(num_cells, std::array<double, 3>{}); // 初始化下一步状态 (全零)

    switch (scheme_internal) { // 根据方案选择执行逻辑
        case TimeScheme_cpp::FORWARD_EULER: { // 前向欧拉法
            // 1. 计算显式 RHS
            StateVector RHS_expl = calculate_rhs_explicit_part(U_current, time_current); // 计算显式RHS
            if (RHS_expl.size() != num_cells) { // 检查RHS大小是否一致
                 throw std::runtime_error("RHS size mismatch in Forward Euler."); // 抛出运行时错误
            }

            // 2. 显式更新步骤 (得到不含摩擦的下一步状态)
            StateVector U_intermediate(num_cells); // 初始化中间状态
            for (size_t i = 0; i < num_cells; ++i) { // 遍历所有单元
                for (int j = 0; j < num_vars; ++j) { // 遍历每个变量
                    U_intermediate[i][j] = U_current[i][j] + dt * RHS_expl[i][j]; // 计算中间状态
                }
            }

            // 3. 应用半隐式摩擦 (使用 U_current 计算系数 τ)
            U_next = apply_friction_semi_implicit(U_intermediate, U_current, dt); // 应用摩擦
            if (U_next.size() != num_cells) { // 检查摩擦后状态大小是否一致
                throw std::runtime_error("Friction output size mismatch in Forward Euler."); // 抛出运行时错误
            }
            break; // 结束当前case
        }
        case TimeScheme_cpp::RK2_SSP: { // 二阶SSP龙格-库塔法
            // Stage 1
            StateVector RHS1 = calculate_rhs_explicit_part(U_current, time_current); // 计算第一阶段RHS
            if (RHS1.size() != num_cells) throw std::runtime_error("RHS1 size mismatch in RK2_SSP."); // 抛出运行时错误

            StateVector U_s1(num_cells); // 初始化第一阶段中间状态
            for (size_t i = 0; i < num_cells; ++i) { // 遍历所有单元
                for (int j = 0; j < num_vars; ++j) { // 遍历每个变量
                    U_s1[i][j] = U_current[i][j] + dt * RHS1[i][j]; // 计算第一阶段中间状态
                }
            }

            // Stage 2
            StateVector RHS2 = calculate_rhs_explicit_part(U_s1, time_current + dt); // 计算第二阶段RHS
            if (RHS2.size() != num_cells) throw std::runtime_error("RHS2 size mismatch in RK2_SSP."); // 抛出运行时错误

            StateVector U_rk_explicit_only(num_cells); // 初始化仅包含显式项的RK结果
            for (size_t i = 0; i < num_cells; ++i) { // 遍历所有单元
                for (int j = 0; j < num_vars; ++j) { // 遍历每个变量
                    U_rk_explicit_only[i][j] = 0.5 * U_current[i][j] + 0.5 * (U_s1[i][j] + dt * RHS2[i][j]); // 计算RK组合结果
                }
            }

            // 摩擦步骤 (算子分裂)
            U_next = apply_friction_semi_implicit(U_rk_explicit_only, U_current, dt); // 应用摩擦
            if (U_next.size() != num_cells) { // 检查摩擦后状态大小是否一致
                 throw std::runtime_error("Friction output size mismatch in RK2_SSP."); // 抛出运行时错误
            }
            break; // 结束当前case
        }
        default: // 其他未实现的方案
            throw std::runtime_error("Unknown or unimplemented time integration scheme."); // 抛出运行时错误
    }
    return U_next; // 返回下一步状态
} // 结束方法体

} // namespace HydroCore