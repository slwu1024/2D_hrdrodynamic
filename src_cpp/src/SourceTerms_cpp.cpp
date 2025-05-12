// src_cpp/src/SourceTerms_cpp.cpp
#include "SourceTerms_cpp.h" // 包含对应的头文件
#include <iostream> // 包含输入输出流 (用于调试)
#include <stdexcept> // 包含标准异常

namespace HydroCore { // 定义HydroCore命名空间

SourceTermCalculator_cpp::SourceTermCalculator_cpp(double gravity, double min_depth_param) // 构造函数实现
    : g(gravity), min_depth(min_depth_param) { // 初始化列表
    // std::cout << "C++ SourceTermCalculator_cpp initialized (g=" << g << ", min_depth=" << min_depth << ")" << std::endl; // 打印初始化信息
} // 结束构造函数

std::vector<std::array<double, 3>> SourceTermCalculator_cpp::apply_friction_semi_implicit_all_cells( // 应用半隐式摩擦 (所有单元)
    const std::vector<std::array<double, 3>>& U_input_all, // 输入状态 (所有单元)
    const std::vector<std::array<double, 3>>& U_coeffs_all, // 用于计算系数的状态 (所有单元)
    double dt, // 时间步长
    const std::vector<double>& manning_n_values_all // 每个单元的曼宁糙率
) { // 开始方法体
    size_t num_cells = U_input_all.size(); // 获取单元数量
    if (num_cells == 0) { // 如果没有单元
        return {}; // 返回空vector
    }
    if (U_coeffs_all.size() != num_cells || manning_n_values_all.size() != num_cells) { // 检查输入数组大小是否一致
        throw std::invalid_argument("Input array sizes do not match in apply_friction_semi_implicit_all_cells."); // 抛出无效参数异常
    }

    std::vector<std::array<double, 3>> U_output_all(num_cells); // 初始化输出数组

    for (size_t i = 0; i < num_cells; ++i) { // 遍历所有单元
        double h_in = U_input_all[i][0]; // 获取输入水深
        double hu_in = U_input_all[i][1]; // 获取输入x方向动量
        double hv_in = U_input_all[i][2]; // 获取输入y方向动量

        // --- 从 U_coeffs 计算摩擦系数 τ ---
        double h_coeff = std::max(U_coeffs_all[i][0], min_depth); // 获取用于计算系数的水深 (限制最小深度)

        double u_coeff = 0.0; // 初始化系数u
        double v_coeff = 0.0; // 初始化系数v
        bool wet_mask_coeff = (U_coeffs_all[i][0] > min_depth); // 判断系数状态是否为湿 (注意这里用 U_coeffs_all[i][0] 而不是 h_coeff)

        if (wet_mask_coeff) { // 如果系数状态为湿
            // 使用 h_coeff (已限制最小值的) 作为分母，而不是原始的 U_coeffs_all[i][0]
            u_coeff = U_coeffs_all[i][1] / h_coeff; // 计算系数u
            v_coeff = U_coeffs_all[i][2] / h_coeff; // 计算系数v
        }

        double speed_coeff = std::sqrt(u_coeff * u_coeff + v_coeff * v_coeff); // 计算系数流速大小

        // --- 计算摩擦项 τ = -g * n^2 * |V| / h^(4/3) ---
        double tau = 0.0; // 初始化tau
        double epsilon_speed = 1e-6; // 避免速度为零时产生 NaN 的小速度阈值

        // 只在满足条件的湿单元计算 tau (原始水深 > min_depth 且速度 > epsilon_speed)
        // 注意：wet_mask_coeff 是基于 U_coeffs_all[i][0] 的
        if (wet_mask_coeff && speed_coeff > epsilon_speed) { // 如果系数状态为湿且速度大于阈值
            double h_pow_4_3 = std::pow(h_coeff, 4.0 / 3.0); // 计算h_coeff的4/3次方
            if (h_pow_4_3 < 1e-12) { // 防止除零
                h_pow_4_3 = 1e-12; // 设置一个很小的值
            }
            tau = -g * manning_n_values_all[i] * manning_n_values_all[i] * speed_coeff / h_pow_4_3; // 计算tau
        }

        // --- 应用半隐式公式 U_out = U_in / (1 - dt * tau) for momentum ---
        double denominator = 1.0 - dt * tau; // 计算分母 (理论上 >= 1)
        denominator = std::max(denominator, 1e-6); // 限制分母的最小值，防止数值问题

        // 对于输入状态 U_input，计算其对应的速度
        double u_in = 0.0; // 初始化输入u
        double v_in = 0.0; // 初始化输入v
        // 使用原始的 h_in 来判断是否湿，而不是 h_coeff
        if (h_in > min_depth) { // 如果输入状态为湿
            double h_in_div = std::max(h_in, 1e-12); // 用于除法的安全水深 (h_in本身，加小量防零)
            u_in = hu_in / h_in_div; // 计算输入u
            v_in = hv_in / h_in_div; // 计算输入v
        }

        double u_out = u_in / denominator; // 计算输出u
        double v_out = v_in / denominator; // 计算输出v

        // --- 构造输出的守恒量 ---
        U_output_all[i][0] = h_in; // 水深在摩擦步骤中不变
        U_output_all[i][1] = u_out * h_in; // 更新x方向动量
        U_output_all[i][2] = v_out * h_in; // 更新y方向动量
    }
    return U_output_all; // 返回所有单元应用摩擦后的状态
} // 结束方法体
} // namespace HydroCore