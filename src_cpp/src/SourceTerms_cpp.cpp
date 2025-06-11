// src_cpp/src/SourceTerms_cpp.cpp
#include "SourceTerms_cpp.h" // 包含对应的头文件
#include <iostream> // 包含输入输出流 (用于调试)
#include <stdexcept> // 包含标准异常
#include <omp.h>

namespace HydroCore { // 定义HydroCore命名空间

SourceTermCalculator_cpp::SourceTermCalculator_cpp(double gravity, double min_depth_param) // 构造函数实现
    : g(gravity), min_depth(min_depth_param) { // 初始化列表
    // std::cout << "C++ SourceTermCalculator_cpp initialized (g=" << g << ", min_depth=" << min_depth << ")" << std::endl; // 打印初始化信息
} // 结束构造函数

std::vector<std::array<double, 3>> SourceTermCalculator_cpp::apply_friction_semi_implicit_all_cells(
    const std::vector<std::array<double, 3>>& U_input_all,
    const std::vector<std::array<double, 3>>& U_coeffs_all,
    double dt,
    const std::vector<double>& manning_n_values_all
) {
    size_t num_cells = U_input_all.size();
    if (num_cells == 0) {
        return {};
    }
    if (U_coeffs_all.size() != num_cells || manning_n_values_all.size() != num_cells) {
        throw std::invalid_argument("Input array sizes do not match in apply_friction_semi_implicit_all_cells.");
    }

    std::vector<std::array<double, 3>> U_output_all(num_cells);

    // --- 新增：并行化单元循环 ---
    // 每个单元的摩擦力计算是完全独立的，可以直接并行化。
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_cells; ++i) {
        double h_in = U_input_all[i][0];
        double hu_in = U_input_all[i][1];
        double hv_in = U_input_all[i][2];

        double h_coeff = std::max(U_coeffs_all[i][0], min_depth);

        double u_coeff = 0.0;
        double v_coeff = 0.0;
        bool wet_mask_coeff = (U_coeffs_all[i][0] > min_depth);

        if (wet_mask_coeff) {
            u_coeff = U_coeffs_all[i][1] / h_coeff;
            v_coeff = U_coeffs_all[i][2] / h_coeff;
        }

        double speed_coeff = std::sqrt(u_coeff * u_coeff + v_coeff * v_coeff);

        double tau = 0.0;
        double epsilon_speed = 1e-6;

        if (wet_mask_coeff && speed_coeff > epsilon_speed) {
            double h_pow_4_3 = std::pow(h_coeff, 4.0 / 3.0);
            if (h_pow_4_3 < 1e-12) {
                h_pow_4_3 = 1e-12;
            }
            tau = -g * manning_n_values_all[i] * manning_n_values_all[i] * speed_coeff / h_pow_4_3;
        }

        double denominator = 1.0 - dt * tau;
        denominator = std::max(denominator, 1e-6);

        double u_in = 0.0;
        double v_in = 0.0;
        if (h_in > min_depth) {
            double h_in_div = std::max(h_in, 1e-12);
            u_in = hu_in / h_in_div;
            v_in = hv_in / h_in_div;
        }

        double u_out = u_in / denominator;
        double v_out = v_in / denominator;

        U_output_all[i][0] = h_in;
        U_output_all[i][1] = u_out * h_in;
        U_output_all[i][2] = v_out * h_in;
    }
    return U_output_all;
}
} // namespace HydroCore