// src_cpp/src/WettingDrying_cpp.cpp
#include "WettingDrying_cpp.h"
#include "Profiler.h"
#include <iostream>
#include <iomanip> // 为了 std::fixed 和 std::setprecision
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string> // 为了 std::to_string

namespace HydroCore { // 定义HydroCore命名空间
    // 初始化类的静态成员调试变量
    bool VFRCalculator_cpp::s_vfr_internal_debug_enabled = false;
    int VFRCalculator_cpp::s_vfr_debug_target_cell_id = -1;
    double VFRCalculator_cpp::s_vfr_debug_target_time_min = -1e9;
    double VFRCalculator_cpp::s_vfr_debug_target_time_max = 1e9;

    // 定义类的静态成员函数
    void VFRCalculator_cpp::set_internal_debug_conditions(bool enable, int target_cell_id, double target_time_min, double target_time_max) {
        s_vfr_internal_debug_enabled = enable;
        s_vfr_debug_target_cell_id = target_cell_id;
        s_vfr_debug_target_time_min = target_time_min;
        s_vfr_debug_target_time_max = target_time_max;
    }

    VFRCalculator_cpp::VFRCalculator_cpp(double min_depth_param,
                                         double min_eta_change_iter_param,
                                         int max_vfr_iters_param,
                                         double relative_h_tolerance_param)
        : min_depth(min_depth_param),
          min_eta_change_iter(min_eta_change_iter_param),
          max_vfr_iters(max_vfr_iters_param),
          relative_h_tolerance(relative_h_tolerance_param),
          epsilon(1e-12) {
    }
// --- 新增的VFR内部调试控制变量 ---
static bool VFR_INTERNAL_DEBUG_ENABLED = false; // 总开关
static int VFR_DEBUG_TARGET_CELL_ID = -1;     // 目标单元ID
static double VFR_DEBUG_TARGET_TIME_MIN = -1e9; // 目标时间范围最小值
static double VFR_DEBUG_TARGET_TIME_MAX = 1e9;  // 目标时间范围最大值
// --- VFR内部调试控制变量结束 ---
// --- 新增：从外部设置VFR内部调试条件的函数 ---
void set_vfr_internal_debug_conditions(bool enable, int target_cell_id, double target_time_min, double target_time_max) { // 新增：设置VFR内部调试条件的函数
    VFR_INTERNAL_DEBUG_ENABLED = enable; // 设置VFR内部调试总开关
    VFR_DEBUG_TARGET_CELL_ID = target_cell_id; // 设置VFR调试目标单元ID
    VFR_DEBUG_TARGET_TIME_MIN = target_time_min; // 设置VFR调试目标时间范围最小值
    VFR_DEBUG_TARGET_TIME_MAX = target_time_max; // 设置VFR调试目标时间范围最大值
}
// --- 函数结束 ---



double VFRCalculator_cpp::get_h_from_eta(double eta, // 根据eta计算h的方法实现
                                        const std::vector<double>& b_sorted_vertices,
                                        double cell_total_area, // (此简化版本未使用cell_total_area)
                                        const std::string& cell_id_for_debug) const {
    if (b_sorted_vertices.size() != 3) { // 检查顶点数量是否为3
        // 在实际应用中，可能需要更通用的处理或抛出错误
        // std::cerr << "Warning (get_h_from_eta): Expected 3 sorted vertices, got " << b_sorted_vertices.size() << std::endl; // 打印警告
        if (b_sorted_vertices.empty()) return 0.0; // 如果为空则返回0
        // 简化处理：如果不是3个点，就按全干或最低点处理
        return std::max(0.0, eta - b_sorted_vertices[0]); // 返回eta与最低点高程的差值（非负）
    }

    double b1 = b_sorted_vertices[0]; // 获取排序后的顶点高程b1
    double b2 = b_sorted_vertices[1]; // 获取排序后的顶点高程b2
    double b3 = b_sorted_vertices[2]; // 获取排序后的顶点高程b3

    if (eta <= b1 + epsilon) return 0.0; // 全干 (加上epsilon以处理浮点比较)

    double h_avg; // 声明平均水深
    // 注意：这里的公式是基于您的Python版本中的简化公式 (2-56)
    // 如果要使用精确几何体积/面积，这里的逻辑会更复杂，需要调用 calculate_wet_volume / calculate_wet_surface_area
    if (eta <= b2 + epsilon) { // 部分淹没1
        double denominator = 2.0 * (b2 - b1 + epsilon) * (b3 - b1 + epsilon); // 计算分母
        if (std::abs(denominator) < epsilon) { // 如果分母接近零
            // 这种情况表示 b1, b2, b3 非常接近，单元几乎是平的
            // 或者 b2=b1 或 b3=b1，导致退化
            h_avg = eta - b1; // 近似为 (eta - 最低点)
        } else { // 否则正常计算
            h_avg = std::pow(eta - b1, 3) / denominator; // 计算平均水深
        }
    } else if (eta <= b3 + epsilon) { // 部分淹没2
        double term_numerator = eta * eta + eta * b3 - 3.0 * eta * b1 - b1 * b3 + b1 * b2 + b1 * b1; // 计算分子项
        double denominator = 3.0 * (b3 - b1 + epsilon); // 计算分母
         if (std::abs(denominator) < epsilon) { // 如果分母接近零
            // b3=b1，单元非常平坦
            h_avg = eta - (b1 + b2) / 2.0; // 近似
        } else { // 否则正常计算
            h_avg = term_numerator / denominator; // 计算平均水深
        }
    } else { // 全淹没
        h_avg = eta - (b1 + b2 + b3) / 3.0; // 计算平均水深
    }
    return std::max(0.0, h_avg); // 返回非负的平均水深
} // 结束方法

std::array<double, 2> VFRCalculator_cpp::linear_interpolate( // 线性插值辅助方法实现
    const std::array<double, 2>& p1_coords, double p1_z,
    const std::array<double, 2>& p2_coords, double p2_z,
    double target_z) const {
    double x1 = p1_coords[0], y1 = p1_coords[1]; // 点1坐标
    double x2 = p2_coords[0], y2 = p2_coords[1]; // 点2坐标

    double delta_z = p2_z - p1_z; // 计算高程差
    if (std::abs(delta_z) < epsilon) { // 如果高程几乎相等
        if (std::abs(target_z - p1_z) < epsilon) { // 如果目标高程也接近
            return {(x1 + x2) / 2.0, (y1 + y2) / 2.0}; // 返回中点
        }
        // 目标高程不在此（几乎平坦的）线上，返回一个端点或表示错误
        // std::cerr << "Warning (linear_interpolate): target_z not on flat segment." << std::endl;
        return p1_coords; // 返回点1坐标
    }

    double t = (target_z - p1_z) / delta_z; // 计算插值比例
    t = std::max(0.0, std::min(1.0, t)); // 将t限制在[0,1]之间 (内插)

    return {x1 + t * (x2 - x1), y1 + t * (y2 - y1)}; // 返回插值点坐标
} // 结束方法

double VFRCalculator_cpp::polygon_area(const std::vector<std::array<double, 2>>& vertices) const { // 计算多边形面积辅助方法实现
    size_t n = vertices.size(); // 获取顶点数量
    if (n < 3) return 0.0; // 点或线段没有面积

    double area = 0.0; // 初始化面积
    for (size_t i = 0; i < n; ++i) { // 遍历顶点
        size_t j = (i + 1) % n; // 下一个顶点索引
        area += vertices[i][0] * vertices[j][1]; // 累加 x_i * y_{i+1}
        area -= vertices[j][0] * vertices[i][1]; // 减去 x_{i+1} * y_i
    }
    return std::abs(area) / 2.0; // 返回面积绝对值的一半
} // 结束方法

double VFRCalculator_cpp::calculate_wet_surface_area( // 计算水面面积的辅助方法实现
    double eta,
    const std::vector<double>& b_sorted_vertices, // [b0, b1, b2]
    const std::vector<Node_cpp>& cell_nodes_sorted, // [node_b0, node_b1, node_b2]
    double cell_total_area) const {
    if (b_sorted_vertices.size() != 3 || cell_nodes_sorted.size() != 3) { // 检查顶点数量
        // std::cerr << "Warning (calculate_wet_surface_area): Expected 3 vertices/nodes." << std::endl; // 打印警告
        return 0.0; // 返回0
    }

    double b0 = b_sorted_vertices[0]; // 获取排序后的顶点高程b0
    double b1 = b_sorted_vertices[1]; // 获取排序后的顶点高程b1
    double b2 = b_sorted_vertices[2]; // 获取排序后的顶点高程b2

    // 假设 cell_nodes_sorted 已经与 b_sorted_vertices 的顺序对应
    std::array<double, 2> v0_coords = {cell_nodes_sorted[0].x, cell_nodes_sorted[0].y}; // 顶点0坐标
    std::array<double, 2> v1_coords = {cell_nodes_sorted[1].x, cell_nodes_sorted[1].y}; // 顶点1坐标
    std::array<double, 2> v2_coords = {cell_nodes_sorted[2].x, cell_nodes_sorted[2].y}; // 顶点2坐标

    if (eta <= b0 + epsilon) { // 情况1: 全干
        return 0.0; // 面积为0
    }
    if (eta >= b2 - epsilon) { // 情况2: 全湿
        return cell_total_area; // 面积等于总面积
    }

    std::vector<std::array<double, 2>> wet_polygon_vertices; // 湿区多边形顶点列表

    if (eta <= b1 + epsilon) { // 情况3: 部分淹没，水面低于b1 (只有顶点 v0 被淹)
        // 水线交点 P01 在边 v0-v1 上, P02 在边 v0-v2 上
        wet_polygon_vertices.push_back(v0_coords); // 添加顶点v0
        wet_polygon_vertices.push_back(linear_interpolate(v0_coords, b0, v1_coords, b1, eta)); // 添加交点P01
        wet_polygon_vertices.push_back(linear_interpolate(v0_coords, b0, v2_coords, b2, eta)); // 添加交点P02
    } else { // 情况4: 部分淹没，水面介于b1和b2之间 (顶点v0和v1被淹)
        // 水线交点 P02 在边 v0-v2 上, P12 在边 v1-v2 上
        wet_polygon_vertices.push_back(v0_coords); // 添加顶点v0
        wet_polygon_vertices.push_back(v1_coords); // 添加顶点v1
        wet_polygon_vertices.push_back(linear_interpolate(v1_coords, b1, v2_coords, b2, eta)); // 添加交点P12
        wet_polygon_vertices.push_back(linear_interpolate(v0_coords, b0, v2_coords, b2, eta)); // 添加交点P02
    }
    return polygon_area(wet_polygon_vertices); // 计算并返回湿区面积
} // 结束方法

// get_eta_from_h 函数的实现，内部使用 s_vfr_internal_debug_enabled 等静态成员
double VFRCalculator_cpp::get_eta_from_h(
    double h_avg,
    const std::vector<double>& b_sorted_vertices,
    const std::vector<Node_cpp>& cell_nodes_sorted, // 在此简化版VFR中可能不直接用，但保留接口
    double cell_total_area,                        // 在此简化版VFR中可能不直接用
    double eta_previous_guess, // 直接法不使用
    double current_sim_time,   // 用于可能的调试输出
    int cell_id_int) const {   // 用于可能的调试输出

    // --- 基本参数和有效性检查 ---
    if (b_sorted_vertices.size() != 3) {
        if (b_sorted_vertices.empty()) return 0.0;
        return b_sorted_vertices[0] + std::max(0.0, h_avg);
    }
    double b0 = b_sorted_vertices[0];
    double b1 = b_sorted_vertices[1];
    double b2 = b_sorted_vertices[2];
    double h_avg_non_negative = std::max(0.0, h_avg);

    const double near_dry_threshold = min_depth * 5.0;
    const double elevation_diff_epsilon = epsilon * 100;

    // --- 情况 0: 全干或接近全干 ---
    if (h_avg_non_negative < near_dry_threshold) {
        return b0 + h_avg_non_negative;
    }

    // --- 计算临界水深 h_at_b1 和 h_at_b2 ---
    // h_at_b1: 当 eta = b1 时的平均水深
    double h_at_b1_direct = get_h_from_eta(b1, b_sorted_vertices, cell_total_area, "DirectH@b1");

    // h_at_b2: 当 eta = b2 时的平均水深
    double h_at_b2_direct = get_h_from_eta(b2, b_sorted_vertices, cell_total_area, "DirectH@b2");

    // 保证 h_at_b2 >= h_at_b1 (理论上 H(eta) 单调不减)
    if (h_at_b2_direct < h_at_b1_direct - epsilon) {
        // 如果发生这种情况，说明 get_h_from_eta(eta,...) 的VFR公式可能有问题，或者数值不稳定
        // 作为一种保护，强制它们至少相等
        h_at_b2_direct = h_at_b1_direct;
    }

    double eta_result;

    // --- 根据 h_avg 与临界水深的关系选择反解公式 ---
    if (h_avg_non_negative <= h_at_b1_direct + epsilon * std::abs(h_at_b1_direct)) { // 情况 1: 0 <= h_avg <= H(b1) => b0 < eta <= b1
        double K_term = 2.0 * (b1 - b0) * (b2 - b0);
        if (std::abs(K_term) < epsilon) { // 退化：b0=b1 或 b0=b2
            eta_result = b0 + h_avg_non_negative; // 近似为平底
        } else {
            double term_inside_cbrt = h_avg_non_negative * K_term;
            eta_result = b0 + std::cbrt(term_inside_cbrt);
        }
    } else if (h_avg_non_negative <= h_at_b2_direct + epsilon * std::abs(h_at_b2_direct)) { // 情况 2: H(b1) < h_avg <= H(b2) => b1 < eta <= b2
        // 解二次方程: A*eta^2 + B*eta + C = 0
        // 从 h_avg = (eta^2 + eta*b2 - 3*eta*b0 - b0*b2 + b0*b1 + b0*b0) / (3 * (b2-b0)) 整理
        // eta^2 + eta*(b2 - 3*b0) + (b0*b1 + b0^2 - b0*b2 - 3*h_avg*(b2-b0)) = 0
        double A_quad = 1.0;
        double B_quad = b2 - 3.0 * b0;
        double C_quad_const_part = b0 * b1 + b0 * b0 - b0 * b2;
        double K_factor_for_h = 3.0 * (b2 - b0);

        if (std::abs(K_factor_for_h) < epsilon) { // 退化: b0=b2 (意味着 b0=b1=b2)
            eta_result = b0 + h_avg_non_negative; // 近似为平底
        } else {
            double C_quad = C_quad_const_part - h_avg_non_negative * K_factor_for_h;
            double discriminant = B_quad * B_quad - 4.0 * A_quad * C_quad;

            if (discriminant >= -epsilon) { // 允许非常小的负判别式（浮点误差）
                discriminant = std::max(0.0, discriminant); // 确保非负
                double eta_sol1 = (-B_quad + std::sqrt(discriminant)) / (2.0 * A_quad);
                double eta_sol2 = (-B_quad - std::sqrt(discriminant)) / (2.0 * A_quad);

                // 选择落在 (b1, b2] 区间的解。
                // H(eta) 在此区间通常是单调的。
                // 我们期望 eta 随 h_avg 增加。
                // 通常较大的根是物理相关的，但最好通过代回验证或检查区间。
                bool s1_in_range = (eta_sol1 > b1 - epsilon && eta_sol1 <= b2 + epsilon);
                bool s2_in_range = (eta_sol2 > b1 - epsilon && eta_sol2 <= b2 + epsilon);

                if (s1_in_range && s2_in_range) {
                    // 如果两个都在范围内，理论上不应该。但如果发生，选一个。
                    // 检查哪个反算回来的h更接近h_avg
                    double h_check_s1 = get_h_from_eta(eta_sol1, b_sorted_vertices, cell_total_area, "DirectChkS1");
                    double h_check_s2 = get_h_from_eta(eta_sol2, b_sorted_vertices, cell_total_area, "DirectChkS2");
                    if (std::abs(h_check_s1 - h_avg_non_negative) < std::abs(h_check_s2 - h_avg_non_negative)) {
                        eta_result = eta_sol1;
                    } else {
                        eta_result = eta_sol2;
                    }
                } else if (s1_in_range) {
                    eta_result = eta_sol1;
                } else if (s2_in_range) {
                    eta_result = eta_sol2;
                } else {
                    // 如果两个根都不在 (b1, b2] 内，这表明区间判断或公式应用可能有问题
                    // 或者h_avg的值恰好使得解落在了区间边界之外一点点（由于浮点误差）
                    // 此时选择一个最接近的解，并将其限制在区间内可能是一种策略
                    // 或者采用线性插值作为更安全的后备
                    double err1 = std::abs(get_h_from_eta(eta_sol1, b_sorted_vertices, cell_total_area, "") - h_avg_non_negative);
                    double err2 = std::abs(get_h_from_eta(eta_sol2, b_sorted_vertices, cell_total_area, "") - h_avg_non_negative);
                    eta_result = (err1 < err2) ? eta_sol1 : eta_sol2;
                    // 强制限制在区间 [b1, b2]
                    eta_result = std::max(b1, std::min(b2, eta_result));
                }
            } else { // 判别式为负，无实根
                // 这理论上不应该发生，如果h_avg确实在[H(b1), H(b2)]。
                // 如果发生，说明VFR公式或临界h的计算与h_avg的区间判断不匹配。
                // 使用线性插值作为最差情况的后备
                if (std::abs(h_at_b2_direct - h_at_b1_direct) > epsilon && std::abs(b2-b1) > epsilon) {
                     eta_result = b1 + (h_avg_non_negative - h_at_b1_direct) * (b2-b1) / (h_at_b2_direct - h_at_b1_direct);
                } else { // 无法线性插值（例如h_at_b1=h_at_b2 或 b1=b2）
                     eta_result = (b1 + b2) / 2.0; // 取区间中点或b1
                }
                // 强制限制在区间 [b1, b2]
                eta_result = std::max(b1, std::min(b2, eta_result));
            }
        }
    } else { // 情况 3: 全淹没 (h_avg > H(b2)) => eta > b2
        eta_result = h_avg_non_negative + (b0 + b1 + b2) / 3.0;
    }

    // 最终确保 eta_result 不低于最低点 b0 (并处理eta可能因浮点误差略小于b0的情况)
    eta_result = std::max(b0, eta_result);
    if (eta_result < b0 + epsilon && h_avg_non_negative > epsilon) { // 如果eta非常接近b0但h不为0，则修正为b0+h
        eta_result = b0 + h_avg_non_negative;
    }


    // (移除对比打印逻辑，因为现在只用直接法)
    // 可以在这里加入对最终结果的单向调试打印（如果需要）
    bool debug_this_direct_call = false;
    if (s_vfr_internal_debug_enabled && cell_id_int == s_vfr_debug_target_cell_id) {
        if (current_sim_time >= s_vfr_debug_target_time_min && current_sim_time <= s_vfr_debug_target_time_max) {
            debug_this_direct_call = true;
        }
    }
    if (debug_this_direct_call) {
        double h_check_final = get_h_from_eta(eta_result, b_sorted_vertices, cell_total_area, "DirectFinalCheck");
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "VFR_DIRECT_ONLY: Cell=" << cell_id_int << ", T=" << current_sim_time
                  << ", h_avg_in=" << h_avg_non_negative
                  << ", DirectEta=" << eta_result
                  << ", h_recalc=" << h_check_final
                  << ", Diff(h_recalc-h_in)=" << (h_check_final - h_avg_non_negative) << std::endl;
    }

    return eta_result;
}
} // namespace HydroCore