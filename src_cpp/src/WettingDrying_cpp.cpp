// src_cpp/src/WettingDrying_cpp.cpp
#include "WettingDrying_cpp.h"
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
    const std::vector<Node_cpp>& cell_nodes_sorted,
    double cell_total_area,
    double eta_previous_guess,
    double current_sim_time,
    int cell_id_int) const {

    bool should_print_vfr_internal_debug = false; // 初始化本地调试打印标志
    // 使用类的静态成员变量进行判断
    if (s_vfr_internal_debug_enabled && cell_id_int == s_vfr_debug_target_cell_id) { // 如果全局VFR调试开启且当前单元是目标单元
        if (current_sim_time >= s_vfr_debug_target_time_min && current_sim_time <= s_vfr_debug_target_time_max) { // 且当前时间在调试时间范围内
            should_print_vfr_internal_debug = true; // 允许为此单元和时间点打印VFR内部调试信息
        }
    }

    if (b_sorted_vertices.size() != 3 || cell_nodes_sorted.size() != 3) {
        if (should_print_vfr_internal_debug) std::cout << "    VFR_DETAIL_WARN (get_eta_from_h): Vertex/Node count not 3 for cell " << cell_id_int << ". b_size=" << b_sorted_vertices.size() << ", node_size=" << cell_nodes_sorted.size() << ". Returning b0 + h_avg or 0." << std::endl; // VFR调试打印：顶点/节点数不为3的警告
        if (b_sorted_vertices.empty()) return 0.0; // 如果顶点列表为空，返回0
        return b_sorted_vertices[0] + std::max(0.0, h_avg); // 否则返回最低点高程加上非负平均水深
    }
    double b0 = b_sorted_vertices[0];
    double b1 = b_sorted_vertices[1];
    double b2 = b_sorted_vertices[2];

    double h_avg_non_negative = std::max(0.0, h_avg); // 确保平均水深非负
    double practical_small_depth_threshold = 0.0005;

    if (h_avg_non_negative < practical_small_depth_threshold) {
        if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印
            std::cout << std::fixed << std::setprecision(8); // 设置输出精度
            std::cout << "    VFR_DETAIL_INFO (Skipping Newton): Cell=" << cell_id_int << ", Time=" << current_sim_time
                      << ", Target h_avg_nn (" << h_avg_non_negative << ") < threshold (" << practical_small_depth_threshold
                      << "). Returning b0 + h_avg_nn = "
                      << (b0 + h_avg_non_negative) << std::endl; // VFR调试打印：目标平均水深小于阈值，跳过牛顿法，直接返回b0+h_avg_nn
        }
        return b0 + h_avg_non_negative; // 返回最低点高程加上非负平均水深
    }

    if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印
        std::cout << std::fixed << std::setprecision(8); // 设置输出精度
        std::cout << "  VFR_DETAIL_INPUT (Newton Attempt): Cell=" << cell_id_int << ", Time=" << current_sim_time
                  << ", Target_h_avg=" << h_avg_non_negative
                  << ", Eta_Guess_In=" << eta_previous_guess
                  << ", b_verts={" << b0 << "," << b1 << "," << b2 << "}" << std::endl; // VFR调试打印：准备尝试牛顿法
    }

    double eta_k_newton; // 牛顿法当前水位迭代值
    if (eta_previous_guess >= (b0 - epsilon) &&
        eta_previous_guess < (b0 + (b2-b0) + h_avg_non_negative + 0.1) && // 修改：确保初始猜测的上限更合理 (b0 + (单元最大高差) + 水深 + 一点余量)
        h_avg_non_negative >= practical_small_depth_threshold * 0.5) {
        eta_k_newton = eta_previous_guess; // 使用上一时刻的水位作为初始猜测
    } else {
        eta_k_newton = b0 + h_avg_non_negative; // 否则使用 (最低点高程 + 平均水深) 作为初始猜测
    }
    eta_k_newton = std::max(b0 - epsilon, eta_k_newton); // 确保初始猜测不低于最低点
    if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印
         std::cout << "    VFR_NEWTON_START: Initial eta_k = " << eta_k_newton << std::endl; // VFR调试打印：牛顿法初始eta_k
    }

    double f_k_newton = 1.0; // 初始化f(eta_k)
    bool converged_newton_abs = false; // 绝对收敛标志
    bool converged_newton_rel = false; // 相对收敛标志
    int iter_count_newton = 0; // 牛顿法迭代计数

    for (iter_count_newton = 0; iter_count_newton < max_vfr_iters; ++iter_count_newton) { // 牛顿法迭代循环
        double h_calc_k_newton = get_h_from_eta(eta_k_newton, b_sorted_vertices, cell_total_area, std::to_string(cell_id_int) + "_newton_internal"); // 根据当前eta计算h
        f_k_newton = h_calc_k_newton - h_avg_non_negative; // 计算f(eta_k) = h_calc - h_target

        converged_newton_abs = std::abs(f_k_newton) < min_depth * relative_h_tolerance; // 检查绝对收敛
        converged_newton_rel = (std::abs(h_avg_non_negative) > epsilon) ? (std::abs(f_k_newton / h_avg_non_negative) < relative_h_tolerance) : (std::abs(f_k_newton) < epsilon) ; // 检查相对收敛

        if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印
            std::cout << "    VFR_NEWTON_ITER[" << iter_count_newton << "]: eta_k=" << eta_k_newton
                      << ", h_calc_k=" << h_calc_k_newton << ", f_k=" << f_k_newton
                      << ", conv_abs=" << converged_newton_abs << ", conv_rel=" << converged_newton_rel << std::endl; // VFR调试打印：牛顿法迭代详情
        }

        if (converged_newton_abs || converged_newton_rel) { // 如果任一收敛条件满足
            if (should_print_vfr_internal_debug) std::cout << "    VFR_NEWTON_CONVERGED at iter " << iter_count_newton << std::endl; // VFR调试打印：牛顿法收敛
            return std::max(b0 - epsilon, eta_k_newton); // 返回收敛的水位（确保不低于最低点）
        }

        double Aw_k_newton = calculate_wet_surface_area(eta_k_newton, b_sorted_vertices, cell_nodes_sorted, cell_total_area); // 计算当前水面面积
        double df_deta_k_newton = (cell_total_area > epsilon) ? (Aw_k_newton / cell_total_area) : epsilon; // 计算导数 df/d_eta (即 Aw/Acell)

        double eta_k_next_newton; // 下一步迭代的水位
        if (std::abs(df_deta_k_newton) < epsilon) { // 如果导数接近零
             eta_k_next_newton = eta_k_newton - ((f_k_newton > 0) ? 1.0 : -1.0) * min_eta_change_iter * 0.1; // 使用一个小的固定步长调整
        } else { // 否则正常牛顿迭代
            double delta_eta_newton = f_k_newton / df_deta_k_newton; // 计算水位调整量

            double max_delta_eta_abs = (b2 - b0 + 0.1) * 0.5;
            max_delta_eta_abs = std::max(max_delta_eta_abs, min_eta_change_iter * 5.0); // 限制最大调整量的绝对值
            delta_eta_newton = std::max(-max_delta_eta_abs, std::min(max_delta_eta_abs, delta_eta_newton)); // 应用限制
            eta_k_next_newton = eta_k_newton - delta_eta_newton; // 计算下一步水位

             if ((f_k_newton > epsilon && delta_eta_newton < -epsilon && df_deta_k_newton > epsilon) || // 检查是否为矛盾步骤（f_k > 0 但 eta 减小，或 f_k < 0 但 eta 增大，当导数>0时）
                (f_k_newton < -epsilon && delta_eta_newton > epsilon && df_deta_k_newton > epsilon)) {
                if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印
                    std::cout << "      VFR_NEWTON_WARN: Contradictory step. Forcing conservative. f_k=" << f_k_newton << ", delta_eta=" << delta_eta_newton << ", df_deta=" << df_deta_k_newton << std::endl; // VFR调试打印：牛顿法步骤矛盾，强制保守
                }
                eta_k_next_newton = eta_k_newton - ((f_k_newton > 0) ? 1.0 : -1.0) * min_eta_change_iter * 0.5; // 强制一个小的保守步长
            }
        }
        eta_k_next_newton = std::max(b0 - epsilon, eta_k_next_newton); // 确保下一步水位不低于最低点

        double reasonable_h_max_for_cell = (b2 - b0) + std::max(min_depth, h_avg_non_negative * 1.5); // 计算一个合理的单元最大水深
        double eta_upper_bound_strict = b0 + reasonable_h_max_for_cell; // 计算严格的水位上限
        if (eta_k_next_newton > eta_upper_bound_strict) { // 如果计算出的下一步水位超过上限
            if (should_print_vfr_internal_debug) std::cout << "      VFR_NEWTON_INFO: eta_k_next (" << eta_k_next_newton << ") capped by eta_upper_bound_strict (" << eta_upper_bound_strict << ")" << std::endl; // VFR调试打印：下一步水位被上限限制
            eta_k_next_newton = eta_upper_bound_strict; // 将下一步水位限制在上限
        }


        if (std::abs(eta_k_next_newton - eta_k_newton) < min_eta_change_iter && iter_count_newton > 0) { // 如果水位变化小于最小变化阈值
            if (should_print_vfr_internal_debug) std::cout << "    VFR_NEWTON_MIN_CHANGE_MET at iter " << iter_count_newton << std::endl; // VFR调试打印：牛顿法满足最小变化量
            eta_k_newton = eta_k_next_newton;

            double h_final_check = get_h_from_eta(eta_k_newton, b_sorted_vertices, cell_total_area, std::to_string(cell_id_int) + "_newton_final_check"); // 最终检查计算的水深
            f_k_newton = h_final_check - h_avg_non_negative; // 计算最终的f_k
            converged_newton_abs = std::abs(f_k_newton) < min_depth * relative_h_tolerance; // 检查绝对收敛
            converged_newton_rel = (std::abs(h_avg_non_negative) > epsilon) ? (std::abs(f_k_newton / h_avg_non_negative) < relative_h_tolerance) : (std::abs(f_k_newton) < epsilon) ; // 检查相对收敛
            if (converged_newton_abs || converged_newton_rel) { // 如果任一收敛条件满足
                 if (should_print_vfr_internal_debug) std::cout << "    VFR_NEWTON_CONVERGED (after min_change) at iter " << iter_count_newton << std::endl; // VFR调试打印：牛顿法在满足最小变化量后收敛
                 return std::max(b0 - epsilon, eta_k_newton); // 返回收敛的水位
            }
            break; // 如果最小变化量满足但f_k仍大，则跳出牛顿法，尝试二分法
        }
        eta_k_newton = eta_k_next_newton; // 更新当前水位为下一步计算的水位
    }

    if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印
         double h_calc_at_final_eta_newton = get_h_from_eta(eta_k_newton, b_sorted_vertices, cell_total_area, std::to_string(cell_id_int) + "_newton_fail"); // 计算牛顿法失败时的水深
         f_k_newton = h_calc_at_final_eta_newton - h_avg_non_negative; // 计算牛顿法失败时的f_k
         std::cout << "  VFR_NEWTON_FAILED or MIN_CHANGE_EXIT_NO_CONV: Cell=" << cell_id_int << ", Time=" << current_sim_time
                  << ". Target_h_avg=" << h_avg_non_negative
                  << ", Newton_eta_final=" << eta_k_newton
                  << ", Newton_h_calc=" << h_calc_at_final_eta_newton
                  << ", Newton_f_k=" << f_k_newton
                  << ". Switching to Bisection." << std::endl; // VFR调试打印：牛顿法失败或满足最小变化量但未收敛，切换到二分法
    }

    double eta_low = b0; // 二分法搜索下界
    double eta_high_guess1 = b0 + (b2 - b0) + h_avg_non_negative + 0.1; // 修改：二分法上界的一个猜测 (最高点高程 + 平均水深 + 余量)
    double eta_high_guess2 = (iter_count_newton == max_vfr_iters) ? eta_k_newton : (b0 + h_avg_non_negative * 2.0); // 如果牛顿法达到最大迭代次数，eta_k可能很大；否则用一个较保守的猜测

    eta_high_guess2 = std::max(eta_high_guess2, b0 + h_avg_non_negative + min_depth);

    double eta_high = std::max(eta_high_guess1, eta_high_guess2); // 取两个猜测中的较大值作为上界
    eta_high = std::max(eta_high, eta_low + min_depth);

    double eta_mid = 0.0; // 二分法中点
    double f_mid = 0.0; // 二分法中点的f值
    int iter_count_bisection = 0; // 二分法迭代计数
    const int max_bisection_iters = 30;
    const double bisection_tol_eta = min_eta_change_iter * 0.1;
    const double bisection_tol_f = min_depth * relative_h_tolerance * 0.1;

    double f_low = get_h_from_eta(eta_low, b_sorted_vertices, cell_total_area, std::to_string(cell_id_int) + "_bisec_low") - h_avg_non_negative; // 计算下界的f值
    double f_high = get_h_from_eta(eta_high, b_sorted_vertices, cell_total_area, std::to_string(cell_id_int) + "_bisec_high") - h_avg_non_negative; // 计算上界的f值

    int expand_count = 0; // 扩大搜索区间计数
    while (f_low * f_high >= 0 && expand_count < 5) { // 如果f_low和f_high同号，尝试扩大搜索区间
        if (std::abs(f_low) < bisection_tol_f) {
             eta_mid = eta_low;
             if (should_print_vfr_internal_debug) std::cout << "    VFR_BISECTION_BRACKET_LOW_CONV: f_low already close to zero. eta_mid=" << eta_mid << std::endl; // VFR调试打印：下界f值已接近零
             goto bisection_finished;
        }
        if (std::abs(f_high) < bisection_tol_f) {
             eta_mid = eta_high;
             if (should_print_vfr_internal_debug) std::cout << "    VFR_BISECTION_BRACKET_HIGH_CONV: f_high already close to zero. eta_mid=" << eta_mid << std::endl; // VFR调试打印：上界f值已接近零
             goto bisection_finished;
        }

        eta_high += (b2 - b0 + h_avg_non_negative) * (0.5 * (expand_count + 1)); // 修改：更积极地扩大上界 // 每次增加 (单元最大高差+水深) 的倍数
        f_high = get_h_from_eta(eta_high, b_sorted_vertices, cell_total_area, std::to_string(cell_id_int) + "_bisec_high_expand") - h_avg_non_negative; // 重新计算扩大后的f_high
        expand_count++; // 增加扩大计数
        if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印
            std::cout << "    VFR_BISECTION_EXPAND: f_low*f_high >=0. Expanded eta_high to " << eta_high
                      << ", new f_high=" << f_high << std::endl; // VFR调试打印：二分法区间扩大
        }
    }

    if (f_low * f_high >= 0) { // 如果多次扩大后f_low和f_high仍然同号
        if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印
            std::cerr << "VFR_BISECTION_ERR: Cell=" << cell_id_int << ", Time=" << current_sim_time
                      << ". Could not bracket root for bisection. f_low("<<eta_low<<")=" << f_low << ", f_high("<<eta_high<<")=" << f_high
                      << ". Returning Newton's last eta or b0+h_avg." << std::endl; // VFR调试打印：二分法无法包围根
        }

        if (iter_count_newton == max_vfr_iters || std::abs(f_k_newton) > std::abs(f_low) || std::abs(f_k_newton) > std::abs(f_high) ) { // 修改：如果牛顿法完全失败，或其误差比当前二分法边界的误差还大
             return b0 + h_avg_non_negative; // 返回最保守的估计 (最低点高程 + 平均水深)
        }
        return std::max(b0 - epsilon, eta_k_newton); // 否则返回牛顿法的最后结果
    }


    for (iter_count_bisection = 0; iter_count_bisection < max_bisection_iters; ++iter_count_bisection) { // 二分法迭代循环
        eta_mid = eta_low + (eta_high - eta_low) / 2.0; // 计算中点
        f_mid = get_h_from_eta(eta_mid, b_sorted_vertices, cell_total_area, std::to_string(cell_id_int) + "_bisec_mid") - h_avg_non_negative; // 计算中点的f值

        if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印
            std::cout << "    VFR_BISECTION_ITER[" << iter_count_bisection << "]: eta_low=" << eta_low << " (f=" << f_low
                      << "), eta_high=" << eta_high << " (f=" << f_high
                      << "), eta_mid=" << eta_mid << " (f=" << f_mid << ")" << std::endl; // VFR调试打印：二分法迭代详情
        }

        if (std::abs(f_mid) < bisection_tol_f || (eta_high - eta_low) / 2.0 < bisection_tol_eta) { // 如果f_mid足够小或区间足够小
            if (should_print_vfr_internal_debug) std::cout << "    VFR_BISECTION_CONVERGED at iter " << iter_count_bisection << std::endl; // VFR调试打印：二分法收敛
            goto bisection_finished; // 跳到二分法结束标签
        }

        if ((f_low * f_mid) < 0) { // 如果解在 [eta_low, eta_mid] 区间
            eta_high = eta_mid; // 更新上界
            f_high = f_mid; // 更新上界的f值
        } else { // 否则解在 [eta_mid, eta_high] 区间
            eta_low = eta_mid; // 更新下界
            f_low = f_mid; // 更新下界的f值
        }
    }

bisection_finished:; // 二分法结束标签
    if (iter_count_bisection == max_bisection_iters && !(std::abs(f_mid) < bisection_tol_f || (eta_high - eta_low) / 2.0 < bisection_tol_eta)) { // 如果二分法达到最大迭代次数但未收敛
        if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印
             std::cerr << "VFR_BISECTION_WARN: Cell=" << cell_id_int << ", Time=" << current_sim_time
                       << ". Bisection did not converge in " << max_bisection_iters << " iters. "
                       << "Final eta_mid=" << eta_mid << ", f_mid=" << f_mid
                       << ". Using this result." << std::endl; // VFR调试打印：二分法未收敛，但仍使用当前结果
        }
    }

    if (should_print_vfr_internal_debug && iter_count_bisection < max_bisection_iters) { // 如果启用了VFR内部调试打印且二分法正常结束
        std::cout << "  VFR_DETAIL_RETURN (Bisection Converged): Cell=" << cell_id_int << ", Final eta_mid = " << std::max(b0 - epsilon, eta_mid) << std::endl << std::endl; // VFR调试打印：二分法收敛并返回结果
    } else if (should_print_vfr_internal_debug) { // 如果启用了VFR内部调试打印但二分法未正常结束（例如达到最大迭代）
        std::cout << "  VFR_DETAIL_RETURN (Bisection MaxIter or Other): Cell=" << cell_id_int << ", Final eta_mid = " << std::max(b0 - epsilon, eta_mid) << std::endl << std::endl; // VFR调试打印：二分法达到最大迭代或其他情况并返回结果
    }


    return std::max(b0 - epsilon, eta_mid); // 确保返回的水位不低于最低点
}
} // namespace HydroCore