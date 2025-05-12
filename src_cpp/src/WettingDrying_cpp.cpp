// src_cpp/src/WettingDrying_cpp.cpp
#include "WettingDrying_cpp.h" // 包含对应的头文件
#include <iostream> // 包含输入输出流 (用于调试)
#include <stdexcept> // 包含标准异常 (例如 std::runtime_error)
#include <algorithm> // 包含算法如 std::sort (虽然这里传入的是已排序的)
#include <vector> // 包含vector容器

namespace HydroCore { // 定义HydroCore命名空间

VFRCalculator_cpp::VFRCalculator_cpp(double min_depth_param, // 构造函数实现
                                     double min_eta_change_iter_param,
                                     int max_vfr_iters_param,
                                     double relative_h_tolerance_param)
    : min_depth(min_depth_param), // 初始化列表
      min_eta_change_iter(min_eta_change_iter_param),
      max_vfr_iters(max_vfr_iters_param),
      relative_h_tolerance(relative_h_tolerance_param),
      epsilon(1e-12) { // 初始化epsilon
    // std::cout << "C++ VFRCalculator_cpp initialized." << std::endl; // 打印初始化信息
} // 结束构造函数

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

double VFRCalculator_cpp::get_eta_from_h( // 根据h计算eta的方法实现
    double h_avg,
    const std::vector<double>& b_sorted_vertices,
    const std::vector<Node_cpp>& cell_nodes_sorted,
    double cell_total_area,
    double eta_previous_guess, // 注意：这里要用 double，而不是 Node_cpp* 或其他
    const std::string& cell_id_for_debug) const {

    if (b_sorted_vertices.size() != 3 || cell_nodes_sorted.size() != 3) { // 检查顶点数量
        // std::cerr << "Warning (get_eta_from_h): Expected 3 vertices/nodes for cell " << cell_id_for_debug << std::endl; // 打印警告
         if (b_sorted_vertices.empty()) return 0.0; // 如果为空则返回0
        return b_sorted_vertices[0]; // 返回最低点高程
    }
    double b0 = b_sorted_vertices[0]; // 获取排序后的顶点高程b0

    if (h_avg < min_depth / 10.0) { // 如果目标平均水深非常小
        return b0; // 返回最低点高程
    }

    double eta_k; // 声明当前迭代的eta
    // 设置初始猜测值 eta_k
    if (eta_previous_guess >= b0 - epsilon) { // 如果提供了上一步的eta且合理 (注意这里用 eta_previous_guess, 是一个double)
         eta_k = eta_previous_guess; // 使用上一步的值
    } else { // 否则，使用默认猜测
        eta_k = (b_sorted_vertices[0] + b_sorted_vertices[1] + b_sorted_vertices[2]) / 3.0 + h_avg; // 形心高程 + 平均水深
        eta_k = std::max(b0, eta_k); // 确保不低于最低点
    }


    double f_k = 1.0; // 初始化一个非零值
    bool converged_abs = false; // 绝对误差收敛标志
    bool converged_rel = false; // 相对误差收敛标志
    int iter_count = 0; // 初始化迭代次数

    for (iter_count = 0; iter_count < max_vfr_iters; ++iter_count) { // 开始迭代
        double h_calc_k = get_h_from_eta(eta_k, b_sorted_vertices, cell_total_area, cell_id_for_debug); // 计算当前eta对应的平均水深
        f_k = h_calc_k - h_avg; // 计算目标函数值

        converged_abs = std::abs(f_k) < min_depth * relative_h_tolerance; // 检查绝对误差收敛
        converged_rel = std::abs(f_k / (h_avg + epsilon)) < relative_h_tolerance; // 检查相对误差收敛
        if (converged_abs || converged_rel) { // 如果任一满足
            break; // 达到收敛，跳出循环
        }

        double Aw_k = calculate_wet_surface_area(eta_k, b_sorted_vertices, cell_nodes_sorted, cell_total_area); // 计算水面面积
        double df_deta_k = (cell_total_area > epsilon) ? (Aw_k / cell_total_area) : epsilon; // 计算导数

        double eta_k_next; // 声明下一个迭代的eta
        if (std::abs(df_deta_k) < epsilon) { // 如果导数接近零
            // std::cout << "Debug (get_eta_from_h, cell " << cell_id_for_debug << "): Derivative near zero. eta_k=" << eta_k << ", Aw_k=" << Aw_k << ", f_k=" << f_k << std::endl; // 打印调试信息
            eta_k_next = eta_k - ( (f_k > 0) ? 1.0 : -1.0 ) * min_eta_change_iter * 10.0; // 尝试小步长调整
        } else { // 正常牛顿步
            double delta_eta = f_k / df_deta_k; // 计算牛顿步长
            // 限制步长
            double max_delta_eta = (b_sorted_vertices[2] - b_sorted_vertices[0] + 1.0) * 0.5; // 最大允许步长
            delta_eta = std::max(-max_delta_eta, std::min(max_delta_eta, delta_eta)); // 限制步长
            eta_k_next = eta_k - delta_eta; // 更新eta
        }
        eta_k_next = std::max(b0, eta_k_next); // 确保不低于最低点

        if (std::abs(eta_k_next - eta_k) < min_eta_change_iter) { // 如果eta变化量小于阈值
            eta_k = eta_k_next; // 更新eta
            break; // 认为收敛
        }
        eta_k = eta_k_next; // 更新eta，进行下一次迭代
    }

    if (iter_count == max_vfr_iters && !(converged_abs || converged_rel) ) { // 如果达到最大迭代次数仍未收敛
         // std::cerr << "Warning (get_eta_from_h, cell " << cell_id_for_debug
         //           << "): VFR Newton's method did not converge in " << max_vfr_iters << " iterations. "
         //           << "Target h_avg=" << h_avg << ", final eta=" << eta_k
         //           << ", calculated h=" << get_h_from_eta(eta_k, b_sorted_vertices, cell_total_area, "") // 避免递归debug打印
         //           << ", f_k=" << f_k << std::endl; // 打印警告
    }
    return std::max(b0, eta_k); // 返回最终计算得到的eta，并确保不小于最低点高程
} // 结束方法

} // namespace HydroCore