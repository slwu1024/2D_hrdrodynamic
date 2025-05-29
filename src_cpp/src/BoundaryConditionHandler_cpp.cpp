// src_cpp/src/BoundaryConditionHandler_cpp.cpp
#include "BoundaryConditionHandler_cpp.h" // 包含对应的头文件
#include "TimeIntegrator_cpp.h"
#include <stdexcept> // 包含标准异常
#include <iostream>  // 包含输入输出流 (用于调试)
#include <algorithm> // 包含算法如 std::lower_bound, std::sort
#include <limits>    // 包含数值极值
#include <cmath>     // 包含数学函数 std::abs, std::sqrt, std::pow
#include <iomanip>

namespace HydroCore { // 定义HydroCore命名空间

BoundaryConditionHandler_cpp::BoundaryConditionHandler_cpp( // 构造函数实现
    const Mesh_cpp* mesh_ptr_param,
    FluxCalculator_cpp* flux_calc_ptr_param,
    Reconstruction_cpp* recon_ptr_param,
    double gravity_param,
    double min_depth_param)
    : mesh_ptr_internal(mesh_ptr_param), // 初始化列表
      flux_calculator_ptr_internal(flux_calc_ptr_param),
      reconstruction_ptr_internal(recon_ptr_param),
      g_internal(gravity_param),
      min_depth_internal(min_depth_param),
      epsilon(1e-12) { // 初始化epsilon

    if (!mesh_ptr_internal || !flux_calculator_ptr_internal || !reconstruction_ptr_internal) { // 检查指针是否有效
        throw std::invalid_argument("BoundaryConditionHandler_cpp: Mesh, FluxCalculator, or Reconstruction pointer is null."); // 抛出无效参数异常
    }
    preprocess_boundaries(); // 在构造时预处理边界信息
    // std::cout << "C++ BoundaryConditionHandler_cpp constructed." << std::endl; // 打印构造信息
} // 结束构造函数

void BoundaryConditionHandler_cpp::preprocess_boundaries() {
    if (!mesh_ptr_internal) return;
    marker_total_lengths_internal.clear(); // 清空旧的长度映射
    for (const auto& he : mesh_ptr_internal->half_edges) { // 遍历所有半边
        // 只为那些是边界半边且具有有效原始 .poly 线段ID的半边计算总长度
        if (he.twin_half_edge_id == -1 && he.original_poly_segment_id != -1) { // 检查是否为边界且有原始ID
            marker_total_lengths_internal[he.original_poly_segment_id] += he.length; // 使用原始线段ID作为键累加长度
        }
    }
}

void BoundaryConditionHandler_cpp::set_boundary_definitions(const std::map<int, BoundaryDefinition_cpp>& definitions) { // 设置边界定义方法实现
    bc_definitions_internal = definitions; // 直接赋值
} // 结束方法

void BoundaryConditionHandler_cpp::set_waterlevel_timeseries_data(const std::map<int, std::vector<TimeseriesPoint_cpp>>& ts_data) { // 设置水位时间序列数据方法实现
    waterlevel_ts_data_internal.clear(); // 清空旧数据
    for(const auto& pair : ts_data) { // 遍历传入的数据
        waterlevel_ts_data_internal[pair.first] = pair.second; // 赋值
        // 确保时间序列是排序的 (如果Python端没有保证)
        // std::sort(waterlevel_ts_data_internal[pair.first].begin(), waterlevel_ts_data_internal[pair.first].end(),
        //           [](const TimeseriesPoint_cpp& a, const TimeseriesPoint_cpp& b){ return a.time < b.time; });
    }
} // 结束方法

void BoundaryConditionHandler_cpp::set_discharge_timeseries_data(const std::map<int, std::vector<TimeseriesPoint_cpp>>& ts_data) { // 设置流量时间序列数据方法实现
    discharge_ts_data_internal.clear(); // 清空旧数据
    for(const auto& pair : ts_data) { // 遍历传入的数据
        discharge_ts_data_internal[pair.first] = pair.second; // 赋值
        // 确保时间序列是排序的
        // std::sort(discharge_ts_data_internal[pair.first].begin(), discharge_ts_data_internal[pair.first].end(),
        //           [](const TimeseriesPoint_cpp& a, const TimeseriesPoint_cpp& b){ return a.time < b.time; });
    }
} // 结束方法

PrimitiveVars_cpp BoundaryConditionHandler_cpp::conserved_to_primitive(const std::array<double, 3>& U_cell) const { // 守恒量转原始量实现
    double h = U_cell[0]; // 获取水深
    if (h < min_depth_internal) { // 如果水深小于最小深度
        return {h, 0.0, 0.0}; // 返回水深和零速度
    }
    double h_for_division = std::max(h, epsilon); // 用于除法的安全水深
    return {h, U_cell[1] / h_for_division, U_cell[2] / h_for_division}; // 返回原始变量
} // 结束方法

double BoundaryConditionHandler_cpp::get_timeseries_value(int marker, double time_current, BoundaryType_cpp type_for_map_selection) const { // 获取时间序列值实现
    const std::map<int, std::vector<TimeseriesPoint_cpp>>* target_map = nullptr; // 目标数据map指针

    if (type_for_map_selection == BoundaryType_cpp::WATERLEVEL) { // 如果是水位时间序列
        target_map = &waterlevel_ts_data_internal; // 指向水位数据
    } else if (type_for_map_selection == BoundaryType_cpp::TOTAL_DISCHARGE) { // 如果是总流量时间序列
        target_map = &discharge_ts_data_internal; // 指向流量数据
    } else { // 其他类型
        return std::numeric_limits<double>::quiet_NaN(); // 返回NaN
    }

    auto it_map = target_map->find(marker); // 查找对应标记的数据
    if (it_map == target_map->end() || it_map->second.empty()) { // 如果未找到或数据为空
        // std::cerr << "Warning (get_timeseries_value): No timeseries data for marker " << marker << std::endl; // 打印警告
        return std::numeric_limits<double>::quiet_NaN(); // 返回NaN
    }

    const std::vector<TimeseriesPoint_cpp>& series = it_map->second; // 获取时间序列数据
    if (series.empty()) return std::numeric_limits<double>::quiet_NaN(); // 如果序列为空则返回NaN

    // 确保系列是按时间排序的 (应该在set数据时完成，但这里可以再次检查或处理)
    // For simplicity, assuming sorted here.

    // 找到第一个时间点 >= time_current 的位置
    auto it_upper = std::lower_bound(series.begin(), series.end(), time_current, // 使用lower_bound查找
                                     [](const TimeseriesPoint_cpp& p, double val) { // lambda比较函数
                                         return p.time < val; // 比较时间
                                     }); // 结束查找

    if (it_upper == series.begin()) { // 如果 time_current 小于或等于序列中的第一个时间点
        return series.front().value; // 返回第一个值 (不外插)
    }
    if (it_upper == series.end()) { // 如果 time_current 大于或等于序列中的最后一个时间点
        return series.back().value; // 返回最后一个值 (不外插)
    }

    const TimeseriesPoint_cpp& p_upper = *it_upper; // 上界点
    const TimeseriesPoint_cpp& p_lower = *(it_upper - 1); // 下界点

    if (std::abs(p_upper.time - p_lower.time) < epsilon) { // 如果两个时间点非常接近
        return p_upper.value; // 返回上界点的值
    }

    double t_ratio = (time_current - p_lower.time) / (p_upper.time - p_lower.time); // 计算插值比例
    return p_lower.value + t_ratio * (p_upper.value - p_lower.value); // 返回插值结果
} // 结束方法


// std::array<double, 3> BoundaryConditionHandler_cpp::handle_wall_boundary( // 处理墙体边界实现
//     const PrimitiveVars_cpp& W_L_reconstructed_iface, // 参数1: 重构后的左侧界面原始变量
//     const HalfEdge_cpp& he // 参数2: 边界半边
//     ) const { // const成员函数
//     PrimitiveVars_cpp W_ghost; // 声明虚拟单元状态
//     W_ghost.h = W_L_reconstructed_iface.h; // 虚拟单元水深等于重构后的内部界面水深
//     // 根据重构后的内部界面速度计算法向和切向速度
//     double unL = W_L_reconstructed_iface.u * he.normal[0] + W_L_reconstructed_iface.v * he.normal[1]; // 计算内部法向速度
//     double utL = -W_L_reconstructed_iface.u * he.normal[1] + W_L_reconstructed_iface.v * he.normal[0]; // 计算内部切向速度
//     // 构造虚拟单元速度 (法向速度反号，切向速度不变)
//     W_ghost.u = (-unL) * he.normal[0] - utL * he.normal[1]; // 虚拟单元x速度
//     W_ghost.v = (-unL) * he.normal[1] + utL * he.normal[0]; // 虚拟单元y速度
//     // 使用重构后的内部界面状态 W_L_reconstructed_iface 和构造的 W_ghost 计算通量
//     return flux_calculator_ptr_internal->calculate_hllc_flux(W_L_reconstructed_iface, W_ghost, he.normal); // 计算通量
// }
std::array<double, 3> BoundaryConditionHandler_cpp::handle_wall_boundary( // 处理墙体边界实现
    const PrimitiveVars_cpp& W_L_reconstructed_iface, // 参数1: 重构后的左侧界面原始变量
    const HalfEdge_cpp& he // 参数2: 边界半边
    ) const { // const成员函数

    PrimitiveVars_cpp W_ghost; // 声明虚拟单元状态
    W_ghost.h = W_L_reconstructed_iface.h; // 虚拟单元水深等于重构后的内部界面水深

    // 根据重构后的内部界面速度计算法向和切向速度
    double unL = W_L_reconstructed_iface.u * he.normal[0] + W_L_reconstructed_iface.v * he.normal[1]; // 计算内部法向速度
    double utL = -W_L_reconstructed_iface.u * he.normal[1] + W_L_reconstructed_iface.v * he.normal[0]; // 计算内部切向速度

    // 构造虚拟单元速度 (法向速度反号，切向速度不变)
    // 目标: un_ghost = -unL, ut_ghost = utL
    double un_ghost_target = -unL; // 目标法向速度
    double ut_ghost_target = utL;  // 目标切向速度

    W_ghost.u = un_ghost_target * he.normal[0] - ut_ghost_target * he.normal[1]; // 虚拟单元x速度
    W_ghost.v = un_ghost_target * he.normal[1] + ut_ghost_target * he.normal[0]; // 虚拟单元y速度



    // 使用重构后的内部界面状态 W_L_reconstructed_iface 和构造的 W_ghost 计算通量
    return flux_calculator_ptr_internal->calculate_hllc_flux(W_L_reconstructed_iface, W_ghost, he.normal); // 计算通量
}

std::array<double, 3> BoundaryConditionHandler_cpp::handle_free_outflow_boundary( // 处理自由出流边界实现
    const PrimitiveVars_cpp& W_L_reconstructed_iface, // 参数1: 重构后的左侧界面原始变量
    const HalfEdge_cpp& he // 参数2: 边界半边
    ) const { // const成员函数
    // 自由出流：虚拟单元状态 W_ghost 等于重构后的内部界面状态 W_L_reconstructed_iface
    PrimitiveVars_cpp W_ghost = W_L_reconstructed_iface; // 虚拟单元状态等于左侧界面状态
    // 使用重构后的内部界面状态 W_L_reconstructed_iface 和 W_ghost 计算通量
    return flux_calculator_ptr_internal->calculate_hllc_flux(W_L_reconstructed_iface, W_ghost, he.normal); // 计算通量
}


std::array<double, 3> BoundaryConditionHandler_cpp::handle_waterlevel_boundary( // 处理水位边界实现 (修改后)
    const StateVector& U_state_all, // 参数1: 所有单元的当前守恒状态 (可能需要用于速度外插等，但h已处理)
    int cell_L_id, // 参数2: 内部单元的ID
    const PrimitiveVars_cpp& W_L_flux, // 参数3: *修改* - 传入静水重构后的左侧界面状态 {h_L_star, u_L, v_L}
    const HalfEdge_cpp& he, // 参数4: 边界半边
    double target_eta // 参数5: 目标水位
    ) const { // const成员函数
    // W_L_flux 已经是界面上基于静水重构的左侧状态

    // 1. 估算边界处底高程 b_bnd (逻辑不变)
    // ... (计算 b_bnd 的代码保持不变) ...
    const Node_cpp* n_origin = mesh_ptr_internal->get_node_by_id(he.origin_node_id); // 获取起点
    const HalfEdge_cpp* he_next = mesh_ptr_internal->get_half_edge_by_id(he.next_half_edge_id); // 获取下一半边
    const Node_cpp* n_end = nullptr; // 初始化终点为空
    if (he_next) n_end = mesh_ptr_internal->get_node_by_id(he_next->origin_node_id); // 获取终点
    double b_bnd; // 声明边界底高程
    if (n_origin && n_end) { // 如果起点和终点都有效
        b_bnd = (n_origin->z_bed + n_end->z_bed) / 2.0; // 取平均值
    } else { // 否则
        const Cell_cpp* cell_L = mesh_ptr_internal->get_cell_by_id(cell_L_id); // 获取左单元
        b_bnd = cell_L ? cell_L->z_bed_centroid : 0.0; // 使用单元形心底高程或0
        // std::cerr << "Warning (waterlevel_bc): Could not determine b_bnd accurately for he " << he.id << std::endl; // 打印警告
    } // 结束底高程估算

    // 2. 计算边界虚拟水深 h_ghost
    double h_ghost = std::max(0.0, target_eta - b_bnd); // 计算边界水深
    if (h_ghost < min_depth_internal) { // 如果边界处为干
        // 当边界干时，表现应类似墙体
        return handle_wall_boundary(W_L_flux, he); // 使用静水重构的W_L_flux按墙体处理
    } // 结束干边界处理

    // 3. 构造 W_ghost (虚拟单元状态)
    // 假设缓流，边界外的速度与内部（重构界面）速度相同，但水深由指定水位决定。
    PrimitiveVars_cpp W_ghost = {h_ghost, W_L_flux.u, W_L_flux.v}; // 构造虚拟单元状态，使用 W_L_flux 中的速度

    // 4. 计算通量
    // 直接使用传入的 W_L_flux 和构造的 W_ghost
    return flux_calculator_ptr_internal->calculate_hllc_flux(W_L_flux, W_ghost, he.normal); // 计算通量
} // 结束水位处理函数

std::array<double, 3> BoundaryConditionHandler_cpp::handle_total_discharge_boundary(
    const StateVector& U_state_all,
    int cell_L_id,
    const PrimitiveVars_cpp& W_L_flux,
    const HalfEdge_cpp& he,
    double target_Q_total_from_csv,
    int original_segment_id_for_length,
    const BoundaryDefinition_cpp& bc_def_for_hint) const {

    bool enable_detailed_debug = true; // 强制开启调试

    // ... (获取 total_length_for_this_marker 的代码不变) ...
    double total_length_for_this_marker = 0.0;
    auto it_len = marker_total_lengths_internal.find(original_segment_id_for_length);
    if (it_len != marker_total_lengths_internal.end()) {
        total_length_for_this_marker = it_len->second;
    }
    if (total_length_for_this_marker < epsilon) {
        if (std::abs(target_Q_total_from_csv) > epsilon) {
             std::cerr << "Warning (total_discharge_bc): Total length for marker "
                       << original_segment_id_for_length << " is near zero (" << total_length_for_this_marker
                       << "), but target_Q is " << target_Q_total_from_csv
                       << ". Treating as wall for HE_ID " << he.id << "." << std::endl;
        }
        return handle_wall_boundary(W_L_flux, he);
    }

    double qn_physical = target_Q_total_from_csv / total_length_for_this_marker;

    PrimitiveVars_cpp W_ghost;
    bool is_physical_inflow = qn_physical > epsilon;
    bool internal_is_dryish = W_L_flux.h < min_depth_internal * 1.5;

    // --- 鬼单元水深设定 (与最新版本一致，这部分应该是合理的) ---
    if (is_physical_inflow) {
        double qn_abs = std::abs(qn_physical);
        if (qn_abs < epsilon) {
            W_ghost.h = 0.0;
        } else {
            double h_characteristic_for_inflow = std::cbrt((qn_abs * qn_abs) / g_internal);
            W_ghost.h = std::max(h_characteristic_for_inflow, min_depth_internal * 5.0);
        }
    } else {
        W_ghost.h = std::max(0.0, W_L_flux.h);
    }
    if (W_ghost.h < min_depth_internal / 10.0) {
        W_ghost.h = 0.0;
    }
    // --- 水深设定结束 ---

    double h_for_vel_calc = std::max(W_ghost.h, epsilon);
    double vn_magnitude_abs = 0.0;
    if (W_ghost.h > epsilon) { // 只有在水深大于epsilon时才计算速度大小
        vn_magnitude_abs = std::abs(qn_physical / h_for_vel_calc);
    }

    // --- 尝试恢复“旧的”法向速度符号逻辑 ---
    // 这个逻辑的目标是让 un_ghost_projection_on_normal 对于入流时为负 (如你之前的调试输出)
    double un_ghost_projection_on_normal_candidate;

    if (W_ghost.h < epsilon) { // 如果鬼单元干
        un_ghost_projection_on_normal_candidate = 0.0;
    } else {
        if (is_physical_inflow) {
            un_ghost_projection_on_normal_candidate = -vn_magnitude_abs; // <--- 入流时，投影为负
        } else { // 出流 (qn_physical < 0) 或静止
            // 对于出流，qn_physical 为负，vn_magnitude_abs 为正
            // 如果希望投影也为负 (表示水从N界面流向鬼区域，与N方向相反)
            // 或者为正 (表示水从鬼区域流向N界面，与N方向相同，但流量为负)
            // 为了匹配你之前的 Fh = -qn_physical (qn为负时Fh为正)，可能投影也是负的
            un_ghost_projection_on_normal_candidate = -vn_magnitude_abs; // 假设出流时也用负投影（待验证）
                                                                       // 或者直接用 qn_physical / h_for_vel_calc，它本身会是负值
                                                                       // un_ghost_projection_on_normal_candidate = qn_physical / h_for_vel_calc;
        }
    }

    // (可选) 应用 flow_direction_hint 来调整或确认 un_ghost_projection_on_normal_candidate 的符号
    // 这里暂时省略，以简化和集中测试核心符号逻辑
    // 如果你之前的版本有复杂的hint逻辑影响法向速度符号，需要在这里恢复

    double final_un_ghost_projection_on_normal = un_ghost_projection_on_normal_candidate;

    // 限制速度
    const double MAX_ABS_UN_G = 25.0;
    if (std::abs(final_un_ghost_projection_on_normal) > MAX_ABS_UN_G) {
        if (enable_detailed_debug)
             std::cout << "  WARN: final_un_ghost_projection_on_normal (before clamp) "
                       << final_un_ghost_projection_on_normal << " (orig mag: " << vn_magnitude_abs << ")"
                       << " clamped to " << MAX_ABS_UN_G << std::endl;
        final_un_ghost_projection_on_normal = std::copysign(MAX_ABS_UN_G, final_un_ghost_projection_on_normal);
    }


    double vt_g_on_he_tangent;
    if (is_physical_inflow && internal_is_dryish) {
        vt_g_on_he_tangent = 0.0;
    } else {
        vt_g_on_he_tangent = -W_L_flux.u * he.normal[1] + W_L_flux.v * he.normal[0];
    }

    if (W_ghost.h < min_depth_internal) {
        final_un_ghost_projection_on_normal = 0.0;
        vt_g_on_he_tangent = 0.0;
    }

    W_ghost.u = final_un_ghost_projection_on_normal * he.normal[0] - vt_g_on_he_tangent * he.normal[1];
    W_ghost.v = final_un_ghost_projection_on_normal * he.normal[1] + vt_g_on_he_tangent * he.normal[0];

    if (enable_detailed_debug) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "[DEBUG_TOTAL_DISCHARGE_RESTORED_LOGIC] HE_ID: " << he.id << std::endl;
        std::cout << "  qn_physical: " << qn_physical << ", is_physical_inflow: " << is_physical_inflow << std::endl;
        std::cout << "  W_ghost.h: " << W_ghost.h << ", h_for_vel_calc: " << h_for_vel_calc << std::endl;
        std::cout << "  vn_magnitude_abs: " << vn_magnitude_abs << std::endl;
        std::cout << "  un_ghost_projection_on_normal_candidate: " << un_ghost_projection_on_normal_candidate << std::endl;
        std::cout << "  final_un_ghost_projection_on_normal (after clamp): " << final_un_ghost_projection_on_normal << std::endl;
        std::cout << "  Final W_ghost: h=" << W_ghost.h << ", u_g=" << W_ghost.u << ", v_g=" << W_ghost.v << std::endl;
        std::cout << std::defaultfloat;
    }

    return flux_calculator_ptr_internal->calculate_hllc_flux(W_L_flux, W_ghost, he.normal);
}


std::array<double, 3> BoundaryConditionHandler_cpp::calculate_boundary_flux( // 计算边界通量实现
    const StateVector & U_state_all, // 所有单元状态
    int cell_L_id,                   // 左单元ID
    const HalfEdge_cpp& he,          // 当前边界半边
    double time_current              // 当前时间
    ) const { // const 成员函数

    auto it_def = bc_definitions_internal.find(he.boundary_marker); // 查找边界标记对应的定义
    BoundaryDefinition_cpp bc_def; // 声明边界定义对象
    // ... (获取 bc_def 的逻辑保持不变) ...
    if (it_def != bc_definitions_internal.end()) { // 如果找到定义
        bc_def = it_def->second; // 使用找到的定义
    } else { // 如果未找到 (检查是否有默认标记，例如标记0)
        auto it_default_def = bc_definitions_internal.find(0); // 查找默认边界标记(0)的定义
        if (it_default_def != bc_definitions_internal.end()) { // 如果找到默认定义
            bc_def = it_default_def->second; // 使用默认定义
        } else { // 如果连默认定义都没有
            bc_def.type = BoundaryType_cpp::WALL; // 默认为墙体类型
        } // 结束默认查找
    } // 结束定义查找
    // --- 添加调试打印：检查特定边界标记 ---
    if (he.boundary_marker == 10) { // 仅在早期且标记为10时打印
        std::cout << "[DEBUG_BC_FLUX_ENTRY] Time: " << time_current << ", HE_ID: " << he.id
                  << ", Cell_L_ID: " << cell_L_id
                  << ", OrigSegID: " << he.original_poly_segment_id // 期望是 5
                  << ", Marker: " << he.boundary_marker           // 期望是 10
                  << ", Normal: (" << he.normal[0] << "," << he.normal[1] << ")"
                  << ", MidPoint: (" << he.mid_point[0] << "," << he.mid_point[1] << ")"
                  << std::endl;
        // 打印调用此函数时的左单元状态
        std::cout << "  Cell_L U(h,hu,hv) for BC_FLUX_ENTRY: " << U_state_all[cell_L_id][0] << ", "
                  << U_state_all[cell_L_id][1] << ", "
                  << U_state_all[cell_L_id][2] << std::endl;
    }
    // --- 调试打印结束 ---

    auto [W_L_recons_iface, _ /* W_R is not used for boundary */] = // 获取重构状态
        reconstruction_ptr_internal->get_reconstructed_interface_states( // 获取重构状态
            U_state_all, cell_L_id, -1, he, true // 传入状态，左单元ID，右单元ID(-1表示边界)，半边，是边界(true)
        ); // 结束获取状态

    const Cell_cpp* cell_L_ptr = mesh_ptr_internal->get_cell_by_id(cell_L_id); // 获取左单元指针
    if (!cell_L_ptr) { // 如果获取失败
         return {0.0, 0.0, 0.0}; // 返回零通量
    } // 结束检查

    double z_face; // 声明界面底高程
    const Node_cpp* n_origin = mesh_ptr_internal->get_node_by_id(he.origin_node_id); // 获取起点
    const HalfEdge_cpp* he_next_ptr = mesh_ptr_internal->get_half_edge_by_id(he.next_half_edge_id); // 获取下一半边
    const Node_cpp* n_end = nullptr; // 初始化终点为空
    if (he_next_ptr) n_end = mesh_ptr_internal->get_node_by_id(he_next_ptr->origin_node_id); // 获取终点

    if (n_origin && n_end) { // 如果起点终点都有效
        z_face = (n_origin->z_bed + n_end->z_bed) / 2.0; // 取平均值
    } else { // 否则
        z_face = cell_L_ptr->z_bed_centroid; // 使用单元形心底高程作为近似
    } // 结束界面底高程计算

    double eta_L_center = U_state_all[cell_L_id][0] + cell_L_ptr->z_bed_centroid; // 计算左单元中心水位
    double eta_L_at_face = eta_L_center; // 初始化界面左侧水位为中心值

    if (reconstruction_ptr_internal->get_scheme_type() != ReconstructionScheme_cpp::FIRST_ORDER) { // 如果不是一阶
         bool gradients_available = true; // 假设梯度已在RHS计算中准备好
         try{ // 尝试获取梯度
              const auto& grad_W_L = reconstruction_ptr_internal->get_gradient_for_cell(cell_L_id); // 获取左单元梯度
              std::array<double, 2> grad_eta_L = {grad_W_L[0][0] + cell_L_ptr->b_slope_x, grad_W_L[0][1] + cell_L_ptr->b_slope_y}; // 左单元水位梯度
              std::array<double, 2> vec_L_to_face = {he.mid_point[0] - cell_L_ptr->centroid[0], he.mid_point[1] - cell_L_ptr->centroid[1]}; // 左单元到界面向量
              eta_L_at_face += grad_eta_L[0] * vec_L_to_face[0] + grad_eta_L[1] * vec_L_to_face[1]; // 外插得到界面左侧水位
         } catch (const std::exception& e) { // 捕获异常 (例如梯度未准备好)
            gradients_available = false; // 标记梯度不可用
         }
    } // 结束高阶外插

    double h_L_star = std::max(0.0, eta_L_at_face - z_face); // 计算静水重构的界面左侧水深
    PrimitiveVars_cpp W_L_flux = {h_L_star, W_L_recons_iface.u, W_L_recons_iface.v}; // 使用h_L_star和重构的速度
    // --- 添加调试打印：W_L_flux ---
    if (he.boundary_marker == 10 && time_current < 0.1) {
        std::cout << "  [DEBUG_BC_FLUX_WLFLUX] W_L_flux (input to specific BC handler): h=" << W_L_flux.h
                  << ", u=" << W_L_flux.u << ", v=" << W_L_flux.v << std::endl;
    }
    // --- 调试打印结束 ---
    std::array<double, 3> numerical_flux_cartesian_result; // 用于存储结果


    switch (bc_def.type) {
        case BoundaryType_cpp::WALL:
            numerical_flux_cartesian_result = handle_wall_boundary(W_L_flux, he);
            break; // 添加 break
        case BoundaryType_cpp::FREE_OUTFLOW:
            numerical_flux_cartesian_result = handle_free_outflow_boundary(W_L_flux, he);
            break; // 添加 break
        case BoundaryType_cpp::WATERLEVEL: { // 注意这里枚举名已改为 WATERLEVEL
            double target_eta = get_timeseries_value(he.original_poly_segment_id, time_current, BoundaryType_cpp::WATERLEVEL);
            if (std::isnan(target_eta)) {
                numerical_flux_cartesian_result = handle_wall_boundary(W_L_flux, he);
            } else {
                numerical_flux_cartesian_result = handle_waterlevel_boundary(U_state_all, cell_L_id, W_L_flux, he, target_eta);
            }
            break; // 添加 break
        }
        case BoundaryType_cpp::TOTAL_DISCHARGE: { // 注意这里枚举名已改为 TOTAL_DISCHARGE
            double target_Q = get_timeseries_value(he.original_poly_segment_id, time_current, BoundaryType_cpp::TOTAL_DISCHARGE);
            // --- 添加调试打印：target_Q ---
            if (he.boundary_marker == 10 && time_current < 0.1) {
                 std::cout << "  [DEBUG_BC_FLUX_TARGETQ] Target_Q from timeseries for OrigSegID "
                           << he.original_poly_segment_id << ": " << target_Q << std::endl;
            }
            // --- 调试打印结束 ---
            if (std::isnan(target_Q)) {
                numerical_flux_cartesian_result = handle_wall_boundary(W_L_flux, he);
            } else {
                // 将 bc_def 传递给 handle_total_discharge_boundary
                numerical_flux_cartesian_result = handle_total_discharge_boundary(
                    U_state_all, cell_L_id, W_L_flux, he, target_Q,
                    he.original_poly_segment_id, // 这个是 original_segment_id_for_length
                    bc_def);                       // 传递获取到的 bc_def
            }
            break; // 添加 break
        }
        default:
            numerical_flux_cartesian_result = handle_wall_boundary(W_L_flux, he);
    }

    // --- 添加调试打印：最终从边界条件处理器返回的通量 ---
    if (he.boundary_marker == 10 && time_current < 0.1) {
        std::cout << "  [DEBUG_BC_FLUX_RESULT] Final Flux (Cartesian) for HE_ID " << he.id << ": Fh=" << numerical_flux_cartesian_result[0]
                  << ", Fhu=" << numerical_flux_cartesian_result[1] << ", Fhv=" << numerical_flux_cartesian_result[2] << std::endl;
    }
    // --- 调试打印结束 ---
    return numerical_flux_cartesian_result;
} // 结束方法

} // namespace HydroCore