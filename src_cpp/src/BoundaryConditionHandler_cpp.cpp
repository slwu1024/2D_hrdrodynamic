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
    const PrimitiveVars_cpp& W_L_flux, // 静水重构后的左侧界面状态 {h_L_star, u_L, v_L}
    const HalfEdge_cpp& he,            // 边界半边，其 he.normal 指向外部
    double target_Q_total_from_csv,    // 从CSV读取的流量值 (正代表入流，负代表出流)
    int original_segment_id_for_length, // 用于查找边界长度的原始ID
    const BoundaryDefinition_cpp& bc_def_for_hint
) const {
    // bool enable_detailed_debug = (original_segment_id_for_length == 5); // 只对我们的目标线段进行详细打印
    bool enable_detailed_debug = true; // <--- 修改: 强制启用调试打印

    if (enable_detailed_debug) {
        std::cout << "[DEBUG_HANDLE_TOTAL_DISCHARGE_ENTRY] HE_ID: " << he.id << ", Cell_L_ID: " << cell_L_id
                  << ", OrigSegID: " << original_segment_id_for_length << std::endl;
        std::cout << "  Target_Q_CSV: " << target_Q_total_from_csv << std::endl;
        std::cout << "  W_L_flux (static recon from calculate_boundary_flux): h=" << W_L_flux.h
                  << ", u=" << W_L_flux.u << ", v=" << W_L_flux.v << std::endl;
        std::cout << "  he.normal: (" << he.normal[0] << "," << he.normal[1] << ")" << std::endl;
        if (bc_def_for_hint.has_flow_direction_hint) {
            std::cout << "  FlowHint: (" << bc_def_for_hint.flow_direction_hint_x << "," << bc_def_for_hint.flow_direction_hint_y << ")" << std::endl;
        }
    }

    // --- 1. 计算单宽流量 qn_target_physical ---
    double total_length_marker = 0.0;
    auto it_len = marker_total_lengths_internal.find(original_segment_id_for_length);
    if (it_len != marker_total_lengths_internal.end()) {
        total_length_marker = it_len->second;
    }

    if (enable_detailed_debug) {
        std::cout << "  TotalLengthForMarker (OrigSegID " << original_segment_id_for_length << "): " << total_length_marker << std::endl;
    }

    // 如果边界长度过小，无法分配流量，则视为墙体
    if (total_length_marker < epsilon) {
        // std::cerr << "Warning (total_discharge_bc): Total length for marker "
        //           << original_segment_id_for_length << " is near zero. Treating as wall." << std::endl;
        return handle_wall_boundary(W_L_flux, he); // 长度为0，按墙处理
    }

    // qn_target_physical: 用户期望的物理法向单宽流量。
    // 正值表示入流 (水流进入计算域)，负值表示出流。
    double qn_target_physical = target_Q_total_from_csv / total_length_marker;
    if (enable_detailed_debug) {
        std::cout << "  qn_target_physical: " << qn_target_physical << std::endl;
    }

    // --- 2. 初始化鬼单元状态 ---
    PrimitiveVars_cpp W_ghost_final; // 最终用于HLLC的鬼单元原始变量

    // --- 3. 判断入流且内部接近干涸的情况 ---
    bool is_inflow = qn_target_physical > epsilon; // 判断是否为入流
    // W_L_flux.h 是经过静水重构后的内部界面水深，用它判断内部是否干
    bool internal_is_dry_or_very_shallow = (W_L_flux.h < min_depth_internal * 1.5); // 可以调整这个阈值，比如 min_depth_internal * 2.0
    if (enable_detailed_debug) {
        std::cout << "  is_inflow: " << is_inflow << ", internal_is_dry_or_very_shallow: " << internal_is_dry_or_very_shallow << std::endl;
    }
    if (is_inflow && internal_is_dry_or_very_shallow) { // 情况 A
        if (enable_detailed_debug) std::cout << "  Branch A: Inflow to dry/shallow." << std::endl;
        double qn_inflow_magnitude = std::abs(qn_target_physical);
        if (qn_inflow_magnitude < epsilon) {
            W_ghost_final.h = 0.0; W_ghost_final.u = 0.0; W_ghost_final.v = 0.0;
        } else {
            double h_critical = std::cbrt((qn_inflow_magnitude * qn_inflow_magnitude) / g_internal);
            W_ghost_final.h = std::max(h_critical, min_depth_internal);
            double un_ghost_physical;
            if (W_ghost_final.h < epsilon) {
                un_ghost_physical = 0.0; W_ghost_final.h = 0.0;
            } else {
                 un_ghost_physical = qn_inflow_magnitude / W_ghost_final.h;
            }
            // --- 调试打印 un_ghost_projection_on_normal 的计算 ---
            double un_ghost_projection_on_normal = -un_ghost_physical; // 默认
            if (bc_def_for_hint.has_flow_direction_hint) {
                double dot_product = bc_def_for_hint.flow_direction_hint_x * he.normal[0] +
                                     bc_def_for_hint.flow_direction_hint_y * he.normal[1];
                if (enable_detailed_debug) std::cout << "    Hint Dot Product: " << dot_product << std::endl;
                if (dot_product >= 0) { // 流量提示与法向量同向
                    un_ghost_projection_on_normal = un_ghost_physical;
                } else { // 流量提示与法向量反向
                    un_ghost_projection_on_normal = -un_ghost_physical;
                }
            }
            if (enable_detailed_debug) std::cout << "    un_ghost_physical: " << un_ghost_physical << ", un_ghost_projection_on_normal: " << un_ghost_projection_on_normal << std::endl;
            // --- 结束调试 ---
            double ut_ghost_projection_on_tangent = 0.0;
            W_ghost_final.u = un_ghost_projection_on_normal * he.normal[0] - ut_ghost_projection_on_tangent * he.normal[1];
            W_ghost_final.v = un_ghost_projection_on_normal * he.normal[1] + ut_ghost_projection_on_tangent * he.normal[0];
        }
    } else { // 情况 B
        if (enable_detailed_debug) std::cout << "  Branch B: Outflow or inflow to wet." << std::endl;
        double h_ghost_calc = W_L_flux.h; // 对入流到湿区和出流都用 W_L_flux.h 作为基础
        W_ghost_final.h = std::max(0.0, h_ghost_calc);
        double un_ghost_physical;
        double h_ghost_for_vel_calc = std::max(W_ghost_final.h, epsilon);
        if (std::abs(qn_target_physical) < epsilon && W_ghost_final.h < min_depth_internal) {
            un_ghost_physical = 0.0; W_ghost_final.h = 0.0;
        } else if (h_ghost_for_vel_calc < epsilon && std::abs(qn_target_physical) > epsilon) {
            if (enable_detailed_debug) std::cout << "  WARN: h_ghost is zero but q_n non-zero. Treating as wall." << std::endl;
            return handle_wall_boundary(W_L_flux, he);
        } else {
            un_ghost_physical = qn_target_physical / h_ghost_for_vel_calc;
        }
        // ... (速度限制逻辑不变) ...
        const double MAX_ABS_UN_PHYSICAL = 20.0;
        if (std::abs(un_ghost_physical) > MAX_ABS_UN_PHYSICAL) {
            if (enable_detailed_debug) std::cout << "    WARN: un_ghost_physical " << un_ghost_physical << " clamped." << std::endl;
            un_ghost_physical = std::copysign(MAX_ABS_UN_PHYSICAL, un_ghost_physical);
        }
        // --- 调试打印 un_ghost_projection_on_normal 的计算 ---
        double un_ghost_projection_on_normal = -un_ghost_physical; // 默认
        if (bc_def_for_hint.has_flow_direction_hint) {
            double dot_product = bc_def_for_hint.flow_direction_hint_x * he.normal[0] +
                                 bc_def_for_hint.flow_direction_hint_y * he.normal[1];
            if (enable_detailed_debug) std::cout << "    Hint Dot Product: " << dot_product << std::endl;
            if (dot_product >= 0) {
                un_ghost_projection_on_normal = un_ghost_physical;
            } else {
                un_ghost_projection_on_normal = -un_ghost_physical;
            }
        }
        if (enable_detailed_debug) std::cout << "    un_ghost_physical: " << un_ghost_physical << ", un_ghost_projection_on_normal: " << un_ghost_projection_on_normal << std::endl;
        // --- 结束调试 ---
        double ut_ghost_projection_on_tangent = -W_L_flux.u * he.normal[1] + W_L_flux.v * he.normal[0];
        W_ghost_final.u = un_ghost_projection_on_normal * he.normal[0] - ut_ghost_projection_on_tangent * he.normal[1];
        W_ghost_final.v = un_ghost_projection_on_normal * he.normal[1] + ut_ghost_projection_on_tangent * he.normal[0];
    }

    if (W_ghost_final.h < 0.0) W_ghost_final.h = 0.0;
    if (W_ghost_final.h < min_depth_internal) {
        W_ghost_final.u = 0.0; W_ghost_final.v = 0.0;
    }

    if (enable_detailed_debug) {
        std::cout << "  Final W_ghost_final for HLLC: h=" << W_ghost_final.h
                  << ", u=" << W_ghost_final.u << ", v=" << W_ghost_final.v << std::endl;
        std::cout << "  Input to HLLC (WL, WR, N): WL(h,u,v)=(" << W_L_flux.h << "," << W_L_flux.u << "," << W_L_flux.v << ")"
                  << " WR(h,u,v)=(" << W_ghost_final.h << "," << W_ghost_final.u << "," << W_ghost_final.v << ")"
                  << " Normal=(" << he.normal[0] << "," << he.normal[1] << ")" << std::endl;
    }

    std::array<double, 3> flux_result = flux_calculator_ptr_internal->calculate_hllc_flux(W_L_flux, W_ghost_final, he.normal);

    if (enable_detailed_debug) {
        std::cout << "  HLLC Flux Result (Cartesian): Fh=" << flux_result[0]
                  << ", Fhu=" << flux_result[1] << ", Fhv=" << flux_result[2] << std::endl;
    }
    return flux_result;
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