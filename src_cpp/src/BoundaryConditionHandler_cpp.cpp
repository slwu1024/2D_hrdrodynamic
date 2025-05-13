// src_cpp/src/BoundaryConditionHandler_cpp.cpp
#include "BoundaryConditionHandler_cpp.h" // 包含对应的头文件
#include "TimeIntegrator_cpp.h"
#include <stdexcept> // 包含标准异常
#include <iostream>  // 包含输入输出流 (用于调试)
#include <algorithm> // 包含算法如 std::lower_bound, std::sort
#include <limits>    // 包含数值极值
#include <cmath>     // 包含数学函数 std::abs, std::sqrt, std::pow

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

void BoundaryConditionHandler_cpp::preprocess_boundaries() { // 预处理边界方法实现
    if (!mesh_ptr_internal) return; // 如果网格指针无效则返回

    marker_total_lengths_internal.clear(); // 清空标记总长度
    for (const auto& he : mesh_ptr_internal->half_edges) { // 遍历所有半边
        if (he.twin_half_edge_id == -1) { // 如果是边界边 (没有孪生半边)
            marker_total_lengths_internal[he.boundary_marker] += he.length; // 累加对应标记的总长度
        }
    }
} // 结束方法

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

    if (type_for_map_selection == BoundaryType_cpp::WATERLEVEL_TIMESERIES) { // 如果是水位时间序列
        target_map = &waterlevel_ts_data_internal; // 指向水位数据
    } else if (type_for_map_selection == BoundaryType_cpp::TOTAL_DISCHARGE_TIMESERIES) { // 如果是总流量时间序列
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
    W_ghost.u = (-unL) * he.normal[0] - utL * he.normal[1]; // 虚拟单元x速度
    W_ghost.v = (-unL) * he.normal[1] + utL * he.normal[0]; // 虚拟单元y速度
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

std::array<double, 3> BoundaryConditionHandler_cpp::handle_total_discharge_boundary( // 处理总流量边界实现 (修改后)
    const StateVector& U_state_all, // 参数1
    int cell_L_id, // 参数2
    const PrimitiveVars_cpp& W_L_flux, // 参数3: *修改* - 传入静水重构后的左侧界面状态 {h_L_star, u_L, v_L}
    const HalfEdge_cpp& he, // 参数4
    double target_Q_total, // 参数5
    int marker // 参数6
    ) const { // const成员函数
    // W_L_flux 已经是界面上基于静水重构的左侧状态

    // ... (计算 total_length_marker 和 qn_bnd 的逻辑不变) ...
    double total_length_marker = 0.0; // 初始化标记总长度
    auto it_len = marker_total_lengths_internal.find(marker); // 查找标记总长度
    if (it_len != marker_total_lengths_internal.end()) { // 如果找到
        total_length_marker = it_len->second; // 获取总长度
    } // 结束查找
    if (total_length_marker < epsilon) { // 如果总长度过小
        return handle_wall_boundary(W_L_flux, he); // 按墙体处理
    } // 结束长度检查
    double qn_bnd = target_Q_total / total_length_marker; // 计算法向单宽流量

    // ... (计算 h_bnd_ghost 的逻辑不变) ...
    double h_bnd_ghost; // 声明边界虚拟水深
    if (qn_bnd > 0) { // 如果是入流
        // 使用临界流深作为入流水深估计 (或者其他方法)
        h_bnd_ghost = std::cbrt(qn_bnd * qn_bnd / g_internal); // 计算临界水深
        h_bnd_ghost = std::max(h_bnd_ghost, min_depth_internal); // 保证最小水深
    } else { // 出流
        // 出流水深通常由内部决定，这里使用静水重构后的内部界面水深
        h_bnd_ghost = W_L_flux.h; // 假设出流水深等于内部静水重构界面水深 h_L_star
    } // 结束水深计算

    if (h_bnd_ghost < min_depth_internal) { // 如果计算出的边界水深过小
        return handle_wall_boundary(W_L_flux, he); // 按墙体处理
    } // 结束干边界处理

    // ... (计算 un_bnd_ghost, ut_bnd_ghost, W_ghost 的逻辑不变，但使用 W_L_flux 中的速度) ...
    double h_ghost_div = std::max(h_bnd_ghost, epsilon); // 用于除法的安全水深
    double un_bnd_ghost = qn_bnd / h_ghost_div; // 计算边界法向速度
    // 切向速度假设与重构后的内部界面切向速度相同
    double ut_bnd_ghost = -W_L_flux.u * he.normal[1] + W_L_flux.v * he.normal[0]; // 计算边界切向速度 (使用 W_L_flux 中的 u,v)

    PrimitiveVars_cpp W_ghost; // 声明虚拟单元状态
    W_ghost.h = h_bnd_ghost; // 设置水深
    W_ghost.u = un_bnd_ghost * he.normal[0] - ut_bnd_ghost * he.normal[1]; // 计算x方向速度
    W_ghost.v = un_bnd_ghost * he.normal[1] + ut_bnd_ghost * he.normal[0]; // 计算y方向速度

    // 计算通量
    // 直接使用传入的 W_L_flux 和构造的 W_ghost
    return flux_calculator_ptr_internal->calculate_hllc_flux(W_L_flux, W_ghost, he.normal); // 计算通量
} // 结束流量处理函数


std::array<double, 3> BoundaryConditionHandler_cpp::calculate_boundary_flux( // 计算边界通量实现
    const StateVector & U_state_all, // 所有单元状态
    int cell_L_id,                   // 左单元ID
    const HalfEdge_cpp& he,          // 当前边界半边
    double time_current              // 当前时间
    ) const { // const 成员函数

    auto it_def = bc_definitions_internal.find(he.boundary_marker); // 查找边界标记对应的定义
    BoundaryDefinition_cpp bc_def; // 声明边界定义对象
    if (it_def != bc_definitions_internal.end()) { // 如果找到定义
        bc_def = it_def->second; // 使用找到的定义
    } else { // 如果未找到 (检查是否有默认标记，例如标记0)
        auto it_default_def = bc_definitions_internal.find(0); // 查找默认边界标记(0)的定义
        if (it_default_def != bc_definitions_internal.end()) { // 如果找到默认定义
            bc_def = it_default_def->second; // 使用默认定义
            // std::cout << "Info (BC): Marker " << he.boundary_marker << " not defined, using default BC (marker 0)." << std::endl; // 打印信息
        } else { // 如果连默认定义都没有
            // std::cerr << "Warning (BC): Boundary marker " << he.boundary_marker << " (and default marker 0) not found. Using WALL." << std::endl; // 打印警告
            bc_def.type = BoundaryType_cpp::WALL; // 默认为墙体类型
        } // 结束默认查找
    } // 结束定义查找

    // 获取内部单元在界面上的重构状态 W_L = [h, u, v]
    auto [W_L_recons_iface, _ /* W_R is not used for boundary */] =
        reconstruction_ptr_internal->get_reconstructed_interface_states( // 获取重构状态
            U_state_all, cell_L_id, -1, he, true // 传入状态，左单元ID，右单元ID(-1表示边界)，半边，是边界(true)
        ); // 结束获取状态

    // --- 计算界面左侧的静水平衡状态 (h_L_star) ---
    const Cell_cpp* cell_L_ptr = mesh_ptr_internal->get_cell_by_id(cell_L_id); // 获取左单元指针
    if (!cell_L_ptr) { // 如果获取失败
         // std::cerr << "Error (BC): Could not get cell_L_ptr for ID " << cell_L_id << std::endl; // 打印错误
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
        // std::cerr << "Warning (BC): Using cell centroid elevation for face " << he.id << " elevation." << std::endl; // 打印警告
    } // 结束界面底高程计算

    double eta_L_center = U_state_all[cell_L_id][0] + cell_L_ptr->z_bed_centroid; // 计算左单元中心水位
    double eta_L_at_face = eta_L_center; // 初始化界面左侧水位为中心值

    // 如果是高阶格式，进行水位外插
    if (reconstruction_ptr_internal->get_scheme_type() != ReconstructionScheme_cpp::FIRST_ORDER) { // 如果不是一阶
         bool gradients_available = true; // 假设梯度已在RHS计算中准备好
         try{ // 尝试获取梯度
              const auto& grad_W_L = reconstruction_ptr_internal->get_gradient_for_cell(cell_L_id); // 获取左单元梯度
              // 计算水位梯度 grad(eta) = grad(h) + grad(b)
              std::array<double, 2> grad_eta_L = {grad_W_L[0][0] + cell_L_ptr->b_slope_x, grad_W_L[0][1] + cell_L_ptr->b_slope_y}; // 左单元水位梯度
              std::array<double, 2> vec_L_to_face = {he.mid_point[0] - cell_L_ptr->centroid[0], he.mid_point[1] - cell_L_ptr->centroid[1]}; // 左单元到界面向量
              eta_L_at_face += grad_eta_L[0] * vec_L_to_face[0] + grad_eta_L[1] * vec_L_to_face[1]; // 外插得到界面左侧水位
         } catch (const std::exception& e) { // 捕获异常 (例如梯度未准备好)
            // std::cerr << "Warning (BC): Could not get gradients for cell " << cell_L_id << " for BC extrapolation. Using cell center eta. Error: " << e.what() << std::endl; // 打印警告
            gradients_available = false; // 标记梯度不可用
         }
    } // 结束高阶外插

    double h_L_star = std::max(0.0, eta_L_at_face - z_face); // 计算静水重构的界面左侧水深
    // 构建用于通量计算的左侧状态 W_L_flux
    PrimitiveVars_cpp W_L_flux = {h_L_star, W_L_recons_iface.u, W_L_recons_iface.v}; // 使用h_L_star和重构的速度

    // --- 根据边界类型调用相应的处理函数 ---
    switch (bc_def.type) { // 判断边界类型
        case BoundaryType_cpp::WALL: // 墙体边界
            // handle_wall_boundary 需要 W_L_flux (包含了静水重构信息)
            return handle_wall_boundary(W_L_flux, he); // 调用墙体处理函数
        case BoundaryType_cpp::FREE_OUTFLOW: // 自由出流边界
            // handle_free_outflow_boundary 需要 W_L_flux
            return handle_free_outflow_boundary(W_L_flux, he); // 调用自由出流处理函数
        case BoundaryType_cpp::WATERLEVEL_TIMESERIES: { // 水位时间序列边界
            double target_eta = get_timeseries_value(he.boundary_marker, time_current, BoundaryType_cpp::WATERLEVEL_TIMESERIES); // 获取目标水位
            if (std::isnan(target_eta)) { // 如果获取失败 (返回NaN)
                // std::cerr << "Warning (BC): Failed to get waterlevel timeseries value for marker " << he.boundary_marker << ". Treating as WALL." << std::endl; // 打印警告
                return handle_wall_boundary(W_L_flux, he); // 退化为墙体
            } // 结束NaN检查
            // handle_waterlevel_boundary 需要 W_L_flux
            return handle_waterlevel_boundary(U_state_all, cell_L_id, W_L_flux, he, target_eta); // 调用水位处理函数 (注意: W_L_recons_iface参数改为W_L_flux)
        } // 结束水位边界处理
        case BoundaryType_cpp::TOTAL_DISCHARGE_TIMESERIES: { // 总流量时间序列边界
            double target_Q = get_timeseries_value(he.boundary_marker, time_current, BoundaryType_cpp::TOTAL_DISCHARGE_TIMESERIES); // 获取目标流量
            if (std::isnan(target_Q)) { // 如果获取失败
                 // std::cerr << "Warning (BC): Failed to get discharge timeseries value for marker " << he.boundary_marker << ". Treating as WALL." << std::endl; // 打印警告
                return handle_wall_boundary(W_L_flux, he); // 退化为墙体
            } // 结束NaN检查
            // handle_total_discharge_boundary 需要 W_L_flux
            return handle_total_discharge_boundary(U_state_all, cell_L_id, W_L_flux, he, target_Q, he.boundary_marker); // 调用流量处理函数 (注意: W_L_recons_iface参数改为W_L_flux)
        } // 结束流量边界处理
        default: // 未知或未定义边界类型
             // std::cerr << "Warning (BC): Unknown boundary type " << static_cast<int>(bc_def.type) << " for marker " << he.boundary_marker << ". Treating as WALL." << std::endl; // 打印警告
            return handle_wall_boundary(W_L_flux, he); // 默认为墙体
    } // 结束switch
} // 结束方法

} // namespace HydroCore