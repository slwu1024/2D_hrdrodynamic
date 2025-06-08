// src_cpp/src/HydroModelCore_cpp.cpp
#include "HydroModelCore_cpp.h" // 包含对应的头文件
#include "WettingDrying_cpp.h" // <--- 确保包含了这个头文件
#include "FluxCalculator_cpp.h"
#include "Profiler.h"
#include <stdexcept> // 包含标准异常
#include <iostream>  // 包含输入输出流
#include <algorithm> // 包含算法
#include <cmath>     // 包含数学函数
#include <vector>    // 包含vector容器
#include <array>     // 包含array容器
#include <limits>    // 为了 numeric_limits
#include <set>      // 需要包含
#include <iomanip> // 新增：为了std::fixed 和 std::setprecision

namespace HydroCore { // HydroCore命名空间开始

HydroModelCore_cpp::HydroModelCore_cpp() // 构造函数实现 (无参数)
    : gravity_internal(9.81), min_depth_internal(1e-6), cfl_number_internal(0.5), // 初始化列表
      total_time_internal(10.0), output_dt_internal(1.0), max_dt_internal(1.0),
      current_time_internal(0.0), step_count_internal(0), epsilon(1e-12),
      current_recon_scheme(ReconstructionScheme_cpp::FIRST_ORDER),
      current_riemann_solver(RiemannSolverType_cpp::HLLC),
      current_time_scheme(TimeScheme_cpp::FORWARD_EULER),
      last_calculated_dt_internal(0.0),
      model_fully_initialized_flag(false), // 初始化为未完成
      initial_conditions_set_flag(false), boundary_conditions_set_flag(false) {
    // 指针成员默认为 nullptr (由 unique_ptr 自动处理)
    // std::cout << "C++ HydroModelCore_cpp constructed (empty)." << std::endl; // 打印构造信息
    // std::cout << "  Call initialize_model_from_files to load mesh and setup components." << std::endl; // 打印提示信息
} // 结束构造函数

HydroModelCore_cpp::~HydroModelCore_cpp() { // 析构函数实现
    // std::unique_ptr 会自动管理内存
    // std::cout << "C++ HydroModelCore_cpp destroyed." << std::endl; // 打印析构信息
} // 结束析构函数

void HydroModelCore_cpp::initialize_model_from_files(
    const std::string& node_filepath,
    const std::string& cell_filepath,
    const std::string& edge_filepath,
    const std::vector<double>& cell_manning_values, // Python端传入的曼宁值
    double gravity, double min_depth, double cfl,
    double total_t, double output_dt_interval, double max_dt_val,
    ReconstructionScheme_cpp recon_scheme,
    RiemannSolverType_cpp riemann_solver,
    TimeScheme_cpp time_scheme) {
    PROFILE_FUNCTION(); // 记录整个初始化函数的耗时

    std::cout << "C++ HydroModelCore_cpp: Starting full model initialization from files." << std::endl; // 打印开始信息
    // 1. 创建并加载网格
    mesh_internal_ptr = std::make_unique<Mesh_cpp>(); // 创建Mesh_cpp对象
    mesh_internal_ptr->load_mesh_from_files(node_filepath, cell_filepath, edge_filepath, cell_manning_values); // 调用Mesh_cpp的方法加载网格数据
    std::cout << "  Mesh loaded into C++ HydroModelCore." << std::endl; // 打印网格加载完成信息

    // 2. 设置模拟参数
    gravity_internal = gravity; // 设置重力加速度
    min_depth_internal = min_depth; // 设置最小水深
    cfl_number_internal = cfl; // 设置CFL数
    total_time_internal = total_t; // 设置总模拟时长
    output_dt_internal = output_dt_interval; // 设置输出时间间隔
    max_dt_internal = max_dt_val; // 设置最大时间步长
    std::cout << "  Simulation parameters set." << std::endl; // 打印参数设置完成信息

    // 3. 设置数值方案
    current_recon_scheme = recon_scheme; // 设置重构方案
    current_riemann_solver = riemann_solver; // 设置黎曼求解器
    current_time_scheme = time_scheme; // 设置时间积分方案
    std::cout << "  Numerical schemes chosen." << std::endl; // 打印方案选择完成信息

    // 4. 创建计算组件 (顺序可能重要，例如BoundaryHandler需要FluxCalc和Recon)
    flux_calculator_ptr = std::make_unique<FluxCalculator_cpp>(gravity_internal, min_depth_internal, current_riemann_solver); // 创建通量计算器
    source_term_calculator_ptr = std::make_unique<SourceTermCalculator_cpp>(gravity_internal, min_depth_internal); // 创建源项计算器
    vfr_calculator_ptr = std::make_unique<VFRCalculator_cpp>(min_depth_internal); // 创建VFR计算器
    reconstruction_ptr = std::make_unique<Reconstruction_cpp>(current_recon_scheme, mesh_internal_ptr.get(), gravity_internal, min_depth_internal); // 创建重构器
    boundary_handler_ptr = std::make_unique<BoundaryConditionHandler_cpp>( // 创建边界条件处理器
        mesh_internal_ptr.get(), flux_calculator_ptr.get(), reconstruction_ptr.get(), gravity_internal, min_depth_internal
    ); // 结束创建
    time_integrator_ptr = std::make_unique<TimeIntegrator_cpp>( // 创建时间积分器
        current_time_scheme, // 时间积分方案
        [this](const StateVector& U, double t) { return this->_calculate_rhs_explicit_part_internal(U, t); }, // RHS函数
        [this](const StateVector& Ui, const StateVector& Uc, double d_t) { return this->_apply_friction_semi_implicit_internal(Ui, Uc, d_t); } // 摩擦函数
    ); // 结束创建
    std::cout << "  Computational components created." << std::endl; // 打印组件创建完成信息

    // 5. 初始化曼宁系数值 (从已加载到Mesh_cpp的单元中)
    _initialize_manning_from_mesh_internal(); // 调用内部方法初始化曼宁值
    std::cout << "  Manning coefficients initialized from mesh." << std::endl; // 打印曼宁值初始化完成信息

    model_fully_initialized_flag = true; // 标记模型已完全初始化
    // initial_conditions_set_flag 和 boundary_conditions_set_flag 将由各自的setter设置
    std::cout << "C++ HydroModelCore_cpp: Full model initialization complete. Ready for ICs and BCs." << std::endl; // 打印完全初始化完成信息
} // 结束函数


void HydroModelCore_cpp::_initialize_manning_from_mesh_internal() { // 从网格初始化曼宁系数(内部)实现
    if (mesh_internal_ptr && !mesh_internal_ptr->cells.empty()) { // 如果网格指针有效且单元不为空
        manning_n_values_internal.resize(mesh_internal_ptr->cells.size()); // 调整曼宁值数组大小
        for(size_t i = 0; i < mesh_internal_ptr->cells.size(); ++i) { // 遍历所有单元
            manning_n_values_internal[i] = mesh_internal_ptr->cells[i].manning_n; // 从单元对象中获取曼宁值
        }
    } else { // 否则
        manning_n_values_internal.clear(); // 清空曼宁值数组
        std::cerr << "Warning (_initialize_manning_from_mesh_internal): Mesh not available or no cells, Manning values not set." << std::endl; // 打印警告
    }
} // 结束函数

void HydroModelCore_cpp::setup_internal_flow_source(
    const std::string& line_name,
    const std::vector<int>& poly_node_ids_for_line_py,
    const std::vector<TimeseriesPoint_cpp>& q_timeseries, // 接收时程数据
    const std::array<double, 2>& direction_py
) {
    if (!mesh_internal_ptr) { // 检查网格指针
        std::cerr << "ERROR (setup_internal_flow_source): Mesh not initialized for line '" << line_name << "'." << std::endl;
        return;
    }
    if (poly_node_ids_for_line_py.size() < 2) { // 检查节点ID数量
        std::cerr << "ERROR (setup_internal_flow_source): poly_node_ids_for_line must have at least 2 nodes for line '" << line_name << "'." << std::endl;
        return;
    }
    if (q_timeseries.empty()) { // 检查时程数据是否为空
        std::cerr << "WARNING (setup_internal_flow_source): q_timeseries is empty for line '" << line_name << "'. No flow will be applied." << std::endl;
        // 即使时程为空，也可能需要设置方向和识别边，只是流量始终为0
    }

    std::cout << "C++ HydroModelCore: Setting up internal flow source '" << line_name << "'." << std::endl; // 打印设置信息
    std::cout << "  Poly line nodes (original 1-based): ";
    for (int id : poly_node_ids_for_line_py) std::cout << id << " ";
    std::cout << std::endl;
    std::cout << "  Direction: (" << direction_py[0] << "," << direction_py[1] << ")" << std::endl;
    std::cout << "  Received " << q_timeseries.size() << " timeseries points." << std::endl;

    // 存储时程数据和方向
    internal_q_timeseries_data_map_internal[line_name] = q_timeseries;
    // 如果时程数据非空，确保它是按时间排序的 (或者依赖Python端排序)
    if (!internal_q_timeseries_data_map_internal[line_name].empty()) {
        std::sort(internal_q_timeseries_data_map_internal[line_name].begin(),
                  internal_q_timeseries_data_map_internal[line_name].end(),
                  [](const TimeseriesPoint_cpp& a, const TimeseriesPoint_cpp& b) {
                      return a.time < b.time;
                  });
    }
    internal_flow_directions_map_internal[line_name] = direction_py;

    // 清空这条线之前可能存在的边信息 (如果允许重复调用setup修改同一条线)
    all_internal_flow_edges_info_map_internal[line_name].clear();
    double current_line_total_length = 0.0; // 这条线的总长度

    std::set<int> line_specific_mesh_node_ids; // 这条线上的网格节点ID

    for (size_t i = 0; i < poly_node_ids_for_line_py.size() - 1; ++i) { // 遍历构成线的每条子线段
        int p_start_id_poly_1_based = poly_node_ids_for_line_py[i];
        int p_end_id_poly_1_based = poly_node_ids_for_line_py[i + 1];
        int actual_mesh_start_node_id = p_start_id_poly_1_based - 1;
        int actual_mesh_end_node_id = p_end_id_poly_1_based - 1;

        const Node_cpp* n_p_start = mesh_internal_ptr->get_node_by_id(actual_mesh_start_node_id);
        const Node_cpp* n_p_end = mesh_internal_ptr->get_node_by_id(actual_mesh_end_node_id);

        if (!n_p_start || !n_p_end) {
            std::cerr << "ERROR (setup_internal_flow_source line '" << line_name << "'): Could not find mesh nodes for polyline segment using mesh IDs "
                      << actual_mesh_start_node_id << " or " << actual_mesh_end_node_id << std::endl;
            continue;
        }
        // (这部分打印可以保留或简化)
        std::cout << "  Processing segment for line '" << line_name << "' from original poly_node " << p_start_id_poly_1_based
                  << " (mesh_id " << actual_mesh_start_node_id << ", coords: " << n_p_start->x << "," << n_p_start->y << ")"
                  << " to original poly_node " << p_end_id_poly_1_based << " (mesh_id " << actual_mesh_end_node_id
                  << ", coords: " << n_p_end->x << "," << n_p_end->y << ")" << std::endl;

        line_specific_mesh_node_ids.insert(n_p_start->id);
        line_specific_mesh_node_ids.insert(n_p_end->id);

        for (const auto& node : mesh_internal_ptr->nodes) { // 寻找线段上的中间节点
            double dist_start_node = std::sqrt(std::pow(node.x - n_p_start->x, 2) + std::pow(node.y - n_p_start->y, 2));
            double dist_end_node = std::sqrt(std::pow(node.x - n_p_end->x, 2) + std::pow(node.y - n_p_end->y, 2));
            double dist_start_end = std::sqrt(std::pow(n_p_start->x - n_p_end->x, 2) + std::pow(n_p_start->y - n_p_end->y, 2));
            if (std::abs(dist_start_end) < 1e-9) {
                 if (std::abs(dist_start_node) < 1e-9) {
                    line_specific_mesh_node_ids.insert(node.id);
                 }
            } else if (std::abs(dist_start_node + dist_end_node - dist_start_end) < 1e-6 * dist_start_end) {
                line_specific_mesh_node_ids.insert(node.id);
            }
        }
    }
    std::cout << "  Line '" << line_name << "': Identified " << line_specific_mesh_node_ids.size() << " mesh nodes on the polyline." << std::endl;

    std::set<std::pair<int, int>> processed_physical_edges_for_this_line; // 避免重复处理这条线的物理边

    for (const auto& he : mesh_internal_ptr->half_edges) { // 遍历所有半边
        if (he.twin_half_edge_id == -1) continue; // 只关心内部边

        const Node_cpp* n1 = mesh_internal_ptr->get_node_by_id(he.origin_node_id);
        const HalfEdge_cpp* next_he_ptr = mesh_internal_ptr->get_half_edge_by_id(he.next_half_edge_id);
        if (!n1 || !next_he_ptr) continue;
        const Node_cpp* n2 = mesh_internal_ptr->get_node_by_id(next_he_ptr->origin_node_id);
        if (!n2) continue;

        bool n1_on_this_line = line_specific_mesh_node_ids.count(n1->id); // 检查节点是否属于当前处理的这条线
        bool n2_on_this_line = line_specific_mesh_node_ids.count(n2->id);

        if (n1_on_this_line && n2_on_this_line) { // 这条边是当前内部流量线的一部分
            std::pair<int, int> phys_edge_key = (n1->id < n2->id) ? std::make_pair(n1->id, n2->id) : std::make_pair(n2->id, n1->id);
            if (processed_physical_edges_for_this_line.count(phys_edge_key)) {
                continue;
            }
            processed_physical_edges_for_this_line.insert(phys_edge_key);

            const HalfEdge_cpp* twin_he_ptr = mesh_internal_ptr->get_half_edge_by_id(he.twin_half_edge_id);
            if (!twin_he_ptr) continue;

            int cell_id1 = he.cell_id;
            int cell_id2 = twin_he_ptr->cell_id;
            double dot_product = he.normal[0] * direction_py[0] + he.normal[1] * direction_py[1];

            int source_cell_id_val;
            int sink_cell_id_val;
            const Cell_cpp* source_cell_ptr_val = nullptr;
            const Cell_cpp* sink_cell_ptr_val = nullptr;

            if (dot_product > 1e-3) {
                source_cell_id_val = cell_id2;
                sink_cell_id_val = cell_id1;
                source_cell_ptr_val = mesh_internal_ptr->get_cell_by_id(cell_id2);
                sink_cell_ptr_val = mesh_internal_ptr->get_cell_by_id(cell_id1);
            } else if (dot_product < -1e-3) {
                source_cell_id_val = cell_id1;
                sink_cell_id_val = cell_id2;
                source_cell_ptr_val = mesh_internal_ptr->get_cell_by_id(cell_id1);
                sink_cell_ptr_val = mesh_internal_ptr->get_cell_by_id(cell_id2);
            } else {
                std::cout << "  Line '" << line_name << "': Skipping edge (" << n1->id << "-" << n2->id << ") as flow direction is nearly perpendicular." << std::endl;
                continue;
            }

            if (source_cell_ptr_val && sink_cell_ptr_val) {
                all_internal_flow_edges_info_map_internal[line_name].push_back({ // 添加到对应名称的vector中
                    source_cell_id_val,
                    sink_cell_id_val,
                    source_cell_ptr_val,
                    sink_cell_ptr_val,
                    he.length
                });
                current_line_total_length += he.length;
                std::cout << "  Line '" << line_name << "': Found segment HE_ID=" << he.id << " (len=" << he.length
                          << "), Source: " << source_cell_id_val << ", Sink: " << sink_cell_id_val << std::endl;
            } else {
                std::cerr << "ERROR (setup_internal_flow_source line '" << line_name << "'): Could not get valid cell pointers for edge (HE_ID=" << he.id << ")" << std::endl;
            }
        }
    }

    internal_flow_line_total_lengths_map_internal[line_name] = current_line_total_length; // 存储这条线的总长度

    if (current_line_total_length < 1e-6) {
        std::cerr << "WARNING (setup_internal_flow_source line '" << line_name << "'): Total length of identified segments is zero. No flow will be effectively applied." << std::endl;
        // 即使长度为0，也保留时程和方向，只是单位长度流量会是inf/nan，需要在apply时处理
    }

    std::cout << "  Line '" << line_name << "': Identified " << all_internal_flow_edges_info_map_internal[line_name].size() << " segments for source/sink." << std::endl;
    std::cout << "  Line '" << line_name << "': Total length = " << current_line_total_length << std::endl;
}

void HydroModelCore_cpp::setup_internal_point_source_cpp(
    const std::string& name,
    const std::array<double, 2>& coordinates,
    const std::vector<TimeseriesPoint_cpp>& q_timeseries
) {
    if (!mesh_internal_ptr) { // 检查网格指针是否有效
        std::cerr << "ERROR (setup_internal_point_source_cpp): Mesh not initialized for point source '" << name << "'." << std::endl; // 打印错误
        return; // 返回
    }

    std::cout << "C++ HydroModelCore: Setting up internal point source '" << name << "' at coordinates ("
              << coordinates[0] << ", " << coordinates[1] << ")." << std::endl; // 打印设置信息
    std::cout << "  Received " << q_timeseries.size() << " timeseries points for Q." << std::endl; // 打印接收到的时程点数

    int target_cell_id = mesh_internal_ptr->find_cell_containing_point(coordinates[0], coordinates[1]); // 查找包含该坐标的单元

    if (target_cell_id == -1) { // 如果未找到单元
        std::cerr << "WARNING (setup_internal_point_source_cpp): Could not find a cell containing coordinates ("
                  << coordinates[0] << ", " << coordinates[1] << ") for point source '" << name << "'. This source will be inactive." << std::endl; // 打印警告
        // --- 新增的详细调试信息 ---
        std::cerr << "  Debugging find_cell_containing_point for (" << coordinates[0] << ", " << coordinates[1] << "):" << std::endl;
        double min_dist_sq_to_centroid = std::numeric_limits<double>::max();
        int closest_cell_id = -1;
        for(const auto& cell_debug : mesh_internal_ptr->cells){
            double dx_c = cell_debug.centroid[0] - coordinates[0];
            double dy_c = cell_debug.centroid[1] - coordinates[1];
            double dist_sq = dx_c * dx_c + dy_c * dy_c;
            if(dist_sq < min_dist_sq_to_centroid){
                min_dist_sq_to_centroid = dist_sq;
                closest_cell_id = cell_debug.id;
            }
        }
        if(closest_cell_id != -1){
            const Cell_cpp* closest_c_ptr = mesh_internal_ptr->get_cell_by_id(closest_cell_id);
            if(closest_c_ptr){
                std::cerr << "    Closest cell found by centroid distance is ID: " << closest_cell_id
                          << ", Centroid: (" << closest_c_ptr->centroid[0] << ", " << closest_c_ptr->centroid[1] << ")"
                          << ", Distance_sq: " << min_dist_sq_to_centroid << std::endl;
                if (closest_c_ptr->node_ids.size() == 3) {
                    const Node_cpp* n0_dbg = mesh_internal_ptr->get_node_by_id(closest_c_ptr->node_ids[0]);
                    const Node_cpp* n1_dbg = mesh_internal_ptr->get_node_by_id(closest_c_ptr->node_ids[1]);
                    const Node_cpp* n2_dbg = mesh_internal_ptr->get_node_by_id(closest_c_ptr->node_ids[2]);
                    if (n0_dbg && n1_dbg && n2_dbg) {
                        std::cerr << "      Closest cell nodes: "
                                  << "N0(" << n0_dbg->x << "," << n0_dbg->y << "), "
                                  << "N1(" << n1_dbg->x << "," << n1_dbg->y << "), "
                                  << "N2(" << n2_dbg->x << "," << n2_dbg->y << ")" << std::endl;
                    }
                }
            }
        } else {
            std::cerr << "    Could not find any closest cell (mesh might be empty or other issue)." << std::endl;
        }
        // --- 详细调试信息结束 ---
    } else { // 如果找到了单元
        std::cout << "  Point source '" << name << "' will be applied to cell ID: " << target_cell_id << std::endl; // 打印目标单元ID
    }

    PointSourceInfo_cpp ps_info; // 创建点源信息对象
    ps_info.name = name; // 设置名称
    ps_info.target_cell_id = target_cell_id; // 设置目标单元ID
    ps_info.q_timeseries = q_timeseries; // 设置流量时程

    // 如果时程数据非空，确保它是按时间排序的
    if (!ps_info.q_timeseries.empty()) { // 如果时程非空
        std::sort(ps_info.q_timeseries.begin(), ps_info.q_timeseries.end(),
                  [](const TimeseriesPoint_cpp& a, const TimeseriesPoint_cpp& b) { // lambda排序函数
                      return a.time < b.time; // 按时间升序排序
                  });
    }
    internal_point_sources_info_internal.push_back(ps_info); // 将点源信息添加到内部存储列表
}

StateVector HydroModelCore_cpp::_apply_point_sources_internal(const StateVector& U_input, double dt, double time_current) {
    PROFILE_FUNCTION();
    StateVector U_output = U_input; // 复制输入状态作为输出基础

    if (internal_point_sources_info_internal.empty()) { // 如果没有配置点源
        return U_output; // 直接返回
    }

    // 仅在需要时打印调试信息 (可以根据需要注释掉)
    // std::cout << "DEBUG _apply_point_sources_internal at t=" << time_current << ", dt=" << dt << std::endl;

    for (const auto& ps_info : internal_point_sources_info_internal) { // 遍历所有点源
        if (ps_info.target_cell_id == -1) { // 如果该点源没有有效的目标单元
            // std::cout << "  Skipping point source '" << ps_info.name << "' as target_cell_id is -1." << std::endl;
            continue; // 跳过此点源
        }

        // 从时程插值获取当前点源流量 Q_point (m^3/s)
        double current_Q_point = get_timeseries_value_internal(ps_info.q_timeseries, time_current); // 调用内部方法获取时程值
        if (std::isnan(current_Q_point)) { // 如果插值失败
            // std::cerr << "WARNING (_apply_point_sources_internal line '" << ps_info.name << "'): Could not get timeseries value for Q at t=" << time_current << ". Assuming Q=0." << std::endl;
            current_Q_point = 0.0; // 假设流量为0
        }

        const Cell_cpp* target_cell = mesh_internal_ptr->get_cell_by_id(ps_info.target_cell_id); // 获取目标单元指针
        if (!target_cell || target_cell->area < epsilon) { // 如果目标单元无效或面积过小
            // std::cerr << "WARNING (_apply_point_sources_internal): Invalid target cell or zero area for point source '" << ps_info.name << "' (cell_id: " << ps_info.target_cell_id << "). Skipping." << std::endl;
            continue; // 跳过
        }

        // 计算体积变化 (m^3)
        double delta_V_total = current_Q_point * dt; // 体积变化 = 流量 * 时间步长

        // 计算由于点源导致的水深变化 (dU_h)
        double dU_h_point_source = delta_V_total / target_cell->area; // 水深变化 = 体积变化 / 单元面积

        // 应用到目标单元的水深
        U_output[ps_info.target_cell_id][0] += dU_h_point_source; // 更新水深守恒量

        // (仅质量源，不修改动量 U[1], U[2])

        // 调试打印 (可选)点源流量，成功运行已经屏蔽
        // if (std::abs(current_Q_point) > epsilon) { // 仅当流量不为零时打印
        //     std::cout << "  PointSource '" << ps_info.name << "' (Cell " << ps_info.target_cell_id
        //               << "): Q=" << current_Q_point << " m^3/s, dH=" << dU_h_point_source << " m (Area=" << target_cell->area << ")" << std::endl;
        // }
    }
    return U_output; // 返回修改后的状态向量
}

StateVector HydroModelCore_cpp::_apply_internal_flow_source_terms(const StateVector& U_input, double dt, double time_current) {
    PROFILE_FUNCTION();
    StateVector U_output = U_input; // 复制输入状态

    if (all_internal_flow_edges_info_map_internal.empty()) { // 如果没有任何已定义的内部流量线
        return U_output;
    }


    // 遍历所有已定义的内部流量线 (通过map的键，即line_name)
    for (auto const& [line_name, edge_info_list_for_line] : all_internal_flow_edges_info_map_internal) {
        if (edge_info_list_for_line.empty()) { // 如果这条线没有识别出任何边
            // std::cout << "  Skipping internal flow line '" << line_name << "' as it has no associated edges." << std::endl;
            continue;
        }

        // 1. 获取这条线的时程数据、总长度和方向
        auto it_ts = internal_q_timeseries_data_map_internal.find(line_name);
        auto it_len = internal_flow_line_total_lengths_map_internal.find(line_name);
        auto it_dir = internal_flow_directions_map_internal.find(line_name);

        if (it_ts == internal_q_timeseries_data_map_internal.end() ||
            it_len == internal_flow_line_total_lengths_map_internal.end() ||
            it_dir == internal_flow_directions_map_internal.end()) {
            std::cerr << "ERROR (_apply_internal_flow_source_terms): Missing data (timeseries, length, or direction) for line '" << line_name << "'. Skipping." << std::endl;
            continue;
        }

        const std::vector<TimeseriesPoint_cpp>& q_timeseries = it_ts->second;
        double total_length_for_line = it_len->second;
        const std::array<double, 2>& direction_for_line = it_dir->second;

        // 2. 从时程插值获取当前总流量 Q_total
        double current_target_Q_total = get_timeseries_value_internal(q_timeseries, time_current);
        if (std::isnan(current_target_Q_total)) { // 如果插值失败 (例如时间超出范围且未外插)
            // std::cerr << "WARNING (_apply_internal_flow_source_terms line '" << line_name << "'): Could not get timeseries value for Q at t=" << time_current << ". Assuming Q=0." << std::endl;
            current_target_Q_total = 0.0; // 或者其他默认处理
        }

        // 3. 计算单位长度流量
        double current_Q_per_unit_length = 0.0;
        if (total_length_for_line > epsilon) { // 避免除以零
            current_Q_per_unit_length = current_target_Q_total / total_length_for_line;
        } else if (std::abs(current_target_Q_total) > epsilon) { // 长度为0但流量不为0，警告
            std::cerr << "WARNING (_apply_internal_flow_source_terms line '" << line_name << "'): Total length is zero but target Q is " << current_target_Q_total << ". Cannot apply flow." << std::endl;
        }




        // 4. 遍历这条线上的所有边并施加源项
        for (const auto& edge_info : edge_info_list_for_line) {
            double edge_length = edge_info.edge_length;
            double Q_edge_total = current_Q_per_unit_length * edge_length; // 这条边承担的总流量

            int source_cell_id = edge_info.source_cell_id;
            int sink_cell_id = edge_info.sink_cell_id;
            const Cell_cpp* source_cell_ptr = edge_info.source_cell_ptr;
            const Cell_cpp* sink_cell_ptr = edge_info.sink_cell_ptr;

            if (!source_cell_ptr || !sink_cell_ptr) { // 应该在setup时就保证了
                continue;
            }

            // (这部分打印可以根据需要保留或移除)
            // std::cout << "    Applying to edge (len=" << edge_length << "): Q_edge=" << Q_edge_total
            //           << ". SourceCell=" << source_cell_id << ", SinkCell=" << sink_cell_id << std::endl;

            double delta_V_total = Q_edge_total * dt;

            double dU_h_source = (source_cell_ptr->area > epsilon) ? (delta_V_total / source_cell_ptr->area) : 0.0;
            double dU_h_sink   = (sink_cell_ptr->area > epsilon) ? (delta_V_total / sink_cell_ptr->area) : 0.0;

            double dU_hu_source = (source_cell_ptr->area > epsilon) ? (delta_V_total * direction_for_line[0] / source_cell_ptr->area) : 0.0;
            double dU_hv_source = (source_cell_ptr->area > epsilon) ? (delta_V_total * direction_for_line[1] / source_cell_ptr->area) : 0.0;

            double dU_hu_sink = (sink_cell_ptr->area > epsilon) ? (delta_V_total * direction_for_line[0] / sink_cell_ptr->area) : 0.0;
            double dU_hv_sink = (sink_cell_ptr->area > epsilon) ? (delta_V_total * direction_for_line[1] / sink_cell_ptr->area) : 0.0;

            U_output[source_cell_id][0] += dU_h_source;
            U_output[source_cell_id][1] += dU_hu_source;
            U_output[source_cell_id][2] += dU_hv_source;

            U_output[sink_cell_id][0] -= dU_h_sink;
            U_output[sink_cell_id][1] -= dU_hu_sink;
            U_output[sink_cell_id][2] -= dU_hv_sink;

            // (这部分打印也可以根据需要保留或移除)
            // std::cout << "      SourceCell " << source_cell_id << ": dU_h=" << dU_h_source << ", dU_hu=" << dU_hu_source << ", dU_hv=" << dU_hv_source << std::endl;
            // std::cout << "      SinkCell " << sink_cell_id << ": dU_h=" << -dU_h_sink << ", dU_hu=" << -dU_hu_sink << ", dU_hv=" << -dU_hv_sink << std::endl;
        }
    }
    return U_output;
}

double HydroModelCore_cpp::get_timeseries_value_internal(const std::vector<TimeseriesPoint_cpp>& series, double time_current) const {
    if (series.empty()) { // 如果时间序列为空
        // std::cerr << "Warning (get_timeseries_value_internal): Timeseries is empty." << std::endl; // 打印警告
        return std::numeric_limits<double>::quiet_NaN(); // 返回NaN
    }

    // 假设系列已按时间排序 (Python端应确保)
    auto it_upper = std::lower_bound(series.begin(), series.end(), time_current,
                                     [](const TimeseriesPoint_cpp& p, double val) {
                                         return p.time < val;
                                     });

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
}
void HydroModelCore_cpp::set_initial_conditions_cpp(const StateVector& U_initial) { // 设置初始条件(C++)实现
    if (!model_fully_initialized_flag) { // 如果模型未完全初始化
        throw std::runtime_error("Model core not initialized before setting initial conditions."); // 抛出运行时错误
    }
    if (!mesh_internal_ptr) { throw std::runtime_error("Mesh not loaded for initial conditions."); } // 如果网格未加载
    if (U_initial.size() != mesh_internal_ptr->cells.size()) { // 如果初始条件大小与单元数量不符
        throw std::invalid_argument("Initial conditions size mismatch with number of cells."); // 抛出无效参数异常
    }
    U_state_all_internal = U_initial; // 设置内部守恒量 (h, hu, hv)

    // VFRCalculator的检查是好的，但对于初始eta的计算，它不是必须的
    // if (!vfr_calculator_ptr) {
    //     throw std::runtime_error("VFRCalculator not initialized, cannot compute initial eta in C++.");
    // }

    eta_previous_internal.resize(mesh_internal_ptr->cells.size()); // 调整eta数组大小

    for (size_t i = 0; i < mesh_internal_ptr->cells.size(); ++i) { // 遍历所有单元
        const Cell_cpp& cell = mesh_internal_ptr->cells[i]; // 获取当前单元
        // 直接使用 h_cell + z_bed_centroid 作为初始的 eta_previous_internal。
        // U_state_all_internal[i][0] 就是从Python传递过来的，已经根据配置计算好的水深h。
        eta_previous_internal[i] = U_state_all_internal[i][0] + cell.z_bed_centroid; // 水深+单元形心底高程

        // 对于干单元的特殊处理 (如果初始水深为0或非常小，则eta等于底高程中较低的一个值)
        if (U_state_all_internal[i][0] < min_depth_internal / 10.0) { // 如果单元初始为干
            double min_b_cell_node = cell.z_bed_centroid; // 默认为形心底高程
            bool node_found = false; // 标记是否找到节点
            if (!cell.node_ids.empty()) { // 如果单元有节点
                 min_b_cell_node = std::numeric_limits<double>::max(); // 初始化为最大值
                 for(int node_id : cell.node_ids) { // 遍历单元的节点ID
                    const Node_cpp* node = mesh_internal_ptr->get_node_by_id(node_id); // 获取节点对象
                    if(node) { // 如果节点有效
                        min_b_cell_node = std::min(min_b_cell_node, node->z_bed); // 更新最低点高程
                        node_found = true; // 标记找到节点
                    }
                 }
                 if (!node_found) min_b_cell_node = cell.z_bed_centroid; // 如果遍历后仍未找到有效节点（不太可能），则用形心
            }
            eta_previous_internal[i] = min_b_cell_node; // 设置为节点最低点高程或单元形心底高程
        }
    }

    initial_conditions_set_flag = true; // 标记初始条件已设置
    std::cout << "C++ HydroModelCore_cpp: Initial U set. Initial eta_previous_internal directly calculated from U and z_bed_centroid." << std::endl; // 打印信息
} // 结束函数

void HydroModelCore_cpp::setup_boundary_conditions_cpp( // 设置边界条件(C++)实现
    const std::map<int, BoundaryDefinition_cpp>& bc_definitions,
    const std::map<int, std::vector<TimeseriesPoint_cpp>>& waterlevel_ts_data,
    const std::map<int, std::vector<TimeseriesPoint_cpp>>& discharge_ts_data
) {
    if (!model_fully_initialized_flag || !boundary_handler_ptr) { // 如果模型未完全初始化或边界处理器无效
        throw std::runtime_error("Model or BoundaryConditionHandler not initialized before setting boundary conditions."); // 抛出运行时错误
    }
    boundary_handler_ptr->set_boundary_definitions(bc_definitions); // 设置边界定义
    boundary_handler_ptr->set_waterlevel_timeseries_data(waterlevel_ts_data); // 设置水位时间序列数据
    boundary_handler_ptr->set_discharge_timeseries_data(discharge_ts_data); // 设置流量时间序列数据
    boundary_conditions_set_flag = true; // 标记边界条件已设置
    std::cout << "C++ HydroModelCore_cpp: Boundary conditions configured in handler." << std::endl; // 打印信息
} // 结束函数


StateVector HydroModelCore_cpp::_calculate_rhs_explicit_part_internal(const StateVector& U_current_rhs, double time_current_rhs) {
    PROFILE_FUNCTION(); // 自动使用函数名 "_calculate_rhs_explicit_part_internal"
    if (!mesh_internal_ptr || U_current_rhs.size() != mesh_internal_ptr->cells.size()) {
        throw std::runtime_error("Invalid mesh or U_current_rhs size in RHS calculation.");
    }
    if (!reconstruction_ptr || !flux_calculator_ptr || !boundary_handler_ptr) {
        throw std::runtime_error("RHS calculation cannot proceed: core components not initialized.");
    }

    StateVector RHS_vector(mesh_internal_ptr->cells.size(), {0.0, 0.0, 0.0});

    bool gradients_available = false;
    if (reconstruction_ptr->get_scheme_type() != ReconstructionScheme_cpp::FIRST_ORDER) {
        PROFILE_SCOPE("RHS_Reconstruction_Prepare"); // 计时重构准备步骤
        try {
            reconstruction_ptr->prepare_for_step(U_current_rhs);
            gradients_available = true;
        } catch (const std::exception& e) {
            std::cerr << "Error during reconstruction prepare_for_step: " << e.what() << std::endl;
            throw;
        }
    }

    // --- 定义我们关心的单元ID和时间范围 ---
    const int problem_cell_id_1 = 1385; // 从ParaView中找到的ID
    const int problem_cell_id_2 = 1419;   // 从ParaView中找到的ID
    const double target_time_min_debug = 4.99; // 目标时间范围开始
    const double target_time_max_debug = 5.01; // 目标时间范围结束
    const double min_eta_to_debug = 0.3; // 新增：只有当这些问题单元的水位已经比较高时才开始详细打印HLLC
    // --- 定义结束 ---

    bool time_is_in_debug_range_rhs = (time_current_rhs >= target_time_min_debug && time_current_rhs <= target_time_max_debug);

    // --- 为整个边循环创建一个总的计时作用域 ---
    {
        PROFILE_SCOPE("RHS_EdgeLoop_Total"); // 计时所有边的处理

        for (const auto& he : mesh_internal_ptr->half_edges) {
            // ... (获取 cell_L_ptr) ...
            const Cell_cpp* cell_L_ptr = mesh_internal_ptr->get_cell_by_id(he.cell_id);
            if (!cell_L_ptr) continue; // Should not happen if mesh is consistent

            // 1. 获取左单元中心状态
            const auto& U_L_center_cons = U_current_rhs[cell_L_ptr->id];
            PrimitiveVars_cpp W_L_center = reconstruction_ptr->conserved_to_primitive(U_L_center_cons); // 使用Reconstruction的转换函数
            double Z_L_centroid = cell_L_ptr->z_bed_centroid;
            double eta_L_center = U_L_center_cons[0] + Z_L_centroid; // h + z_bed

            PrimitiveVars_cpp W_L_flux, W_R_flux; // 用于HLLC的最终界面状态
            std::array<double, 3> numerical_flux_cartesian;
            std::array<double, 3> source_term_L_interface = {0.0, 0.0, 0.0}; // 初始化界面源项对L的贡献
            std::array<double, 3> source_term_R_interface = {0.0, 0.0, 0.0}; // 初始化界面源项对R的贡献

            if (he.twin_half_edge_id != -1) { // --- 处理内部边 ---
                // --- 添加调试打印：边界边RHS更新 ---
                bool is_target_discharge_boundary = (he.boundary_marker == 10); // 假设10是你的流量边界标记
                if (is_target_discharge_boundary && time_current_rhs < 0.1) { // 只在早期且为目标边界时打印
                    std::cout << "[DEBUG_RHS_UPDATE_BND] Time: " << time_current_rhs
                              << ", Boundary HE_ID: " << he.id
                              << ", Cell_L_ID: " << cell_L_ptr->id
                              << ", Marker: " << he.boundary_marker << std::endl;
                }
                // --- 调试打印结束 ---
                if (static_cast<unsigned int>(he.id) >= static_cast<unsigned int>(he.twin_half_edge_id)) { continue; } // 避免重复

                const HalfEdge_cpp* he_twin_ptr = mesh_internal_ptr->get_half_edge_by_id(he.twin_half_edge_id);
                const Cell_cpp* cell_R_ptr = mesh_internal_ptr->get_cell_by_id(he_twin_ptr->cell_id);
                // ... (检查 cell_R_ptr有效性) ...

                // 2. 获取右单元中心状态
                const auto& U_R_center_cons = U_current_rhs[cell_R_ptr->id];
                PrimitiveVars_cpp W_R_center = reconstruction_ptr->conserved_to_primitive(U_R_center_cons);
                double Z_R_centroid = cell_R_ptr->z_bed_centroid;
                double eta_R_center = U_R_center_cons[0] + Z_R_centroid;

                // 3. 定义界面底高程 (Audusse et al. 2005, eq. 4.2 - Z_ij = max(Z_i, Z_j))
                double Z_interface = std::max(Z_L_centroid, Z_R_centroid);
                // 或者使用你原来的 z_face (边中点节点的底高程平均)
                // double Z_interface = z_face; // (你需要确保z_face在这里已计算)
                // 为了与文献方法一致，我们先用 max(Z_L, Z_R)

                // 4. 静水重构左右两侧的水深 (Audusse et al. 2005, eq. 4.3 - h*_ij = (eta_i - Z_ij)+ )
                // 注意，文献中的 h*_ij 是指从单元i看界面ij的值。我们的he从L到R。
                // h_L_star 是从 L 看 he 的值，h_R_star 是从 R 看 he_twin 的值 (即从R看he的值)
                double h_L_star = std::max(0.0, eta_L_center - Z_interface);
                double h_R_star = std::max(0.0, eta_R_center - Z_interface);

                W_L_flux.h = h_L_star;
                W_L_flux.u = W_L_center.u; // 一阶：使用单元中心速度
                W_L_flux.v = W_L_center.v;
                if (h_L_star < min_depth_internal * 1.1) { // 阈值可以调整
                    W_L_flux.u = 0.0; W_L_flux.v = 0.0;
                }

                W_R_flux.h = h_R_star;
                W_R_flux.u = W_R_center.u; // 一阶：使用单元中心速度
                W_R_flux.v = W_R_center.v;
                if (h_R_star < min_depth_internal * 1.1) {
                    W_R_flux.u = 0.0; W_R_flux.v = 0.0;
                }

                // 5. 计算数值通量

                numerical_flux_cartesian = flux_calculator_ptr->calculate_hllc_flux(W_L_flux, W_R_flux, he.normal);


                // 6. 计算界面源项 (Audusse et al. 2005, eq. 4.4)
                // S(U_i, U*_ij, n_ij) = (0, g/2 * ( (h*_ij)^2 - h_i^2 ) * n_ij)
                // 对于单元 L，它"失去" S(U_L, U*_L_to_R, he.normal)
                // U*_L_to_R 是指 (h_L_star, h_L_star*u_L_center, h_L_star*v_L_center)
                // 但源项只作用于动量，且只与水深有关
                source_term_L_interface[1] = 0.5 * gravity_internal * (h_L_star * h_L_star - U_L_center_cons[0] * U_L_center_cons[0]) * he.normal[0];
                source_term_L_interface[2] = 0.5 * gravity_internal * (h_L_star * h_L_star - U_L_center_cons[0] * U_L_center_cons[0]) * he.normal[1];

                // 对于单元 R，它"得到" S(U_R, U*_R_to_L, he_twin.normal)
                // U*_R_to_L 是指 (h_R_star, h_R_star*u_R_center, h_R_star*v_R_center)
                // he_twin.normal = -he.normal
                source_term_R_interface[1] = 0.5 * gravity_internal * (h_R_star * h_R_star - U_R_center_cons[0] * U_R_center_cons[0]) * (-he.normal[0]);
                source_term_R_interface[2] = 0.5 * gravity_internal * (h_R_star * h_R_star - U_R_center_cons[0] * U_R_center_cons[0]) * (-he.normal[1]);

                // 打印更新RHS前的状态
                if (is_target_discharge_boundary && time_current_rhs < 0.1) {
                    std::cout << "  Flux from BC Handler (Cartesian): Fh=" << numerical_flux_cartesian[0]
                              << ", Fhu=" << numerical_flux_cartesian[1] << ", Fhv=" << numerical_flux_cartesian[2] << std::endl;
                    std::cout << "  Source_L_Interface (Cartesian, if any): S_hu=" << source_term_L_interface[1] // 假设你这里计算了
                              << ", S_hv=" << source_term_L_interface[2] << std::endl;
                    std::cout << "  RHS for Cell " << cell_L_ptr->id << " BEFORE [h,hu,hv]: ("
                              << RHS_vector[cell_L_ptr->id][0] << ","
                              << RHS_vector[cell_L_ptr->id][1] << ","
                              << RHS_vector[cell_L_ptr->id][2] << ")" << std::endl;
                }
                // 7. 更新RHS (通量贡献 + 界面源项贡献)
                // (符号约定：F指向外部为负，S指向内部为正，或者F和S都按流出单元为负)
                // Audusse (4.5): dU_i/dt = ... - F_ij + S_ij ...
                // 如果 F_ij 是流出 i 的通量，S_ij 是作用在 i 上通过界面 ij 的源项
                for (int k = 0; k < 3; ++k) {
                    double flux_val_times_length = numerical_flux_cartesian[k] * he.length;
                    double source_L_val_times_length = source_term_L_interface[k] * he.length; // 这是作用在L上的源项
                    double source_R_val_times_length = source_term_R_interface[k] * he.length; // 这是作用在R上的源项

                    if (cell_L_ptr->area > epsilon) {
                        RHS_vector[cell_L_ptr->id][k] -= flux_val_times_length / cell_L_ptr->area; // 通量流出L
                        RHS_vector[cell_L_ptr->id][k] += source_L_val_times_length / cell_L_ptr->area; // 源项作用于L
                    }
                    if (cell_R_ptr->area > epsilon) {
                        RHS_vector[cell_R_ptr->id][k] += flux_val_times_length / cell_R_ptr->area; // 通量流入R
                        RHS_vector[cell_R_ptr->id][k] += source_R_val_times_length / cell_R_ptr->area; // 源项作用于R
                    }
                }
                // 打印更新RHS后的状态
                if (is_target_discharge_boundary && time_current_rhs < 0.1) {
                    std::cout << "  RHS for Cell " << cell_L_ptr->id << " AFTER [h,hu,hv]: ("
                              << RHS_vector[cell_L_ptr->id][0] << ","
                              << RHS_vector[cell_L_ptr->id][1] << ","
                              << RHS_vector[cell_L_ptr->id][2] << ")" << std::endl;
                }

            } else { // --- 处理边界边 ---

                // 1. 界面底高程 (对于边界，可以认为外部单元底高程与内部单元相同，或使用实际边界高程)
                // 简单起见，Z_interface = Z_L_centroid; (这使得 h_L_star = h_L_center)
                // 或者，如果边界上有精确的 z_face: Z_interface = z_face; (你需要先计算z_face)
                // 我们这里先用 Z_L_centroid，这意味着边界上的静水重构水深等于单元中心水深
                double Z_interface_bnd = Z_L_centroid; // 或 z_face

                // 2. 静水重构左侧状态
                double h_L_star_bnd = std::max(0.0, eta_L_center - Z_interface_bnd);
                W_L_flux.h = h_L_star_bnd;
                W_L_flux.u = W_L_center.u;
                W_L_flux.v = W_L_center.v;
                if (h_L_star_bnd < min_depth_internal * 1.1) {
                    W_L_flux.u = 0.0; W_L_flux.v = 0.0;
                }

                // 3. 计算边界通量 (传递重构后的W_L_flux)
                // 你需要修改 boundary_handler_ptr->calculate_boundary_flux 的接口
                // 使其接受 PrimitiveVars_cpp W_L_reconstructed 作为输入，而不是U_state_all
                // 或者，让它内部自己做与这里类似的静水重构。
                // 为了保持一致性，最好是传递 W_L_flux。
                // 假设你已经修改了接口：
                // numerical_flux_cartesian = boundary_handler_ptr->calculate_boundary_flux_reconstructed(
                // W_L_flux, cell_L_ptr->id, he, time_current_rhs);

                // 如果暂时不改接口，它内部的逻辑是：
                // W_L_recons_iface 来自 reconstruction_ptr->get_reconstructed_interface_states(U_current_rhs, cell_L_ptr->id, -1, he, true)
                // eta_L_at_face (一阶时是 eta_L_center)
                // h_L_star = std::max(0.0, eta_L_at_face - z_face); (z_face是边界中点高程)
                // 这与我们上面用 Z_interface_bnd = z_face 得到的结果是一致的。
                // 所以，暂时可以不修改 BoundaryConditionHandler 的调用。
                numerical_flux_cartesian = boundary_handler_ptr->calculate_boundary_flux(
                        U_current_rhs, cell_L_ptr->id, he, time_current_rhs);


                // 4. 计算边界界面源项 (只作用于单元L)
                // S(U_L, U*_L_bnd, he.normal)
                // U*_L_bnd 是 (h_L_star_bnd, ...)
                source_term_L_interface[1] = 0.5 * gravity_internal * (h_L_star_bnd * h_L_star_bnd - U_L_center_cons[0] * U_L_center_cons[0]) * he.normal[0];
                source_term_L_interface[2] = 0.5 * gravity_internal * (h_L_star_bnd * h_L_star_bnd - U_L_center_cons[0] * U_L_center_cons[0]) * he.normal[1];

                // 5. 更新RHS
                for (int k = 0; k < 3; ++k) {
                    double flux_val_times_length = numerical_flux_cartesian[k] * he.length;
                    double source_L_val_times_length = source_term_L_interface[k] * he.length;
                    if (cell_L_ptr->area > epsilon) {
                        RHS_vector[cell_L_ptr->id][k] -= flux_val_times_length / cell_L_ptr->area;
                        RHS_vector[cell_L_ptr->id][k] += source_L_val_times_length / cell_L_ptr->area;
                    }
                }
            }
        }
    }

    // --- 移除原来的中心化底坡源项 ---
    // for (size_t i = 0; i < mesh_internal_ptr->cells.size(); ++i) {
    // const Cell_cpp& cell = mesh_internal_ptr->cells[i];
    // double h_center = U_current_rhs[i][0];
    // if (h_center >= min_depth_internal / 10.0) {
    // double gx = gravity_internal; // gx 应该是 g (9.81)
    // RHS_vector[i][1] += -gx * h_center * cell.b_slope_x; // 这是压力梯度项的一部分，现在通过界面源项处理了
    // RHS_vector[i][2] += -gx * h_center * cell.b_slope_y;
    // }
    // }
    return RHS_vector;
}

StateVector HydroModelCore_cpp::_apply_friction_semi_implicit_internal(const StateVector& U_input_friction, // 应用半隐式摩擦(内部)实现
                                                                    const StateVector& U_coeffs_friction, double dt_friction) {
    PROFILE_FUNCTION();
    if (!source_term_calculator_ptr) { // 如果源项计算器无效
        throw std::runtime_error("SourceTermCalculator not initialized for friction."); // 抛出运行时错误
    }
    if (manning_n_values_internal.size() != U_input_friction.size()) { // 如果曼宁值数量不匹配
        _initialize_manning_from_mesh_internal(); // 尝试重新初始化
        if (manning_n_values_internal.size() != U_input_friction.size()) { // 再次检查
             throw std::runtime_error("Manning values size mismatch for friction after re-init."); // 抛出运行时错误
        }
    }
    return source_term_calculator_ptr->apply_friction_semi_implicit_all_cells( // 调用源项计算器应用摩擦
        U_input_friction, U_coeffs_friction, dt_friction, manning_n_values_internal
    ); // 结束调用
} // 结束函数

double HydroModelCore_cpp::_calculate_dt_internal() {
    PROFILE_FUNCTION();
    if (!mesh_internal_ptr || U_state_all_internal.size() != mesh_internal_ptr->cells.size()) {
        return max_dt_internal;
    }
    double min_dt_inv_term = 0.0;
    int problematic_cell_id_for_dt = -1;
    double h_at_problem_cell = 0.0;
    double u_at_problem_cell = 0.0;
    double v_at_problem_cell = 0.0;
    double cfl_term_at_problem_cell = 0.0;

    // 定义一个阈值，比如 min_depth_internal 的5倍。低于此阈值的水深将受到特殊处理。
    const double shallow_water_threshold_for_dt = std::max(min_depth_internal * 50.0, 1e-4); // 新增：为dt计算定义一个浅水阈值
    // 定义在浅水区dt计算中允许的最大速度（如果不想完全置零）
    const double max_speed_in_shallow_for_dt = 0.1; // 例如0.1 m/s，非常小 // 新增：为dt计算在浅水区设置一个最大速度

    for (size_t i = 0; i < mesh_internal_ptr->cells.size(); ++i) {
        const Cell_cpp& cell = mesh_internal_ptr->cells[i];
        double h = U_state_all_internal[i][0];

        // 1. 如果水深严格小于 min_depth_internal，完全跳过 (保持之前的逻辑，但可以更早跳出)
        if (h < min_depth_internal) { // 如果水深小于最小水深，则跳过
            continue;
        }

        double u, v;
        // 2. 特殊处理浅水单元 (h < shallow_water_threshold_for_dt 但 h >= min_depth_internal)
        if (h < shallow_water_threshold_for_dt) { // 如果水深小于我们定义的浅水阈值
            // 对于非常浅的水，对流速度的贡献在dt计算中应该被限制或忽略
            // 以避免 hu/h 或 hv/h 产生虚假的大速度
            // 方案A: 直接将流速视为0，只考虑波速
            u = 0.0; // 强制u为0
            v = 0.0; // 强制v为0
            // 方案B: 限制速度大小 (如下所示，但方案A可能更简单直接)
            /*
            double h_div_robust = std::max(h, epsilon);
            u = U_state_all_internal[i][1] / h_div_robust;
            v = U_state_all_internal[i][2] / h_div_robust;
            double speed_sq = u*u + v*v;
            if (speed_sq > max_speed_in_shallow_for_dt * max_speed_in_shallow_for_dt) {
                double scale_factor = max_speed_in_shallow_for_dt / std::sqrt(speed_sq);
                u *= scale_factor;
                v *= scale_factor;
            }
            */
        } else { // 3. 对于水深足够的单元，正常计算流速
            double h_div = std::max(h, epsilon); // 使用一个安全的分母
            u = U_state_all_internal[i][1] / h_div; // 正常计算u
            v = U_state_all_internal[i][2] / h_div; // 正常计算v
        }

        // 即使是正常计算的u,v，如果h本身也比较大但产生了异常大的速度，也应该有个总的限制
        // (这部分可以保留，但上面的浅水处理应该更有效)
        double speed_sq_check = u * u + v * v; // 检查速度平方
        double general_max_speed_sq = 20.0 * 20.0; // 一般情况下的最大速度平方 (例如20m/s)
        if (speed_sq_check > general_max_speed_sq) { // 如果速度平方超过一般最大值
             // 这可能表明即使水深不小，hu或hv也异常大了，可能来自其他地方的数值问题
             // std::cout << "C++ DEBUG _calculate_dt: Cell " << i << " (h=" << h
             //           << ") has speed_sq=" << speed_sq_check << " > general_max_speed_sq. Clamping u,v for dt calc." << std::endl; // 调试信息：单元速度平方超过一般最大值，为dt计算限制u,v
             double scale = std::sqrt(general_max_speed_sq / speed_sq_check); // 计算缩放因子
             u *= scale; // 缩放u
             v *= scale; // 缩放v
        }


        double c_wave = std::sqrt(gravity_internal * h); // 波速计算不变
        double sum_lambda_L_over_area_cell = 0.0;

        for (int he_id : cell.half_edge_ids_list) {
            const HalfEdge_cpp* he = mesh_internal_ptr->get_half_edge_by_id(he_id);
            if (!he || he->length < epsilon) continue;
            double un = u * he->normal[0] + v * he->normal[1]; // 使用（可能）修正后的 u, v
            double lambda_max_edge = std::abs(un) + c_wave;
            sum_lambda_L_over_area_cell += lambda_max_edge * he->length;
        }

        if (cell.area > epsilon && sum_lambda_L_over_area_cell > epsilon) {
            double current_cell_inv_term = sum_lambda_L_over_area_cell / cell.area;
            if (current_cell_inv_term > min_dt_inv_term) {
                min_dt_inv_term = current_cell_inv_term;
                problematic_cell_id_for_dt = static_cast<int>(i);
                h_at_problem_cell = h;          // 记录的是原始的h
                u_at_problem_cell = u;          // 记录的是用于dt计算的u（可能被修正过）
                v_at_problem_cell = v;          // 记录的是用于dt计算的v（可能被修正过）
                cfl_term_at_problem_cell = current_cell_inv_term;
            }
        }
    }

    if (min_dt_inv_term < epsilon) {
        return max_dt_internal;
    }

    double calculated_dt = cfl_number_internal / min_dt_inv_term;

    // static int dt_calc_print_counter = 0;
    // if (problematic_cell_id_for_dt != -1 && dt_calc_print_counter % 100 == 0) { // 每100次dt计算打印一次
    //     const Cell_cpp& p_cell_info = mesh_internal_ptr->cells[problematic_cell_id_for_dt];
    //     double ph_orig_info = U_state_all_internal[problematic_cell_id_for_dt][0];
    //     double phu_orig_info = U_state_all_internal[problematic_cell_id_for_dt][1];
    //     double phv_orig_info = U_state_all_internal[problematic_cell_id_for_dt][2];
    //
    //     std::cout << std::fixed << std::setprecision(10); // 保证精度
    //     std::cout << "C++ DT_LIMITING_CELL_INFO: Time=" << current_time_internal
    //               << ", Target_dt_before_max_dt_clip=" << calculated_dt // 这是CFL计算出的dt
    //               << ", Problem_Cell_ID=" << problematic_cell_id_for_dt
    //               << ", Orig_h=" << ph_orig_info << ", Orig_hu=" << phu_orig_info << ", Orig_hv=" << phv_orig_info
    //               << ", h_for_dt_calc=" << h_at_problem_cell // 用于dt计算的h
    //               << ", u_eff_in_dt=" << u_at_problem_cell // 用于dt计算的u (可能修正后)
    //               << ", v_eff_in_dt=" << v_at_problem_cell // 用于dt计算的v (可能修正后)
    //               << ", cell_area=" << p_cell_info.area
    //               << ", cell_CFL_term_sum_lambda_L_over_A=" << cfl_term_at_problem_cell // (sum |un_i|+c_i * L_i) / Area_cell
    //               << std::endl;
    // }
    // dt_calc_print_counter++;

    return std::min(calculated_dt, max_dt_internal);
}


void HydroModelCore_cpp::_handle_dry_cells_and_update_eta_internal() { // 处理干单元并更新水位(内部)实现
    PROFILE_FUNCTION();


    if (!mesh_internal_ptr || !vfr_calculator_ptr || U_state_all_internal.size() != mesh_internal_ptr->cells.size()) {
        return;
    }
    if (eta_previous_internal.size() != mesh_internal_ptr->cells.size()) {
        eta_previous_internal.assign(mesh_internal_ptr->cells.size(), 0.0);
        std::cerr << "Warning: eta_previous_internal resized in _handle_dry_cells." << std::endl;
    }

    std::vector<double> eta_new(mesh_internal_ptr->cells.size());


    for (size_t i = 0; i < mesh_internal_ptr->cells.size(); ++i) {
        Cell_cpp& cell = mesh_internal_ptr->cells[i];
        std::array<double, 3>& U_cell = U_state_all_internal[i];


        const double momentum_zeroing_h_threshold = min_depth_internal * 10.0; // 例如，1e-6
        // 这个阈值可以根据情况调整，比如 5*min_depth 或 10*min_depth

        if (U_cell[0] < momentum_zeroing_h_threshold) { // 如果水深小于动量清零阈值
            U_cell[1] = 0.0; // 清零 x 方向动量
            U_cell[2] = 0.0; // 清零 y 方向动量

            // 如果水深甚至小于 min_depth_internal，则也将水深设为0
            if (U_cell[0] < min_depth_internal) {
                U_cell[0] = 0.0;
            }
            // 否则，保留 U_cell[0] 的值（它在 min_depth_internal 和 momentum_zeroing_h_threshold 之间）
            // 但其动量已经被清零。
        }


        double h_avg_non_negative_for_vfr = std::max(0.0, U_cell[0]); // 计算用于VFR的非负平均水深 // 新增：获取非负平均水深

        if (h_avg_non_negative_for_vfr >= min_depth_internal / 10.0) { // 湿单元或接近湿 // 修改：使用h_avg_non_negative_for_vfr判断
            std::vector<double> b_sorted_cell_vfr;
            std::vector<Node_cpp> nodes_sorted_cell_vfr;
            for(int node_id : cell.node_ids) {
                const Node_cpp* node = mesh_internal_ptr->get_node_by_id(node_id);
                if(node) nodes_sorted_cell_vfr.push_back(*node);
            }
            std::sort(nodes_sorted_cell_vfr.begin(), nodes_sorted_cell_vfr.end(),
                      [](const Node_cpp& a, const Node_cpp& b) { return a.z_bed < b.z_bed; });
            for(const auto& sorted_node : nodes_sorted_cell_vfr) {
                b_sorted_cell_vfr.push_back(sorted_node.z_bed);
            }


            if (b_sorted_cell_vfr.size() == 3) { // 确保是三角形
                 eta_new[i] = vfr_calculator_ptr->get_eta_from_h(
                    h_avg_non_negative_for_vfr, b_sorted_cell_vfr, nodes_sorted_cell_vfr, cell.area, eta_previous_internal[i], // 修改：传入h_avg_non_negative_for_vfr
                    this->current_time_internal,
                    static_cast<int>(i)
                 );
            } else { // 非三角形单元的备用逻辑
                 double base_elev_for_eta = b_sorted_cell_vfr.empty() ? cell.z_bed_centroid : b_sorted_cell_vfr[0]; // 获取基准高程
                 eta_new[i] = base_elev_for_eta + h_avg_non_negative_for_vfr; // eta = bed + h // 修改：使用h_avg_non_negative_for_vfr
            }

        } else { // 干单元
             double min_b_cell = std::numeric_limits<double>::max();
            bool found_node = false;
            for(int node_id : cell.node_ids) {
                const Node_cpp* node = mesh_internal_ptr->get_node_by_id(node_id);
                if(node) { min_b_cell = std::min(min_b_cell, node->z_bed); found_node = true; }
            }
            eta_new[i] = found_node ? min_b_cell : cell.z_bed_centroid;
        }
    }
    eta_previous_internal = eta_new;
}

bool HydroModelCore_cpp::advance_one_step() {
    PROFILE_FUNCTION(); // 计时整个 advance_one_step 函数

    if (!model_fully_initialized_flag || !initial_conditions_set_flag || !boundary_conditions_set_flag) {
        std::string error_msg = "Model not fully initialized/configured before calling advance_one_step. Flags: ";
        error_msg += "model_fully_initialized=" + std::string(model_fully_initialized_flag ? "true" : "false");
        error_msg += ", initial_conditions_set=" + std::string(initial_conditions_set_flag ? "true" : "false");
        error_msg += ", boundary_conditions_set=" + std::string(boundary_conditions_set_flag ? "true" : "false");
        throw std::runtime_error(error_msg);
    }
    if (!time_integrator_ptr || U_state_all_internal.empty()) {
         throw std::runtime_error("Time integrator or initial state not ready for step.");
    }

    double dt = 0.0;
    { // 为 dt 计算添加计时作用域
        PROFILE_SCOPE("CalculateDtInternal");
        dt = _calculate_dt_internal();
    }

    double actual_output_dt = output_dt_internal;
    // 确保如果 output_dt 设置为0或非常小，它不会干扰总时间判断
    if (actual_output_dt <= epsilon && total_time_internal > epsilon) {
         actual_output_dt = total_time_internal + 1.0; // 使其大于总时间，不触发中间输出对齐
    }

    // 调整dt以精确到达总时间或输出时间点
    if (current_time_internal + dt >= total_time_internal - epsilon) {
        dt = total_time_internal - current_time_internal;
        if (dt < epsilon / 10.0) dt = 0; // 如果剩余时间非常小，则认为已到达
    } else if (actual_output_dt > epsilon) { // 仅当有意义的输出间隔时才尝试对齐
        // 计算下一个理想的输出时间点
        double num_outputs_passed = std::floor(current_time_internal / actual_output_dt + epsilon);
        double next_ideal_output_time = (num_outputs_passed + 1.0) * actual_output_dt;

        if (current_time_internal + dt >= next_ideal_output_time - epsilon) { // 如果当前步会跨过下一个输出点
            dt = next_ideal_output_time - current_time_internal; // 调整dt以精确到达
             if (dt < epsilon / 10.0 && next_ideal_output_time < total_time_internal - epsilon) { // 如果调整后的dt太小，且还没到总时间
                // 尝试跳到下下个输出点，以避免dt过小
                // （这个逻辑可能需要根据具体需求调整，或者接受小的dt）
                // next_ideal_output_time = (num_outputs_passed + 2.0) * actual_output_dt;
                // dt = next_ideal_output_time - current_time_internal;
             }
             // 再次检查调整后的dt是否会超出总时间
             if (current_time_internal + dt >= total_time_internal - epsilon) {
                 dt = total_time_internal - current_time_internal;
                 if (dt < epsilon / 10.0) dt = 0;
             }
        }
    }

    // 如果计算出的 dt 仍然非常小，但模拟尚未结束，这可能表示卡顿
    if (dt < epsilon / 10.0 && current_time_internal < total_time_internal - epsilon) {
        if (dt < 0) dt = 0; // 确保dt非负
        // 卡顿检测：如果dt变得极小，但模拟还没结束，强制结束或警告
        if (this->last_calculated_dt_internal < epsilon / 100.0 && dt < epsilon / 100.0) { // 连续两次dt极小
            std::cerr << "C++ WARNING advance_one_step: Calculated dt (" << dt
                      << ") and previous dt (" << this->last_calculated_dt_internal
                      << ") are effectively zero at t = " << current_time_internal
                      << "s, but total_time is " << total_time_internal
                      << "s. Simulation might be stalled. Forcing current_time to total_time to end." << std::endl;
            current_time_internal = total_time_internal; // 强制结束
            dt = 0; // 确保不前进
        }
    }

    this->last_calculated_dt_internal = dt;

    if (this->last_calculated_dt_internal > epsilon / 100.0) { // 仅当dt足够大时才执行步骤
        StateVector U_after_explicit_step;
        {
            PROFILE_SCOPE("TimeIntegrator_Step");
            U_after_explicit_step = time_integrator_ptr->step(U_state_all_internal, this->last_calculated_dt_internal, current_time_internal);
        }

        StateVector U_after_sources = U_after_explicit_step;
        {
            PROFILE_SCOPE("InternalFlowSourceTerms");
            U_after_sources = _apply_internal_flow_source_terms(U_after_sources, this->last_calculated_dt_internal, current_time_internal);
        }
        {
            PROFILE_SCOPE("InternalPointSourceTerms");
             U_after_sources = _apply_point_sources_internal(U_after_sources, this->last_calculated_dt_internal, current_time_internal);
        }
        U_state_all_internal = U_after_sources;

        {
            PROFILE_SCOPE("HandleDryCellsAndUpdateEta");
            _handle_dry_cells_and_update_eta_internal();
        }
    } else if (current_time_internal < total_time_internal - epsilon) {
         // 如果dt太小但模拟没结束，仅打印警告，但不前进时间或步数
         // std::cout << "C++ advance_one_step: dt (" << this->last_calculated_dt_internal
         //           << ") is too small to advance at t = " << current_time_internal << ". Step not taken." << std::endl;
    }


    // 物理合理性检查 (这部分可以根据需要开启或关闭，或者移到VFR之后)
    const double MAX_PHYSICAL_H = 1000.0;
    const double MAX_PHYSICAL_SPEED = 50.0;
    for (size_t i = 0; i < U_state_all_internal.size(); ++i) {
        bool corrected = false;
        if (U_state_all_internal[i][0] > MAX_PHYSICAL_H) {
            // std::cerr << "WARNING (Post-Step Check): Cell " << i << " h=" << U_state_all_internal[i][0] << " > MAX_H. Clamping." << std::endl;
            U_state_all_internal[i][0] = MAX_PHYSICAL_H; U_state_all_internal[i][1] = 0.0; U_state_all_internal[i][2] = 0.0; corrected = true;
        }
        if (U_state_all_internal[i][0] < 0.0 && std::abs(U_state_all_internal[i][0]) > epsilon) {
             // std::cerr << "WARNING (Post-Step Check): Cell " << i << " h=" << U_state_all_internal[i][0] << " < 0. Setting to 0." << std::endl;
             U_state_all_internal[i][0] = 0.0; U_state_all_internal[i][1] = 0.0; U_state_all_internal[i][2] = 0.0; corrected = true;
        }
        if (U_state_all_internal[i][0] > min_depth_internal) {
            double h_chk = U_state_all_internal[i][0]; double u_chk = U_state_all_internal[i][1]/h_chk; double v_chk = U_state_all_internal[i][2]/h_chk;
            double speed_sq_chk = u_chk*u_chk + v_chk*v_chk;
            if (speed_sq_chk > MAX_PHYSICAL_SPEED * MAX_PHYSICAL_SPEED) {
                // if (!corrected) std::cerr << "WARNING (Post-Step Check): Cell " << i << " speed > MAX_SPEED. Clamping." << std::endl;
                double scale = MAX_PHYSICAL_SPEED / std::sqrt(speed_sq_chk);
                U_state_all_internal[i][1] *= scale; U_state_all_internal[i][2] *= scale;
            }
        } else if (U_state_all_internal[i][0] < min_depth_internal) {
             if (std::abs(U_state_all_internal[i][1]) > epsilon || std::abs(U_state_all_internal[i][2]) > epsilon) {
                 U_state_all_internal[i][1] = 0.0; U_state_all_internal[i][2] = 0.0;
             }
        }
    }

    // 仅当dt有效时才更新时间和步数
    if (this->last_calculated_dt_internal > epsilon / 100.0) {
        current_time_internal += this->last_calculated_dt_internal;
        step_count_internal++;
    }

    bool simulation_should_continue = !is_simulation_finished();

    if (!simulation_should_continue) { // 如果模拟在本步之后结束了
        // 检查是否已经打印过总结，避免重复打印
        // (如果 advance_one_step 可能在结束后被意外多调用一次)
        // 可以用一个成员变量标记，或者更简单地，依赖 Profiler::reset_summary()
        #ifdef ENABLE_PROFILING
            // Profiler::results 是 Profiler 命名空间下的全局变量
            if (!Profiler::results.empty()) { // 仅当有分析结果时才打印和重置
                std::cout << "\nINFO: Simulation finished based on is_simulation_finished() in advance_one_step." << std::endl;
                std::cout << "Final time: " << current_time_internal << ", Total steps: " << step_count_internal << std::endl;
                Profiler::print_summary();
                Profiler::reset_summary(); // 重置以便下次Python脚本运行（如果适用）或单元测试
            }
        #else
            // 如果分析未启用，也打印一个结束信息
            std::cout << "\nINFO: Simulation finished based on is_simulation_finished() in advance_one_step (profiling disabled)." << std::endl;
            std::cout << "Final time: " << current_time_internal << ", Total steps: " << step_count_internal << std::endl;
        #endif
    }
    return simulation_should_continue; // 返回模拟是否应该继续
}

void HydroModelCore_cpp::run_simulation_to_end() {
    PROFILE_FUNCTION(); // 记录整个模拟循环的总时间
    std::cout << "C++ Simulation starting (via run_simulation_to_end)..." << std::endl;
    bool should_continue = true;
    while(should_continue) {
        should_continue = advance_one_step(); // advance_one_step 内部会在结束时打印
        if (step_count_internal % 100 == 0 && should_continue) { // 每100步打印进度 (且模拟未结束)
            std::cout << "  Progress (run_simulation_to_end): Step " << step_count_internal
                      << ", Time: " << current_time_internal
                      << ", dt: " << get_last_dt() << std::endl;
        }
    }
    // advance_one_step 应该已经在最后一次调用时打印了总结
    // 因此这里不需要再次调用 Profiler::print_summary()
    // 只需打印一个总的完成信息
    std::cout << "C++ Simulation finished (via run_simulation_to_end) call. Final state reached at time "
              << current_time_internal << " after " << step_count_internal << " steps." << std::endl;
} // 结束函数

} // namespace HydroCore