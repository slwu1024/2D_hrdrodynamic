// src_cpp/src/HydroModelCore_cpp.cpp
// --- 新增：编译时检查 ---
#ifdef _OPENMP
#include <omp.h>
#else
    // 如果 CMakeLists.txt 正确配置，这里不应该被触发。
    // 如果触发了，说明编译器没有收到OpenMP标志，直接报错可以防止生成错误的单线程版本。
    #error "OpenMP is not enabled. Please check compiler flags (e.g., -fopenmp or /openmp)."
#endif

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

// --- 新增：实现 set_num_threads 方法 ---
void HydroModelCore_cpp::set_num_threads(int num_threads) {
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
        // 使用 omp_get_max_threads() 来获取将要使用的线程数
        std::cout << "C++ Info: Number of OpenMP threads has been set to " << omp_get_max_threads() << "." << std::endl;
    }
    else {
        // 如果传入0或负数，恢复OpenMP的默认行为（通常是使用所有可用核心）
        // 在某些系统上，需要通过环境变量重置，但 omp_set_num_threads(omp_get_num_procs()) 通常可行
        // 为了简单起见，我们先打印信息，让用户知道将使用默认值
        int default_threads = omp_get_num_procs(); // 获取处理器核心数
        omp_set_num_threads(default_threads); // 设置为核心数
        std::cout << "C++ Info: num_threads <= 0. OpenMP threads set to default (available cores: " << omp_get_max_threads() << ")." << std::endl;
    }
#else
    std::cout << "C++ Warning: OpenMP is not supported or enabled in this build. Model will run on a single thread." << std::endl;
#endif
}

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
    PROFILE_FUNCTION();
    if (!mesh_internal_ptr || U_current_rhs.size() != mesh_internal_ptr->cells.size()) {
        throw std::runtime_error("Invalid mesh or U_current_rhs size in RHS calculation.");
    }
    if (!reconstruction_ptr || !flux_calculator_ptr || !boundary_handler_ptr) {
        throw std::runtime_error("RHS calculation cannot proceed: core components not initialized.");
    }

    const size_t num_cells = mesh_internal_ptr->cells.size();
    StateVector RHS_vector(num_cells, { 0.0, 0.0, 0.0 });

    bool gradients_available = false;
    if (reconstruction_ptr->get_scheme_type() != ReconstructionScheme_cpp::FIRST_ORDER) {
        PROFILE_SCOPE("RHS_Reconstruction_Prepare");
        reconstruction_ptr->prepare_for_step(U_current_rhs);
        gradients_available = true;
    }

    // --- 并行化单元循环 ---
    // 使用 #pragma omp parallel for 指令来并行化这个主循环。
    // 每个线程会处理一部分单元。
    // `schedule(dynamic)` 可能对负载不均衡的网格有好处，`static` 开销更小。可以先从默认(通常是static)开始。
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_cells; ++i) {
        const Cell_cpp* cell_L_ptr = &mesh_internal_ptr->cells[i];

        // 每个单元的RHS贡献是独立的，所以我们可以在循环内部计算
        std::array<double, 3> rhs_contribution_for_cell_i = { 0.0, 0.0, 0.0 };

        // 1. 获取左单元中心状态 (这里 cell_L_ptr 就是当前单元 i)
        const auto& U_L_center_cons = U_current_rhs[cell_L_ptr->id];
        PrimitiveVars_cpp W_L_center = reconstruction_ptr->conserved_to_primitive(U_L_center_cons);
        double Z_L_centroid = cell_L_ptr->z_bed_centroid;
        double eta_L_center = U_L_center_cons[0] + Z_L_centroid;

        // 2. 遍历当前单元的所有半边
        for (int he_id : cell_L_ptr->half_edge_ids_list) {
            const HalfEdge_cpp* he = mesh_internal_ptr->get_half_edge_by_id(he_id);
            if (!he) continue;

            PrimitiveVars_cpp W_L_flux;
            std::array<double, 3> numerical_flux_cartesian;
            std::array<double, 3> source_term_L_interface = { 0.0, 0.0, 0.0 };

            if (he->twin_half_edge_id != -1) { // --- 处理内部边 ---
                const HalfEdge_cpp* he_twin_ptr = mesh_internal_ptr->get_half_edge_by_id(he->twin_half_edge_id);
                const Cell_cpp* cell_R_ptr = mesh_internal_ptr->get_cell_by_id(he_twin_ptr->cell_id);

                // 获取右单元中心状态
                const auto& U_R_center_cons = U_current_rhs[cell_R_ptr->id];
                PrimitiveVars_cpp W_R_center = reconstruction_ptr->conserved_to_primitive(U_R_center_cons);
                double Z_R_centroid = cell_R_ptr->z_bed_centroid;
                double eta_R_center = U_R_center_cons[0] + Z_R_centroid;

                // 静水重构
                double Z_interface = std::max(Z_L_centroid, Z_R_centroid);
                double h_L_star = std::max(0.0, eta_L_center - Z_interface);
                double h_R_star = std::max(0.0, eta_R_center - Z_interface);

                W_L_flux.h = h_L_star;
                W_L_flux.u = W_L_center.u;
                W_L_flux.v = W_L_center.v;
                if (h_L_star < min_depth_internal * 1.1) { W_L_flux.u = 0.0; W_L_flux.v = 0.0; }

                PrimitiveVars_cpp W_R_flux;
                W_R_flux.h = h_R_star;
                W_R_flux.u = W_R_center.u;
                W_R_flux.v = W_R_center.v;
                if (h_R_star < min_depth_internal * 1.1) { W_R_flux.u = 0.0; W_R_flux.v = 0.0; }

                // 计算通量
                numerical_flux_cartesian = flux_calculator_ptr->calculate_hllc_flux(W_L_flux, W_R_flux, he->normal);

                // 计算界面源项 (只为当前单元L计算)
                source_term_L_interface[1] = 0.5 * gravity_internal * (h_L_star * h_L_star - U_L_center_cons[0] * U_L_center_cons[0]) * he->normal[0];
                source_term_L_interface[2] = 0.5 * gravity_internal * (h_L_star * h_L_star - U_L_center_cons[0] * U_L_center_cons[0]) * he->normal[1];

            }
            else { // --- 处理边界边 ---
                // 静水重构
                double Z_interface_bnd = Z_L_centroid; // 或 z_face
                double h_L_star_bnd = std::max(0.0, eta_L_center - Z_interface_bnd);

                W_L_flux.h = h_L_star_bnd;
                W_L_flux.u = W_L_center.u;
                W_L_flux.v = W_L_center.v;
                if (h_L_star_bnd < min_depth_internal * 1.1) { W_L_flux.u = 0.0; W_L_flux.v = 0.0; }

                // 计算边界通量
                numerical_flux_cartesian = boundary_handler_ptr->calculate_boundary_flux(
                    U_current_rhs, cell_L_ptr->id, *he, time_current_rhs);

                // 计算边界界面源项
                source_term_L_interface[1] = 0.5 * gravity_internal * (h_L_star_bnd * h_L_star_bnd - U_L_center_cons[0] * U_L_center_cons[0]) * he->normal[0];
                source_term_L_interface[2] = 0.5 * gravity_internal * (h_L_star_bnd * h_L_star_bnd - U_L_center_cons[0] * U_L_center_cons[0]) * he->normal[1];
            }

            // 累加这条边对当前单元 i 的贡献
            // 通量流出为负，源项作用为正
            for (int k = 0; k < 3; ++k) {
                rhs_contribution_for_cell_i[k] -= numerical_flux_cartesian[k] * he->length;
                rhs_contribution_for_cell_i[k] += source_term_L_interface[k] * he->length;
            }
        } // 结束边的循环

        // 所有边的贡献计算完毕后，除以面积，并赋值给主RHS向量
        if (cell_L_ptr->area > epsilon) {
            for (int k = 0; k < 3; ++k) {
                RHS_vector[i][k] = rhs_contribution_for_cell_i[k] / cell_L_ptr->area;
            }
        }
    } // 结束并行化的单元循环

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

    // --- 修改：手动实现 reduction(max:...) ---
    double global_max_inv_term = 0.0; // 全局最大值

    const double shallow_water_threshold_for_dt = std::max(min_depth_internal * 50.0, 1e-4);
    const double max_speed_in_shallow_for_dt = 0.1;
    const size_t num_cells = mesh_internal_ptr->cells.size();

    // 删除了 #pragma omp parallel for reduction(...)
#pragma omp parallel
    {
        double local_max_inv_term = 0.0; // 每个线程的局部最大值

#pragma omp for schedule(static) nowait
        // --- 修改：循环变量类型 ---
        for (long long i = 0; i < num_cells; ++i) { // <--- 修改类型
            const Cell_cpp& cell = mesh_internal_ptr->cells[i];
            double h = U_state_all_internal[i][0];

            if (h < min_depth_internal) {
                continue;
            }

            double u, v;
            if (h < shallow_water_threshold_for_dt) {
                u = 0.0;
                v = 0.0;
            }
            else {
                double h_div = std::max(h, epsilon);
                u = U_state_all_internal[i][1] / h_div;
                v = U_state_all_internal[i][2] / h_div;
            }

            double speed_sq_check = u * u + v * v;
            double general_max_speed_sq = 20.0 * 20.0;
            if (speed_sq_check > general_max_speed_sq) {
                double scale = std::sqrt(general_max_speed_sq / speed_sq_check);
                u *= scale;
                v *= scale;
            }

            double c_wave = std::sqrt(gravity_internal * h);
            double sum_lambda_L_over_area_cell = 0.0;

            for (int he_id : cell.half_edge_ids_list) {
                const HalfEdge_cpp* he = mesh_internal_ptr->get_half_edge_by_id(he_id);
                if (!he || he->length < epsilon) continue;
                double un = u * he->normal[0] + v * he->normal[1];
                double lambda_max_edge = std::abs(un) + c_wave;
                sum_lambda_L_over_area_cell += lambda_max_edge * he->length;
            }

            if (cell.area > epsilon && sum_lambda_L_over_area_cell > epsilon) {
                double current_cell_inv_term = sum_lambda_L_over_area_cell / cell.area;
                if (current_cell_inv_term > local_max_inv_term) {
                    local_max_inv_term = current_cell_inv_term;
                }
            }
        }

        // --- 手动合并结果 ---
        // critical 确保一次只有一个线程写入全局变量
#pragma omp critical
        {
            if (local_max_inv_term > global_max_inv_term) {
                global_max_inv_term = local_max_inv_term;
            }
        }
    } // 结束并行区域

    if (global_max_inv_term < epsilon) {
        return max_dt_internal;
    }

    double calculated_dt = cfl_number_internal / global_max_inv_term;

    return std::min(calculated_dt, max_dt_internal);
}


void HydroModelCore_cpp::_handle_dry_cells_and_update_eta_internal() {
    PROFILE_FUNCTION();

    if (!mesh_internal_ptr || !vfr_calculator_ptr || U_state_all_internal.size() != mesh_internal_ptr->cells.size()) {
        return;
    }
    if (eta_previous_internal.size() != mesh_internal_ptr->cells.size()) {
        eta_previous_internal.assign(mesh_internal_ptr->cells.size(), 0.0);
        std::cerr << "Warning: eta_previous_internal resized in _handle_dry_cells." << std::endl;
    }

    const size_t num_cells = mesh_internal_ptr->cells.size();
    std::vector<double> eta_new(num_cells);

    // 这是一个完美的并行循环，因为每次迭代只写入 U_state_all_internal[i] 和 eta_new[i]，
    // 并且读取的数据 (如cell.node_ids, eta_previous_internal[i]) 之间没有冲突。
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_cells; ++i) {
        Cell_cpp& cell = mesh_internal_ptr->cells[i];
        std::array<double, 3>& U_cell = U_state_all_internal[i];

        const double momentum_zeroing_h_threshold = min_depth_internal * 10.0;

        if (U_cell[0] < momentum_zeroing_h_threshold) {
            U_cell[1] = 0.0;
            U_cell[2] = 0.0;
            if (U_cell[0] < min_depth_internal) {
                U_cell[0] = 0.0;
            }
        }

        double h_avg_non_negative_for_vfr = std::max(0.0, U_cell[0]);

        if (h_avg_non_negative_for_vfr >= min_depth_internal / 10.0) {
            std::vector<double> b_sorted_cell_vfr;
            std::vector<Node_cpp> nodes_sorted_cell_vfr;
            for (int node_id : cell.node_ids) {
                const Node_cpp* node = mesh_internal_ptr->get_node_by_id(node_id);
                if (node) nodes_sorted_cell_vfr.push_back(*node);
            }
            std::sort(nodes_sorted_cell_vfr.begin(), nodes_sorted_cell_vfr.end(),
                [](const Node_cpp& a, const Node_cpp& b) { return a.z_bed < b.z_bed; });
            for (const auto& sorted_node : nodes_sorted_cell_vfr) {
                b_sorted_cell_vfr.push_back(sorted_node.z_bed);
            }

            if (b_sorted_cell_vfr.size() == 3) {
                eta_new[i] = vfr_calculator_ptr->get_eta_from_h(
                    h_avg_non_negative_for_vfr, b_sorted_cell_vfr, nodes_sorted_cell_vfr, cell.area, eta_previous_internal[i],
                    this->current_time_internal, static_cast<int>(i)
                );
            }
            else {
                double base_elev_for_eta = b_sorted_cell_vfr.empty() ? cell.z_bed_centroid : b_sorted_cell_vfr[0];
                eta_new[i] = base_elev_for_eta + h_avg_non_negative_for_vfr;
            }

        }
        else { // 干单元
            double min_b_cell = std::numeric_limits<double>::max();
            bool found_node = false;
            for (int node_id : cell.node_ids) {
                const Node_cpp* node = mesh_internal_ptr->get_node_by_id(node_id);
                if (node) { min_b_cell = std::min(min_b_cell, node->z_bed); found_node = true; }
            }
            eta_new[i] = found_node ? min_b_cell : cell.z_bed_centroid;
        }
    } // 结束并行循环

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