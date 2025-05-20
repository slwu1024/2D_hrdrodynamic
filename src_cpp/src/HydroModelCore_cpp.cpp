// src_cpp/src/HydroModelCore_cpp.cpp
#include "HydroModelCore_cpp.h" // 包含对应的头文件
#include "WettingDrying_cpp.h" // <--- 确保包含了这个头文件
#include "FluxCalculator_cpp.h"
#include <stdexcept> // 包含标准异常
#include <iostream>  // 包含输入输出流
#include <algorithm> // 包含算法
#include <cmath>     // 包含数学函数
#include <vector>    // 包含vector容器
#include <array>     // 包含array容器
#include <limits>    // 为了 numeric_limits
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

void HydroModelCore_cpp::initialize_model_from_files( // 从文件初始化模型实现
    const std::string& node_filepath, // 节点文件路径
    const std::string& cell_filepath, // 单元文件路径
    const std::string& edge_filepath, // 边文件路径
    const std::vector<double>& cell_manning_values, // 单元曼宁系数值
    double gravity, double min_depth, double cfl, // 模拟参数
    double total_t, double output_dt_interval, double max_dt_val, // 时间参数
    ReconstructionScheme_cpp recon_scheme, // 数值方案
    RiemannSolverType_cpp riemann_solver,
    TimeScheme_cpp time_scheme) {

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
    if (!mesh_internal_ptr || U_current_rhs.size() != mesh_internal_ptr->cells.size()) {
        throw std::runtime_error("Invalid mesh or U_current_rhs size in RHS calculation.");
    }
    if (!reconstruction_ptr || !flux_calculator_ptr || !boundary_handler_ptr) {
        throw std::runtime_error("RHS calculation cannot proceed: core components not initialized.");
    }

    StateVector RHS_vector(mesh_internal_ptr->cells.size(), {0.0, 0.0, 0.0});

    bool gradients_available = false;
    if (reconstruction_ptr->get_scheme_type() != ReconstructionScheme_cpp::FIRST_ORDER) {
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



    for (const auto& he : mesh_internal_ptr->half_edges) {
        // ... (获取 cell_L_ptr) ...
        const Cell_cpp* cell_L_ptr = mesh_internal_ptr->get_cell_by_id(he.cell_id);
        // ...

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
    const double shallow_water_threshold_for_dt = min_depth_internal * 5.0; // 新增：为dt计算定义一个浅水阈值
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

    // 保持 CRITICAL_DT_INFO 日志，触发阈值 5e-6 仍然适用
    if (calculated_dt < 0.000005 && problematic_cell_id_for_dt != -1) {
        const Cell_cpp& p_cell = mesh_internal_ptr->cells[problematic_cell_id_for_dt];
        double ph_orig = U_state_all_internal[problematic_cell_id_for_dt][0];
        double phu_orig = U_state_all_internal[problematic_cell_id_for_dt][1];
        double phv_orig = U_state_all_internal[problematic_cell_id_for_dt][2];

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "C++ CRITICAL_DT_INFO (Threshold 5e-6): Time=" << current_time_internal
                  << ", Calculated dt=" << calculated_dt
                  << ". Problem Cell ID: " << problematic_cell_id_for_dt
                  << ", Orig_h=" << ph_orig << ", Orig_hu=" << phu_orig << ", Orig_hv=" << phv_orig
                  << ", h_of_prob_cell=" << h_at_problem_cell // 之前记录的h (原始h)
                  << ", u_eff_in_dt=" << u_at_problem_cell // 之前记录的u (可能修正后)
                  << ", v_eff_in_dt=" << v_at_problem_cell // 之前记录的v (可能修正后)
                  << ", cell_area=" << p_cell.area
                  << ", cell_CFL_term_val=" << cfl_term_at_problem_cell
                  << std::endl;
    }

    return std::min(calculated_dt, max_dt_internal);
}


void HydroModelCore_cpp::_handle_dry_cells_and_update_eta_internal() { // 处理干单元并更新水位(内部)实现
    // const int problem_cell_vfr_1 = 1385; // 在函数开始处定义 // 注释掉旧的
    // const int problem_cell_vfr_2 = 1419; // 注释掉旧的
    // bool time_in_vfr_debug_range = (current_time_internal >= 4.99 && current_time_internal <= 5.01); // current_time_internal 需要能被访问 // 注释掉旧的

    // --- 修改：针对 t=0 时的调试 ---
    const int problem_cell_vfr_1 = 0; // 假设单元ID 0 是一个你想观察的单元（例如初始为湿的单元） // 新增：设置调试单元ID
    const int problem_cell_vfr_2 = -1; // 暂时禁用第二个问题单元的调试 // 新增：禁用第二个调试单元
    bool time_in_vfr_debug_range = (std::abs(current_time_internal - 0.0) < epsilon); // 仅在 t=0 时激活调试 // 新增：设置调试时间范围为t=0

    if (!mesh_internal_ptr || !vfr_calculator_ptr || U_state_all_internal.size() != mesh_internal_ptr->cells.size()) {
        return;
    }
    if (eta_previous_internal.size() != mesh_internal_ptr->cells.size()) {
        eta_previous_internal.assign(mesh_internal_ptr->cells.size(), 0.0);
        std::cerr << "Warning: eta_previous_internal resized in _handle_dry_cells." << std::endl;
    }

    std::vector<double> eta_new(mesh_internal_ptr->cells.size());

    // const double vfr_debug_time_start = 4.99960; // 注释掉旧的
    // const double vfr_debug_time_end   = 4.99980; // 注释掉旧的
    const double vfr_debug_time_start = -0.001; // 确保 t=0 在此窗口内 // 新增：设置VFR调试起始时间
    const double vfr_debug_time_end   = 0.001;  // 确保 t=0 在此窗口内 // 新增：设置VFR调试结束时间

    for (size_t i = 0; i < mesh_internal_ptr->cells.size(); ++i) {
        Cell_cpp& cell = mesh_internal_ptr->cells[i];
        std::array<double, 3>& U_cell = U_state_all_internal[i];

        // --- 修改：更积极地清零浅水区的动量 ---
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
        // --- 修改结束 ---

        bool is_this_problem_cell_for_vfr = (static_cast<int>(i) == problem_cell_vfr_1 || static_cast<int>(i) == problem_cell_vfr_2);
        bool is_time_in_vfr_debug_window = (this->current_time_internal >= vfr_debug_time_start && this->current_time_internal <= vfr_debug_time_end);

        if (is_this_problem_cell_for_vfr && is_time_in_vfr_debug_window) {
            std::cout << std::fixed << std::setprecision(8); // 设置输出精度 // 新增：设置输出精度
            std::cout << "DEBUG_VFR_CALL_INPUT: Time=" << this->current_time_internal << ", Cell=" << i
                      << ", h_avg_in=" << U_cell[0]
                      << ", eta_guess_in=" << eta_previous_internal[i];
            const auto* cell_ptr_for_bverts = mesh_internal_ptr->get_cell_by_id(i);
            if (cell_ptr_for_bverts && !cell_ptr_for_bverts->node_ids.empty()) {
                std::vector<double> temp_b_verts;
                for(int node_id_b : cell_ptr_for_bverts->node_ids) {
                    const Node_cpp* node_b = mesh_internal_ptr->get_node_by_id(node_id_b);
                    if(node_b) temp_b_verts.push_back(node_b->z_bed);
                }
                std::sort(temp_b_verts.begin(), temp_b_verts.end());
                std::cout << ", b_verts={";
                for(size_t k_b=0; k_b<temp_b_verts.size(); ++k_b) std::cout << temp_b_verts[k_b] << (k_b==temp_b_verts.size()-1 ? "" : ",");
                std::cout << "}";
            }
            std::cout << std::endl;
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

            if (is_this_problem_cell_for_vfr && is_time_in_vfr_debug_window) {
                VFRCalculator_cpp::set_internal_debug_conditions(true, static_cast<int>(i), this->current_time_internal - 0.00001, this->current_time_internal + 0.00001);
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

            if (is_this_problem_cell_for_vfr && is_time_in_vfr_debug_window) {
                VFRCalculator_cpp::set_internal_debug_conditions(false, -1, -1e9, 1e9);
            }

            if (is_this_problem_cell_for_vfr && is_time_in_vfr_debug_window) {
                std::cout << "DEBUG_VFR_CALL_OUTPUT: Time=" << this->current_time_internal << ", Cell=" << i
                          << ", eta_new_out=" << eta_new[i] << std::endl;
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

    double dt = _calculate_dt_internal();
    double actual_output_dt = output_dt_internal;
    if (actual_output_dt <= epsilon && total_time_internal > epsilon) {
         actual_output_dt = total_time_internal + 1.0;
    }

    if (current_time_internal + dt >= total_time_internal - epsilon) {
        dt = total_time_internal - current_time_internal;
        if (dt < epsilon / 10.0) dt = 0;
    } else if (actual_output_dt > epsilon) {
        double num_outputs_passed = std::floor(current_time_internal / actual_output_dt + epsilon);
        double next_ideal_output_time = (num_outputs_passed + 1.0) * actual_output_dt;
        if (current_time_internal + dt >= next_ideal_output_time - epsilon) {
            dt = next_ideal_output_time - current_time_internal;
             if (dt < epsilon / 10.0 && next_ideal_output_time < total_time_internal - epsilon) {
                next_ideal_output_time = (num_outputs_passed + 2.0) * actual_output_dt;
                dt = next_ideal_output_time - current_time_internal;
             }
             if (current_time_internal + dt >= total_time_internal - epsilon) {
                 dt = total_time_internal - current_time_internal;
                 if (dt < epsilon / 10.0) dt = 0;
             }
        }
    }

    if (dt < epsilon / 10.0 && current_time_internal < total_time_internal - epsilon) {
        if (dt < 0) dt = 0;
    }

    this->last_calculated_dt_internal = dt;


    // Stall detection / prevention: 卡顿检测/预防
    if (this->last_calculated_dt_internal < epsilon / 100.0) { // 如果计算出的dt非常小 // 新增：卡顿检测逻辑
        if (current_time_internal < total_time_internal - epsilon) { // 但模拟尚未结束
            std::cerr << "C++ WARNING advance_one_step: Calculated dt is effectively zero (" << this->last_calculated_dt_internal
                      << ") at t = " << current_time_internal
                      << "s, but total_time is " << total_time_internal
                      << "s. Simulation might be stalled. Forcing current_time to total_time to end." << std::endl; // 记录警告信息：dt过小可能导致卡顿，强制结束
            current_time_internal = total_time_internal; // 强制当前时间等于总时间以结束循环
        }
    } // 新增：卡顿检测逻辑结束

    if (this->last_calculated_dt_internal > epsilon / 100.0) {
        try {
            U_state_all_internal = time_integrator_ptr->step(U_state_all_internal, this->last_calculated_dt_internal, current_time_internal);
        } catch (const std::exception& e) {
            std::cerr << "Error during time integration step: " << e.what() << std::endl; throw;
        }
        _handle_dry_cells_and_update_eta_internal();
    }

    const double MAX_PHYSICAL_H = 1000.0; // 假设物理上可能的最大水深是1000米 (根据你的问题调整)
    const double MAX_PHYSICAL_SPEED = 50.0; // 假设物理上可能的最大流速是50m/s

    for (size_t i = 0; i < U_state_all_internal.size(); ++i) {
        bool corrected = false; // 标记此单元是否被修正
        // 检查水深 h
        if (U_state_all_internal[i][0] > MAX_PHYSICAL_H) {
            std::cerr << "WARNING (Post-Step Check): Cell " << i << " h=" << U_state_all_internal[i][0]
                      << " exceeded MAX_PHYSICAL_H=" << MAX_PHYSICAL_H << ". Clamping h and zeroing momentum." << std::endl;
            U_state_all_internal[i][0] = MAX_PHYSICAL_H; // 限制水深
            U_state_all_internal[i][1] = 0.0; // 清零hu
            U_state_all_internal[i][2] = 0.0; // 清零hv
            corrected = true;
        }
        if (U_state_all_internal[i][0] < 0.0 && std::abs(U_state_all_internal[i][0]) > epsilon) {
             std::cerr << "WARNING (Post-Step Check): Cell " << i << " h=" << U_state_all_internal[i][0]
                       << " is negative. Setting to 0 and zeroing momentum." << std::endl;
             U_state_all_internal[i][0] = 0.0;
             U_state_all_internal[i][1] = 0.0;
             U_state_all_internal[i][2] = 0.0;
             corrected = true;
        }

        // 如果水深经过修正或者本身就有效，再检查流速
        if (U_state_all_internal[i][0] > min_depth_internal) { // 只对湿单元检查流速
            double h_check = U_state_all_internal[i][0];
            double u_check = U_state_all_internal[i][1] / h_check;
            double v_check = U_state_all_internal[i][2] / h_check;
            double speed_check_sq = u_check * u_check + v_check * v_check;
            if (speed_check_sq > MAX_PHYSICAL_SPEED * MAX_PHYSICAL_SPEED) {
                if (!corrected) { /* ... print warning ... */ }
                double scale = MAX_PHYSICAL_SPEED / std::sqrt(speed_check_sq);
                U_state_all_internal[i][1] *= scale; // 缩放hu
                U_state_all_internal[i][2] *= scale; // 缩放hv
            }
        } else if (U_state_all_internal[i][0] < min_depth_internal) { // 对于干/极浅单元，确保动量为零
             if (std::abs(U_state_all_internal[i][1]) > epsilon || std::abs(U_state_all_internal[i][2]) > epsilon) {
                 /* ... print warning (optional) ... */
                 U_state_all_internal[i][1] = 0.0;
                 U_state_all_internal[i][2] = 0.0;
             }
        }
    }
    current_time_internal += this->last_calculated_dt_internal;
    step_count_internal++;
    return !is_simulation_finished();
}

void HydroModelCore_cpp::run_simulation_to_end() { // 执行整个模拟循环实现
    std::cout << "C++ Simulation starting..." << std::endl; // 打印开始信息
    while(advance_one_step()) { // 循环执行单步积分
        if (step_count_internal % 100 == 0) { // 每100步打印进度
            std::cout << "  Progress: Step " << step_count_internal << ", Time: " << current_time_internal
                      << ", dt: " << get_last_dt() << std::endl; // 打印进度和dt
        }
    }
    std::cout << "C++ Simulation finished at time " << current_time_internal
              << " after " << step_count_internal << " steps." << std::endl; // 打印结束信息
} // 结束函数

} // namespace HydroCore