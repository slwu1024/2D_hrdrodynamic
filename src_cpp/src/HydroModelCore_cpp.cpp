// src_cpp/src/HydroModelCore_cpp.cpp
#include "HydroModelCore_cpp.h" // 包含对应的头文件
#include <stdexcept> // 包含标准异常
#include <iostream>  // 包含输入输出流
#include <algorithm> // 包含算法
#include <cmath>     // 包含数学函数
#include <vector>    // 包含vector容器
#include <array>     // 包含array容器
#include <limits>    // 为了 numeric_limits

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
    } // 结束检查
    if (!mesh_internal_ptr) { throw std::runtime_error("Mesh not loaded for initial conditions."); } // 如果网格未加载
    if (U_initial.size() != mesh_internal_ptr->cells.size()) { // 如果初始条件大小与单元数量不符
        throw std::invalid_argument("Initial conditions size mismatch with number of cells."); // 抛出无效参数异常
    } // 结束检查
    U_state_all_internal = U_initial; // 设置内部守恒量 (h, hu, hv)

    if (!vfr_calculator_ptr) { // 确保VFR计算器已创建 (虽然下面可能不用它来设置初始eta，但其他地方可能需要)
        throw std::runtime_error("VFRCalculator not initialized, cannot compute initial eta in C++."); // 抛出运行时错误
    } // 结束检查

    eta_previous_internal.resize(mesh_internal_ptr->cells.size()); // 调整eta数组大小

    // ************* 关键修改 *************
    // 对于 uniform_elevation 类型的初始条件，我们期望初始的 eta_previous_internal
    // 直接是给定的水面高程。U_initial 中的 h 已经是根据这个高程和底床计算的。
    // 我们需要一种方式从Python端知道初始条件类型，或者在这里基于U_initial[0]和z_bed反推。
    // 一个更稳健的做法是，如果Python端已经明确知道初始水位是 uniform_elevation，
    // 那么 eta_previous_internal 应该直接等于那个值。
    // 假设我们能从某个地方获取到初始的 uniform_water_surface_elevation
    // (这可能需要从Python端传递一个额外的参数给这个函数，或者在HydroModelCore中存储这个值)
    //
    // 简化处理：我们假设U_initial[0]是根据某个统一水位计算的。
    // eta = h + z_bed_centroid。这只是一个近似，但对于初始状态可能足够。
    // 更准确的是，应该从Python知道初始设置的eta_initial值。

    // **一个更直接的方法，如果Python端已经设置了初始水位是 params['initial_water_surface_elevation']**
    // **那么 eta_previous_internal 的每个值都应该是这个。**
    // **但是，set_initial_conditions_cpp 只接收 U_initial。**
    // **所以，这里我们还是基于 U_initial[0] (水深) 和 z_bed_centroid 来近似初始水位。**
    // **这解释了为什么VFR被用来“修正”它，但对于uniform_elevation，这个修正是多余的，且可能出错。**

    for (size_t i = 0; i < mesh_internal_ptr->cells.size(); ++i) { // 遍历所有单元
        const Cell_cpp& cell = mesh_internal_ptr->cells[i]; // 获取当前单元
        // 直接使用 h_cell + z_bed_centroid 作为初始的 eta_previous_internal，
        // 因为 U_initial[i][0] (水深) 是基于初始的 uniform_elevation 计算的。
        eta_previous_internal[i] = U_state_all_internal[i][0] + cell.z_bed_centroid; // 水深+单元形心底高程

        // 对于干单元的特殊处理 (如果初始水深为0，则eta等于底高程)
        if (U_state_all_internal[i][0] < min_depth_internal / 10.0) { // 如果单元初始为干
            double min_b_cell = std::numeric_limits<double>::max(); // 初始化最低点高程为最大值
            bool found_node_cpp = false; // 标记是否找到节点
            for(int node_id : cell.node_ids) { // 遍历单元的节点ID
                const Node_cpp* node = mesh_internal_ptr->get_node_by_id(node_id); // 获取节点对象
                if(node) { min_b_cell = std::min(min_b_cell, node->z_bed); found_node_cpp = true; } // 更新最低点高程
            } // 结束节点遍历
            eta_previous_internal[i] = found_node_cpp ? min_b_cell : cell.z_bed_centroid; // 设置为节点最低点高程或单元形心底高程
        } // 结束干单元处理
    } // 结束单元遍历

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


StateVector HydroModelCore_cpp::_calculate_rhs_explicit_part_internal(const StateVector& U_current_rhs, double time_current_rhs) { // 计算显式RHS(内部)实现
    if (!mesh_internal_ptr || U_current_rhs.size() != mesh_internal_ptr->cells.size()) { // 检查网格和状态大小
        throw std::runtime_error("Invalid mesh or U_current_rhs size in RHS calculation."); // 抛出运行时错误
    }
    if (!reconstruction_ptr || !flux_calculator_ptr || !boundary_handler_ptr) { // 检查核心组件是否已初始化
        throw std::runtime_error("RHS calculation cannot proceed: core components not initialized."); // 抛出运行时错误
    }

    StateVector RHS_vector(mesh_internal_ptr->cells.size(), {0.0, 0.0, 0.0}); // 初始化RHS向量

    bool gradients_available = false; // 标记梯度是否可用
    if (reconstruction_ptr->get_scheme_type() != ReconstructionScheme_cpp::FIRST_ORDER) { // 如果不是一阶方案
        try { // 尝试准备重构步骤
            reconstruction_ptr->prepare_for_step(U_current_rhs); // 调用重构器的prepare_for_step方法
            gradients_available = true; // 标记梯度已可用
        } catch (const std::exception& e) { // 捕获异常
            std::cerr << "Error during reconstruction prepare_for_step: " << e.what() << std::endl; // 打印错误信息
            throw; // 重新抛出异常
        }
    }

    for (const auto& he : mesh_internal_ptr->half_edges) { // 遍历所有半边
        if (he.cell_id < 0 || static_cast<size_t>(he.cell_id) >= mesh_internal_ptr->cells.size()) { // 检查半边所属单元ID是否有效
            continue; // 跳过
        }
        const Cell_cpp* cell_L_ptr = mesh_internal_ptr->get_cell_by_id(he.cell_id); // 获取左单元指针
        if (!cell_L_ptr) { // 如果无法获取左单元
            continue; // 跳过
        }

        std::array<double, 3> numerical_flux_cartesian; // 存储笛卡尔坐标系下的数值通量

        if (he.twin_half_edge_id != -1) { // --- 处理内部边 ---
            if (static_cast<unsigned int>(he.id) >= static_cast<unsigned int>(he.twin_half_edge_id)) { // 避免重复计算
                continue; // 跳过
            }
            const HalfEdge_cpp* he_twin_ptr = mesh_internal_ptr->get_half_edge_by_id(he.twin_half_edge_id); // 获取孪生半边指针
            if (!he_twin_ptr || he_twin_ptr->cell_id < 0 || static_cast<size_t>(he_twin_ptr->cell_id) >= mesh_internal_ptr->cells.size()) { // 检查孪生半边及其单元ID是否有效
                continue; // 跳过
            }
            const Cell_cpp* cell_R_ptr = mesh_internal_ptr->get_cell_by_id(he_twin_ptr->cell_id); // 获取右单元指针
            if (!cell_R_ptr) { // 如果无法获取右单元
                continue; // 跳过
            }

            auto [W_L_recons_at_iface, W_R_recons_at_iface] = reconstruction_ptr->get_reconstructed_interface_states( // 获取界面左右两侧重构后的原始变量
                U_current_rhs, cell_L_ptr->id, cell_R_ptr->id, he, false
            );

            double z_face; // 声明界面底高程
            const Node_cpp* n_origin = mesh_internal_ptr->get_node_by_id(he.origin_node_id); // 获取起点
            const HalfEdge_cpp* he_next_ptr = mesh_internal_ptr->get_half_edge_by_id(he.next_half_edge_id); // 获取下一半边
            const Node_cpp* n_end = nullptr; // 初始化终点为空
            if (he_next_ptr) n_end = mesh_internal_ptr->get_node_by_id(he_next_ptr->origin_node_id); // 获取终点

            if (n_origin && n_end) { // 如果起点和终点都有效
                z_face = (n_origin->z_bed + n_end->z_bed) / 2.0; // 取平均值
            } else { // 否则
                z_face = (cell_L_ptr->z_bed_centroid + cell_R_ptr->z_bed_centroid) / 2.0; // 使用单元形心底高程平均值
            }

            double eta_L_center = U_current_rhs[cell_L_ptr->id][0] + cell_L_ptr->z_bed_centroid; // 左单元中心水位
            double eta_R_center = U_current_rhs[cell_R_ptr->id][0] + cell_R_ptr->z_bed_centroid; // 右单元中心水位
            double eta_L_at_face = eta_L_center; // 初始化界面左侧水位
            double eta_R_at_face = eta_R_center; // 初始化界面右侧水位

            if (gradients_available) { // 如果梯度可用 (高阶)
                const auto& grad_W_L = reconstruction_ptr->get_gradient_for_cell(cell_L_ptr->id); // 获取左单元梯度
                const auto& grad_W_R = reconstruction_ptr->get_gradient_for_cell(cell_R_ptr->id); // 获取右单元梯度
                std::array<double, 2> grad_eta_L = {grad_W_L[0][0] + cell_L_ptr->b_slope_x, grad_W_L[0][1] + cell_L_ptr->b_slope_y}; // 左单元水位梯度
                std::array<double, 2> grad_eta_R = {grad_W_R[0][0] + cell_R_ptr->b_slope_x, grad_W_R[0][1] + cell_R_ptr->b_slope_y}; // 右单元水位梯度
                std::array<double, 2> vec_L_to_face = {he.mid_point[0] - cell_L_ptr->centroid[0], he.mid_point[1] - cell_L_ptr->centroid[1]}; // 左单元到界面向量
                std::array<double, 2> vec_R_to_face = {he.mid_point[0] - cell_R_ptr->centroid[0], he.mid_point[1] - cell_R_ptr->centroid[1]}; // 右单元到界面向量
                eta_L_at_face += grad_eta_L[0] * vec_L_to_face[0] + grad_eta_L[1] * vec_L_to_face[1]; // 更新界面左侧水位
                eta_R_at_face += grad_eta_R[0] * vec_R_to_face[0] + grad_eta_R[1] * vec_R_to_face[1]; // 更新界面右侧水位
            }

            double h_L_star = std::max(0.0, eta_L_at_face - z_face); // 计算静水重构左侧水深
            double h_R_star = std::max(0.0, eta_R_at_face - z_face); // 计算静水重构右侧水深
            PrimitiveVars_cpp W_L_flux = {h_L_star, W_L_recons_at_iface.u, W_L_recons_at_iface.v}; // 构建左侧通量状态
            PrimitiveVars_cpp W_R_flux = {h_R_star, W_R_recons_at_iface.u, W_R_recons_at_iface.v}; // 构建右侧通量状态 (修正: 使用 W_R_recons_at_iface.u, W_R_recons_at_iface.v)

            try { // 尝试计算通量
                numerical_flux_cartesian = flux_calculator_ptr->calculate_hllc_flux(W_L_flux, W_R_flux, he.normal); // 计算HLLC通量
            } catch (const std::exception& e) { // 捕获异常
                std::cerr << "Error HLLC flux internal edge " << he.id << ": " << e.what() << std::endl; throw; // 打印错误并重新抛出
            }

            for (int k = 0; k < 3; ++k) { // 遍历通量分量
                double flux_term = numerical_flux_cartesian[k] * he.length; // 计算通量项
                if (cell_L_ptr->area > epsilon) RHS_vector[cell_L_ptr->id][k] -= flux_term / cell_L_ptr->area; // 更新左单元RHS
                if (cell_R_ptr->area > epsilon) RHS_vector[cell_R_ptr->id][k] += flux_term / cell_R_ptr->area; // 更新右单元RHS
            }

        } else { // --- 处理边界边 ---
            try { // 尝试计算边界通量
                numerical_flux_cartesian = boundary_handler_ptr->calculate_boundary_flux( // 调用边界处理器计算通量
                    U_current_rhs, cell_L_ptr->id, he, time_current_rhs
                );
            } catch (const std::exception& e) { // 捕获异常
                std::cerr << "Error boundary flux edge " << he.id << ": " << e.what() << std::endl; throw; // 打印错误并重新抛出
            }
            for (int k = 0; k < 3; ++k) { // 遍历通量分量
                double flux_term = numerical_flux_cartesian[k] * he.length; // 计算通量项
                if (cell_L_ptr->area > epsilon) RHS_vector[cell_L_ptr->id][k] -= flux_term / cell_L_ptr->area; // 更新左单元RHS
            }
        }
    }
    // --- **添加底坡源项** --- // <--- 这是你需要添加的部分
    for (size_t i = 0; i < mesh_internal_ptr->cells.size(); ++i) { // 再次遍历所有单元
        const Cell_cpp& cell = mesh_internal_ptr->cells[i]; // 获取当前单元
        double h_center = U_current_rhs[i][0]; // 获取单元中心水深

        if (h_center >= min_depth_internal / 10.0) { // 只对湿单元计算源项 (用小一点的阈值避免干湿交界问题)
            // 源项 S_b = [0, -g*h*(∂b/∂x), -g*h*(∂b/∂y)]
            // 假设 Mesh_cpp::precompute_cell_geometry_cpp() 计算的是 ∂b/∂x 和 ∂b/∂y
            double gx = gravity_internal; // 使用成员变量 g_internal
            RHS_vector[i][1] += -gx * h_center * cell.b_slope_x; // x方向动量源项
            RHS_vector[i][2] += -gx * h_center * cell.b_slope_y; // y方向动量源项
        } // 结束湿单元判断
    } // 结束源项添加循环
    return RHS_vector; // 返回RHS向量
} // 结束函数

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

double HydroModelCore_cpp::_calculate_dt_internal() { // 计算CFL时间步长(内部)实现
    if (!mesh_internal_ptr || U_state_all_internal.size() != mesh_internal_ptr->cells.size()) { // 检查网格和状态大小
        return max_dt_internal; // 返回最大dt
    }
    double min_dt_inv_term = 0.0; // 初始化dt倒数项的最大值

    for (size_t i = 0; i < mesh_internal_ptr->cells.size(); ++i) { // 遍历所有单元
        const Cell_cpp& cell = mesh_internal_ptr->cells[i]; // 获取当前单元
        double h = U_state_all_internal[i][0]; // 获取水深
        if (h < min_depth_internal) continue; // 跳过干单元

        double h_div = std::max(h, epsilon); // 安全水深
        double u = U_state_all_internal[i][1] / h_div; // 计算u速度
        double v = U_state_all_internal[i][2] / h_div; // 计算v速度
        double c_wave = std::sqrt(gravity_internal * h); // 计算波速
        double sum_lambda_L_over_area_cell = 0.0; // 初始化单元的 (特征速度*边长/面积) 总和

        for (int he_id : cell.half_edge_ids_list) { // 遍历单元的半边ID
            const HalfEdge_cpp* he = mesh_internal_ptr->get_half_edge_by_id(he_id); // 获取半边对象
            if (!he || he->length < epsilon) continue; // 跳过无效或零长度半边
            double un = u * he->normal[0] + v * he->normal[1]; // 计算法向速度
            double lambda_max_edge = std::abs(un) + c_wave; // 计算该边最大特征速度
            sum_lambda_L_over_area_cell += lambda_max_edge * he->length; // 累加
        }
        if (cell.area > epsilon && sum_lambda_L_over_area_cell > epsilon) { // 如果面积和累加值有效
            min_dt_inv_term = std::max(min_dt_inv_term, sum_lambda_L_over_area_cell / cell.area); // 更新全局最大值
        }
    }
    if (min_dt_inv_term < epsilon) { // 如果全局最大值非常小
        return max_dt_internal; // 返回最大dt
    }
    double calculated_dt = cfl_number_internal / min_dt_inv_term; // 计算dt
    return std::min(calculated_dt, max_dt_internal); // 返回计算dt和最大dt中的较小者
} // 结束函数

void HydroModelCore_cpp::_handle_dry_cells_and_update_eta_internal() { // 处理干单元并更新水位(内部)实现
    if (!mesh_internal_ptr || !vfr_calculator_ptr || U_state_all_internal.size() != mesh_internal_ptr->cells.size()) { // 检查依赖组件和状态大小
        return; // 返回
    }
    if (eta_previous_internal.size() != mesh_internal_ptr->cells.size()) { // 如果上一时刻水位大小无效
        eta_previous_internal.assign(mesh_internal_ptr->cells.size(), 0.0); // 初始化为0
        std::cerr << "Warning: eta_previous_internal resized in _handle_dry_cells." << std::endl; // 打印警告
    }

    std::vector<double> eta_new(mesh_internal_ptr->cells.size()); // 初始化新的水位数组
    for (size_t i = 0; i < mesh_internal_ptr->cells.size(); ++i) { // 遍历所有单元
        Cell_cpp& cell = mesh_internal_ptr->cells[i]; // 获取当前单元 (引用)
        std::array<double, 3>& U_cell = U_state_all_internal[i]; // 获取当前单元状态 (引用)

        if (U_cell[0] < min_depth_internal) { // 如果水深小于最小深度 (干单元处理)
            U_cell[0] = 0.0; U_cell[1] = 0.0; U_cell[2] = 0.0; // 水深和动量设为0
        }

        if (U_cell[0] >= min_depth_internal / 10.0) { // 如果单元为湿 (用稍小阈值触发VFR)
            std::vector<double> b_sorted_cell; // 存储单元排序后的顶点底高程
            std::vector<Node_cpp> nodes_sorted_cell; // 存储单元排序后的节点对象
            for(int node_id : cell.node_ids) { // 遍历单元的节点ID
                const Node_cpp* node = mesh_internal_ptr->get_node_by_id(node_id); // 获取节点对象
                if(node) nodes_sorted_cell.push_back(*node); // 如果节点有效则添加到列表
            }
            std::sort(nodes_sorted_cell.begin(), nodes_sorted_cell.end(), // 按底高程排序节点
                      [](const Node_cpp& a, const Node_cpp& b) { return a.z_bed < b.z_bed; });
            for(const auto& sorted_node : nodes_sorted_cell) { // 遍历排序后的节点
                b_sorted_cell.push_back(sorted_node.z_bed); // 添加底高程到列表
            }

            if (b_sorted_cell.size() == 3) { // 如果是三角形单元
                 eta_new[i] = vfr_calculator_ptr->get_eta_from_h( // 调用VFR计算eta
                    U_cell[0], b_sorted_cell, nodes_sorted_cell, cell.area, eta_previous_internal[i]
                 ); // 结束调用
            } else { // 其他情况
                 eta_new[i] = b_sorted_cell.empty() ? 0.0 : b_sorted_cell[0]; // 设置为最低点高程或0
            }
        } else { // 如果单元为干
            double min_b_cell = std::numeric_limits<double>::max(); // 初始化最低点高程为最大值
            bool found_node = false; // 标记是否找到节点
            for(int node_id : cell.node_ids) { // 遍历单元的节点ID
                const Node_cpp* node = mesh_internal_ptr->get_node_by_id(node_id); // 获取节点对象
                if(node) { min_b_cell = std::min(min_b_cell, node->z_bed); found_node = true; } // 更新最低点高程
            }
            eta_new[i] = found_node ? min_b_cell : 0.0; // 设置为最低点高程或0
        }
    }
    eta_previous_internal = eta_new; // 更新内部存储的上一时刻水位
} // 结束函数

bool HydroModelCore_cpp::advance_one_step() { // 执行一个完整的时间步实现
    if (!model_fully_initialized_flag || !initial_conditions_set_flag || !boundary_conditions_set_flag) { // 如果模型未完全准备好
        throw std::runtime_error("Model not fully initialized/configured before calling advance_one_step."); // 抛出运行时错误
    }
    if (!time_integrator_ptr || U_state_all_internal.empty()) { // 如果时间积分器或初始状态无效
         throw std::runtime_error("Time integrator or initial state not ready for step."); // 抛出运行时错误
    }

    double dt = _calculate_dt_internal(); // 计算CFL时间步长
    double actual_output_dt = output_dt_internal; // 获取输出时间间隔
    if (actual_output_dt <= epsilon && total_time_internal > epsilon) { // 如果输出间隔无效但总时间有效
         actual_output_dt = total_time_internal + 1.0; // 设置一个永远不会达到的输出间隔
    }

    if (current_time_internal + dt >= total_time_internal - epsilon) { // 如果下一步将超过或接近总时间
        dt = total_time_internal - current_time_internal; // 调整dt恰好到达总时间
        if (dt < epsilon / 10.0) dt = 0; // 如果调整后dt过小则设为0
    } else if (actual_output_dt > epsilon) { // 如果设置了有效的输出间隔
        double num_outputs_passed = std::floor(current_time_internal / actual_output_dt + epsilon); // 已过的输出周期数
        double next_ideal_output_time = (num_outputs_passed + 1.0) * actual_output_dt; // 下一个理想输出时间点
        if (current_time_internal + dt >= next_ideal_output_time - epsilon) { // 如果下一步将超过或接近下一个输出时间点
            dt = next_ideal_output_time - current_time_internal; // 调整dt恰好到达输出时间点
             if (dt < epsilon / 10.0 && next_ideal_output_time < total_time_internal - epsilon) { // 如果调整后dt过小且未到总时间
                next_ideal_output_time = (num_outputs_passed + 2.0) * actual_output_dt; // 跳到再下一个输出点
                dt = next_ideal_output_time - current_time_internal; // 重新调整dt
             }
             if (current_time_internal + dt >= total_time_internal - epsilon) { // 再次确保不超过总时间
                 dt = total_time_internal - current_time_internal; // 调整dt
                 if (dt < epsilon / 10.0) dt = 0; // 如果调整后dt过小则设为0
             }
        }
    }

    if (dt < epsilon / 10.0 && current_time_internal < total_time_internal - epsilon) { // 如果dt极小但未到总时间
        if (dt < 0) dt = 0; // 确保dt非负
    }

    this->last_calculated_dt_internal = dt; // 记录实际使用的dt

    if (dt <= epsilon / 100.0 && current_time_internal >= total_time_internal - epsilon) { // 如果dt为0或极小且已达总时间
        return false; // 指示模拟结束
    }

    if (this->last_calculated_dt_internal > epsilon / 100.0) { // 只有当dt大于极小值时才执行积分
        try { // 尝试时间积分
            U_state_all_internal = time_integrator_ptr->step(U_state_all_internal, this->last_calculated_dt_internal, current_time_internal); // 执行一步积分
        } catch (const std::exception& e) { // 捕获异常
            std::cerr << "Error during time integration step: " << e.what() << std::endl; throw; // 打印错误并重新抛出
        }
        _handle_dry_cells_and_update_eta_internal(); // 处理干湿单元并更新水位
    }

    current_time_internal += this->last_calculated_dt_internal; // 更新当前时间
    step_count_internal++; // 增加步数
    return !is_simulation_finished(); // 返回模拟是否继续
} // 结束函数

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