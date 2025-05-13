// bindings.cpp
#include <pybind11/pybind11.h> // 包含pybind11核心库
#include <pybind11/stl.h>       // 用于自动转换STL容器如 std::vector, std::map
#include <pybind11/functional.h> // 用于自动转换 std::function
#include <pybind11/numpy.h>     // 用于处理NumPy数组
#include <pybind11/chrono.h>    // 如果处理时间 (可选)

// 包含所有相关的C++头文件
#include "MeshData_cpp.h"
#include "FluxCalculator_cpp.h"
#include "SourceTerms_cpp.h"         // <--- 确保已包含
#include "WettingDrying_cpp.h"       // <--- 确保已包含
#include "Reconstruction_cpp.h"
#include "TimeIntegrator_cpp.h"
#include "BoundaryConditionHandler_cpp.h"
#include "HydroModelCore_cpp.h"

namespace py = pybind11; // 定义pybind11命名空间别名
using namespace HydroCore; // 使用你的核心命名空间

PYBIND11_MODULE(hydro_model_cpp, m) { // 定义Python模块，名称为 hydro_model_cpp
    m.doc() = "Python bindings for C++ Hydrodynamic Model core"; // 模块文档字符串

    // --- 绑定核心数据结构 (Node, HalfEdge, Cell) ---
    py::class_<Node_cpp>(m, "Node_cpp") // 绑定 Node_cpp 类
        .def(py::init<int, double, double, double, int>(), // 绑定构造函数
             py::arg("id") = -1, py::arg("x") = 0.0, py::arg("y") = 0.0, // 参数名及默认值
             py::arg("z_bed") = 0.0, py::arg("marker") = 0) // 参数名及默认值
        .def_readwrite("id", &Node_cpp::id)        // 绑定 id 属性 (可读写)
        .def_readwrite("x", &Node_cpp::x)          // 绑定 x 属性 (可读写)
        .def_readwrite("y", &Node_cpp::y)          // 绑定 y 属性 (可读写)
        .def_readwrite("z_bed", &Node_cpp::z_bed)  // 绑定 z_bed 属性 (可读写)
        .def_readwrite("marker", &Node_cpp::marker); // 绑定 marker 属性 (可读写)

    py::class_<HalfEdge_cpp>(m, "HalfEdge_cpp") // 绑定 HalfEdge_cpp 类
        .def(py::init<int>(), py::arg("id") = -1) // 绑定构造函数
        .def_readwrite("id", &HalfEdge_cpp::id)                   // 绑定 id 属性
        .def_readwrite("origin_node_id", &HalfEdge_cpp::origin_node_id) // 绑定起点ID属性
        .def_readwrite("twin_half_edge_id", &HalfEdge_cpp::twin_half_edge_id) // 绑定孪生半边ID属性
        .def_readwrite("next_half_edge_id", &HalfEdge_cpp::next_half_edge_id) // 绑定下一半边ID属性
        .def_readwrite("prev_half_edge_id", &HalfEdge_cpp::prev_half_edge_id) // 绑定上一半边ID属性
        .def_readwrite("cell_id", &HalfEdge_cpp::cell_id)             // 绑定所属单元ID属性
        .def_readwrite("length", &HalfEdge_cpp::length)               // 绑定长度属性
        .def_readwrite("normal", &HalfEdge_cpp::normal)               // 绑定法向量属性 (std::array<double, 2>)
        .def_readwrite("mid_point", &HalfEdge_cpp::mid_point)         // 绑定中点属性 (std::array<double, 2>)
        .def_readwrite("boundary_marker", &HalfEdge_cpp::boundary_marker); // 绑定边界标记属性

    py::class_<Cell_cpp>(m, "Cell_cpp") // 绑定 Cell_cpp 类
        .def(py::init<int>(), py::arg("id") = -1) // 绑定构造函数
        .def_readwrite("id", &Cell_cpp::id)                   // 绑定 id 属性
        .def_readwrite("node_ids", &Cell_cpp::node_ids)       // 绑定节点ID列表属性 (std::vector<int>)
        .def_readwrite("half_edge_ids_list", &Cell_cpp::half_edge_ids_list) // 绑定半边ID列表属性 (std::vector<int>)
        .def_readwrite("area", &Cell_cpp::area)               // 绑定面积属性
        .def_readwrite("centroid", &Cell_cpp::centroid)           // 绑定形心属性 (std::array<double, 2>)
        .def_readwrite("z_bed_centroid", &Cell_cpp::z_bed_centroid) // 绑定形心底高程属性
        .def_readwrite("b_slope_x", &Cell_cpp::b_slope_x)       // 绑定x方向底坡属性
        .def_readwrite("b_slope_y", &Cell_cpp::b_slope_y)       // 绑定y方向底坡属性
        .def_readwrite("manning_n", &Cell_cpp::manning_n);     // 绑定曼宁系数属性

    py::class_<Mesh_cpp>(m, "Mesh_cpp") // 绑定 Mesh_cpp 类 (保持原名)
        .def(py::init<>()) // 默认构造函数
        .def("get_num_nodes", [](const Mesh_cpp& self) { return self.nodes.size(); }, // 获取节点数量
             "Returns the number of nodes in the mesh.") // 方法文档字符串
        .def("get_num_cells", [](const Mesh_cpp& self) { return self.cells.size(); }, // 获取单元数量
             "Returns the number of cells in the mesh.") // 方法文档字符串
        .def("get_num_half_edges", [](const Mesh_cpp& self) { return self.half_edges.size(); }, // 获取半边数量
             "Returns the number of half-edges in the mesh.") // 方法文档字符串
        .def("get_node", [](const Mesh_cpp& self, int node_id) { // 通过ID获取节点 (返回副本)
            const Node_cpp* node = self.get_node_by_id(node_id); // 调用C++ getter
            if (!node) throw py::index_error("Node ID not found: " + std::to_string(node_id)); // 抛出异常
            return *node; // 返回节点的副本
        }, py::arg("node_id"), py::return_value_policy::copy, // 返回策略：副本
           "Returns a copy of the node with the specified ID.") // 方法文档字符串
        .def("get_cell", [](const Mesh_cpp& self, int cell_id) { // 通过ID获取单元 (返回副本)
            const Cell_cpp* cell = self.get_cell_by_id(cell_id); // 调用C++ getter
            if (!cell) throw py::index_error("Cell ID not found: " + std::to_string(cell_id)); // 抛出异常
            return *cell; // 返回单元的副本
        }, py::arg("cell_id"), py::return_value_policy::copy, // 返回策略：副本
           "Returns a copy of the cell with the specified ID.") // 方法文档字符串
        .def("get_half_edge", [](const Mesh_cpp& self, int he_id) { // 通过ID获取半边 (返回副本)
            const HalfEdge_cpp* he = self.get_half_edge_by_id(he_id); // 调用C++ getter
            if (!he) throw py::index_error("HalfEdge ID not found: " + std::to_string(he_id)); // 抛出异常
            return *he; // 返回半边的副本
        }, py::arg("he_id"), py::return_value_policy::copy, // 返回策略：副本
           "Returns a copy of the half-edge with the specified ID."); // 方法文档字符串


    py::class_<PrimitiveVars_cpp>(m, "PrimitiveVars_cpp") // 绑定 PrimitiveVars_cpp 结构体
        .def(py::init<double, double, double>(), // 绑定构造函数
             py::arg("h") = 0.0, py::arg("u") = 0.0, py::arg("v") = 0.0) // 参数名及默认值
        .def_readwrite("h", &PrimitiveVars_cpp::h) // 绑定 h 属性
        .def_readwrite("u", &PrimitiveVars_cpp::u) // 绑定 u 属性
        .def_readwrite("v", &PrimitiveVars_cpp::v); // 绑定 v 属性

    py::enum_<RiemannSolverType_cpp>(m, "RiemannSolverType_cpp") // 绑定黎曼求解器枚举
        .value("HLLC", RiemannSolverType_cpp::HLLC) // HLLC求解器
        .export_values(); // 导出枚举值到模块

    py::enum_<ReconstructionScheme_cpp>(m, "ReconstructionScheme_cpp") // 绑定重构方案枚举
        .value("FIRST_ORDER", ReconstructionScheme_cpp::FIRST_ORDER) // 一阶
        .value("SECOND_ORDER_LIMITED", ReconstructionScheme_cpp::SECOND_ORDER_LIMITED) // 二阶限制
        .export_values(); // 导出枚举值到模块

    py::enum_<TimeScheme_cpp>(m, "TimeScheme_cpp") // 绑定时间积分方案枚举
        .value("FORWARD_EULER", TimeScheme_cpp::FORWARD_EULER) // 前向欧拉
        .value("RK2_SSP", TimeScheme_cpp::RK2_SSP) // 二阶SSP RK
        .export_values(); // 导出枚举值到模块

    py::enum_<BoundaryType_cpp>(m, "BoundaryType_cpp") // 绑定边界类型枚举
        .value("WALL", BoundaryType_cpp::WALL) // 墙体
        .value("WATERLEVEL_TIMESERIES", BoundaryType_cpp::WATERLEVEL_TIMESERIES) // 水位时间序列
        .value("TOTAL_DISCHARGE_TIMESERIES", BoundaryType_cpp::TOTAL_DISCHARGE_TIMESERIES) // 总流量时间序列
        .value("FREE_OUTFLOW", BoundaryType_cpp::FREE_OUTFLOW) // 自由出流
        .value("UNDEFINED", BoundaryType_cpp::UNDEFINED) // 未定义
        .export_values(); // 导出枚举值到模块

    py::class_<BoundaryDefinition_cpp>(m, "BoundaryDefinition_cpp") // 绑定边界定义结构体
        .def(py::init<>()) // 默认构造函数
        .def_readwrite("type", &BoundaryDefinition_cpp::type); // 绑定 type 属性

    py::class_<TimeseriesPoint_cpp>(m, "TimeseriesPoint_cpp") // 绑定时间序列点结构体
        .def(py::init<>()) // 默认构造函数
        .def(py::init<double, double>(), py::arg("time"), py::arg("value")) // 带参数构造函数
        .def_readwrite("time", &TimeseriesPoint_cpp::time) // 绑定 time 属性
        .def_readwrite("value", &TimeseriesPoint_cpp::value); // 绑定 value 属性

    // --- 绑定计算组件 (即使主要由Core内部使用，绑定它们有助于测试) ---
    py::class_<FluxCalculator_cpp>(m, "FluxCalculator_cpp") // 绑定 FluxCalculator_cpp 类
        .def(py::init<double, double, RiemannSolverType_cpp>(), // 绑定构造函数
             py::arg("gravity"), py::arg("min_depth"), // 参数
             py::arg("solver_type") = RiemannSolverType_cpp::HLLC) // 参数及默认值
        .def("calculate_hllc_flux", &FluxCalculator_cpp::calculate_hllc_flux, // 绑定计算HLLC通量方法
             py::arg("W_L"), py::arg("W_R"), py::arg("normal_vec"), // 参数
             "Calculates HLLC flux."); // 方法文档字符串

    py::class_<SourceTermCalculator_cpp>(m, "SourceTermCalculator_cpp") // *** 新增绑定 ***
        .def(py::init<double, double>(), // 绑定构造函数
             py::arg("gravity"), py::arg("min_depth_param")) // 参数
        .def("apply_friction_semi_implicit_all_cells", // 绑定应用摩擦方法
             &SourceTermCalculator_cpp::apply_friction_semi_implicit_all_cells, // 方法指针
             py::arg("U_input_all"), py::arg("U_coeffs_all"), // 参数
             py::arg("dt"), py::arg("manning_n_values_all"), // 参数
             "Applies semi-implicit friction to all cells."); // 方法文档字符串

    py::class_<VFRCalculator_cpp>(m, "VFRCalculator_cpp") // *** 新增绑定 ***
        .def(py::init<double, double, int, double>(), // 绑定构造函数
             py::arg("min_depth_param") = 1e-6, // 参数及默认值
             py::arg("min_eta_change_iter_param") = 1e-6, // 参数及默认值
             py::arg("max_vfr_iters_param") = 20, // 参数及默认值
             py::arg("relative_h_tolerance_param") = 1e-4) // 参数及默认值
        .def("get_h_from_eta", &VFRCalculator_cpp::get_h_from_eta, // 绑定 get_h_from_eta 方法
             py::arg("eta"), py::arg("b_sorted_vertices"), // 参数
             py::arg("cell_total_area"), py::arg("cell_id_for_debug") = "", // 参数及默认值
             "Calculates h_avg from eta using simplified VFR formulas.") // 方法文档字符串
        .def("get_eta_from_h", &VFRCalculator_cpp::get_eta_from_h, // 绑定 get_eta_from_h 方法
             py::arg("h_avg"), py::arg("b_sorted_vertices"), // 参数
             py::arg("cell_nodes_sorted"), py::arg("cell_total_area"), // 参数
             py::arg("eta_previous_guess"), py::arg("cell_id_for_debug") = "", // 参数及默认值
             "Calculates eta from h_avg using Newton's method with VFR."); // 方法文档字符串

    py::class_<Reconstruction_cpp>(m, "Reconstruction_cpp") // 绑定 Reconstruction_cpp 类
        .def(py::init<ReconstructionScheme_cpp, const Mesh_cpp*, double, double>(), // 绑定构造函数
             py::arg("scheme"), py::arg("mesh_ptr"), py::arg("gravity"), py::arg("min_depth_param")) // 参数
        .def("prepare_for_step", &Reconstruction_cpp::prepare_for_step, // 绑定 prepare_for_step 方法
             py::arg("U_state_all")) // 参数
        .def("get_reconstructed_interface_states", &Reconstruction_cpp::get_reconstructed_interface_states, // 绑定获取重构界面状态方法
             py::arg("U_state_all"), py::arg("cell_L_id"), py::arg("cell_R_id"), // 参数
             py::arg("half_edge_L_to_R"), py::arg("is_boundary")); // 参数

    py::class_<TimeIntegrator_cpp>(m, "TimeIntegrator_cpp") // 绑定 TimeIntegrator_cpp 类
        .def(py::init<TimeScheme_cpp, TimeIntegrator_cpp::RHSFunction, TimeIntegrator_cpp::FrictionFunction, int>(), // 绑定构造函数
             py::arg("scheme"), py::arg("rhs_func"), py::arg("friction_func"), // 参数
             py::arg("num_vars_per_cell") = 3) // 参数及默认值
        .def("step", &TimeIntegrator_cpp::step, // 绑定 step 方法
             py::arg("U_current"), py::arg("dt"), py::arg("time_current")); // 参数

    py::class_<BoundaryConditionHandler_cpp>(m, "BoundaryConditionHandler_cpp") // *** 新增绑定 (如果需要单独测试) ***
        .def(py::init<const Mesh_cpp*, FluxCalculator_cpp*, Reconstruction_cpp*, double, double>(), // 绑定构造函数
             py::arg("mesh_ptr"), py::arg("flux_calc_ptr"), py::arg("recon_ptr"), // 参数
             py::arg("gravity_param"), py::arg("min_depth_param")) // 参数
        .def("set_boundary_definitions", &BoundaryConditionHandler_cpp::set_boundary_definitions, // 绑定设置边界定义方法
             py::arg("definitions")) // 参数
        .def("set_waterlevel_timeseries_data", &BoundaryConditionHandler_cpp::set_waterlevel_timeseries_data, // 绑定设置水位时间序列方法
             py::arg("ts_data")) // 参数
        .def("set_discharge_timeseries_data", &BoundaryConditionHandler_cpp::set_discharge_timeseries_data, // 绑定设置流量时间序列方法
             py::arg("ts_data")) // 参数
        .def("calculate_boundary_flux", &BoundaryConditionHandler_cpp::calculate_boundary_flux, // 绑定计算边界通量方法
             py::arg("U_state_all"), py::arg("cell_L_id"), // 参数
             py::arg("he"), py::arg("time_current")); // 参数


    // --- 绑定核心模型类 HydroModelCore_cpp ---
    py::class_<HydroModelCore_cpp>(m, "HydroModelCore_cpp") // 绑定 HydroModelCore_cpp 类
        .def(py::init<>()) // 绑定无参构造函数

        .def("initialize_model_from_files", &HydroModelCore_cpp::initialize_model_from_files, // 绑定核心初始化方法
            py::arg("node_filepath"), py::arg("cell_filepath"), py::arg("edge_filepath"), // 文件路径参数
            py::arg("cell_manning_values"), // 曼宁值列表参数
            py::arg("gravity"), py::arg("min_depth"), py::arg("cfl"), // 模拟参数
            py::arg("total_t"), py::arg("output_dt_interval"), py::arg("max_dt_val"), // 时间参数
            py::arg("recon_scheme"), py::arg("riemann_solver"), py::arg("time_scheme"), // 数值方案参数
            "Initializes the mesh from files and sets up simulation parameters and schemes.") // 方法文档字符串

        .def("set_initial_conditions_py", // 绑定从NumPy设置初始条件的方法
            [](HydroModelCore_cpp &self, py::array_t<double> U_initial_np) { // lambda函数包装
                if (U_initial_np.ndim() != 2 || U_initial_np.shape(1) != 3) { // 检查输入维度和形状
                    throw std::runtime_error("Initial conditions NumPy array must be Nx3."); // 抛出错误
                }
                std::vector<std::array<double, 3>> U_initial_vec(U_initial_np.shape(0)); // 创建C++向量
                auto r = U_initial_np.unchecked<2>(); // 获取NumPy数组访问器
                for (py::ssize_t i = 0; i < U_initial_np.shape(0); i++) { // 遍历NumPy数组
                    U_initial_vec[i] = {r(i, 0), r(i, 1), r(i, 2)}; // 复制数据到C++向量
                }
                self.set_initial_conditions_cpp(U_initial_vec); // 调用C++内部方法
            }, py::arg("U_initial_np"), "Sets initial conditions from a NumPy array.") // 参数名和文档字符串

        .def("setup_boundary_conditions_cpp", &HydroModelCore_cpp::setup_boundary_conditions_cpp, // 绑定设置边界条件的方法
             py::arg("bc_definitions"), py::arg("waterlevel_ts_data"), py::arg("discharge_ts_data"), // 参数 (使用之前绑定的结构体和枚举)
             "Configures boundary conditions and timeseries data.") // 方法文档字符串

        .def("get_gravity", &HydroModelCore_cpp::get_gravity, // 获取重力加速度
            "Returns the gravity value used.") // 方法文档字符串
        .def("get_min_depth", &HydroModelCore_cpp::get_min_depth, // 获取最小水深
            "Returns the minimum depth threshold used.") // 方法文档字符串
        .def("advance_one_step", &HydroModelCore_cpp::advance_one_step, // 执行一步积分
            "Advances the simulation by one time step, returns True if simulation should continue.") // 方法文档字符串
        .def("run_simulation_to_end", &HydroModelCore_cpp::run_simulation_to_end, // 运行到结束
            "Runs the simulation until the total time is reached.") // 方法文档字符串
        .def("get_current_time", &HydroModelCore_cpp::get_current_time, // 获取当前时间
            "Returns the current simulation time.") // 方法文档字符串
        .def("get_step_count", &HydroModelCore_cpp::get_step_count, // 获取步数
            "Returns the current simulation step count.") // 方法文档字符串
        .def("is_simulation_finished", &HydroModelCore_cpp::is_simulation_finished, // 判断是否结束
            "Returns True if the simulation has reached the total time.") // 方法文档字符串
        .def("get_total_time", &HydroModelCore_cpp::get_total_time, // 获取总时间
            "Returns the total simulation time.") // 方法文档字符串
        .def("get_output_dt", &HydroModelCore_cpp::get_output_dt, // 获取输出间隔
            "Returns the output time interval.") // 方法文档字符串
        .def("get_last_dt", &HydroModelCore_cpp::get_last_dt, // 获取上一时间步长
            "Returns the actual time step used in the last call to advance_one_step.") // 方法文档字符串
        .def("get_mesh_ptr", &HydroModelCore_cpp::get_mesh_ptr, // *** 改回 get_mesh_ptr ***
            py::return_value_policy::reference_internal,
            "Returns a const pointer to the internal mesh object (for inspection only).") // 方法文档字符串

        .def("get_U_state_all_py", // 获取U状态 (NumPy)
            [](const HydroModelCore_cpp &self) { // lambda函数
                StateVector u_vec = self.get_U_state_all_internal_copy(); // 获取C++数据副本
                if (u_vec.empty()) { // 处理空情况
                    // *** 显式指定 C_STYLE ***
                    return py::array_t<double, py::array::c_style>({0,3}); // 返回空Nx3 C风格数组
                } // 结束空处理
                // *** 显式指定 C_STYLE ***
                py::array_t<double, py::array::c_style> result_np({(py::ssize_t)u_vec.size(), (py::ssize_t)3}); // 创建 C 风格 NumPy 数组
                auto buf = result_np.request(); // 获取缓冲区信息
                double *ptr = static_cast<double *>(buf.ptr); // 获取指针
                // 使用 C++ 向量的 data() 获取连续内存指针，确保数据是连续存储的
                if (!u_vec.empty()) { // 再次检查非空，避免对空向量调用 data()
                    std::memcpy(ptr, u_vec.data()->data(), u_vec.size() * 3 * sizeof(double)); // 使用 data()->data() 获取底层指针
                } // 结束拷贝
                return result_np; // 返回NumPy数组
            }, "Returns a NumPy array copy of the current U state [h, hu, hv].") // 方法文档字符串

        .def("get_eta_previous_py", // 获取eta状态 (NumPy)
            [](const HydroModelCore_cpp &self) { // lambda函数
                std::vector<double> eta_vec = self.get_eta_previous_internal_copy(); // 获取C++数据副本
                 if (eta_vec.empty()) { // 处理空情况
                    // *** 显式指定 C_STYLE ***
                    return py::array_t<double, py::array::c_style>(0); // 返回空一维 C 风格数组
                } // 结束空处理
                // *** 显式指定 C_STYLE ***
                py::array_t<double, py::array::c_style> result_np(eta_vec.size()); // 创建 C 风格 NumPy 数组
                auto buf = result_np.request(); // 获取缓冲区信息
                double *ptr = static_cast<double *>(buf.ptr); // 获取指针
                if (!eta_vec.empty()){ // 检查非空
                     std::memcpy(ptr, eta_vec.data(), eta_vec.size() * sizeof(double)); // 内存拷贝
                } // 结束拷贝
                return result_np; // 返回NumPy数组
            }, "Returns a NumPy array copy of the previous water surface elevation (eta) state."); // 方法文档字符串
} // 结束模块定义