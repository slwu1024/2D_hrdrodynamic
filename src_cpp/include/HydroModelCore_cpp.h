// src_cpp/include/HydroModelCore_cpp.h
#ifndef HYDROMODELCORE_CPP_H
#define HYDROMODELCORE_CPP_H

#include <vector> // 包含vector容器
#include <array>  // 包含array容器
#include <string> // 包含string类
#include <memory> // 包含智能指针
#include <map>    // 包含map容器
#include <functional> // 包含std::function
#include <set>    // 新增：为 setup_internal_flow_source 中的 processed_physical_edges
#include <algorithm> // 新增: 为了 std::sort, std::lower_bound
#include <limits>    // 新增: 为了 std::numeric_limits

#include "MeshData_cpp.h" // 网格数据结构
#include "FluxCalculator_cpp.h" // 通量计算器
#include "SourceTerms_cpp.h"    // 源项计算器
#include "WettingDrying_cpp.h"  // VFR计算器
#include "Reconstruction_cpp.h" // 重构器
#include "TimeIntegrator_cpp.h" // 时间积分器
#include "BoundaryConditionHandler_cpp.h" // 边界条件处理器

namespace HydroCore { // HydroCore命名空间开始

using StateVector = std::vector<std::array<double, 3>>; // 定义状态向量类型别名

// --- 新增：用于点源信息的结构体 ---
struct PointSourceInfo_cpp { // 定义点源信息结构体
    std::string name;        // 点源名称
    int target_cell_id;    // 施加点源的目标单元ID (如果为-1，表示未找到单元)
    std::array<double, 2> original_coordinates; // 存储配置文件中指定的原始坐标，用于调试或记录
    std::vector<TimeseriesPoint_cpp> q_timeseries; // 流量时程 (m^3/s), 正值注入，负值抽取
};
// --- 点源信息结构体结束 ---

class HydroModelCore_cpp { // 水动力模型核心类
public: // 公有成员
    HydroModelCore_cpp(); // 修改后的构造函数 (不再接收Mesh_cpp*)
    ~HydroModelCore_cpp(); // 析构函数

    // 新的初始化方法，负责创建和加载网格，并初始化其他组件
    void initialize_model_from_files( // 从文件初始化模型
        const std::string& node_filepath, // 节点文件路径
        const std::string& cell_filepath, // 单元文件路径
        const std::string& edge_filepath, // 边文件路径
        const std::vector<double>& cell_manning_values, // 单元曼宁系数值
        // 以下是原set_simulation_parameters 和 set_numerical_schemes的参数
        double gravity, double min_depth, double cfl, // 模拟参数
        double total_t, double output_dt_interval, double max_dt_val, // 时间参数
        ReconstructionScheme_cpp recon_scheme, // 数值方案
        RiemannSolverType_cpp riemann_solver,
        TimeScheme_cpp time_scheme
    );
    void setup_internal_flow_source(
        const std::string& line_name,                         // 新增：内部流量线的唯一名称
        const std::vector<int>& poly_node_ids_for_line_py,    // .poly文件中的1-based节点ID
        const std::vector<TimeseriesPoint_cpp>& q_timeseries, // 该线的流量时程数据
        const std::array<double, 2>& direction_py           // 流量方向
    );
    // --- 新增：公开的设置内部点源的方法 ---
    void setup_internal_point_source_cpp( // 新增：设置内部点源 (C++)
        const std::string& name,
        const std::array<double, 2>& coordinates, // 从Python接收坐标
        const std::vector<TimeseriesPoint_cpp>& q_timeseries // 对应的流量时程
    );
    // --- 设置内部点源方法结束 ---

    void set_initial_conditions_cpp(const StateVector& U_initial); // 设置初始条件(C++)
    void setup_boundary_conditions_cpp( // 设置边界条件(C++)
        const std::map<int, BoundaryDefinition_cpp>& bc_definitions,
        const std::map<int, std::vector<TimeseriesPoint_cpp>>& waterlevel_ts_data,
        const std::map<int, std::vector<TimeseriesPoint_cpp>>& discharge_ts_data
    );

    double get_gravity() const { return gravity_internal; } // 获取重力加速度
    double get_min_depth() const { return min_depth_internal; } // 获取最小水深

    bool advance_one_step(); // 执行一个完整的时间步
    void run_simulation_to_end(); // 执行整个模拟循环

    double get_current_time() const { return current_time_internal; } // 获取当前时间
    int get_step_count() const { return step_count_internal; } // 获取步数
    bool is_simulation_finished() const { return current_time_internal >= total_time_internal - epsilon; } // 判断模拟是否结束
    double get_total_time() const { return total_time_internal; } // 获取总模拟时长
    double get_output_dt() const { return output_dt_internal; } // 获取输出时间间隔
    double get_last_dt() const { return last_calculated_dt_internal; } // 获取上一个计算的dt

    StateVector get_U_state_all_internal_copy() const { return U_state_all_internal; } // 获取内部守恒量副本
    std::vector<double> get_eta_previous_internal_copy() const { return eta_previous_internal; } // 获取内部上一时刻水位副本
    const Mesh_cpp* get_mesh_ptr() const { return mesh_internal_ptr.get(); } // 获取网格指针 (const)


public: // 回调函数 (保持public或通过其他方式传递给TimeIntegrator)
    StateVector _calculate_rhs_explicit_part_internal(const StateVector& U_current_rhs, double time_current_rhs); // 计算显式RHS(内部)
    StateVector _apply_friction_semi_implicit_internal(const StateVector& U_input_friction, // 应用半隐式摩擦(内部)
                                                   const StateVector& U_coeffs_friction, double dt_friction);
private: // 私有成员
    double _calculate_dt_internal(); // 计算CFL时间步长(内部)
    void _handle_dry_cells_and_update_eta_internal(); // 处理干单元并更新水位(内部)
    void _initialize_manning_from_mesh_internal(); // 从网格初始化曼宁系数(内部)
    StateVector _apply_internal_flow_source_terms(const StateVector& U_input, double dt, double time_current); // 修改：参数名与实现匹配

    // --- 新增：私有的应用内部点源的方法 ---
    StateVector _apply_point_sources_internal(const StateVector& U_input, double dt, double time_current); // 新增：应用点源 (内部)
    // --- 应用内部点源方法结束 ---
    double get_timeseries_value_internal(const std::vector<TimeseriesPoint_cpp>& series, double time_current) const;

    std::unique_ptr<Mesh_cpp> mesh_internal_ptr; // HydroModelCore拥有Mesh对象
    std::unique_ptr<FluxCalculator_cpp> flux_calculator_ptr; // 通量计算器指针
    std::unique_ptr<SourceTermCalculator_cpp> source_term_calculator_ptr; // 源项计算器指针
    std::unique_ptr<Reconstruction_cpp> reconstruction_ptr; // 重构器指针
    std::unique_ptr<VFRCalculator_cpp> vfr_calculator_ptr; // VFR计算器指针
    std::unique_ptr<TimeIntegrator_cpp> time_integrator_ptr; // 时间积分器指针
    std::unique_ptr<BoundaryConditionHandler_cpp> boundary_handler_ptr; // 边界条件处理器指针

    // --- 内部流量源项相关的新结构和成员变量 ---
    struct InternalFlowEdgeInfo {       // 定义一个结构体来存储每条内部流量边的信息
        int source_cell_id;             // 源单元的ID
        int sink_cell_id;               // 汇单元的ID
        const Cell_cpp* source_cell_ptr; // 指向源单元的指针 (在setup时填充)
        const Cell_cpp* sink_cell_ptr;   // 指向汇单元的指针 (在setup时填充)
        double edge_length;             // 这条边的长度
    };
    // 使用 map 来存储每条命名内部流量线的信息
    std::map<std::string, std::vector<InternalFlowEdgeInfo>> all_internal_flow_edges_info_map_internal; // 修改: map的value是vector
    std::map<std::string, std::vector<TimeseriesPoint_cpp>> internal_q_timeseries_data_map_internal;  // 存储流量时程
    std::map<std::string, double> internal_flow_line_total_lengths_map_internal;                     // 存储总长度
    std::map<std::string, std::array<double, 2>> internal_flow_directions_map_internal;              // 存储方向

    // --- 新增：用于存储所有点源配置的成员变量 ---
    std::vector<PointSourceInfo_cpp> internal_point_sources_info_internal; // 新增：存储所有点源信息
    // --- 点源配置成员变量结束 ---

    StateVector U_state_all_internal; // 内部存储的守恒量
    std::vector<double> eta_previous_internal; // 内部存储的上一时刻水位
    std::vector<double> manning_n_values_internal; // 内部存储的曼宁系数值

    double gravity_internal; // 重力加速度
    double min_depth_internal; // 最小水深
    double cfl_number_internal; // CFL数
    double total_time_internal; // 总模拟时长
    double output_dt_internal; // 输出时间间隔
    double max_dt_internal; // 最大时间步长
    double current_time_internal; // 当前模拟时间
    int step_count_internal; // 当前步数
    double epsilon; // 小量
    double last_calculated_dt_internal; // 上一个计算的dt

    ReconstructionScheme_cpp current_recon_scheme; // 当前重构方案
    RiemannSolverType_cpp current_riemann_solver; // 当前黎曼求解器
    TimeScheme_cpp current_time_scheme; // 当前时间积分方案

    bool model_fully_initialized_flag; // 标记模型是否已完全初始化
    bool initial_conditions_set_flag; // 标记初始条件是否已设置
    bool boundary_conditions_set_flag; // 标记边界条件是否已设置
}; // 结束类定义

} // namespace HydroCore
#endif //HYDROMODELCORE_CPP_H