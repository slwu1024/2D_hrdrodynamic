// src_cpp/include/BoundaryConditionHandler_cpp.h
#ifndef BOUNDARYCONDITIONHANDLER_CPP_H // 防止头文件重复包含
#define BOUNDARYCONDITIONHANDLER_CPP_H // 定义头文件宏

#include <vector> // 包含vector容器
#include <string> // 包含string类
#include <map>    // 包含map容器 (用于存储边界定义和时间序列)
#include <array>  // 包含array容器 (用于通量和坐标)
#include <functional> // 可能用于某些高级策略，但暂时不用
#include <memory>   // 包含智能指针 (如果持有指向其他组件的指针)

#include "MeshData_cpp.h"       // Mesh_cpp, Cell_cpp, HalfEdge_cpp, Node_cpp
#include "TimeIntegrator_cpp.h"
#include "FluxCalculator_cpp.h" // FluxCalculator_cpp, PrimitiveVars_cpp
#include "Reconstruction_cpp.h" // Reconstruction_cpp (可能需要其进行边界重构)

namespace HydroCore { // 定义HydroCore命名空间

// 定义边界类型枚举 (与Python版本对应)
enum class BoundaryType_cpp { // 定义边界类型枚举
    WALL,                     // 墙体边界
    WATERLEVEL,    // 水位时间序列边界
    TOTAL_DISCHARGE, // 总流量时间序列边界
    FREE_OUTFLOW,             // 自由出流边界
    UNDEFINED                 // 未定义或默认 (可能也视为墙体)
}; // 结束枚举定义

// 存储单个边界标记的配置
struct BoundaryDefinition_cpp { // 定义边界定义结构体
    BoundaryType_cpp type = BoundaryType_cpp::WALL; // 边界类型，默认为墙体
    // 新增：流量方向提示 (可选)
    bool has_flow_direction_hint = false;         // 是否有流量方向提示
    double flow_direction_hint_x = 0.0;           // x方向分量
    double flow_direction_hint_y = 0.0;           // y方向分量
}; // 结束结构体定义

// 时间序列数据点
struct TimeseriesPoint_cpp { // 定义时间序列数据点结构体
    double time;  // 时间
    double value; // 值 (例如水位或总流量)
}; // 结束结构体定义

class BoundaryConditionHandler_cpp { // 定义边界条件处理器类
public: // 公有成员
    BoundaryConditionHandler_cpp( // 构造函数
        const Mesh_cpp* mesh_ptr_param,              // 指向网格对象的指针
        FluxCalculator_cpp* flux_calc_ptr_param,     // 指向通量计算器的指针
        Reconstruction_cpp* recon_ptr_param,         // 指向重构器的指针
        double gravity_param,                        // 重力加速度
        double min_depth_param                       // 最小水深
    ); // 结束构造函数声明

    // 设置边界定义 (从Python传递的配置)
    // key: 边界标记 (int), value: BoundaryDefinition_cpp
    void set_boundary_definitions(const std::map<int, BoundaryDefinition_cpp>& definitions); // 设置边界定义方法

    // 设置时间序列数据 (从Python传递)
    // key: 边界标记 (int)
    // value: 该标记对应的时间序列数据 (vector of TimeseriesPoint_cpp)
    void set_waterlevel_timeseries_data(const std::map<int, std::vector<TimeseriesPoint_cpp>>& ts_data); // 设置水位时间序列数据方法
    void set_discharge_timeseries_data(const std::map<int, std::vector<TimeseriesPoint_cpp>>& ts_data); // 设置流量时间序列数据方法

    // 计算指定边界半边的数值通量
    // U_L_state: 内部单元的守恒量 [h, hu, hv]
    // he: 边界半边对象 (从内部单元指向外部)
    // time_current: 当前模拟时间
    // 返回值: 笛卡尔坐标系下的通量 [Fh, Fhu, Fhv]
    std::array<double, 3> calculate_boundary_flux( // 计算边界通量方法
        const StateVector & U_state_all, // 接收所有单元的状态 (StateVector是std::vector<std::array<double,3>>)
        int cell_L_id,                  // 内部单元的ID
        const HalfEdge_cpp& he,
        double time_current
    ) const; // const成员函数，理想情况下不修改内部状态，但插值可能涉及查找

private: // 私有成员
    // 辅助函数：将守恒量转换为原始变量
    PrimitiveVars_cpp conserved_to_primitive(const std::array<double, 3>& U_cell) const; // 守恒量转原始量

    // 辅助函数：从存储的时间序列数据中获取特定时间的值 (线性插值)
    double get_timeseries_value(int marker, double time_current, BoundaryType_cpp type_for_map_selection) const; // 获取时间序列值

    // 不同边界类型的具体处理函数
    // --- 修改以下函数的声明 ---
    std::array<double, 3> handle_wall_boundary( // 处理墙体边界
        const PrimitiveVars_cpp& W_L_reconstructed_iface, // 参数1: 重构后的左侧界面原始变量
        const HalfEdge_cpp& he                            // 参数2: 边界半边
    ) const; // const成员函数

    std::array<double, 3> handle_free_outflow_boundary( // 处理自由出流边界
        const PrimitiveVars_cpp& W_L_reconstructed_iface, // 参数1: 重构后的左侧界面原始变量
        const HalfEdge_cpp& he                            // 参数2: 边界半边
    ) const; // const成员函数

    std::array<double, 3> handle_waterlevel_boundary( // 处理水位边界
        const StateVector& U_state_all,                   // 参数1: 所有单元的当前守恒状态 (用于水位外插)
        int cell_L_id,                                    // 参数2: 内部单元的ID
        const PrimitiveVars_cpp& W_L_reconstructed_iface, // 参数3: 重构后的左侧界面原始变量
        const HalfEdge_cpp& he,                           // 参数4: 边界半边
        double target_eta                                 // 参数5: 目标水位
    ) const; // const成员函数

    std::array<double, 3> handle_total_discharge_boundary( // 处理总流量边界
        const StateVector& U_state_all,                   // 参数1: 所有单元的当前守恒状态
        int cell_L_id,                                    // 参数2: 内部单元的ID
        const PrimitiveVars_cpp& W_L_reconstructed_iface, // 参数3: 重构后的左侧界面原始变量
        const HalfEdge_cpp& he,                           // 参数4: 边界半边
        double target_Q_total,                            // 参数5: 目标总流量
        int original_segment_id_for_length, // 参数名与之前一致
        const BoundaryDefinition_cpp& bc_def_for_hint // 新增参数
    ) const; // const成员函数
    // 预处理边界边，计算每个标记的总长度等 (如果需要)
    void preprocess_boundaries(); // 预处理边界 (可选，如果流量分配需要)

    const Mesh_cpp* mesh_ptr_internal;              // 指向网格对象的指针
    FluxCalculator_cpp* flux_calculator_ptr_internal; // 指向通量计算器的指针
    Reconstruction_cpp* reconstruction_ptr_internal; // 指向重构器的指针
    double g_internal;                                // 重力加速度
    double min_depth_internal;                        // 最小水深

    std::map<int, BoundaryDefinition_cpp> bc_definitions_internal; // 存储边界定义
    std::map<int, std::vector<TimeseriesPoint_cpp>> waterlevel_ts_data_internal; // 水位时间序列数据
    std::map<int, std::vector<TimeseriesPoint_cpp>> discharge_ts_data_internal;  // 流量时间序列数据

    // (可选) 预计算的每个标记的边界总长度，用于流量分配
    std::map<int, double> marker_total_lengths_internal; // 标记总长度

    double epsilon; // 小量，用于浮点比较
}; // 结束类定义

} // namespace HydroCore
#endif //BOUNDARYCONDITIONHANDLER_CPP_H // 结束头文件宏