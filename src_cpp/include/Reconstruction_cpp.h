// src_cpp/include/Reconstruction_cpp.h
#ifndef RECONSTRUCTION_CPP_H // 防止头文件重复包含
#define RECONSTRUCTION_CPP_H // 定义头文件宏

#include <vector> // 包含vector容器
#include <array>  // 包含array容器
#include <string> // 包含string类
#include <cmath>  // 包含数学函数
#include <algorithm> // 包含算法

#include "MeshData_cpp.h" // 需要 Mesh_cpp, Cell_cpp, Node_cpp 等
#include "FluxCalculator_cpp.h" // 只需要 PrimitiveVars_cpp 结构体 (如果之前定义在那里)
                               // 或者在此文件内重新定义类似的结构体

namespace HydroCore { // 定义HydroCore命名空间

// 如果 PrimitiveVars_cpp 已经在 FluxCalculator_cpp.h 中定义且被包含，则不需要重复定义
// 否则，可以像这样定义一个类似的，或者直接使用 std::array<double, 3>
// struct PrimitiveState { double h, u, v; };

enum class ReconstructionScheme_cpp { // 定义重构方案枚举
    FIRST_ORDER, // 一阶
    SECOND_ORDER_LIMITED // 代表一个带限制器的二阶方法 (例如Barth-Jespersen)
}; // 结束枚举定义

class Reconstruction_cpp { // 定义重构类
public: // 公有成员
    Reconstruction_cpp( // 构造函数
        ReconstructionScheme_cpp scheme, // 重构方案
        const Mesh_cpp* mesh_ptr,      // 指向Mesh_cpp对象的指针 (const因为重构器不应修改网格)
        double gravity,                // 重力加速度
        double min_depth_param         // 最小水深
    ); // 结束构造函数声明

    // 在每个时间步开始时计算梯度和限制器 (仅高阶方法需要)
    // U_state_all 是当前所有单元的守恒量 [h, hu, hv]
    void prepare_for_step(const std::vector<std::array<double, 3>>& U_state_all); // 准备步骤声明

    // 新增：公有 getter 方法
    ReconstructionScheme_cpp get_scheme_type() const { return scheme_internal; } // 获取重构方案类型
    // 新增：公有 getter 方法，用于获取指定单元的（限制后的）梯度
    // 返回值：指向单元梯度的 const 引用，避免不必要的拷贝
    // 梯度格式：[var_idx(0:h,1:u,2:v)][dim(0:x,1:y)]
    const std::array<std::array<double, 2>, 3>& get_gradient_for_cell(int cell_id) const; // 获取指定单元的梯度

    // 获取界面左右两侧重构后的原始变量状态 W = [h, u, v]
    // U_state_all: 当前所有单元的守恒量
    // cell_L_id, cell_R_id: 左右单元的ID
    // he_L_origin_node_id, he_L_end_node_id: 左半边的起点和终点ID (用于确定界面位置)
    // normal_vec: 从L到R的法向量 (如果需要的话，但通常界面值重构不直接用法向量)
    // is_boundary: 标记是否为边界边
    // 返回值: pair 的第一个是 W_L_interface, 第二个是 W_R_interface (如果内部边) 或空的 W_R
    std::pair<PrimitiveVars_cpp, PrimitiveVars_cpp> get_reconstructed_interface_states( // 获取重构界面状态声明
        const std::vector<std::array<double, 3>>& U_state_all, // 所有单元的守恒量
        int cell_L_id, // 左单元ID
        int cell_R_id, // 右单元ID (-1 如果是边界)
        const HalfEdge_cpp& half_edge_L_to_R, // 从L指向R(或外部)的半边对象 (const引用)
        bool is_boundary // 是否为边界
    ) const; // const成员函数

private: // 私有成员
    // 辅助函数：守恒量转原始量
    PrimitiveVars_cpp conserved_to_primitive(const std::array<double, 3>& U_cell) const; // 守恒量转原始量
    // 辅助函数：原始量转守恒量 (如果需要)
    // std::array<double, 3> primitive_to_conserved(const PrimitiveVars_cpp& W_cell) const;

    // 辅助函数：计算所有单元的原始变量
    std::vector<PrimitiveVars_cpp> get_all_primitive_states( // 获取所有原始状态
        const std::vector<std::array<double, 3>>& U_state_all
    ) const; // const成员函数

    // (高阶) 计算无限制梯度 (Green-Gauss)
    // W_state_all 是所有单元的原始变量
    // 返回值: gradients_all[cell_idx][var_idx_primitive(0:h,1:u,2:v)][dim(0:x,1:y)]
    std::vector<std::array<std::array<double, 2>, 3>> calculate_gradients_green_gauss( // 计算Green-Gauss梯度
        const std::vector<PrimitiveVars_cpp>& W_state_all
    ) const; // const成员函数

    // (高阶) 计算Barth-Jespersen限制器 phi (0到1)
    // 返回值: limiters_phi_all[cell_idx][var_idx_primitive]
    std::vector<std::array<double, 3>> calculate_barth_jespersen_limiters( // 计算Barth-Jespersen限制器
        const std::vector<PrimitiveVars_cpp>& W_state_all, // 所有单元的原始变量
        const std::vector<std::array<std::array<double, 2>, 3>>& unlimited_gradients_all // 所有单元的无限制梯度
    ) const; // const成员函数

    ReconstructionScheme_cpp scheme_internal; // 内部存储的重构方案
    const Mesh_cpp* mesh;                     // 指向网格对象的指针
    double g_internal;                        // 重力加速度
    double min_depth_internal;                // 最小水深

    // 存储每个单元、每个原始变量(h, u, v)的梯度 [grad_x, grad_y]
    // gradients[cell_id][var_idx][dim_idx]
    std::vector<std::array<std::array<double, 2>, 3>> gradients_primitive; // 存储原始变量的梯度
    // (注意：Python版本中 self.gradients 可能是直接作用于守恒量，这里我们选择作用于原始量，更常见)
    double epsilon; // 小量，用于浮点比较和避免除零
}; // 结束类定义

} // namespace HydroCore
#endif //RECONSTRUCTION_CPP_H // 结束头文件宏