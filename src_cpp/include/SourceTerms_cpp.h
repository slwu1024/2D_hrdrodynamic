// src_cpp/include/SourceTerms_cpp.h
#ifndef SOURCETERMS_CPP_H // 防止头文件重复包含
#define SOURCETERMS_CPP_H // 定义头文件宏

#include <vector> // 包含vector容器
#include <array>  // 包含array容器
#include <cmath>  // 包含数学函数如sqrt, pow
#include <algorithm> // 包含算法如std::max

// 如果 SourceTerms_cpp 需要 MeshData_cpp.h 中的结构或枚举，请在此处包含
// #include "MeshData_cpp.h" // (当前 SourceTerms 似乎不需要直接访问 Mesh 结构)

namespace HydroCore { // 定义HydroCore命名空间

    // 假设守恒量 U 是一个包含3个double的数组 [h, hu, hv]
    // 我们可以使用 std::array<double, 3> 或者一个简单的结构体
    // 为了与 FluxCalculator_cpp 中的 PrimitiveVars_cpp 保持某种对应性，
    // 但这里处理的是守恒量，所以暂时不定义新结构体，直接用 std::vector<std::array<double,3>> 或类似方式传递

    class SourceTermCalculator_cpp { // 定义源项计算器类
    public: // 公有成员
        SourceTermCalculator_cpp(double gravity, double min_depth_param); // 构造函数

        // 应用半隐式摩擦
        // U_input 和 U_coeffs 是守恒量 [h, hu, hv]
        // 输入: U_input_all (num_cells x 3), U_coeffs_all (num_cells x 3), manning_n_values (num_cells)
        // 输出: U_output_all (num_cells x 3)
        std::vector<std::array<double, 3>> apply_friction_semi_implicit_all_cells( // 应用半隐式摩擦 (所有单元)
            const std::vector<std::array<double, 3>>& U_input_all, // 输入状态 (所有单元)
            const std::vector<std::array<double, 3>>& U_coeffs_all, // 用于计算系数的状态 (所有单元)
            double dt, // 时间步长
            const std::vector<double>& manning_n_values_all // 每个单元的曼宁糙率
        ); // 结束方法声明

        // 如果以后添加底坡源项的显式计算，可以在这里添加方法
        // std::vector<std::array<double, 3>> calculate_bed_slope_term_all_cells(
        // const std::vector<std::array<double, 3>>& U_state_all,
        // const Mesh_cpp& mesh // 可能需要网格信息来获取底坡
        // );

    private: // 私有成员
        double g;         // 重力加速度
        double min_depth; // 最小水深阈值
    }; // 结束类定义

} // namespace HydroCore
#endif //SOURCETERMS_CPP_H // 结束头文件宏