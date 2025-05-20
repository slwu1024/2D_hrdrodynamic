// src_cpp/include/WettingDrying_cpp.h
#ifndef WETTINGDRYING_CPP_H // 防止头文件重复包含
#define WETTINGDRYING_CPP_H // 定义头文件宏

#include <vector> // 包含vector容器
#include <array>  // 包含array容器
#include <cmath>  // 包含数学函数如sqrt, pow, abs
#include <algorithm> // 包含算法如std::max, std::min, std::sort
#include <numeric>   // 包含数值操作如std::abs for collections (though cmath::abs is usually fine for doubles)

#include "MeshData_cpp.h" // 需要 Node_cpp 来获取顶点坐标和高程

namespace HydroCore { // 定义HydroCore命名空间

class VFRCalculator_cpp { // 定义VFR计算器类
public: // 公有成员
    VFRCalculator_cpp(double min_depth_param = 1e-6, // 构造函数，参数及默认值
                      double min_eta_change_iter_param = 1e-6,
                      int max_vfr_iters_param = 20,
                      double relative_h_tolerance_param = 1e-4);

    // 根据水面高程 eta 计算单元平均水深 h_avg
    double get_h_from_eta( // 根据eta计算h的方法
        double eta, // 当前水面高程
        const std::vector<double>& b_sorted_vertices, // 已排序的顶点底高程 [b1, b2, b3]
        double cell_total_area, // 单元总面积
        const std::string& cell_id_for_debug = "" // 用于调试的单元ID (可选)
    ) const; // const成员函数，不修改对象状态

    // 根据单元平均水深 h_avg 计算水面高程 eta (通过牛顿法)
    double get_eta_from_h( // 根据h计算eta的方法
        double h_avg, // 单元平均水深
        const std::vector<double>& b_sorted_vertices, // 已排序的顶点底高程 [b1, b2, b3]
        const std::vector<Node_cpp>& cell_nodes_sorted, // 与b_sorted_vertices对应的已排序节点对象 (用于精确几何)
        double cell_total_area, // 单元总面积
        double eta_previous_guess, // 上一步的eta作为初始猜测 (重要：传入double)
        // 新增的参数需要在这里声明
        double current_sim_time, // 新增：当前模拟时间
        int cell_id_int          // 新增：当前单元的整数ID
    ) const; // const成员函数

    // 将其声明为静态成员函数
    static void set_internal_debug_conditions(bool enable, int target_cell_id, double target_time_min, double target_time_max); // 修改：声明为静态成员函数并重命名

private: // 私有成员
    // 辅助函数：计算给定eta下的水面面积 Aw (精确几何版)
    double calculate_wet_surface_area( // 计算水面面积的辅助方法
        double eta, // 当前水面高程
        const std::vector<double>& b_sorted_vertices, // 已排序的顶点底高程
        const std::vector<Node_cpp>& cell_nodes_sorted, // 与b_sorted_vertices对应的已排序节点对象
        double cell_total_area // 单元总面积
    ) const; // const成员函数

    // 辅助函数：线性插值 (用于 _calculate_wet_surface_area)
    std::array<double, 2> linear_interpolate( // 线性插值的辅助方法
        const std::array<double, 2>& p1_coords, double p1_z, // 点1坐标和高程
        const std::array<double, 2>& p2_coords, double p2_z, // 点2坐标和高程
        double target_z // 目标高程
    ) const; // const成员函数

    // 辅助函数：计算多边形面积 (用于 _calculate_wet_surface_area)
    double polygon_area(const std::vector<std::array<double, 2>>& vertices) const; // 计算多边形面积的辅助方法

    double min_depth; // 最小水深阈值
    double min_eta_change_iter; // eta迭代绝对收敛阈值
    int max_vfr_iters; // 最大迭代次数
    double relative_h_tolerance; // 相对水深误差收敛阈值
    double epsilon; // 用于避免除零的小量

    // 将调试变量声明为静态成员变量
    static bool s_vfr_internal_debug_enabled; // 修改：声明为静态成员变量
    static int s_vfr_debug_target_cell_id;    // 修改：声明为静态成员变量
    static double s_vfr_debug_target_time_min; // 修改：声明为静态成员变量
    static double s_vfr_debug_target_time_max;  // 修改：声明为静态成员变量
}; // 结束类定义

} // namespace HydroCore
#endif //WETTINGDRYING_CPP_H // 结束头文件宏