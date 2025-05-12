// src_cpp/include/FluxCalculator_cpp.h
#ifndef FLUXCALCULATOR_CPP_H // 防止头文件重复包含
#define FLUXCALCULATOR_CPP_H // 定义头文件宏

#include <vector> // 包含vector容器
#include <array>  // 包含array容器
#include <string> // 包含string类
#include <cmath>  // 包含数学函数如sqrt
#include <algorithm> // 包含算法如min, max

namespace HydroCore { // 定义HydroCore命名空间

    // 用于在C++内部清晰地表示原始变量，可以被Pybind11绑定层转换为Python对象
    struct PrimitiveVars_cpp { // 定义原始变量结构体
        double h; // 水深
        double u; // x方向速度
        double v; // y方向速度
    }; // 结束结构体定义

    enum class RiemannSolverType_cpp { // 定义黎曼求解器类型枚举
        HLLC // HLLC求解器
        // 以后可以添加其他类型
    }; // 结束枚举定义

    class FluxCalculator_cpp { // 定义通量计算器类
    public: // 公有成员
        FluxCalculator_cpp(double gravity, double min_depth_param, RiemannSolverType_cpp solver_type = RiemannSolverType_cpp::HLLC); // 构造函数

        // HLLC求解器，返回笛卡尔坐标系下的通量 [Fh, Fhu, Fhv]
        std::array<double, 3> calculate_hllc_flux( // 计算HLLC通量方法
            const PrimitiveVars_cpp& W_L,        // 左侧原始变量
            const PrimitiveVars_cpp& W_R,        // 右侧原始变量
            const std::array<double, 2>& normal_vec // 从左到右的单位法向量 [nx, ny]
        ); // 结束方法声明

        // 新增：公有 getter 方法
        RiemannSolverType_cpp get_solver_type() const { return solver_type_internal; } // 获取求解器类型

    private: // 私有成员
        double g;         // 重力加速度
        double min_depth; // 最小水深阈值
        RiemannSolverType_cpp solver_type_internal; // 内部存储的求解器类型
    }; // 结束类定义

} // namespace HydroCore
#endif //FLUXCALCULATOR_CPP_H // 结束头文件宏