// src_cpp/include/TimeIntegrator_cpp.h
#ifndef TIMEINTEGRATOR_CPP_H // 防止头文件重复包含
#define TIMEINTEGRATOR_CPP_H // 定义头文件宏

#include <vector> // 包含vector容器
#include <array>  // 包含array容器
#include <functional> // 包含std::function，用于存储可调用对象 (如成员函数指针或lambda)
#include <string> // 包含string类

// 我们需要一个前向声明或包含 HydroModelCore_cpp.h (如果它定义了状态类型和RHS函数签名)
// 暂时，我们可以假设状态是 std::vector<std::array<double, 3>>
// 并定义RHS和摩擦函数的签名

namespace HydroCore { // 定义HydroCore命名空间

// 定义时间积分方案枚举 (与Python版本对应)
enum class TimeScheme_cpp { // 定义时间积分方案枚举
    FORWARD_EULER, // 前向欧拉法
    RK2_SSP        // 二阶SSP龙格-库塔法
    // RK3_SSP 等可以后续添加
}; // 结束枚举定义

// 定义状态类型别名，方便使用
using StateVector = std::vector<std::array<double, 3>>; // 定义状态向量类型别名

// 前向声明核心模型类，以避免循环依赖 (如果RHS函数是其成员)
// class HydroModelCore_cpp; // 如果RHS和friction是HydroModelCore_cpp的成员函数，则需要它

class TimeIntegrator_cpp { // 定义时间积分器类
public: // 公有成员
    // RHS函数类型: (输入当前状态U, 当前时间t) -> 返回RHS贡献
    using RHSFunction = std::function<StateVector(const StateVector&, double)>; // 定义RHS函数类型

    // 摩擦函数类型: (输入待施加摩擦的状态U_in, 用于计算系数的状态U_coeffs, 时间步长dt) -> 返回施加摩擦后的状态U_out
    using FrictionFunction = std::function<StateVector(const StateVector&, const StateVector&, double)>; // 定义摩擦函数类型

    TimeIntegrator_cpp( // 构造函数
        TimeScheme_cpp scheme, // 时间积分方案
        RHSFunction rhs_func, // RHS计算函数
        FrictionFunction friction_func, // 摩擦计算函数
        int num_vars_per_cell = 3 // 每个单元的变量数 (通常是h, hu, hv)
    ); // 结束构造函数声明

    // 执行一个时间积分步骤
    // 返回下一时刻的状态
    StateVector step( // 执行一步时间积分的方法
        const StateVector& U_current, // 当前状态
        double dt, // 时间步长
        double time_current // 当前时间
    ) const; // const成员函数，不修改对象内部状态 (rhs_func和friction_func是外部提供的)

private: // 私有成员
    TimeScheme_cpp scheme_internal; // 内部存储的时间积分方案
    RHSFunction calculate_rhs_explicit_part; // 存储的RHS计算函数
    FrictionFunction apply_friction_semi_implicit; // 存储的摩擦计算函数
    int num_vars; // 每个单元的变量数
}; // 结束类定义

} // namespace HydroCore
#endif //TIMEINTEGRATOR_CPP_H // 结束头文件宏