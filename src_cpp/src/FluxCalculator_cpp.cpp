// src_cpp/src/FluxCalculator_cpp.cpp
#include "FluxCalculator_cpp.h" // 包含对应的头文件
#include "Profiler.h"
#include <iostream> // 包含输入输出流
#include <iomanip> // 为了 std::fixed 和 std::setprecision

// 定义一个简单的调试标志，可以在外部控制是否打印
// 在实际项目中，这可以通过更复杂的日志系统或编译开关来控制
// static bool enable_debug_print_for_specific_wall = false; // 可以考虑从外部设置
// 为了方便，我们直接在函数内判断法向量

// --- 新增的调试控制变量 ---
static bool DEBUG_HLLC_PRINT_ENABLED = false; // 总开关
static double DEBUG_TARGET_X_MIN = -1e9; // 目标X坐标范围最小值
static double DEBUG_TARGET_X_MAX = 1e9;  // 目标X坐标范围最大值
static double DEBUG_TARGET_TIME_MIN = -1e9; // 目标时间范围最小值
static double DEBUG_TARGET_TIME_MAX = 1e9;   // 目标时间范围最大值
static int DEBUG_TARGET_HE_ID = -1;      // 目标半边ID (可选)
// --- 调试控制变量结束 ---

namespace HydroCore { // 定义HydroCore命名空间
    // 初始化静态成员调试变量
    bool FluxCalculator_cpp::s_debug_print_enabled = false; // 修改：定义并初始化静态成员变量
    double FluxCalculator_cpp::s_debug_target_x_min = -1e9; // 修改：定义并初始化静态成员变量
    double FluxCalculator_cpp::s_debug_target_x_max = 1e9;  // 修改：定义并初始化静态成员变量
    double FluxCalculator_cpp::s_debug_target_time_min = -1e9; // 修改：定义并初始化静态成员变量
    double FluxCalculator_cpp::s_debug_target_time_max = 1e9;   // 修改：定义并初始化静态成员变量
    int FluxCalculator_cpp::s_debug_target_he_id = -1;      // 修改：定义并初始化静态成员变量

FluxCalculator_cpp::FluxCalculator_cpp(double gravity, double min_depth_param, RiemannSolverType_cpp solver_type) // 构造函数实现
    : g(gravity), min_depth(min_depth_param), solver_type_internal(solver_type) { // 初始化列表
    // std::cout << "C++ FluxCalculator_cpp initialized (g=" << g << ", min_depth=" << min_depth << ")" << std::endl; // 打印初始化信息（可选）
} // 结束构造函数

    // 新增一个函数来从外部设置调试参数 (可选，或者直接在需要的地方修改上面的静态变量)
    // 定义静态成员函数
void FluxCalculator_cpp::set_debug_conditions(bool enable, double x_min, double x_max, double t_min, double t_max, int he_id) { // 修改：定义为类的静态成员函数
    s_debug_print_enabled = enable; // 修改：操作静态成员变量
    s_debug_target_x_min = x_min;   // 修改：操作静态成员变量
    s_debug_target_x_max = x_max;   // 修改：操作静态成员变量
    s_debug_target_time_min = t_min; // 修改：操作静态成员变量
    s_debug_target_time_max = t_max; // 修改：操作静态成员变量
    s_debug_target_he_id = he_id;     // 修改：操作静态成员变量
}
std::array<double, 3> FluxCalculator_cpp::calculate_hllc_flux(
    const PrimitiveVars_cpp& W_L_param, // 修改：参数名，以区分内部使用的W_L
    const PrimitiveVars_cpp& W_R_param, // 修改：参数名
    const std::array<double, 2>& normal_vec
) {

    // PROFILE_FUNCTION();


    PrimitiveVars_cpp W_L = W_L_param; // 拷贝一份输入参数，以便修改
    PrimitiveVars_cpp W_R = W_R_param; // 拷贝一份输入参数

    // 新增：对传入HLLC的状态进行最终检查和修正
    const double hllc_input_dry_threshold = min_depth * 1.5; // HLLC输入干判断阈值，可以比重构的阈值略宽松或一致

    if (W_L.h < hllc_input_dry_threshold) { // 如果左侧水深小于HLLC干阈值
        // W_L.h = std::max(0.0, W_L.h); // 确保非负 (重构应该已经做过，但再次确保)
        if (W_L.h < min_depth) W_L.h = 0.0; // 如果严格小于min_depth，则视为完全干
        W_L.u = 0.0; // 清零流速
        W_L.v = 0.0; // 清零流速
    }
    if (W_R.h < hllc_input_dry_threshold) { // 如果右侧水深小于HLLC干阈值
        // W_R.h = std::max(0.0, W_R.h); // 确保非负
        if (W_R.h < min_depth) W_R.h = 0.0; // 如果严格小于min_depth，则视为完全干
        W_R.u = 0.0; // 清零流速
        W_R.v = 0.0; // 清零流速
    }
    // 至此，W_L 和 W_R 是经过输入保护修正后的状态

    // 后续的 dryL, dryR 判断以及 unL, utL, cL 等计算都基于修正后的 W_L, W_R
    bool dryL = (W_L.h < min_depth); // 使用 min_depth 进行严格的干判断
    bool dryR = (W_R.h < min_depth); // 使用 min_depth 进行严格的干判断



    if (dryL && dryR) {
        // if (should_print_debug_local) std::cout << "  HLLC: Both dry, returning zero flux." << std::endl;
        return {0.0, 0.0, 0.0};
    }


    double unL, utL, unR, utR;
    double cL, cR;

    // --- 使用修正后的 W_L, W_R 计算旋转速度 ---
    unL = W_L.u * normal_vec[0] + W_L.v * normal_vec[1];
    utL = -W_L.u * normal_vec[1] + W_L.v * normal_vec[0];
    unR = W_R.u * normal_vec[0] + W_R.v * normal_vec[1];
    utR = -W_R.u * normal_vec[1] + W_R.v * normal_vec[0];



    // --- 干区处理，使用修正后的 W_L, W_R 和 dryL, dryR ---
    if (dryL) { // 左干
        W_L.h = 0.0; cL = 0.0; // W_L.h 已经被设为0或极小值，这里再次确保
        if (!dryR) { // 右湿
            if (unR < 0) {
                unL = 0.0;
                utL = 0.0;
            } else {
                unL = -unR;
                utL = utR;
            }
            cR = std::sqrt(g * W_R.h); // W_R.h 是修正后的
        } else { // 右也干
            cR = 0.0;
        }
    } else if (dryR) { // 左湿右干
        W_R.h = 0.0; cR = 0.0; // W_R.h 已经被设为0或极小值
        // !dryL 隐含为真
        if (unL > 0) {
            unR = 0.0;
            utR = 0.0;
        } else {
            unR = -unL;
            utR = utL;
        }
        cL = std::sqrt(g * W_L.h); // W_L.h 是修正后的
    } else { // 两侧都湿 (基于修正后的 W_L, W_R 判断仍然湿)
        cL = std::sqrt(g * W_L.h);
        cR = std::sqrt(g * W_R.h);
    }


    // --- (此处省略之前提供的HLLC完整计算逻辑，它应该在这一系列修正之后) ---
    // 例如:
    double sqrt_hL = (W_L.h > 0) ? std::sqrt(W_L.h) : 0.0;
    double sqrt_hR = (W_R.h > 0) ? std::sqrt(W_R.h) : 0.0;
    // ...一直到 return F_hllc_cartesian;

    // 确保你将上面省略的HLLC计算部分（从sqrt_hL开始到最后）粘贴回来
    // 这只是为了展示在HLLC最开始的地方加入输入保护。

    // ... (粘贴HLLC计算的剩余部分) ...
    double sqrt_sum = sqrt_hL + sqrt_hR;

    double un_roe, h_roe_for_c, c_roe;
    if (sqrt_sum < 1e-9) {
        un_roe = 0.5 * (unL + unR);
        h_roe_for_c = 0.5 * (W_L.h + W_R.h);
    } else {
        un_roe = (sqrt_hL * unL + sqrt_hR * unR) / sqrt_sum;
        h_roe_for_c = 0.5 * (W_L.h + W_R.h);
    }
    c_roe = (h_roe_for_c > 0) ? std::sqrt(g * h_roe_for_c) : 0.0;

    double sL_wet = un_roe - c_roe;
    double sR_wet = un_roe + c_roe;
    double sL_simple = unL - cL;
    double sR_simple = unR + cR;

    double sL_davis = std::min(sL_simple, sL_wet);
    double sR_davis = std::max(sR_simple, sR_wet);

    double sL_final, sR_final;
    if (dryL) {
        sL_final = unR - 2 * cR;
    } else {
        sL_final = sL_davis;
    }
    if (dryR) {
        sR_final = unL + 2 * cL;
    } else {
        sR_final = sR_davis;
    }

    if (sL_final > sR_final - 1e-9) {
        double s_avg = 0.5 * (sL_final + sR_final);
        sL_final = s_avg - 1e-6;
        sR_final = s_avg + 1e-6;
        if (sL_final > sR_final) {
             sL_final = std::min(sL_simple,sR_simple) - 1e-6;
             sR_final = std::max(sL_simple,sR_simple) + 1e-6;
        }
    }

    double PL = 0.5 * g * W_L.h * W_L.h;
    double PR = 0.5 * g * W_R.h * W_R.h;

    double den_s_star = W_L.h * (sL_final - unL) - W_R.h * (sR_final - unR);
    double s_star;
    double epsilon_den = 1e-9;

    if (std::abs(den_s_star) < epsilon_den) {
        s_star = un_roe;
    } else {
        s_star = (PR - PL + W_L.h * unL * (sL_final - unL) - W_R.h * unR * (sR_final - unR)) / den_s_star;
    }

    std::array<double, 3> FL_nt = {W_L.h * unL, W_L.h * unL * unL + PL, W_L.h * unL * utL};
    std::array<double, 3> FR_nt = {W_R.h * unR, W_R.h * unR * unR + PR, W_R.h * unR * utR};
    std::array<double, 3> F_hllc_nt;

    if (sL_final >= 0) {
        F_hllc_nt = FL_nt;
    } else if (sR_final <= 0) {
        F_hllc_nt = FR_nt;
    } else {
        std::array<double, 3> UL_nt = {W_L.h, W_L.h * unL, W_L.h * utL};
        std::array<double, 3> UR_nt = {W_R.h, W_R.h * unR, W_R.h * utR};
        if (s_star >= 0) {
            double h_starL_num = W_L.h * (sL_final - unL);
            double h_starL_den = sL_final - s_star;
            double h_starL = (std::abs(h_starL_den) < epsilon_den) ? W_L.h : h_starL_num / h_starL_den;
            h_starL = std::max(0.0, h_starL);
            std::array<double, 3> U_starL_nt = {h_starL, h_starL * s_star, h_starL * utL};
            for(int k=0; k<3; ++k) F_hllc_nt[k] = FL_nt[k] + sL_final * (U_starL_nt[k] - UL_nt[k]);
        } else {
            double h_starR_num = W_R.h * (sR_final - unR);
            double h_starR_den = sR_final - s_star;
            double h_starR = (std::abs(h_starR_den) < epsilon_den) ? W_R.h : h_starR_num / h_starR_den;
            h_starR = std::max(0.0, h_starR);
            std::array<double, 3> U_starR_nt = {h_starR, h_starR * s_star, h_starR * utR};
            for(int k=0; k<3; ++k) F_hllc_nt[k] = FR_nt[k] + sR_final * (U_starR_nt[k] - UR_nt[k]);
        }
    }
    double Fh_n = F_hllc_nt[0];
    double Fun_n = F_hllc_nt[1];
    double Fut_n = F_hllc_nt[2];

    std::array<double, 3> F_hllc_cartesian;
    F_hllc_cartesian[0] = Fh_n;
    F_hllc_cartesian[1] = Fun_n * normal_vec[0] - Fut_n * normal_vec[1];
    F_hllc_cartesian[2] = Fun_n * normal_vec[1] + Fut_n * normal_vec[0];


    return F_hllc_cartesian;
}

} // namespace HydroCore