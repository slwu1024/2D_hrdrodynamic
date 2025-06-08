// src_cpp/include/Profiler.h
#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>    // 用于计时
#include <string>    // 用于名称
#include <map>       // 用于存储结果
#include <vector>    // 用于排序
#include <iostream>  // 用于打印总结
#include <iomanip>   // 用于格式化输出 (std::setw, std::fixed, std::setprecision)
#include <algorithm> // 用于排序

namespace Profiler {

    struct ProfileResult {
        std::chrono::duration<double, std::micro> total_time{0}; // 总时间，使用微秒存储
        long long call_count = 0;                                // 调用次数
        std::chrono::duration<double, std::micro> min_time{std::chrono::duration<double, std::micro>::max()}; // 最短时间
        std::chrono::duration<double, std::micro> max_time{std::chrono::duration<double, std::micro>::min()}; // 最长时间

        ProfileResult() = default; // 默认构造函数
    };

    // 全局或静态存储分析结果
    // 使用 extern 声明，定义将在 .cpp 文件中
    extern std::map<std::string, ProfileResult> results;

    // 记录一次耗时
    void record_duration(const std::string& name, std::chrono::steady_clock::duration duration);

    // 打印总结报告
    void print_summary();

    // 重置/清空分析数据
    void reset_summary();

    // ScopedTimer 类定义
    class ScopedTimer {
    public:
        ScopedTimer(const std::string& name);
        ~ScopedTimer();

    private:
        std::string m_name;
        std::chrono::time_point<std::chrono::steady_clock> m_start_time;
        bool m_stopped; // 防止重复记录
    };

} // namespace Profiler

// 宏定义，方便在代码中启用/禁用分析
#ifdef ENABLE_PROFILING
// ##__LINE__ 用于创建基于行号的唯一变量名，防止在同一作用域多次使用宏时重定义
#define PROFILE_SCOPE(name_str) Profiler::ScopedTimer timer_##__LINE__(name_str)
// __func__ 是一个预定义标识符，在C++11及以后版本中代表当前函数名
#define PROFILE_FUNCTION() Profiler::ScopedTimer func_timer_##__LINE__(__func__)
#else
#define PROFILE_SCOPE(name_str)
#define PROFILE_FUNCTION()
#endif

#endif // PROFILER_H