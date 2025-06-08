// src_cpp/src/Profiler.cpp
#include "Profiler.h" // 包含头文件

namespace Profiler {

// 定义静态成员变量
std::map<std::string, ProfileResult> results;

ScopedTimer::ScopedTimer(const std::string& name) : m_name(name), m_stopped(false) {
    m_start_time = std::chrono::steady_clock::now(); // 记录开始时间
}

ScopedTimer::~ScopedTimer() {
    if (!m_stopped) { // 确保只记录一次
        auto end_time = std::chrono::steady_clock::now();          // 记录结束时间
        auto duration = end_time - m_start_time;                  // 计算耗时
        record_duration(m_name, duration);                        // 调用记录函数
        m_stopped = true;
    }
}

void record_duration(const std::string& name, std::chrono::steady_clock::duration duration_steady) {
    // 将 steady_clock::duration 转换为更方便用于算术和显示的 double 微秒
    auto duration_us = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(duration_steady);

    results[name].total_time += duration_us;
    results[name].call_count++;
    if (duration_us < results[name].min_time) {
        results[name].min_time = duration_us;
    }
    if (duration_us > results[name].max_time) {
        results[name].max_time = duration_us;
    }
}

void print_summary() {
    std::cout << "\n--- C++ Performance Profiling Summary ---\n";
    // 为了更好的可读性，可以按总时间排序
    std::vector<std::pair<std::string, ProfileResult>> sorted_results(results.begin(), results.end());
    std::sort(sorted_results.begin(), sorted_results.end(),
              [](const auto& a, const auto& b) {
                  return a.second.total_time > b.second.total_time; // 按总时间降序
              });

    // 打印表头
    const int name_width = 55; // 根据最长函数名调整
    const int time_width = 18;
    const int count_width = 12;

    std::cout << std::left << std::setw(name_width) << "Function/Scope"
              << std::right << std::setw(time_width) << "Total Time (ms)"
              << std::setw(count_width) << "Calls"
              << std::setw(time_width) << "Avg Time (us)"
              << std::setw(time_width) << "Min Time (us)"
              << std::setw(time_width) << "Max Time (us)" << std::endl;
    std::cout << std::string(name_width + time_width * 4 + count_width, '-') << std::endl;

    for (const auto& pair : sorted_results) {
        double total_ms = pair.second.total_time.count() / 1000.0; // 总时间转换为毫秒
        double avg_us = (pair.second.call_count > 0) ? pair.second.total_time.count() / pair.second.call_count : 0.0;
        // 处理min/max的初始值
        double min_us = (pair.second.min_time == std::chrono::duration<double, std::micro>::max()) ? 0.0 : pair.second.min_time.count();
        double max_us = (pair.second.max_time == std::chrono::duration<double, std::micro>::min()) ? 0.0 : pair.second.max_time.count();


        std::cout << std::left << std::setw(name_width) << pair.first
                  << std::right << std::fixed << std::setprecision(3) << std::setw(time_width) << total_ms
                  << std::setw(count_width) << pair.second.call_count
                  << std::fixed << std::setprecision(3) << std::setw(time_width) << avg_us
                  << std::fixed << std::setprecision(3) << std::setw(time_width) << min_us
                  << std::fixed << std::setprecision(3) << std::setw(time_width) << max_us
                  << std::endl;
    }
    std::cout << "------------------------------------------\n";
}

void reset_summary() {
    results.clear(); // 清空map
}

} // namespace Profiler