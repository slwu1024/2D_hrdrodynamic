cmake_minimum_required(VERSION 3.12)
project(hydro_cpp_core LANGUAGES CXX)



set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Python和Pybind11的配置 ---
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(src_cpp/extern/pybind11)

# --- 定义你的C++核心逻辑库 ---
set(HYDRO_CORE_SOURCES
        src_cpp/src/MeshData_cpp.cpp
        src_cpp/src/FluxCalculator_cpp.cpp
        src_cpp/src/SourceTerms_cpp.cpp
        src_cpp/src/WettingDrying_cpp.cpp
        src_cpp/src/Reconstruction_cpp.cpp
        src_cpp/src/TimeIntegrator_cpp.cpp
        src_cpp/src/HydroModelCore_cpp.cpp
        src_cpp/src/BoundaryConditionHandler_cpp.cpp
        src_cpp/src/Profiler.cpp
)
add_library(hydro_core_logic STATIC ${HYDRO_CORE_SOURCES})

option(ENABLE_HYDRO_PROFILING "Enable internal C++ performance profiling" OFF) # 默认关闭
if(ENABLE_HYDRO_PROFILING)
    target_compile_definitions(hydro_core_logic PUBLIC ENABLE_PROFILING) # 为核心逻辑库添加宏定义
    # 如果绑定代码或其他目标也需要这个宏（比如从Python调用print_summary）:
    # target_compile_definitions(hydro_model_cpp PUBLIC ENABLE_PROFILING)
    message(STATUS "C++ Performance profiling ENABLED.")
else()
    message(STATUS "C++ Performance profiling DISABLED.")
endif()

# 为 hydro_core_logic 设置包含目录
target_include_directories(hydro_core_logic PUBLIC
        # PUBLIC 表示链接到 hydro_core_logic 的目标也会自动获得这些包含路径
        ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/include
)
# 如果 hydro_core_logic 自身需要链接到 Pybind11 的某些仅头文件部分 (通常不需要)
# target_link_libraries(hydro_core_logic INTERFACE pybind11::headers)


# --- 创建Python扩展模块 ---
pybind11_add_module(hydro_model_cpp SHARED
        src_cpp/bindings/bindings.cpp
)



# 为 hydro_model_cpp 模块设置包含目录
target_include_directories(hydro_model_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/include # 确保绑定代码能找到你的核心头文件
        ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/extern/eigen
)

# 将Python模块链接到你的C++核心逻辑库和必要的Pybind11/Python库
target_link_libraries(hydro_model_cpp PRIVATE
        hydro_core_logic   # 链接到你自己的核心逻辑
        Python::Module     # 链接到Python模块支持
        pybind11::module   # 链接到Pybind11模块支持
)


# 这个安装指令至关重要，它决定了编译出的 C++ 模块 (.pyd 或 .so) 会被安装到哪里。
# DESTINATION hydro_model 表示将它安装到名为 "hydro_model" 的包目录下。
# 这与你的 Python 包名 `src/hydro_model` 匹配，scikit-build 会处理好这个映射。
# 最终，hydro_model_cpp.pyd 会和 __init__.py 等文件放在一起。
install(TARGETS hydro_model_cpp
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
)

# --- (可选) 为独立的C++测试创建可执行文件 ---
# 为 main.cpp 的路径检查添加 'src_cpp' 前缀
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/main.cpp")
    # 为源文件路径添加 'src_cpp' 前缀
    set(CPP_TEST_SOURCES src_cpp/main.cpp) # 通常测试程序只包含main.cpp，然后链接核心库
    add_executable(cpp_tests ${CPP_TEST_SOURCES})
    # 为包含目录路径添加 'src_cpp' 前缀
    target_include_directories(cpp_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/include     # 包含您项目定义的头文件
            ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/extern/eigen # 如果 main.cpp 或其包含的头文件需要Eigen
    ) # 结束设置

    # 将 cpp_tests 链接到您的核心逻辑库
    target_link_libraries(cpp_tests PRIVATE hydro_core_logic) # 链接到核心逻辑库
endif()