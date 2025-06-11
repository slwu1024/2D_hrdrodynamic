# CMakeLists.txt

cmake_minimum_required(VERSION 3.12)
project(hydro_cpp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Python、Pybind11 和 OpenMP 的配置 ---
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(OpenMP REQUIRED)
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

# --- 将 OpenMP 编译选项应用到核心逻辑库 ---
if(OpenMP_FOUND)
    message(STATUS "OpenMP found, applying compiler options to hydro_core_logic.")
    # 只设置编译选项
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(hydro_core_logic PUBLIC -fopenmp)
    elseif(MSVC)
        target_compile_options(hydro_core_logic PUBLIC /openmp)
    endif()
else()
    message(WARNING "OpenMP not found, compilation will be single-threaded.")
endif()


option(ENABLE_HYDRO_PROFILING "Enable internal C++ performance profiling" OFF)
if(ENABLE_HYDRO_PROFILING)
    target_compile_definitions(hydro_core_logic PUBLIC ENABLE_PROFILING)
    message(STATUS "C++ Performance profiling ENABLED.")
else()
    message(STATUS "C++ Performance profiling DISABLED.")
endif()

target_include_directories(hydro_core_logic PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/include
)

# --- 创建Python扩展模块 ---
pybind11_add_module(hydro_model_cpp SHARED
        src_cpp/bindings/bindings.cpp
)

target_include_directories(hydro_model_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/extern/eigen
)

# --- 将Python模块链接到库和 OpenMP 运行时 ---
target_link_libraries(hydro_model_cpp PRIVATE
        hydro_core_logic
        Python::Module
        pybind11::module
        OpenMP::OpenMP_CXX # 在这里链接运行时库
)

# 安装指令
install(TARGETS hydro_model_cpp
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
)

# --- (可选) 为独立的C++测试创建可执行文件 ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/main.cpp")
    set(CPP_TEST_SOURCES src_cpp/main.cpp)
    add_executable(cpp_tests ${CPP_TEST_SOURCES})
    target_include_directories(cpp_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/extern/eigen
    )
    # 为测试程序链接核心库和OpenMP运行时
    target_link_libraries(cpp_tests PRIVATE
        hydro_core_logic
        OpenMP::OpenMP_CXX
    )
endif()