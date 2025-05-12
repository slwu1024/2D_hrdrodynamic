cmake_minimum_required(VERSION 3.12)
project(hydro_cpp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Python和Pybind11的配置 ---
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(extern/pybind11)

# --- 定义你的C++核心逻辑库 ---
set(HYDRO_CORE_SOURCES
        src/MeshData_cpp.cpp
        src/FluxCalculator_cpp.cpp
        src/SourceTerms_cpp.cpp
        src/WettingDrying_cpp.cpp
        src/Reconstruction_cpp.cpp
        src/TimeIntegrator_cpp.cpp
        src/HydroModelCore_cpp.cpp
        src/BoundaryConditionHandler_cpp.cpp
)
add_library(hydro_core_logic STATIC ${HYDRO_CORE_SOURCES})

# 为 hydro_core_logic 设置包含目录
target_include_directories(hydro_core_logic PUBLIC
        # PUBLIC 表示链接到 hydro_core_logic 的目标也会自动获得这些包含路径
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# 如果 hydro_core_logic 自身需要链接到 Pybind11 的某些仅头文件部分 (通常不需要)
# target_link_libraries(hydro_core_logic INTERFACE pybind11::headers)


# --- 创建Python扩展模块 ---
pybind11_add_module(hydro_model_cpp SHARED
        bindings/bindings.cpp
)



# 为 hydro_model_cpp 模块设置包含目录
target_include_directories(hydro_model_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include # 确保绑定代码能找到你的核心头文件
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen # <--- 新增
        # Pybind11的包含目录通常通过链接 pybind11::module 自动处理
)

# 将Python模块链接到你的C++核心逻辑库和必要的Pybind11/Python库
target_link_libraries(hydro_model_cpp PRIVATE
        hydro_core_logic   # 链接到你自己的核心逻辑
        Python::Module     # 链接到Python模块支持
        pybind11::module   # 链接到Pybind11模块支持
)

# --- 尝试明确设置输出目录 ---
# CMAKE_BINARY_DIR 指向构建目录的根 (例如 _skbuild/win-amd64-3.10/cmake-build)
# 我们希望 .pyd 文件直接在这个构建目录的顶层，或者在一个 scikit-build 容易找到的地方
# 对于 SHARED 库 (Python扩展模块)，通常是 LIBRARY_OUTPUT_DIRECTORY 和 RUNTIME_OUTPUT_DIRECTORY
# pybind11_add_module 内部可能已经设置了这些，但我们可以尝试覆盖或补充

# 获取Python扩展的后缀 (例如 .pyd, .so)
# find_package(Python ...) 应该已经设置了 Python_EXTENSION_SUFFIX
# set(EXTENSION_SUFFIX ${Python_EXTENSION_SUFFIX}) # 在较新的CMake中，Python_EXTENSION_SUFFIX 可能不直接可用
# 或者通过 pybind11 获取
get_target_property(EXTENSION_SUFFIX hydro_model_cpp SUFFIX)

# 设置模块的输出路径，使其直接位于构建目录的根下，或者一个可预测的子目录
# scikit-build 在 --inplace 时会查找这些位置
set_target_properties(hydro_model_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" # .lib 文件 (虽然我们的是.pyd)
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" # .dll / .pyd / .so 文件
        # ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" # .a / .lib (静态库)
        # PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" # .pdb 文件
)
# 注意：pybind11_add_module 本身可能已经设置了这些。
# 覆盖它们可能需要小心，或者有时不需要。
# 主要目标是确保 scikit-build 的 build_ext --inplace 能找到它。

# --- 安装Python扩展模块 ---
# 这个 install 命令是关键，它告诉 CMake 在执行 "install" 目标时要做什么。
# scikit-build 会调用这个 "install" 目标。
install(TARGETS hydro_model_cpp
        LIBRARY DESTINATION "." # 安装到 CMAKE_INSTALL_PREFIX 的根目录
        RUNTIME DESTINATION "."
)

# --- (可选) 为独立的C++测试创建可执行文件 ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    set(CPP_TEST_SOURCES main.cpp) # 通常测试程序只包含main.cpp，然后链接核心库
    add_executable(cpp_tests ${CPP_TEST_SOURCES})
    target_include_directories(cpp_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include     # 包含您项目定义的头文件
            ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen # 如果 main.cpp 或其包含的头文件需要Eigen
            # Pybind11 的头文件通常不需要为独立的C++可执行文件包含，除非main.cpp直接用了Pybind11 API
    ) # 结束设置

    # 将 cpp_tests 链接到您的核心逻辑库
    target_link_libraries(cpp_tests PRIVATE hydro_core_logic) # 链接到核心逻辑库
endif()