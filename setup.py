# setup.py
from skbuild import setup
import os

# --- 控制是否编译CUDA版本 ---
# BUILD_WITH_CUDA = True # 或者 False，根据你的需要
# 动态地从环境变量决定是否启用CUDA会更灵活
enable_cuda_env = os.environ.get('ENABLE_CUDA_BUILD', 'OFF').upper()
BUILD_WITH_CUDA = True if enable_cuda_env == 'ON' else False

cmake_args_list = ['-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON']
if BUILD_WITH_CUDA:
    print("INFO: setup.py - Enabling CUDA in CMake arguments.")
    cmake_args_list.append('-DENABLE_CUDA=ON')
    cuda_arch = os.environ.get('CMAKE_CUDA_ARCHITECTURES')
    if cuda_arch:
        cmake_args_list.append(f'-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}')
        print(f"INFO: setup.py - Setting CMAKE_CUDA_ARCHITECTURES to {cuda_arch}")
else:
    print("INFO: setup.py - Disabling CUDA in CMake arguments.")
    cmake_args_list.append('-DENABLE_CUDA=OFF')

setup(
    name="hydro_model_pkg",
    version="0.1.0",
    author="wsl",
    description="A 2D Hydrodynamic Model with a C++ core",
    license="MIT", # 已在 pyproject.toml 中处理
    cmake_source_dir='src_cpp/',
    cmake_args=cmake_args_list, # 确保这个参数被正确使用
    # packages=[], # 如果没有纯Python包，可以为空或省略
    # py_modules=[], # 如果没有顶层纯Python模块，可以为空或省略
    # install_requires=[], # 运行时依赖在 pyproject.toml 的 [project.dependencies] 中定义
    python_requires='>=3.8',
)