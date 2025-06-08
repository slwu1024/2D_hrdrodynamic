# setup.py
from skbuild import setup
import os # 导入 os 模块

setup(
    name="hydro_model_pkg",
    version="0.1.0",
    author="wsl",
    description="A 2D Hydrodynamic Model with a C++ core",
    license="MIT",

    cmake_source_dir='src_cpp/',

    # 告诉 setuptools 编译后的 C++ 扩展 (由 CMake install 安装的)
    # 应该被认为是包的一部分，并且位于项目的根目录下。
    # cmake_install_dir 的路径是相对于最终包的安装位置的。
    # 对于一个直接导入的 .pyd 文件，我们希望它在顶层。
    # scikit-build 会处理将 _skbuild/<platform>/cmake-install/ 的内容
    # 映射到这个相对路径。
    cmake_install_dir='.',  # <--- 新增或确认这一行

    # packages 和 py_modules 保持为空，因为我们只依赖 CMake 构建的 C++ 扩展
    packages=[],
    py_modules=[],

    cmake_args=['-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON'],

    install_requires=[
        'numpy',
        'pybind11>=2.6',
        'scipy',
        'matplotlib',
        'pandas',
        'pykrige',
        'PyYAML',
        'meshio'
    ],
    python_requires='>=3.8',
)