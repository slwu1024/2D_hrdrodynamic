# setup.py
from skbuild import setup
import os # 导入 os 模块

setup(
    name="hydro_model_pkg",
    version="0.1.0",
    author="wsl",
    description="A 2D Hydrodynamic Model with a C++ core",
    license="MIT",

    cmake_source_dir='src_cpp/', # C++ 源代码在 src_cpp 目录下

    # 告诉 setuptools 编译后的 C++ 扩展 (由 CMake install 安装的)
    # 应该被认为是包的一部分，并且位于项目的根目录下。
    # cmake_install_dir 的路径是相对于最终包的安装位置的。
    # 对于一个直接导入的 .pyd 文件，我们希望它在顶层。
    # scikit-build 会处理将 _skbuild/<platform>/cmake-install/ 的内容
    # 映射到这个相对路径。
    cmake_install_dir='.', # 安装到包的根目录

    # packages 和 py_modules 保持为空，因为我们只依赖 CMake 构建的 C++ 扩展
    packages=[],
    py_modules=[],

    # 修改 cmake_args
    # -G Ninja 尝试强制使用 Ninja 生成器
    # -DCMAKE_MAKE_PROGRAM=ninja 尝试明确告诉 CMake ninja 可执行文件的名字 (它应该会从 PATH 中查找)
    cmake_args=[
        '-G', 'Ninja',  # 指定使用 Ninja 作为 CMake 生成器
        '-DCMAKE_MAKE_PROGRAM=ninja', # 指定 ninja 可执行程序 (应从 PATH 查找)
        '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON' # 开启详细的 Makefile 输出，方便调试
    ],

    install_requires=[ # 列出项目依赖的 Python 包
        'numpy',
        'pybind11>=2.6',
        'scipy',
        'matplotlib',
        'pandas',
        'pykrige',
        'PyYAML',
        'meshio'
    ],
    python_requires='>=3.8', # 指定项目兼容的 Python 版本
)