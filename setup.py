from skbuild import setup # 导入skbuild的setup函数

setup( # 调用setup函数
    name="hydro_model_pkg", # 包名
    version="0.1.0", # 版本号
    author="wsl", # 作者名
    description="A 2D Hydrodynamic Model with a C++ core", # 描述
    license="MIT", # 许可证

    cmake_source_dir='src_cpp/', # CMake源码目录

    packages=[], # 明确告诉setuptools，此setup主要用于C++扩展，没有纯Python包
    py_modules=[], # 同上

    install_requires=[ # 依赖项
        'numpy', # numpy库
        'pybind11>=2.6', # pybind11库
        'scipy', # scipy库
        'matplotlib', # matplotlib库
        'pandas', # pandas库
        'pykrige', # pykrige库
        'PyYAML', # PyYAML库
        'meshio' # meshio库
    ], # 结束依赖项
    python_requires='>=3.8', # Python版本要求
) # 结束setup函数