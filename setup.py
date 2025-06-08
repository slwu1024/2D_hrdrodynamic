# setup.py
# 最终版本，专为 'python setup.py build_ext --inplace' 设计
from skbuild import setup
from setuptools import find_packages

setup(
    # --- 提供包结构信息，这是让 --inplace 找到目标的关键 ---
    # `find_packages` 会找到 `src/hydro_model`
    packages=find_packages(where="src"),

    # `package_dir` 告诉 setuptools，这些包的根目录在 'src' 下
    package_dir={"": "src"},

    # --- C++/scikit-build 配置 ---
    # `cmake_install_dir` 明确告诉 scikit-build，编译产物应该属于 'hydro_model' 包的一部分。
    # 当 --inplace 运行时，它会把产物复制到 src/hydro_model/
    cmake_install_dir="src/hydro_model",
)