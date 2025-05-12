# tests/test_flux_calculator_cpp.py
import numpy as np  # 导入numpy
import unittest  # 导入unittest测试框架
import sys  # 导入sys模块
import os  # 导入os模块

# --- 步骤 1: 将项目根目录添加到 sys.path ---
# 获取当前测试脚本文件所在的目录 (E:\Python project\2D_hrdrodynamic\tests)
current_script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
# 获取项目根目录 (E:\Python project\2D_hrdrodynamic)
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))  # 项目根目录
if project_root_dir not in sys.path:  # 如果根目录不在sys.path中
    sys.path.insert(0, project_root_dir)  # 添加到sys.path的开头
    print(f"Temporarily added project root to sys.path: {project_root_dir}")  # 打印添加的路径

# --- 步骤 2: 定位并添加 .pyd 文件所在的目录到 sys.path ---
# (根据你的Python版本和平台调整 'win-amd64-3.10')
# 优先尝试 cmake-install 目录，其次是 cmake-build 目录
# 路径是相对于项目根目录构建的
pyd_cmake_install_dir = os.path.join(project_root_dir, '_skbuild', 'win-amd64-3.10', 'cmake-install')  # cmake-install目录
pyd_cmake_build_dir = os.path.join(project_root_dir, '_skbuild', 'win-amd64-3.10', 'cmake-build')  # cmake-build目录

pyd_actual_dir = None  # 初始化pyd实际目录
# 检查 hydro_model_cpp.cp310-win_amd64.pyd 是否在这些目录中
# (注意：实际的 .pyd 文件名可能因 Python 版本和平台而略有不同，但cp310-win_amd64是常见的)
pyd_filename = "hydro_model_cpp.cp310-win_amd64.pyd"  # pyd文件名

if os.path.exists(os.path.join(pyd_cmake_install_dir, pyd_filename)):  # 如果在cmake-install目录中找到
    pyd_actual_dir = pyd_cmake_install_dir  # 设置为cmake-install目录
elif os.path.exists(os.path.join(pyd_cmake_build_dir, pyd_filename)):  # 如果在cmake-build目录中找到
    pyd_actual_dir = pyd_cmake_build_dir  # 设置为cmake-build目录

if pyd_actual_dir and pyd_actual_dir not in sys.path:  # 如果找到了pyd目录且不在sys.path中
    sys.path.insert(0, pyd_actual_dir)  # 添加到sys.path
    print(f"Temporarily added .pyd directory to sys.path: {pyd_actual_dir}")  # 打印添加的路径
elif not pyd_actual_dir:  # 如果没有找到pyd目录
    print(f"Warning: Could not find {pyd_filename} in expected _skbuild subdirectories.")  # 打印警告
    print(f"  Checked: {pyd_cmake_install_dir}")  # 打印检查过的路径
    print(f"  Checked: {pyd_cmake_build_dir}")  # 打印检查过的路径
    print("  Consider manually copying the .pyd file to the project root or an install location.")  # 提示手动复制

# --- 步骤 3: 现在尝试导入 ---
try:  # 尝试导入模块
    from src.model.FluxCalculator import FluxCalculator as FluxCalculator_py, RiemannSolvers  # 导入Python版FluxCalculator
    import hydro_model_cpp  # 导入C++模块 (现在应该能从 pyd_actual_dir 找到)
except ImportError as e:  # 捕获导入错误
    print(f"导入错误: {e}")  # 打印错误信息
    print("请确保：")  # 提示信息
    print(f"1. 项目根目录 ({project_root_dir}) 已正确添加到 sys.path 以找到 'src' 包。")  # 提示信息
    print(
        f"2. C++模块 ({pyd_filename}) 已编译并位于一个 sys.path 中的可访问位置 (如 {pyd_actual_dir if pyd_actual_dir else 'expected _skbuild dir'}).")  # 提示信息
    sys.exit(1)  # 退出程序


class TestFluxCalculator(unittest.TestCase):  # 定义测试类
    # ... (您的测试用例代码保持不变) ...
    def setUp(self):  # 测试准备方法
        self.gravity = 9.81  # 重力加速度
        self.min_depth_py = 1e-6  # Python版本最小水深
        self.min_depth_cpp = 1e-6  # C++版本最小水深
        self.rtol = 1e-5  # 相对容差
        self.atol = 1e-7  # 绝对容差

        # 实例化 Python 版本
        self.py_flux_calc = FluxCalculator_py(RiemannSolvers.HLLC, self.gravity, self.min_depth_py)  # 创建Python通量计算器

        # 实例化 C++ 版本
        self.cpp_flux_calc = hydro_model_cpp.FluxCalculator_cpp(self.gravity, self.min_depth_cpp,
                                                                hydro_model_cpp.RiemannSolverType_cpp.HLLC)  # 创建C++通量计算器

    def _run_comparison(self, test_name, W_L_np, W_R_np, normal_np):  # 定义运行比较的辅助方法
        print(f"\n--- Running Test: {test_name} ---")  # 打印测试名称
        # Python 计算
        flux_py = self.py_flux_calc._hllc_solver(W_L_np, W_R_np, normal_np)  # Python计算通量
        print(f"  Python Flux ({test_name}): {flux_py}")  # 打印Python通量

        # C++ 计算
        W_L_cpp_pv = hydro_model_cpp.PrimitiveVars_cpp(W_L_np[0], W_L_np[1], W_L_np[2])  # 创建C++左侧状态
        W_R_cpp_pv = hydro_model_cpp.PrimitiveVars_cpp(W_R_np[0], W_R_np[1], W_R_np[2])  # 创建C++右侧状态
        flux_cpp_tuple = self.cpp_flux_calc.calculate_hllc_flux(W_L_cpp_pv, W_R_cpp_pv, list(normal_np))  # C++计算通量
        flux_cpp = np.array(flux_cpp_tuple)  # 转换为NumPy数组
        print(f"  C++ Flux    ({test_name}): {flux_cpp}")  # 打印C++通量

        try:  # 尝试断言
            np.testing.assert_allclose(flux_py, flux_cpp, rtol=self.rtol, atol=self.atol)  # 断言比较结果
            print(f"  Test '{test_name}' PASSED")  # 打印测试通过信息
        except AssertionError as e:  # 捕获断言错误
            print(f"  Test '{test_name}' FAILED: {e}")  # 打印测试失败信息
            raise  # 重新抛出错误

    def test_still_water(self):  # 测试静水情况
        W_L = np.array([1.0, 0.0, 0.0])  # 左侧状态
        W_R = np.array([1.0, 0.0, 0.0])  # 右侧状态
        normal = np.array([1.0, 0.0])  # 法向量
        self._run_comparison("Still Water", W_L, W_R, normal)  # 运行比较

    def test_dam_break_dry_bed_right(self):  # 测试右侧干底溃坝
        W_L = np.array([1.0, 0.0, 0.0])  # 左侧状态
        W_R = np.array([0.0, 0.0, 0.0])  # 右侧干底
        normal = np.array([1.0, 0.0])  # 法向量
        self._run_comparison("Dam Break (Dry Bed Right)", W_L, W_R, normal)  # 运行比较

    def test_dam_break_dry_bed_left(self):  # 测试左侧干底溃坝
        W_L = np.array([0.0, 0.0, 0.0])  # 左侧干底
        W_R = np.array([1.0, 0.0, 0.0])  # 右侧状态
        normal = np.array([1.0, 0.0])  # 法向量
        self._run_comparison("Dam Break (Dry Bed Left)", W_L, W_R, normal)  # 运行比较

    def test_subcritical_flow_positive_normal(self):  # 测试亚临界正向流动
        W_L = np.array([1.0, 1.0, 0.0])  # 左侧状态
        W_R = np.array([0.8, 0.8, 0.0])  # 右侧状态
        normal = np.array([1.0, 0.0])  # 法向量
        self._run_comparison("Subcritical Flow (Positive Normal)", W_L, W_R, normal)  # 运行比较

    def test_subcritical_flow_negative_normal(self):  # 测试亚临界负向流动
        W_L = np.array([0.8, -0.8, 0.0])  # 左侧状态
        W_R = np.array([1.0, -1.0, 0.0])  # 右侧状态
        normal = np.array([-1.0, 0.0])  # 法向量
        self._run_comparison("Subcritical Flow (Negative Normal)", W_L, W_R, normal)  # 运行比较

    def test_both_dry(self):  # 测试两侧都干
        W_L = np.array([self.min_depth_py / 2, 0.0, 0.0])  # 左侧干
        W_R = np.array([self.min_depth_py / 10, 0.0, 0.0])  # 右侧干
        normal = np.array([1.0, 0.0])  # 法向量
        self._run_comparison("Both Dry", W_L, W_R, normal)  # 运行比较


if __name__ == '__main__':  # 如果是主程序
    print("Running FluxCalculator C++/Python comparison tests...")  # 打印测试开始信息
    # 这里的 print(f"Attempting to import hydro_model_cpp from...") 意义不大了，因为导入在try-except块里
    unittest.main(verbosity=2)  # 运行测试