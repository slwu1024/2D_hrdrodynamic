# tests/test_source_terms_cpp.py
import numpy as np  # 导入numpy
import unittest  # 导入unittest测试框架
import sys  # 导入sys模块
import os  # 导入os模块

# --- sys.path 修改逻辑 (与 test_flux_calculator_cpp.py 中相同) ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))  # 项目根目录
if project_root_dir not in sys.path:  # 如果根目录不在sys.path中
    sys.path.insert(0, project_root_dir)  # 添加到sys.path
    print(f"Temporarily added project root to sys.path: {project_root_dir}")  # 打印添加的路径

pyd_cmake_install_dir = os.path.join(project_root_dir, '_skbuild', 'win-amd64-3.10', 'cmake-install')  # cmake-install目录
pyd_cmake_build_dir = os.path.join(project_root_dir, '_skbuild', 'win-amd64-3.10', 'cmake-build')  # cmake-build目录
pyd_actual_dir = None  # 初始化pyd实际目录
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
# --- 结束 sys.path 修改 ---

try:  # 尝试导入模块
    from src.model.SourceTerms import SourceTermCalculator as SourceTermCalculator_py  # 从Python模型导入
    import hydro_model_cpp  # 导入编译好的C++模块
except ImportError as e:  # 捕获导入错误
    print(f"导入错误 (SourceTerms Test): {e}")  # 打印错误信息
    sys.exit(1)  # 退出程序


class TestSourceTerms(unittest.TestCase):  # 定义测试类

    def setUp(self):  # 测试准备方法
        self.gravity_py = 9.81  # Python重力加速度
        self.min_depth_py = 1e-6  # Python最小水深
        self.gravity_cpp = 9.81  # C++重力加速度
        self.min_depth_cpp = 1e-6  # C++最小水深
        self.rtol = 1e-5  # 相对容差
        self.atol = 1e-7  # 绝对容差

        self.py_st_calc = SourceTermCalculator_py(self.gravity_py, self.min_depth_py)  # 创建Python源项计算器
        self.cpp_st_calc = hydro_model_cpp.SourceTermCalculator_cpp(self.gravity_cpp, self.min_depth_cpp)  # 创建C++源项计算器

    def _convert_to_cpp_format(self, U_numpy_array):  # 将NumPy数组转换为C++期望的列表的列表格式
        """将 (N, 3) NumPy 数组转换为 List[List[float]]"""
        return U_numpy_array.tolist()  # 转换为列表的列表

    def _run_friction_comparison(self, test_name, U_input_np, U_coeffs_np, dt, manning_n_np):  # 定义运行摩擦比较的辅助方法
        print(f"\n--- Running Friction Test: {test_name} ---")  # 打印测试名称

        # Python 计算
        U_output_py = self.py_st_calc.apply_friction_semi_implicit(  # Python计算摩擦
            U_input_np.copy(), U_coeffs_np.copy(), dt, manning_n_np.copy()  # 传入参数
        )  # 结束计算
        print(f"  Python Output ({test_name}):\n{U_output_py}")  # 打印Python输出

        # C++ 计算
        # 将NumPy数组转换为C++函数期望的格式 (std::vector<std::array<double,3>>)
        # Pybind11 通常可以直接处理 list of lists of numbers
        U_input_cpp_fmt = self._convert_to_cpp_format(U_input_np)  # 转换输入状态
        U_coeffs_cpp_fmt = self._convert_to_cpp_format(U_coeffs_np)  # 转换系数状态
        manning_n_list = manning_n_np.tolist()  # 转换曼宁系数值

        U_output_cpp_list_of_lists = self.cpp_st_calc.apply_friction_semi_implicit_all_cells(  # C++计算摩擦
            U_input_cpp_fmt, U_coeffs_cpp_fmt, dt, manning_n_list  # 传入参数
        )  # 结束计算
        U_output_cpp = np.array(U_output_cpp_list_of_lists)  # 转换为NumPy数组
        print(f"  C++ Output    ({test_name}):\n{U_output_cpp}")  # 打印C++输出

        try:  # 尝试断言
            np.testing.assert_allclose(U_output_py, U_output_cpp, rtol=self.rtol, atol=self.atol)  # 断言比较结果
            print(f"  Test '{test_name}' PASSED")  # 打印测试通过信息
        except AssertionError as e:  # 捕获断言错误
            print(f"  Test '{test_name}' FAILED: {e}")  # 打印测试失败信息
            raise  # 重新抛出错误

    def test_friction_single_wet_cell(self):  # 测试单个湿单元摩擦
        U_input = np.array([[1.0, 0.5, 0.2]])  # 输入状态
        U_coeffs = np.array([[1.0, 0.5, 0.2]])  # 系数状态
        dt = 0.1  # 时间步长
        manning_n = np.array([0.025])  # 曼宁系数
        self._run_friction_comparison("Single Wet Cell", U_input, U_coeffs, dt, manning_n)  # 运行比较

    def test_friction_single_dry_cell_input(self):  # 测试单个干单元输入摩擦
        U_input = np.array([[0.0000001, 0.0, 0.0]])  # 输入干单元
        U_coeffs = np.array([[1.0, 0.5, 0.2]])  # 系数湿单元
        dt = 0.1  # 时间步长
        manning_n = np.array([0.025])  # 曼宁系数
        self._run_friction_comparison("Single Dry Cell (Input)", U_input, U_coeffs, dt, manning_n)  # 运行比较

    def test_friction_single_dry_cell_coeffs(self):  # 测试单个干单元系数摩擦
        U_input = np.array([[1.0, 0.5, 0.2]])  # 输入湿单元
        U_coeffs = np.array([[0.0000001, 0.0, 0.0]])  # 系数干单元
        dt = 0.1  # 时间步长
        manning_n = np.array([0.025])  # 曼宁系数
        self._run_friction_comparison("Single Dry Cell (Coeffs)", U_input, U_coeffs, dt, manning_n)  # 运行比较

    def test_friction_multiple_cells_mixed(self):  # 测试多个混合单元摩擦
        U_input = np.array([  # 输入状态
            [1.0, 0.5, 0.2],  # Wet
            [0.5, 0.1, -0.1],  # Wet
            [0.00001, 0.0, 0.0]  # Dryish
        ])  # 结束输入状态
        U_coeffs = np.array([  # 系数状态
            [1.0, 0.5, 0.2],  # Wet (same as input)
            [0.4, 0.0, 0.0],  # Wet (different velocity, lower h)
            [1.0, 1.0, 1.0]  # Wet (coeffs are wet even if input is dryish)
        ])  # 结束系数状态
        dt = 0.05  # 时间步长
        manning_n = np.array([0.02, 0.03, 0.025])  # 曼宁系数
        self._run_friction_comparison("Multiple Cells Mixed", U_input, U_coeffs, dt, manning_n)  # 运行比较

    def test_friction_zero_velocity_coeffs(self):  # 测试系数速度为零的摩擦
        U_input = np.array([[1.0, 0.5, 0.2]])  # 输入状态
        U_coeffs = np.array([[1.0, 0.0, 0.0]])  # 系数速度为零
        dt = 0.1  # 时间步长
        manning_n = np.array([0.025])  # 曼宁系数
        self._run_friction_comparison("Zero Velocity Coeffs", U_input, U_coeffs, dt, manning_n)  # 运行比较


if __name__ == '__main__':  # 如果是主程序
    print("Running SourceTermCalculator C++/Python friction comparison tests...")  # 打印测试开始信息
    unittest.main(verbosity=2)  # 运行测试