# tests/test_time_integrator_cpp.py
import numpy as np  # 导入numpy
import unittest  # 导入unittest测试框架
import sys  # 导入sys模块
import os  # 导入os模块

# --- sys.path 修改逻辑 (同上) ---
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
    # Python版 TimeIntegrator 用于比较签名或概念，但我们不直接比较其输出，因为RHS不同
    # from src.model.TimeIntegrator import TimeIntegrator as TimeIntegrator_py, TimeIntegrationSchemes as TimeSchemes_py
    import hydro_model_cpp  # 导入C++模块

    TimeScheme_cpp_py = hydro_model_cpp.TimeScheme_cpp  # C++ TimeScheme枚举
except ImportError as e:  # 捕获导入错误
    print(f"导入错误 (TimeIntegrator Test): {e}")  # 打印错误信息
    sys.exit(1)  # 退出程序


# 定义虚拟的 RHS 和 Friction 函数 (Python端)
def dummy_rhs_function_py(U_state_list_of_lists, time_current):  # 虚拟RHS函数 (Python)
    # 期望输入是 list of lists, 输出也是 list of lists
    # 简单地返回一个与输入形状相同，但值乘以0.1的数组
    num_cells = len(U_state_list_of_lists)  # 获取单元数量
    num_vars = len(U_state_list_of_lists[0]) if num_cells > 0 else 0  # 获取变量数量

    # Pybind11传递给std::function的Python函数，其参数类型应与std::function签名匹配
    # C++: const StateVector& U_current -> Python: list of lists (or NumPy array if casted)
    # C++: double time_current -> Python: float
    # C++: returns StateVector -> Python: should return list of lists

    rhs_out = []  # 初始化RHS输出
    for i in range(num_cells):  # 遍历单元
        cell_rhs = [val * 0.1 for val in U_state_list_of_lists[i]]  # 计算单元RHS
        rhs_out.append(cell_rhs)  # 添加到输出列表
    # print(f"Python dummy_rhs called at t={time_current}, U_in[0]={U_state_list_of_lists[0] if num_cells >0 else None}") # 打印调试信息
    return rhs_out  # 返回RHS输出


def dummy_friction_function_py(U_input_list_of_lists, U_coeffs_list_of_lists, dt):  # 虚拟摩擦函数 (Python)
    # 简单地返回 U_input 不变 (无摩擦)
    # print(f"Python dummy_friction called with dt={dt}, U_in[0]={U_input_list_of_lists[0] if U_input_list_of_lists else None}") # 打印调试信息
    return U_input_list_of_lists  # 返回输入状态 (无摩擦)


class TestTimeIntegrator(unittest.TestCase):  # 定义测试类

    def test_creation_and_step_call_forward_euler(self):  # 测试前向欧拉法创建和step调用
        print("\n--- Testing TimeIntegrator_cpp (Forward Euler) Creation & Step Call ---")  # 打印测试信息

        cpp_time_integrator = hydro_model_cpp.TimeIntegrator_cpp(  # 创建C++时间积分器
            TimeScheme_cpp_py.FORWARD_EULER,  # 时间积分方案
            dummy_rhs_function_py,  # 虚拟RHS函数
            dummy_friction_function_py,  # 虚拟摩擦函数
            3  # 变量数量
        )  # 结束创建
        self.assertIsNotNone(cpp_time_integrator)  # 断言对象不为空
        print("  TimeIntegrator_cpp (FE) created successfully.")  # 打印创建成功信息

        U_current_py_list = [[1.0, 0.5, 0.2], [0.8, 0.1, -0.1]]  # 当前状态 (Python列表的列表)
        dt_py = 0.1  # 时间步长
        time_current_py = 0.0  # 当前时间

        try:  # 尝试执行step
            U_next_cpp_list_of_lists = cpp_time_integrator.step(U_current_py_list, dt_py, time_current_py)  # 执行一步积分
            self.assertEqual(len(U_next_cpp_list_of_lists), len(U_current_py_list))  # 断言输出长度与输入长度相同
            if U_next_cpp_list_of_lists:  # 如果输出不为空
                self.assertEqual(len(U_next_cpp_list_of_lists[0]), 3)  # 断言每个单元变量数量为3
            print(
                f"  FE Step called. U_next[0] (approx): {U_next_cpp_list_of_lists[0] if U_next_cpp_list_of_lists else 'N/A'}")  # 打印输出结果
            # 粗略验证结果: U_next approx U_current + dt * (0.1 * U_current) (因为dummy_rhs是0.1*U, dummy_friction无作用)
            # U_next[0][0] approx 1.0 + 0.1 * (0.1 * 1.0) = 1.0 + 0.01 = 1.01
            if U_next_cpp_list_of_lists:  # 如果输出不为空
                self.assertAlmostEqual(U_next_cpp_list_of_lists[0][0], 1.0 + dt_py * (0.1 * U_current_py_list[0][0]),
                                       delta=1e-9)  # 断言第一个元素的值
        except Exception as e:  # 捕获异常
            self.fail(f"TimeIntegrator_cpp (FE) step call failed: {e}")  # 标记测试失败

    def test_creation_and_step_call_rk2_ssp(self):  # 测试RK2_SSP创建和step调用
        print("\n--- Testing TimeIntegrator_cpp (RK2_SSP) Creation & Step Call ---")  # 打印测试信息
        cpp_time_integrator = hydro_model_cpp.TimeIntegrator_cpp(  # 创建C++时间积分器
            TimeScheme_cpp_py.RK2_SSP,  # 时间积分方案
            dummy_rhs_function_py,  # 虚拟RHS函数
            dummy_friction_function_py,  # 虚拟摩擦函数
            3  # 变量数量
        )  # 结束创建
        self.assertIsNotNone(cpp_time_integrator)  # 断言对象不为空
        print("  TimeIntegrator_cpp (RK2) created successfully.")  # 打印创建成功信息

        U_current_py_list = [[1.0, 0.5, 0.2], [0.8, 0.1, -0.1]]  # 当前状态
        dt_py = 0.1  # 时间步长
        time_current_py = 0.0  # 当前时间

        try:  # 尝试执行step
            U_next_cpp_list_of_lists = cpp_time_integrator.step(U_current_py_list, dt_py, time_current_py)  # 执行一步积分
            self.assertEqual(len(U_next_cpp_list_of_lists), len(U_current_py_list))  # 断言输出长度与输入长度相同
            if U_next_cpp_list_of_lists:  # 如果输出不为空
                self.assertEqual(len(U_next_cpp_list_of_lists[0]), 3)  # 断言每个单元变量数量为3
            print(
                f"  RK2 Step called. U_next[0] (approx): {U_next_cpp_list_of_lists[0] if U_next_cpp_list_of_lists else 'N/A'}")  # 打印输出结果
            # 粗略验证RK2结果:
            # U_s1 = U_current + dt * (0.1 * U_current) = U_current * (1 + 0.1*dt)
            # U_next approx 0.5*U_current + 0.5*(U_s1 + dt*(0.1*U_s1))
            # U_next approx 0.5*U_current + 0.5*U_s1*(1 + 0.1*dt)
            # U_next approx 0.5*U_current + 0.5*U_current*(1+0.1*dt)^2
            # For U[0][0]=1.0, dt=0.1:
            # U_s1[0][0] = 1.0 * (1 + 0.1*0.1) = 1.01
            # U_next[0][0] approx 0.5*1.0 + 0.5*1.01*(1+0.01) = 0.5 + 0.5*1.01*1.01 = 0.5 + 0.5 * 1.0201 = 0.5 + 0.51005 = 1.01005
            if U_next_cpp_list_of_lists:  # 如果输出不为空
                self.assertAlmostEqual(U_next_cpp_list_of_lists[0][0],
                                       0.5 * U_current_py_list[0][0] + 0.5 * U_current_py_list[0][0] * (
                                                   (1 + 0.1 * dt_py) ** 2), delta=1e-7)  # 断言第一个元素的值
        except Exception as e:  # 捕获异常
            self.fail(f"TimeIntegrator_cpp (RK2) step call failed: {e}")  # 标记测试失败


if __name__ == '__main__':  # 如果是主程序
    print("Running TimeIntegrator_cpp basic functionality tests...")  # 打印测试开始信息
    unittest.main(verbosity=2)  # 运行测试