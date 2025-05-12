# tests/test_vfr_calculator_cpp.py
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
    from src.model.WettingDrying import VFRCalculator as VFRCalculator_py  # 从Python模型导入
    # 如果你的 src.model.MeshData.Node 定义简单且易于实例化，也可以用它
    # from src.model.MeshData import Node as Node_py_actual # 尝试导入实际的Node类
    import hydro_model_cpp  # 导入编译好的C++模块

    Node_cpp_py = hydro_model_cpp.Node_cpp  # 获取绑定的Node_cpp类
except ImportError as e:  # 捕获导入错误
    # ... (错误处理不变) ...
    print(f"导入错误 (VFR Test): {e}")  # 打印错误信息
    sys.exit(1)  # 退出程序


# 为Python版本的测试定义一个简单的Node模拟类
class SimplePyNode:  # 定义简单的Python节点类
    def __init__(self, id, x, y, z_bed):  # 初始化方法
        self.id = id  # 节点ID
        self.x = x  # x坐标
        self.y = y  # y坐标
        self.z_bed = z_bed  # 底高程


class TestVFRCalculator(unittest.TestCase):  # 定义测试类

    def setUp(self):  # 测试准备方法
        # ... (VFRCalculator 参数不变) ...
        self.min_depth = 1e-6  # 最小水深
        self.min_eta_change = 1e-6  # eta最小变化
        self.max_iters = 20  # 最大迭代次数
        self.rel_h_tol = 1e-4  # 相对水深容差
        self.rtol_compare = 1e-5  # 比较结果的相对容差
        self.atol_compare = 1e-7  # 比较结果的绝对容差

        self.py_vfr = VFRCalculator_py(self.min_depth, self.min_eta_change, self.max_iters,
                                       self.rel_h_tol)  # 创建Python VFR计算器
        self.cpp_vfr = hydro_model_cpp.VFRCalculator_cpp(self.min_depth, self.min_eta_change, self.max_iters,
                                                         self.rel_h_tol)  # 创建C++ VFR计算器

        # 使用 SimplePyNode 实例化 self.nodes_py
        self.nodes_py = [  # Python节点对象列表
            SimplePyNode(id=0, x=0.0, y=0.0, z_bed=0.0),  # 节点0
            SimplePyNode(id=1, x=1.0, y=0.0, z_bed=0.1),  # 节点1
            SimplePyNode(id=2, x=0.5, y=1.0, z_bed=0.2)  # 节点2
        ]  # 结束节点列表

        # C++ Node_cpp 对象列表 (保持不变)
        self.nodes_cpp = [  # C++节点对象列表
            Node_cpp_py(0, 0.0, 0.0, 0.0),  # 节点0
            Node_cpp_py(1, 1.0, 0.0, 0.1),  # 节点1
            Node_cpp_py(2, 0.5, 1.0, 0.2)  # 节点2
        ]  # 结束节点列表
        self.b_sorted_py = np.array([0.0, 0.1, 0.2])  # Python排序高程
        self.b_sorted_cpp = [0.0, 0.1, 0.2]  # C++排序高程
        self.cell_total_area_py = 0.5 * np.abs(self.nodes_py[0].x * (self.nodes_py[1].y - self.nodes_py[2].y) + \
                                               self.nodes_py[1].x * (self.nodes_py[2].y - self.nodes_py[0].y) + \
                                               self.nodes_py[2].x * (self.nodes_py[0].y - self.nodes_py[
            1].y))  # 计算面积 (使用SimplePyNode属性)
        self.cell_total_area_cpp = self.cell_total_area_py  # 面积相同

    # ... (test_get_h_from_eta_cases 和 test_get_eta_from_h_cases 方法保持不变) ...
    def test_get_h_from_eta_cases(self):  # 测试get_h_from_eta方法
        test_cases = [  # 测试用例
            ("Dry", -0.1, 0.0),
            ("At b0", 0.0, 0.0),
            ("Part wet 1", 0.05, None),
            ("At b1", 0.1, None),
            ("Part wet 2", 0.15, None),
            ("At b2", 0.2, None),
            ("Fully wet", 0.3, None)
        ]  # 结束测试用例

        for name, eta, expected_h_manual_py in test_cases:  # 遍历测试用例
            with self.subTest(name=name, eta=eta):  # 定义子测试
                h_py = self.py_vfr.get_h_from_eta(eta, self.b_sorted_py, self.cell_total_area_py,
                                                  name + "_py")  # Python计算h
                h_cpp = self.cpp_vfr.get_h_from_eta(eta, self.b_sorted_cpp, self.cell_total_area_cpp,
                                                    name + "_cpp")  # C++计算h

                print(f"\n--- Test get_h_from_eta: {name} (eta={eta:.3f}) ---")  # 打印测试信息
                print(f"  Python h: {h_py:.7e}")  # 打印Python h
                print(f"  C++    h: {h_cpp:.7e}")  # 打印C++ h

                expected_h_to_compare = expected_h_manual_py if expected_h_manual_py is not None else h_py  # 设定比较基准

                self.assertAlmostEqual(h_cpp, expected_h_to_compare,
                                       delta=self.atol_compare + self.rtol_compare * abs(expected_h_to_compare),
                                       msg=f"{name}: C++ h differs from expected/Python h.")  # 断言比较结果
                print(f"  Test '{name}' PASSED for get_h_from_eta")  # 打印测试通过信息

    def test_get_eta_from_h_cases(self):  # 测试get_eta_from_h方法
        etas_for_h_calc = [0.05, 0.1, 0.15, 0.25]  # 用于计算h的eta值

        for i, eta_target_for_h in enumerate(etas_for_h_calc):  # 遍历eta值
            h_avg_target_py = self.py_vfr.get_h_from_eta(eta_target_for_h, self.b_sorted_py,
                                                         self.cell_total_area_py)  # 计算目标平均水深(Python)
            eta_guess_py = eta_target_for_h * 0.9 if h_avg_target_py > self.min_depth else self.b_sorted_py[
                0]  # Python初始猜测值
            eta_guess_cpp = eta_guess_py  # C++初始猜测值

            test_name = f"Case_h_target_{i + 1}_(h_avg={h_avg_target_py:.4f})"  # 定义测试名称

            with self.subTest(name=test_name):  # 定义子测试
                expected_eta_py = 0.0  # 初始化预期eta
                expected_eta_cpp = 0.0  # 初始化预期eta
                if h_avg_target_py < self.min_depth / 10:  # 如果目标水深过小
                    expected_eta_py = self.b_sorted_py[0]  # 预期eta为最低点
                    expected_eta_cpp = self.b_sorted_cpp[0]  # 预期eta为最低点
                    # C++ version (直接使用期望值，因为Python版可能因干单元而不调用迭代)
                    eta_cpp = self.cpp_vfr.get_eta_from_h(h_avg_target_py, self.b_sorted_cpp, self.nodes_cpp,
                                                          self.cell_total_area_cpp, eta_guess_cpp,
                                                          test_name + "_cpp_dry_case")  # C++计算eta
                    expected_eta_cpp = eta_cpp  # C++结果
                else:  # 否则
                    eta_py = self.py_vfr.get_eta_from_h(h_avg_target_py, self.b_sorted_py, self.nodes_py,
                                                        self.cell_total_area_py, eta_guess_py,
                                                        test_name + "_py")  # Python计算eta
                    expected_eta_py = eta_py  # Python结果作为比较基准

                    eta_cpp = self.cpp_vfr.get_eta_from_h(h_avg_target_py, self.b_sorted_cpp, self.nodes_cpp,
                                                          self.cell_total_area_cpp, eta_guess_cpp,
                                                          test_name + "_cpp")  # C++计算eta
                    expected_eta_cpp = eta_cpp  # C++结果

                print(
                    f"\n--- Test get_eta_from_h: {test_name} (h_avg_target={h_avg_target_py:.4e}, eta_guess={eta_guess_py:.3f}) ---")  # 打印测试信息
                print(f"  Python eta: {expected_eta_py:.7e}")  # 打印Python eta
                print(f"  C++    eta: {expected_eta_cpp:.7e}")  # 打印C++ eta

                self.assertAlmostEqual(expected_eta_cpp, expected_eta_py,
                                       delta=self.atol_compare + self.rtol_compare * abs(expected_eta_py),
                                       msg=f"{test_name}: C++ eta differs from Python eta.")  # 断言比较结果
                print(f"  Test '{test_name}' PASSED for get_eta_from_h")  # 打印测试通过信息


if __name__ == '__main__':  # 如果是主程序
    unittest.main(verbosity=2)  # 运行测试