# tests/test_hydro_model_core_cpp.py
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
    import hydro_model_cpp  # 导入C++模块

    # 从C++模块获取需要的类和枚举
    Mesh_cpp_py = hydro_model_cpp.Mesh_cpp  # C++ Mesh类
    ReconScheme_cpp_py = hydro_model_cpp.ReconstructionScheme_cpp  # C++ ReconstructionScheme枚举
    RiemannSolver_cpp_py = hydro_model_cpp.RiemannSolverType_cpp  # C++ RiemannSolverType枚举
    TimeScheme_cpp_py = hydro_model_cpp.TimeScheme_cpp  # C++ TimeScheme枚举
except ImportError as e:  # 捕获导入错误
    print(f"导入错误 (HydroModelCore Test): {e}")  # 打印错误信息
    sys.exit(1)  # 退出程序


def create_test_mesh_cpp():  # 创建测试用C++网格对象
    """Creates a very simple C++ Mesh_cpp object for testing."""
    mesh = Mesh_cpp_py()  # 创建C++ Mesh对象
    # Nodes: id, x, y, z_bed, marker (flat_node_data, num_nodes, num_attrs)
    nodes_data = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0.5, 1, 0, 0], dtype=float).flatten()  # 节点数据
    mesh.load_nodes_from_numpy(nodes_data, 3, 5)  # 加载节点数据
    # Cells: id, n0, n1, n2 (flat_cell_data, num_cells, nodes_per_cell, manning_values)
    cells_conn = np.array([0, 0, 1, 2], dtype=int).flatten()  # 单元连接
    manning = np.array([0.025], dtype=float)  # 曼宁系数
    mesh.load_cells_from_numpy(cells_conn, 1, 3, manning)  # 加载单元数据
    # Edges (flat_edge_data, num_edges, num_edge_attrs)
    edges_data = np.array([0, 0, 1, 1, 1, 1, 2, 1, 2, 2, 0, 1], dtype=int).flatten()  # 边数据
    mesh.precompute_geometry_and_topology(edges_data, 3, 4)  # 预计算几何和拓扑
    return mesh  # 返回C++网格对象


class TestHydroModelCore(unittest.TestCase):  # 定义测试类

    @classmethod  # 类方法，只执行一次
    def setUpClass(cls):  # 测试类准备方法
        """Create a C++ mesh object once for all tests in this class."""
        cls.mesh_cpp_instance = create_test_mesh_cpp()  # 创建C++网格实例
        # 使用 len() 而不是 .size()
        print(
            f"TestHydroModelCore: setUpClass created mesh_cpp_instance (Mesh has {len(cls.mesh_cpp_instance.nodes)} nodes, {len(cls.mesh_cpp_instance.cells)} cells)")  # 打印信息

    # ... (test_creation_and_parameter_setting 方法不变) ...
    def test_creation_and_parameter_setting(self):  # 测试创建和参数设置
        print("\n--- Testing HydroModelCore_cpp Creation & Parameter Setting ---")  # 打印测试信息
        try:  # 尝试创建对象
            model_core = hydro_model_cpp.HydroModelCore_cpp(self.mesh_cpp_instance)  # 创建C++水动力模型核心对象
            self.assertIsNotNone(model_core)  # 断言对象不为空
            print("  HydroModelCore_cpp created successfully.")  # 打印创建成功信息

            model_core.set_simulation_parameters(  # 设置模拟参数
                gravity=9.80, min_depth=1e-5, cfl=0.4,  # 参数列表
                total_t=20.0, output_dt_interval=0.5, max_dt_val=0.2  # 参数列表
            )  # 结束设置
            self.assertAlmostEqual(model_core.get_gravity(), 9.80)  # 断言重力加速度
            self.assertAlmostEqual(model_core.get_min_depth(), 1e-5)  # 断言最小水深
            print("  Simulation parameters set and retrieved successfully.")  # 打印参数设置成功信息

            model_core.set_numerical_schemes(  # 设置数值方案
                ReconScheme_cpp_py.SECOND_ORDER_LIMITED,  # 重构方案
                RiemannSolver_cpp_py.HLLC,  # 黎曼求解器
                TimeScheme_cpp_py.RK2_SSP  # 时间积分方案
            )  # 结束设置
            print("  Numerical schemes set successfully.")  # 打印数值方案设置成功信息

            # 准备虚拟的初始条件
            # 使用 self.mesh_cpp_instance 来访问在 setUpClass 中创建的网格实例
            num_cells_in_mesh = len(self.mesh_cpp_instance.cells)  # 获取网格中的单元数量 (使用 self)
            U_init = [[0.5, 0.1, 0.05]] * num_cells_in_mesh  # 为每个单元创建初始守恒量
            eta_init = [0.5] * num_cells_in_mesh  # 为每个单元创建初始水位

            model_core.set_initial_conditions_placeholder(U_init, eta_init)  # 设置初始条件占位符
            print("  Initial conditions (placeholder) set successfully.")  # 打印初始条件设置成功信息

            # 测试调用 advance_one_step_placeholder
            can_continue = model_core.advance_one_step_placeholder()  # 调用单步执行占位符
            self.assertTrue(can_continue)  # 断言可以继续
            print("  advance_one_step_placeholder called successfully.")  # 打印调用成功信息

        except Exception as e:  # 捕获异常
            self.fail(f"HydroModelCore_cpp test failed: {e}")  # 标记测试失败


if __name__ == '__main__':  # 如果是主程序
    print("Running HydroModelCore_cpp basic functionality tests...")  # 打印测试开始信息
    unittest.main(verbosity=2)  # 运行测试