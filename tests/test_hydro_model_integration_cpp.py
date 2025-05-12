# tests/test_hydro_model_integration_cpp.py
import numpy as np  # 导入numpy
import unittest  # 导入unittest测试框架
import sys  # 导入sys模块
import os  # 导入os模块

# --- sys.path 修改逻辑 (保持与之前一致) ---
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

    # 从C++模块获取所有需要的类和枚举
    Mesh_cpp_py = hydro_model_cpp.Mesh_cpp  # C++ Mesh类
    ReconScheme_cpp_py = hydro_model_cpp.ReconstructionScheme_cpp  # C++ ReconstructionScheme枚举
    RiemannSolver_cpp_py = hydro_model_cpp.RiemannSolverType_cpp  # C++ RiemannSolverType枚举
    TimeScheme_cpp_py = hydro_model_cpp.TimeScheme_cpp  # C++ TimeScheme枚举
    BoundaryType_cpp_py = hydro_model_cpp.BoundaryType_cpp  # C++ BoundaryType枚举
    BoundaryDef_cpp_py = hydro_model_cpp.BoundaryDefinition_cpp  # C++ BoundaryDefinition结构体
    TimeseriesPoint_cpp_py = hydro_model_cpp.TimeseriesPoint_cpp  # C++ TimeseriesPoint结构体
    HydroModelCore_cpp_py = hydro_model_cpp.HydroModelCore_cpp  # C++ HydroModelCore类
except ImportError as e:  # 捕获导入错误
    print(f"导入错误 (HydroModel Integration Test): {e}")  # 打印错误信息
    sys.exit(1)  # 退出程序


def create_simple_test_mesh_for_integration():  # 创建简单测试网格 (用于集成)
    """创建一个包含1个或2个单元的简单C++网格，并定义边界标记。"""
    mesh = Mesh_cpp_py()  # 创建C++ Mesh对象
    # Nodes: id, x, y, z_bed, marker
    # 一个简单的矩形，中间一条边，形成两个三角形单元
    #   2----3
    #   |  / |
    #   | /  |
    #   0----1
    nodes_data = np.array([  # 节点数据
        0, 0.0, 0.0, 0.0, 1,  # Node 0 (marker 1: left boundary)
        1, 1.0, 0.0, 0.0, 2,  # Node 1 (marker 2: bottom boundary)
        2, 0.0, 1.0, 0.0, 1,  # Node 2 (marker 1: left boundary)
        3, 1.0, 1.0, 0.0, 3  # Node 3 (marker 3: top/right boundary)
    ], dtype=float).flatten()  # 结束节点数据
    mesh.load_nodes_from_numpy(nodes_data, 4, 5)  # 加载节点数据

    # Cells: id, n0, n1, n2
    cells_conn = np.array([  # 单元连接
        0, 0, 1, 2,  # Cell 0 (bottom-left triangle)
        1, 1, 3, 2  # Cell 1 (top-right triangle)
    ], dtype=int).flatten()  # 结束单元连接
    manning = np.array([0.025, 0.025], dtype=float)  # 曼宁系数
    mesh.load_cells_from_numpy(cells_conn, 2, 3, manning)  # 加载单元数据

    # Edges: edge_id, n1_id, n2_id, marker
    # 边界边标记与节点标记对应以便识别
    # 内部边 (1,2) 标记为 0
    edges_data = np.array([  # 边数据
        0, 0, 2, 1,  # Left edge (0-2), marker 1
        1, 0, 1, 2,  # Bottom edge (0-1), marker 2
        2, 1, 3, 3,  # Right edge (1-3), marker 3 (假设)
        3, 2, 3, 3,  # Top edge (2-3), marker 3 (假设)
        4, 1, 2, 0  # Internal edge (1-2), marker 0
    ], dtype=int).flatten()  # 结束边数据
    mesh.precompute_geometry_and_topology(edges_data, 5, 4)  # 预计算几何和拓扑
    return mesh  # 返回C++网格对象


class TestHydroModelIntegration(unittest.TestCase):  # 定义测试类

    @classmethod  # 类方法
    def setUpClass(cls):  # 测试类准备方法
        cls.mesh_cpp = create_simple_test_mesh_for_integration()  # 创建C++网格实例
        print(
            f"TestHydroModelIntegration: setUpClass created mesh (Nodes: {len(cls.mesh_cpp.nodes)}, Cells: {len(cls.mesh_cpp.cells)})")  # 打印信息

    def test_model_run_few_steps_all_walls(self):  # 测试模型运行几步 (全墙体边界)
        print("\n--- Test: Model Run (All Walls) ---")  # 打印测试信息
        model = HydroModelCore_cpp_py(self.mesh_cpp)  # 创建C++水动力模型核心对象

        model.set_simulation_parameters(gravity=9.81, min_depth=1e-6, cfl=0.4,  # 设置模拟参数
                                        total_t=0.1, output_dt_interval=0.05, max_dt_val=0.02)  # 参数列表
        model.set_numerical_schemes(ReconScheme_cpp_py.FIRST_ORDER,  # 设置数值方案 (一阶简化测试)
                                    RiemannSolver_cpp_py.HLLC,
                                    TimeScheme_cpp_py.FORWARD_EULER)  # 结束设置

        # 配置边界条件: 所有已知标记都设为 WALL
        bc_defs = {}  # 边界定义字典
        wall_def = BoundaryDef_cpp_py()  # 创建墙体边界定义
        wall_def.type = BoundaryType_cpp_py.WALL  # 设置类型为墙体
        bc_defs[1] = wall_def  # 标记1为墙体
        bc_defs[2] = wall_def  # 标记2为墙体
        bc_defs[3] = wall_def  # 标记3为墙体
        # 标记0是内部边，不需要定义，或者可以定义一个默认的墙体给未匹配的标记
        # bc_defs[0] = wall_def # 通常不需要为内部边定义，但如果get_cell_by_id(0)是默认标记也可以

        model.setup_boundary_conditions_cpp(bc_defs, {}, {})  # 设置边界条件 (空的时间序列)

        # 设置初始条件 (2个单元) - 例如，左边单元水深高，右边低
        U_init = [[1.0, 0.1, 0.0], [0.5, 0.0, 0.0]]  # 初始守恒量
        eta_init = [1.0, 0.5]  # 初始水位
        model.set_initial_conditions_cpp(U_init, eta_init)  # 设置初始条件

        num_steps_to_run = 3  # 定义运行步数
        print(f"  Running {num_steps_to_run} steps...")  # 打印运行信息
        for i in range(num_steps_to_run):  # 循环运行
            if not model.is_simulation_finished():  # 如果模拟未结束
                can_continue = model.advance_one_step()  # 执行一步
                self.assertTrue(can_continue or model.is_simulation_finished(),
                                f"Step {i + 1} should be able to continue or finish.")  # 断言可以继续或结束
                print(f"    Step {model.get_step_count()} done. Time: {model.get_current_time():.4f}")  # 打印步数和时间
            else:  # 如果模拟已结束
                print(f"    Simulation finished early at step {i + 1}, time {model.get_current_time():.4f}")  # 打印提前结束信息
                break  # 跳出循环

        # 获取最终状态并做一些基本检查 (例如，水深非负)
        U_final_list_of_lists = model.get_U_state_all_internal_copy()  # 获取最终状态副本
        U_final_np = np.array(U_final_list_of_lists)  # 转换为NumPy数组
        print(f"  Final U state (first cell): {U_final_np[0] if U_final_np.size > 0 else 'N/A'}")  # 打印最终状态
        self.assertTrue(np.all(U_final_np[:, 0] >= -1e-9), "Water depth should be non-negative.")  # 断言水深非负

        print("  Test 'Model Run (All Walls)' completed execution.")  # 打印测试完成信息

    def test_model_with_timeseries_boundary_setup(self):  # 测试带时间序列边界的设置
        print("\n--- Test: Model Setup with Timeseries BC ---")  # 打印测试信息
        model = HydroModelCore_cpp_py(self.mesh_cpp)  # 创建C++水动力模型核心对象
        model.set_simulation_parameters(9.81, 1e-6, 0.5, 1.0, 0.1, 0.1)  # 设置模拟参数
        model.set_numerical_schemes(ReconScheme_cpp_py.FIRST_ORDER, RiemannSolver_cpp_py.HLLC,
                                    TimeScheme_cpp_py.FORWARD_EULER)  # 设置数值方案

        bc_defs = {}  # 边界定义字典
        # 标记1 (左边界) 设置为水位时间序列
        wl_bc_def = BoundaryDef_cpp_py()  # 创建水位边界定义
        wl_bc_def.type = BoundaryType_cpp_py.WATERLEVEL_TIMESERIES  # 设置类型为水位时间序列
        bc_defs[1] = wl_bc_def  # 标记1为水位时间序列

        # 标记2 (下边界) 设置为墙体
        wall_def = BoundaryDef_cpp_py()  # 创建墙体边界定义
        wall_def.type = BoundaryType_cpp_py.WALL  # 设置类型为墙体
        bc_defs[2] = wall_def  # 标记2为墙体

        # 标记3 (上/右边界) 设置为自由出流
        free_out_def = BoundaryDef_cpp_py()  # 创建自由出流边界定义
        free_out_def.type = BoundaryType_cpp_py.FREE_OUTFLOW  # 设置类型为自由出流
        bc_defs[3] = free_out_def  # 标记3为自由出流

        # 创建虚拟时间序列数据
        wl_ts_data_py = {  # 水位时间序列数据 (Python字典)
            1: [  # 标记1的时间序列
                TimeseriesPoint_cpp_py(time=0.0, value=1.0),  # 时间点0
                TimeseriesPoint_cpp_py(time=0.5, value=1.2),  # 时间点0.5
                TimeseriesPoint_cpp_py(time=1.0, value=0.8)  # 时间点1.0
            ]  # 结束标记1的时间序列
        }  # 结束水位时间序列数据
        # discharge_ts_data_py 暂时为空
        discharge_ts_data_py = {}  # 流量时间序列数据

        try:  # 尝试设置边界条件
            model.setup_boundary_conditions_cpp(bc_defs, wl_ts_data_py, discharge_ts_data_py)  # 设置边界条件
            print("  Boundary conditions with timeseries configured successfully.")  # 打印配置成功信息

            # 设置一个简单的初始条件
            U_init = [[0.8, 0.0, 0.0], [0.8, 0.0, 0.0]]  # 初始守恒量
            eta_init = [0.8, 0.8]  # 初始水位
            model.set_initial_conditions_cpp(U_init, eta_init)  # 设置初始条件

            # 尝试运行一步，看看是否会因为未完全实现的边界类型而出错（目前应该退化为墙）
            print("  Attempting one step with mixed BCs (some are placeholders)...")  # 打印尝试步骤信息
            model.advance_one_step()  # 执行一步
            print("  One step with mixed BCs executed.")  # 打印执行成功信息

        except Exception as e:  # 捕获异常
            self.fail(f"Model setup or step with timeseries BC failed: {e}")  # 标记测试失败


if __name__ == '__main__':  # 如果是主程序
    unittest.main(verbosity=2)  # 运行测试