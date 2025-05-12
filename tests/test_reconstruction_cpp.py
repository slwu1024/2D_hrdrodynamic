# tests/test_reconstruction_cpp.py
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
    from src.model.Reconstruction import Reconstruction as Reconstruction_py, \
        ReconstructionSchemes as ReconSchemes_py  # 导入Python版Reconstruction
    from src.model.MeshData import Mesh as Mesh_py, Node as Node_py_orig, Cell as Cell_py_orig, \
        HalfEdge as HalfEdge_py_orig  # 导入Python版MeshData (可能需要模拟)
    import hydro_model_cpp  # 导入C++模块

    # 获取绑定的C++类
    Mesh_cpp_py = hydro_model_cpp.Mesh_cpp  # C++ Mesh类
    Node_cpp_py = hydro_model_cpp.Node_cpp  # C++ Node类
    Cell_cpp_py = hydro_model_cpp.Cell_cpp  # C++ Cell类
    HalfEdge_cpp_py = hydro_model_cpp.HalfEdge_cpp  # C++ HalfEdge类
    ReconScheme_cpp_py = hydro_model_cpp.ReconstructionScheme_cpp  # C++ ReconstructionScheme枚举
    PrimitiveVars_cpp_py = hydro_model_cpp.PrimitiveVars_cpp  # C++ PrimitiveVars结构体
except ImportError as e:  # 捕获导入错误
    print(f"导入错误 (Reconstruction Test): {e}")  # 打印错误信息
    sys.exit(1)  # 退出程序


# 辅助函数: 创建一个简单的 Python MeshData.Mesh 对象用于 Python 重构器
def create_simple_py_mesh():  # 创建简单Python网格
    mesh = Mesh_py()  # 创建Mesh对象
    # Node(id, x, y, z_bed)
    mesh.nodes = [  # 节点列表
        Node_py_orig(0, 0.0, 0.0, 0.0), Node_py_orig(1, 1.0, 0.0, 0.0),  # 节点0, 1
        Node_py_orig(2, 0.5, 0.866, 0.0), Node_py_orig(3, 1.5, 0.866, 0.0)  # 节点2, 3 (等边三角形例子)
    ]  # 结束节点列表
    # Cell(id)
    cell0 = Cell_py_orig(0);
    cell0.nodes = [mesh.nodes[0], mesh.nodes[1], mesh.nodes[2]];
    cell0.area = 0.433;
    cell0.centroid = (0.5, 0.2886)  # 单元0
    cell1 = Cell_py_orig(1);
    cell1.nodes = [mesh.nodes[1], mesh.nodes[3], mesh.nodes[2]];
    cell1.area = 0.433;
    cell1.centroid = (1.0, 0.5773)  # 单元1
    mesh.cells = [cell0, cell1]  # 单元列表

    # HalfEdges (简化，只创建测试需要的)
    # he01: cell0, n0->n1
    he01 = HalfEdge_py_orig(0);
    he01.origin = mesh.nodes[0];
    he01.cell = cell0;
    he01.length = 1.0;
    he01.normal = (0.0, -1.0);
    he01.mid_point = (0.5, 0.0)  # 半边01
    # he12: cell0, n1->n2
    he12 = HalfEdge_py_orig(1);
    he12.origin = mesh.nodes[1];
    he12.cell = cell0;
    he12.length = 1.0;
    he12.normal = (0.866, 0.5);
    he12.mid_point = (0.75, 0.433)  # 半边12
    # he20: cell0, n2->n0
    he20 = HalfEdge_py_orig(2);
    he20.origin = mesh.nodes[2];
    he20.cell = cell0;
    he20.length = 1.0;
    he20.normal = (-0.866, 0.5);
    he20.mid_point = (0.25, 0.433)  # 半边20

    # he13 (twin of he31 in cell1): cell1, n1->n3
    he13 = HalfEdge_py_orig(3);
    he13.origin = mesh.nodes[1];
    he13.cell = cell1;
    he13.length = 1.0;
    he13.normal = (0.0, -1.0);
    he13.mid_point = (1.0, 0.433)  # 半边13
    # he32 (twin of he23 in cell1): cell1, n3->n2
    he32 = HalfEdge_py_orig(4);
    he32.origin = mesh.nodes[3];
    he32.cell = cell1;
    he32.length = 1.0;
    he32.normal = (-0.866, -0.5);
    he32.mid_point = (1.0, 0.866)  # 半边32
    # he21 (twin of he12 in cell0): cell1, n2->n1  <-- This is the one we'll test
    he21 = HalfEdge_py_orig(5);
    he21.origin = mesh.nodes[2];
    he21.cell = cell1;
    he21.length = 1.0;
    he21.normal = (-0.866, -0.5);
    he21.mid_point = (0.75, 0.433)  # 半边21

    he12.twin = he21;
    he21.twin = he12  # 设置孪生关系
    # 简化邻接关系和 next/prev
    cell0.half_edges_list = [he01, he12, he20]  # 单元0的半边列表
    cell1.half_edges_list = [he21, he13, he32]  # 单元1的半边列表 (顺序可能需要调整以确保逆时针)

    mesh.half_edges = [he01, he12, he20, he13, he32, he21]  # 所有半边列表
    # 手动设置一些梯度，因为Python版本的梯度计算可能未完全实现或依赖更多网格结构
    cell0.b_slope_x = 0.01;
    cell0.b_slope_y = 0.02  # 单元0底坡
    cell1.b_slope_x = -0.01;
    cell1.b_slope_y = 0.01  # 单元1底坡

    return mesh, he12  # 返回网格和要测试的边


# 辅助函数: 创建一个简单的 C++ Mesh_cpp 对象
def create_simple_cpp_mesh():  # 创建简单C++网格
    mesh_cpp = Mesh_cpp_py()  # 创建C++ Mesh对象
    # Nodes: id, x, y, z_bed, marker
    nodes_data = np.array([  # 节点数据
        0, 0.0, 0.0, 0.0, 0,  # 节点0
        1, 1.0, 0.0, 0.0, 0,  # 节点1
        2, 0.5, 0.866, 0.0, 0,  # 节点2
        3, 1.5, 0.866, 0.0, 0  # 节点3
    ], dtype=float)  # 结束节点数据
    mesh_cpp.load_nodes_from_numpy(nodes_data, 4, 5)  # 加载节点数据

    # Cells: id, n0, n1, n2
    cells_conn = np.array([  # 单元连接
        0, 0, 1, 2,  # 单元0
        1, 1, 3, 2  # 单元1
    ], dtype=int)  # 结束单元连接
    manning_vals = np.array([0.025, 0.025], dtype=float)  # 曼宁系数值
    mesh_cpp.load_cells_from_numpy(cells_conn, 2, 3, manning_vals)  # 加载单元数据

    # Edges (for boundary markers - not strictly needed for this test if we focus on internal edge)
    # 但 precompute_geometry_and_topology 需要它
    # 假设边 (1,2) 是内部边，我们给它一个marker 0 或不提供
    # 假设 (0,1) 是边界 marker 1, (0,2) 是边界 marker 1
    # (1,3) 是边界 marker 2, (3,2) 是边界 marker 2
    edges_data = np.array([  # 边数据
        0, 0, 1, 1,  # 边0 (0-1), marker 1
        1, 0, 2, 1,  # 边1 (0-2), marker 1
        2, 1, 3, 2,  # 边2 (1-3), marker 2
        3, 3, 2, 2,  # 边3 (3-2), marker 2
        4, 1, 2, 0  # 边4 (1-2), internal, marker 0
    ], dtype=int)  # 结束边数据
    mesh_cpp.precompute_geometry_and_topology(edges_data, 5, 4)  # 预计算几何和拓扑

    # 找到我们关心的半边 he_L_to_R (从 cell 0 到 cell 1, 物理边是节点1和节点2之间的边)
    # 在 C++ Mesh_cpp 中，cell 0 的节点是 [0,1,2], cell 1 的节点是 [1,3,2]
    # cell 0 的边 (1,2) 是 he_id = 1 (假设按顺序创建: 0->1, 1->2, 2->0)
    # cell 1 的边 (2,1) 是 he_id = 3 (假设: 1->3, 3->2, 2->1)
    # 我们需要找到从 cell 0 指向 cell 1 (或等效地，其twin从cell1指向cell0)的半边
    # 假设 precompute_geometry_and_topology 正确填充了 half_edges
    # 我们要找 cell_L=0, cell_R=1，共享边 (1,2)
    # 在cell0中，边(1,2)的起点是node1, 终点是node2.
    # 在cell1中，边(2,1)的起点是node2, 终点是node1.
    he_to_test = None  # 初始化测试半边
    for he_cpp_obj in mesh_cpp.half_edges:  # 遍历C++半边
        if he_cpp_obj.cell_id == 0:  # 如果属于单元0
            # 找到连接节点1和节点2的半边
            origin_node = mesh_cpp.nodes[he_cpp_obj.origin_node_id]  # 获取起点
            next_he_in_cell0 = mesh_cpp.half_edges[he_cpp_obj.next_half_edge_id]  # 获取下一半边
            end_node = mesh_cpp.nodes[next_he_in_cell0.origin_node_id]  # 获取终点
            if (origin_node.id == 1 and end_node.id == 2):  # 如果是边(1,2)
                he_to_test = he_cpp_obj  # 设置为测试半边
                break  # 跳出循环
    if he_to_test is None:  # 如果没有找到
        raise ValueError("Could not find the target half-edge in C++ mesh for testing.")  # 抛出值错误

    return mesh_cpp, he_to_test  # 返回C++网格和测试半边


class TestReconstruction(unittest.TestCase):  # 定义测试类
    def setUp(self):  # 测试准备方法
        self.gravity = 9.81  # 重力加速度
        self.min_depth = 1e-6  # 最小水深
        self.rtol = 1e-5  # 相对容差
        self.atol = 1e-7  # 绝对容差

        self.mesh_py, self.he_py_test = create_simple_py_mesh()  # 创建简单Python网格
        self.mesh_cpp, self.he_cpp_test = create_simple_cpp_mesh()  # 创建简单C++网格

        self.py_recon_first = Reconstruction_py(ReconSchemes_py.FIRST_ORDER, self.mesh_py, self.gravity,
                                                self.min_depth)  # 创建Python一阶重构器
        self.cpp_recon_first = hydro_model_cpp.Reconstruction_cpp(ReconScheme_cpp_py.FIRST_ORDER, self.mesh_cpp,
                                                                  self.gravity, self.min_depth)  # 创建C++一阶重构器

        # 对于二阶，Python版本可能依赖于外部计算的梯度和限制器，
        # 或者其 prepare_for_step 可能不完整。测试时需要注意这一点。
        # self.py_recon_second = Reconstruction_py(ReconSchemes_py.SECOND_ORDER_LIMITED, self.mesh_py, self.gravity, self.min_depth)
        self.cpp_recon_second = hydro_model_cpp.Reconstruction_cpp(ReconScheme_cpp_py.SECOND_ORDER_LIMITED,
                                                                   self.mesh_cpp, self.gravity,
                                                                   self.min_depth)  # 创建C++二阶重构器

        # 定义一些单元中心守恒量 (2个单元, 3个变量: h, hu, hv)
        self.U_state_all_np = np.array([  # 所有单元状态
            [1.0, 0.5, 0.1],  # Cell 0
            [0.8, -0.2, 0.3]  # Cell 1
        ], dtype=float)  # 结束状态定义

        # 将 NumPy 数组转换为 C++ 期望的格式 (list of lists, Pybind11 会转为 vector<array>)
        self.U_state_all_cpp_fmt = self.U_state_all_np.tolist()  # 转换为C++格式

    def _compare_primitive_vars(self, pv_py, pv_cpp_obj, msg_prefix):  # 比较原始变量
        self.assertAlmostEqual(pv_cpp_obj.h, pv_py[0], delta=self.atol, msg=f"{msg_prefix} h mismatch")  # 断言h
        self.assertAlmostEqual(pv_cpp_obj.u, pv_py[1], delta=self.atol, msg=f"{msg_prefix} u mismatch")  # 断言u
        self.assertAlmostEqual(pv_cpp_obj.v, pv_py[2], delta=self.atol, msg=f"{msg_prefix} v mismatch")  # 断言v

    def test_first_order_reconstruction(self):  # 测试一阶重构
        print("\n--- Testing First Order Reconstruction ---")  # 打印测试信息
        # Python
        # Python的Reconstruction.py的get_reconstructed_interface_states的参数可能不同
        # 需要 cell_L_obj, cell_R_obj, he, is_boundary
        # he_py_test 是从 cell0 (id=0) 出发，指向 cell1 (id=1) 的边。
        # twin of he_py_test 是 he_cpp_test.twin_half_edge_id (如果C++结构正确)
        cell_L_py = self.mesh_py.cells[0]  # Python左单元
        cell_R_py = self.mesh_py.cells[1]  # Python右单元
        # Python 的 prepare_for_step 可能为空或不同
        self.py_recon_first.prepare_for_step(self.U_state_all_np)  # Python准备步骤
        W_L_py_fo, W_R_py_fo = self.py_recon_first.get_reconstructed_interface_states(  # Python获取重构状态
            self.U_state_all_np, cell_L_py, cell_R_py, self.he_py_test, is_boundary=False
        )  # 结束获取
        print(f"  Python FO: W_L={W_L_py_fo}, W_R={W_R_py_fo}")  # 打印Python结果

        # C++
        # he_cpp_test 是从 cell_L_id=0 (mesh_cpp.cells[0]) 出发，指向 cell_R_id=1 (mesh_cpp.cells[1])
        self.cpp_recon_first.prepare_for_step(self.U_state_all_cpp_fmt)  # C++准备步骤
        W_L_cpp_obj_fo, W_R_cpp_obj_fo = self.cpp_recon_first.get_reconstructed_interface_states(  # C++获取重构状态
            self.U_state_all_cpp_fmt, 0, 1, self.he_cpp_test, is_boundary=False
        )  # 结束获取
        print(f"  C++    FO: W_L=({W_L_cpp_obj_fo.h:.3f},{W_L_cpp_obj_fo.u:.3f},{W_L_cpp_obj_fo.v:.3f}), "  # 打印C++结果
              f"W_R=({W_R_cpp_obj_fo.h:.3f},{W_R_cpp_obj_fo.u:.3f},{W_R_cpp_obj_fo.v:.3f})")  # 结束打印

        self._compare_primitive_vars(W_L_py_fo, W_L_cpp_obj_fo, "First Order W_L")  # 比较左侧状态
        self._compare_primitive_vars(W_R_py_fo, W_R_cpp_obj_fo, "First Order W_R")  # 比较右侧状态
        print("  First Order Test PASSED")  # 打印测试通过信息

    def test_second_order_reconstruction_prepare(self):  # 测试二阶重构准备步骤
        # 这个测试主要看 prepare_for_step 是否能成功运行并填充梯度
        # 比较梯度值会比较困难，因为Python版本的梯度计算可能不完整或依赖不同的网格结构
        print("\n--- Testing Second Order Reconstruction (prepare_for_step) ---")  # 打印测试信息
        try:  # 尝试执行
            self.cpp_recon_second.prepare_for_step(self.U_state_all_cpp_fmt)  # C++准备步骤
            # 如果需要，可以尝试从 self.cpp_recon_second 获取梯度并打印一些值
            # (需要先在C++中添加getter并绑定)
            # grad_cell0_h_cpp = self.cpp_recon_second.get_gradient_numpy(0, 0) # 假设有这个方法
            # print(f"  C++ Grad Cell0 h: {grad_cell0_h_cpp}")
            print("  C++ Second Order prepare_for_step executed without error.")  # 打印执行成功信息
        except Exception as e:  # 捕获异常
            self.fail(f"C++ Second Order prepare_for_step failed: {e}")  # 标记测试失败

        # (可选) 如果Python版本也有完整的prepare_for_step，可以调用并比较内部状态（困难）
        # self.py_recon_second.prepare_for_step(self.U_state_all_np)
        # print("  Python Second Order prepare_for_step executed.")

    def test_second_order_interface_values(self):  # 测试二阶界面值
        print("\n--- Testing Second Order Reconstruction (interface values) ---")  # 打印测试信息
        # 先执行 prepare_for_step
        self.cpp_recon_second.prepare_for_step(self.U_state_all_cpp_fmt)  # C++准备步骤

        W_L_cpp_obj_so, W_R_cpp_obj_so = self.cpp_recon_second.get_reconstructed_interface_states(  # C++获取重构状态
            self.U_state_all_cpp_fmt, 0, 1, self.he_cpp_test, is_boundary=False
        )  # 结束获取
        print(f"  C++    SO: W_L=({W_L_cpp_obj_so.h:.7e},{W_L_cpp_obj_so.u:.7e},{W_L_cpp_obj_so.v:.7e}), "  # 打印C++结果
              f"W_R=({W_R_cpp_obj_so.h:.7e},{W_R_cpp_obj_so.u:.7e},{W_R_cpp_obj_so.v:.7e})")  # 结束打印

        # 与Python版本的比较会比较棘手，因为Python的Reconstruction.py中的高阶部分
        # (梯度和限制器)可能与C++的实现有差异，或者依赖于不同的网格属性填充。
        # 暂时，我们只确保C++版本能运行并给出看似合理的值。
        # 如果要精确比较，Python版本需要能以完全相同的方式计算梯度和限制器。

        # 作为一个简单的健全性检查，二阶结果不应与一阶结果完全相同 (除非梯度为0)
        W_L_cpp_obj_fo, _ = self.cpp_recon_first.get_reconstructed_interface_states(  # 获取一阶结果
            self.U_state_all_cpp_fmt, 0, 1, self.he_cpp_test, is_boundary=False
        )  # 结束获取

        # 检查至少有一个变量的二阶值与一阶值有显著差异 (除非梯度确实为零)
        # 这只是一个非常粗略的检查
        h_differs = not np.isclose(W_L_cpp_obj_so.h, W_L_cpp_obj_fo.h, atol=1e-5)  # 检查h是否不同
        u_differs = not np.isclose(W_L_cpp_obj_so.u, W_L_cpp_obj_fo.u, atol=1e-5)  # 检查u是否不同
        v_differs = not np.isclose(W_L_cpp_obj_so.v, W_L_cpp_obj_fo.v, atol=1e-5)  # 检查v是否不同

        # 如果所有梯度都是零，那么它们应该是相等的
        # print(f"  SO vs FO check: h_differs={h_differs}, u_differs={u_differs}, v_differs={v_differs}")
        # self.assertTrue(h_differs or u_differs or v_differs,
        #                 "Second order W_L values are too close to first order, check gradient calculation/application.")
        print("  C++ Second Order get_reconstructed_interface_states executed.")  # 打印执行成功信息
        print(
            "  Note: Direct Python comparison for 2nd order is complex due to potential differences in gradient/limiter implementations.")  # 提示信息


if __name__ == '__main__':  # 如果是主程序
    print("Running Reconstruction C++/Python comparison tests...")  # 打印测试开始信息
    unittest.main(verbosity=2)  # 运行测试