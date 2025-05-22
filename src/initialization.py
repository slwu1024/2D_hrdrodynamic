# initialization.py (修改 load_mesh_data_structure)
# 在文件顶部导入
import numpy as np
try:
    import hydro_model_cpp # 导入编译好的C++模块
except ImportError:
    print("错误: 未找到 C++ 核心模块 'hydro_model_cpp'。请确保已编译并放置在正确路径。")
    # 可以选择退出或回退到纯Python实现 (如果保留了的话)
    exit()

# ... (load_node_data, load_cell_data, load_edge_data 基本不变, 只是返回NumPy数组)
def load_node_data_for_cpp(filepath): # 加载节点数据以供C++使用
    """从 .node 文件加载节点数据, 返回扁平化的NumPy数组和数量信息。"""
    # ... (读取逻辑同原来的 load_node_data) ...
    # 最终需要返回:
    # flat_data = np.array([id0,x0,y0,z0,m0, id1,x1,y1,z1,m1,...])
    # num_nodes, num_attrs (例如5)
    nodes_list_py = [] # 临时Python列表
    num_nodes = 0 # 初始化节点数量
    num_attrs = 0 # 初始化属性数量
    try:
        with open(filepath, 'r') as f: # 打开文件
            header = f.readline().split() # 读取头部
            num_nodes = int(header[0]) # 获取节点数
            # dim = int(header[1]) # 维度
            # point_attrs_count = int(header[2]) # 属性数
            has_marker = int(header[3]) == 1 if len(header) > 3 else False # 是否有标记
            num_attrs = 4 + (1 if has_marker else 0) # id,x,y,z_bed (+marker)

            flat_data_list = [] # 扁平化数据列表
            for i in range(num_nodes): # 遍历节点
                line = f.readline().split() # 读取行
                node_id = int(line[0]) # 节点ID
                x = float(line[1]) # x坐标
                y = float(line[2]) # y坐标
                z_bed = float(line[3]) # z_bed
                marker = int(line[4]) if has_marker and len(line) > 4 else 0 # 标记

                flat_data_list.extend([float(node_id), x, y, z_bed]) # 添加数据
                if has_marker: flat_data_list.append(float(marker)) # 如果有标记则添加
        print(f"  为C++加载了 {num_nodes} 个节点，每个节点 {num_attrs} 个属性。") # 打印信息
        return np.array(flat_data_list, dtype=float), num_nodes, num_attrs # 返回扁平化数组和数量
    except FileNotFoundError: # 文件未找到
        print(f"  错误: 节点文件 {filepath} 未找到。") # 打印错误
        return None, 0, 0 # 返回None
    except Exception as e: # 其他错误
        print(f"  读取节点文件 {filepath} 时出错: {e}") # 打印错误
        return None, 0, 0 # 返回None


def load_cell_data_for_cpp(filepath): # 加载单元数据以供C++使用
    """从 .cell 文件加载单元连接数据, 返回扁平化的NumPy数组和数量信息。"""
    # flat_data = np.array([id0,n0_0,n1_0,n2_0, id1,n0_1,n1_1,n2_1,...])
    # num_cells, nodes_per_cell (例如3)
    elements_list_py = [] # 临时Python列表
    num_elements = 0 # 初始化单元数量
    nodes_per_ele = 0 # 初始化每单元节点数
    try:
        with open(filepath, 'r') as f:
            header = f.readline().split()
            num_elements = int(header[0])
            nodes_per_ele = int(header[1])
            num_cell_attrs = int(header[2]) if len(header) > 2 else 0  # 新增：读取单元属性数量

            if nodes_per_ele != 3:
                print(f"错误: {filepath} ...仅支持3节点单元。")
                return None, 0, 0, None  # 返回 None 代表属性数组

            flat_data_list = []
            cell_attributes_list = []  # 新增：存储单元属性

            for i in range(num_elements):
                line = f.readline().split()
                ele_id = int(line[0])
                node_ids_for_cell = [int(n_id) for n_id in line[1:nodes_per_ele + 1]]
                flat_data_list.append(ele_id)
                flat_data_list.extend(node_ids_for_cell)

                if num_cell_attrs > 0 and len(line) > nodes_per_ele + 1:
                    # 假设第一个属性是区域属性
                    attr_val = float(line[nodes_per_ele + 1])
                    cell_attributes_list.append(attr_val)
                else:
                    cell_attributes_list.append(0.0)  # 默认属性

        print(f"  为C++加载了 {num_elements} 个单元，每个单元 {nodes_per_ele} 个节点，{num_cell_attrs} 个属性。")
        return np.array(flat_data_list, dtype=int), num_elements, nodes_per_ele, np.array(cell_attributes_list,
                                                                                          dtype=float)  # 返回属性数组
    except FileNotFoundError: # 文件未找到
        print(f"  错误: 单元文件 {filepath} 未找到。") # 打印错误
        return None, 0, 0 # 返回None
    except Exception as e: # 其他错误
        print(f"  读取单元文件 {filepath} 时出错: {e}") # 打印错误
        return None, 0, 0,None # 返回None


def load_edge_data_for_cpp(filepath): # 加载边数据以供C++使用
    """从 .edge 文件加载边数据, 返回扁平化的NumPy数组和数量信息。"""
    # flat_data = np.array([id0,n0_0,n1_0,m0, id1,n0_1,n1_1,m1,...])
    # num_edges, num_attrs (例如4)
    num_edges = 0 # 初始化边数量
    num_attrs = 0 # 初始化属性数量
    try:
        with open(filepath, 'r') as f: # 打开文件
            header = f.readline().split() # 读取头部
            num_edges = int(header[0]) # 获取边数量
            has_marker = int(header[1]) == 1 if len(header) > 1 else False # 是否有标记
            num_attrs = 3 + (1 if has_marker else 0) # id,n1,n2 (+marker)

            flat_data_list = [] # 扁平化数据列表
            for i in range(num_edges): # 遍历边
                line = f.readline().split() # 读取行
                edge_id = int(line[0]) # 边ID
                node1_id = int(line[1]) # 节点1 ID
                node2_id = int(line[2]) # 节点2 ID
                marker = int(line[3]) if has_marker and len(line) > 3 else 0 # 标记

                flat_data_list.extend([edge_id, node1_id, node2_id]) # 添加数据
                if has_marker: flat_data_list.append(marker) # 如果有标记则添加
        print(f"  为C++加载了 {num_edges} 条边，每条边 {num_attrs} 个属性。") # 打印信息
        return np.array(flat_data_list, dtype=int), num_edges, num_attrs # 返回扁平化数组和数量
    except FileNotFoundError: # 文件未找到
        print(f"  警告: 边文件 {filepath} 未找到。") # 打印警告
        return None, 0, 0 # 返回None
    except Exception as e: # 其他错误
        print(f"  读取边文件 {filepath} 时出错: {e}") # 打印错误
        return None, 0, 0 # 返回None


def load_mesh_data_structure_cpp(node_file, cell_file, edge_file, cell_manning_values_np: np.ndarray): # 加载网格数据结构(C++)
    """
    从 .node, .cell, .edge 文件加载数据到 C++ Mesh_cpp 对象。
    """
    mesh_cpp_obj = hydro_model_cpp.Mesh_cpp() # 创建C++网格对象
    print(f"开始加载和设置网格结构 (C++ 核心):") # 打印信息

    # 1. 加载节点
    print("步骤 1: 加载节点数据到C++...") # 打印步骤1
    flat_nodes_np, num_nodes, node_attrs = load_node_data_for_cpp(node_file) # 加载节点数据
    if flat_nodes_np is None: return None # 如果失败则返回None
    mesh_cpp_obj.load_nodes_from_numpy(flat_nodes_np, num_nodes, node_attrs) # 调用C++方法加载

    # 2. 加载单元
    print("步骤 2: 加载单元数据到C++...") # 打印步骤2
    flat_cells_np, num_cells, nodes_per_cell, cell_attributes_np = load_cell_data_for_cpp(cell_file)  # 获取属性
    if flat_cells_np is None: return None
    # 确保 manning_values 长度与 num_cells 一致，如果需要传递
    if len(cell_manning_values_np) != num_cells: # 如果长度不一致
        print(f"警告: 提供的曼宁系数值数量 ({len(cell_manning_values_np)}) 与单元数量 ({num_cells}) 不符。将使用默认值或重复值。") # 打印警告
        # C++端 load_cells_from_numpy 会处理不匹配的情况，这里只是Python端的提醒
    mesh_cpp_obj.load_cells_from_numpy(flat_cells_np, num_cells, nodes_per_cell, cell_manning_values_np,
                                       cell_attributes_np)

    # 3. 加载边数据 (用于边界标记)
    print("步骤 3: 加载边数据 (用于边界标记) 到C++...") # 打印步骤3
    flat_edges_np, num_edges, edge_attrs = load_edge_data_for_cpp(edge_file) # 加载边数据
    if flat_edges_np is None: # 如果边文件不存在或读取失败，仍然继续，C++端会处理
        flat_edges_np = np.array([], dtype=int) # 传递空数组
        num_edges = 0 # 边数量为0
        edge_attrs = 0 # 属性数量为0

    # 4. 执行C++端的几何和拓扑预计算
    print("步骤 4: 在C++中预计算几何和拓扑...") # 打印步骤4
    mesh_cpp_obj.precompute_geometry_and_topology(flat_edges_np, num_edges, edge_attrs) # 调用C++方法预计算

    print("C++ 网格加载和设置完成。") # 打印完成信息
    return mesh_cpp_obj # 返回C++网格对象