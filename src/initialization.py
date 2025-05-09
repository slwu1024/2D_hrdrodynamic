# initialization.py
import numpy as np  # 导入 NumPy 用于数值运算
from src.model.MeshData import Mesh, Node, Cell, HalfEdge

# --- 数据加载函数 (与之前基本一致, 细微调整打印信息) ---
def load_node_data(filepath):  # 定义加载节点数据的函数
    """从最终的 .node 文件加载节点数据 (含 Z_bed 和 Marker)"""
    nodes_list = []  # 初始化节点列表
    try:
        with open(filepath, 'r') as f:  # 打开节点文件进行读取
            header = f.readline().split()  # 读取并分割文件头行
            num_nodes = int(header[0])  # 获取节点数量
            # 预期格式: <#nodes> 2 0 1 (节点数 维度 属性数 是否有标记)
            has_marker = int(header[3]) == 1 if len(header) > 3 else False  # 判断是否有节点标记
            print(f"  从 {filepath} 读取 {num_nodes} 个节点 (边界标记: {has_marker})...")  # 打印读取信息
            nodes_list = [None] * num_nodes  # 初始化一个固定大小的列表以存储节点对象
            for i in range(num_nodes):  # 遍历每一行节点数据
                line = f.readline().split()  # 读取并分割数据行
                node_id = int(line[0])  # 获取节点ID
                x = float(line[1])  # 获取x坐标
                y = float(line[2])  # 获取y坐标
                z_bed = float(line[3])  # 获取底高程 Z_bed
                marker = int(line[4]) if has_marker and len(line) > 4 else 0  # 获取节点标记，如果存在
                if node_id != i:  # 检查文件中的节点ID是否与期望的索引一致
                    print(
                        f"    警告: {filepath} 文件中节点 ID 不匹配 (行 {i + 2}, 文件索引 {i}). 期望 {i}, 得到 {node_id}.")  # 打印ID不匹配的警告
                # 假设 node_id 总是从0开始且连续，用列表索引 i 作为实际ID
                nodes_list[i] = Node(id=i, x=x, y=y, z_bed=z_bed)  # 创建 Node 对象并存储，使用行号i作为ID
                nodes_list[i].marker = marker  # 设置节点标记
            print("  节点数据读取成功。")  # 打印成功信息
    except FileNotFoundError:  # 捕获文件未找到异常
        print(f"  错误: 节点文件 {filepath} 未找到。")  # 打印错误信息
        return None  # 返回 None
    except Exception as e:  # 捕获其他异常
        print(f"  读取节点文件 {filepath} 时出错: {e}")  # 打印错误信息
        return None  # 返回 None
    return nodes_list  # 返回节点对象列表


def load_cell_data(filepath):  # 定义加载单元数据的函数
    """从最终的 .cell 文件加载单元数据 (返回节点ID列表的列表)"""
    elements_node_ids = []  # 初始化单元节点ID列表
    try:
        with open(filepath, 'r') as f:  # 打开单元文件进行读取
            header = f.readline().split()  # 读取并分割文件头行
            num_elements = int(header[0])  # 获取单元数量
            nodes_per_ele = int(header[1])  # 获取每个单元的节点数
            if nodes_per_ele != 3:  # 检查每个单元的节点数是否为3 (三角形)
                print(f"  错误: {filepath} 文件指示每个单元有 {nodes_per_ele} 个节点，但期望是 3。")  # 打印错误信息
                return None  # 返回 None
            print(f"  从 {filepath} 读取 {num_elements} 个单元...")  # 打印读取信息
            elements_node_ids = [None] * num_elements  # 初始化一个固定大小的列表以存储单元的节点ID
            for i in range(num_elements):  # 遍历每一行单元数据
                line = f.readline().split()  # 读取并分割数据行
                ele_id = int(line[0])  # 获取单元ID
                node_ids_for_cell = [int(n_id) for n_id in line[1:4]]  # 获取组成该单元的3个节点ID
                if ele_id != i:  # 检查文件中的单元ID是否与期望的索引一致
                    print(
                        f"    警告: {filepath} 文件中单元 ID 不匹配 (行 {i + 2}, 文件索引 {i}). 期望 {i}, 得到 {ele_id}.")  # 打印ID不匹配的警告
                if len(node_ids_for_cell) != 3:  # 再次确认节点ID数量是否为3
                    print(
                        f"  错误: 单元 {ele_id} (文件索引 {i}) 的节点 ID 数量不正确 ({len(node_ids_for_cell)})。")  # 打印错误信息
                    return None  # 返回 None
                elements_node_ids[i] = node_ids_for_cell  # 存储该单元的节点ID列表，使用行号i作为索引
            print("  单元数据读取成功。")  # 打印成功信息
    except FileNotFoundError:  # 捕获文件未找到异常
        print(f"  错误: 单元文件 {filepath} 未找到。")  # 打印错误信息
        return None  # 返回 None
    except Exception as e:  # 捕获其他异常
        print(f"  读取单元文件 {filepath} 时出错: {e}")  # 打印错误信息
        return None  # 返回 None
    return elements_node_ids  # 返回包含各单元节点ID的列表


def load_edge_data(filepath):  # 定义加载边数据的函数
    """从最终的 .edge 文件加载边数据和边界标记"""
    edge_marker_map = {}  # 初始化边标记映射字典，键为排序后的 (节点1ID, 节点2ID) 元组
    try:
        with open(filepath, 'r') as f:  # 打开边文件进行读取
            header = f.readline().split()  # 读取并分割文件头行
            num_edges = int(header[0])  # 获取边的数量
            has_marker = int(header[1]) == 1 if len(header) > 1 else False  # 判断是否有边标记
            print(f"  从 {filepath} 读取 {num_edges} 条边 (边界标记: {has_marker})...")  # 打印读取信息
            # .edge 文件格式: <边ID> <节点1 ID> <节点2 ID> [标记]
            for i in range(num_edges):  # 遍历每一行边数据
                line = f.readline().split()  # 读取并分割数据行
                # edge_id = int(line[0]) # 边ID (通常也从0开始)
                node1_id = int(line[1])  # 获取边的第一个节点ID
                node2_id = int(line[2])  # 获取边的第二个节点ID
                marker = int(line[3]) if has_marker and len(line) > 3 else 0  # 获取边标记，如果存在
                edge_key = tuple(sorted((node1_id, node2_id)))  # 创建排序后的节点ID元组作为键
                edge_marker_map[edge_key] = marker  # 存储边标记到字典
            print("  边数据读取成功。")  # 打印成功信息
            return edge_marker_map  # 返回边标记映射字典
    except FileNotFoundError:  # 捕获文件未找到异常
        print(f"  警告: 边文件 {filepath} 未找到。将无法分配边标记。")  # 打印警告信息 (不视为致命错误)
        return None  # 返回 None
    except Exception as e:  # 捕获其他异常
        print(f"  读取边文件 {filepath} 时出错: {e}")  # 打印错误信息
        return None  # 返回 None


# --- 拓扑构建与几何计算函数 ---
def setup_half_edge_structure_optimized(mesh: Mesh):  # 定义设置半边孪生关系的优化函数
    """优化版本：使用字典高效设置半边数据结构中的twin关系。"""
    print("  设置半边孪生关系...")  # 打印开始设置孪生关系的消息
    edge_to_halfedges_map = {}  # 初始化字典，用于映射物理边到其对应的半边(们)
    for he_idx, he in enumerate(mesh.half_edges):  # 遍历所有半边
        if he.origin is None or he.next is None or he.next.origin is None:  # 跳过不完整的半边
            print(
                f"    警告: 半边 {he_idx} (ID {he.id}) 拓扑不完整，跳过孪生设置。Origin: {he.origin}, Next: {he.next}")  # 打印警告
            continue  # 继续下一条半边
        node1_id = he.origin.id  # 获取半边的起始节点ID
        node2_id = he.next.origin.id  # 获取半边的终止节点（即下一条半边的起始节点）ID
        map_key = tuple(sorted((node1_id, node2_id)))  # 创建排序后的节点ID元组作为键
        if map_key not in edge_to_halfedges_map:  # 如果键不存在于字典中
            edge_to_halfedges_map[map_key] = []  # 为该键创建一个空列表
        edge_to_halfedges_map[map_key].append(he)  # 将当前半边添加到对应键的列表中

    twins_found_count = 0  # 初始化找到的孪生半边计数器 (计物理边数)
    boundary_edges_count = 0  # 初始化边界半边计数器

    twins_found_count = 0
    boundary_edges_count = 0
    processed_physical_edges = set()

    for he in mesh.half_edges:  # 再次遍历所有半边来设置孪生关系
        if he.twin is not None: continue  # 如果已经有孪生，则跳过 (可能被前一个匹配的半边设置了)
        if he.origin is None or he.next is None or he.next.origin is None: continue  # 跳过不完整的半边

        node1_id = he.origin.id  # 获取半边的起始节点ID
        node2_id = he.next.origin.id  # 获取半边的终止节点ID
        map_key = tuple(sorted((node1_id, node2_id)))  # 创建键
        candidates = edge_to_halfedges_map.get(map_key, [])

        candidates = edge_to_halfedges_map.get(map_key, [])  # 获取与当前物理边对应的所有半边

        if len(candidates) == 2:  # 如果找到两条半边，它们是孪生关系
            he1, he2 = candidates[0], candidates[1]  # 获取这两条半边
            # 确保它们是不同的半边对象，并且它们的孪生关系还未被设置
            if he1.id != he2.id and he1.twin is None and he2.twin is None:
                he1.twin = he2  # 设置 he1 的孪生为 he2
                he2.twin = he1  # 设置 he2 的孪生为 he1
                if map_key not in processed_physical_edges:  # 如果这个物理边还没被计数
                    twins_found_count += 1  # 孪生物理边数加1
                    processed_physical_edges.add(map_key)  # 标记为已处理
        elif len(candidates) == 1:  # 如果只找到一条半边，它是边界半边
            # 确保这条候选半边就是当前正在处理的 he
            if candidates[0].id == he.id and he.twin is None:
                he.twin = None  # 明确设置其孪生为 None
                if map_key not in processed_physical_edges:  # 如果这个物理边还没被计数
                    boundary_edges_count += 1  # 边界边数加1
                    processed_physical_edges.add(map_key)  # 标记为已处理
        elif len(candidates) > 2:  # 如果多于两条，说明可能有重复的边或退化单元
            print(
                f"    警告: 物理边 ({node1_id}-{node2_id}) 对应了 {len(candidates)} 条半边。这可能表示网格问题。将不设置孪生。")  # 打印警告
            # 这些半边将不会有 twin，后续检查会发现它们是“边界”
        # 如果 len(candidates) == 0，这是不可能的，因为 he 自身就应该在 map 中

    print(
        f"  孪生关系设置完成。找到 {twins_found_count} 条内部物理边 (即 {twins_found_count * 2} 条内部半边) 和 {boundary_edges_count} 条边界物理边。")  # 打印总结信息

    # 验证半边总数
    total_internal_hes = twins_found_count * 2  # 总内部半边数
    # 边界半边的数量应等于 boundary_edges_count (每个物理边界边对应一条半边)
    # 但上面计数的是物理边，实际的边界半边在遍历时逐个识别
    actual_boundary_hes = 0  # 初始化实际边界半边数
    for he_check in mesh.half_edges:  # 遍历所有半边
        if he_check.origin is not None and he_check.next is not None and he_check.next.origin is not None:  # 确保半边是有效的
            if he_check.twin is None:  # 如果没有孪生
                actual_boundary_hes += 1  # 实际边界半边数加1

    if len(mesh.half_edges) != total_internal_hes + actual_boundary_hes:  # 检查总数是否匹配
        print(
            f"    警告: 半边总数检查不匹配。总半边数: {len(mesh.half_edges)}, 计算的内部半边: {total_internal_hes}, 计算的边界半边: {actual_boundary_hes} (合计: {total_internal_hes + actual_boundary_hes})")  # 打印不匹配的警告
        # 进一步诊断
        unaccounted_hes = []  # 未被计入的半边
        for he_check in mesh.half_edges:  # 遍历检查
            if he_check.origin is None or he_check.next is None or he_check.next.origin is None: continue  # 跳过无效半边
            is_internal_by_twin = he_check.twin is not None and he_check.twin.twin == he_check  # 通过孪生关系判断是否内部
            is_counted_as_boundary = he_check.twin is None  # 是否被认为是边界
            if not is_internal_by_twin and not is_counted_as_boundary:  # 如果既不是内部也不是边界
                unaccounted_hes.append(he_check.id)  # 添加到未计入列表

        if unaccounted_hes:  # 如果有未计入的半边
            print(
                f"    发现 {len(unaccounted_hes)} 条半边既没有有效孪生也不是明确的边界: IDs {unaccounted_hes[:10]}...")  # 打印其ID


def assign_boundary_markers_to_halfedges(mesh, edge_marker_map):  # 定义分配边界标记到半边的函数
    """根据读取的边标记信息为边界半边赋值"""
    if edge_marker_map is None:  # 如果没有边标记映射
        print(
            "  警告: 没有提供边标记映射 (可能 .edge 文件未找到或无标记)，无法分配边界标记到半边。所有边界半边标记将为0。")  # 打印警告
        # 确保所有边界半边的 boundary_marker 初始化为0 (HalfEdge 类中默认是0)
        return  # 直接返回
    print("  分配边界标记到边界半边...")  # 打印开始分配标记的消息
    assigned_count = 0  # 初始化已分配标记的计数器
    missing_marker_count = 0  # 初始化未找到标记的计数器
    for he in mesh.half_edges:  # 遍历所有半边
        if he.twin is None:  # 如果是边界半边 (没有孪生)
            if he.origin is None or he.next is None or he.next.origin is None: continue  # 跳过不完整的边
            node1_id = he.origin.id  # 获取半边的起始节点ID
            node2_id = he.next.origin.id  # 获取半边的终止节点ID
            edge_key = tuple(sorted((node1_id, node2_id)))  # 创建排序后的节点ID元组作为键
            # 从 map 中查找标记
            marker = edge_marker_map.get(edge_key)  # 获取标记，如果找不到则为 None
            if marker is not None:  # 如果找到了标记
                he.boundary_marker = marker  # 设置半边的边界标记
                assigned_count += 1  # 增加已分配计数
            else:  # 如果未找到标记
                # print(f"    警告: 边界半边 {he.id} (节点 {node1_id}-{node2_id}) 未在边标记映射中找到。将使用默认标记 0。") # 打印警告 (可选，可能很多)
                missing_marker_count += 1  # 增加未找到计数
                he.boundary_marker = 0  # 使用默认标记0
    print(
        f"  已为 {assigned_count} 条边界半边分配了从文件读取的标记。有 {missing_marker_count} 条边界半边未在文件中找到对应标记 (已设为0)。")  # 打印总结信息


def precompute_geometry_and_validate(mesh: Mesh, validation_level=1):
    # ... (之前的实现，确保在计算边属性时填充 mid_point) ...
    if not mesh or not mesh.cells: return
    print("  预计算几何属性并验证...")
    # ... (错误/警告计数器初始化) ...

    for cell_id, cell in enumerate(mesh.cells):
        if cell is None or cell.half_edge is None or not cell.nodes or len(cell.nodes) != 3:
            # print(f"    跳过无效单元 {cell_id} 的几何计算。")
            continue  # 已在构建时处理过，这里是双重检查

        # 节点坐标和高程已通过 cell.nodes 访问
        node_coords_x = [n.x for n in cell.nodes]
        node_coords_y = [n.y for n in cell.nodes]
        node_z_bed = [n.z_bed for n in cell.nodes]

        area_signed = 0.5 * (node_coords_x[0] * (node_coords_y[1] - node_coords_y[2]) + \
                             node_coords_x[1] * (node_coords_y[2] - node_coords_y[0]) + \
                             node_coords_x[2] * (node_coords_y[0] - node_coords_y[1]))

        if abs(area_signed) < 1e-12:
            # warnings_small_area += 1 ...
            cell.area = 0.0;  # ...
            # 零面积单元的边处理
            for he_in_cell in cell.half_edges_list:  # 使用 cell.half_edges_list
                if he_in_cell.origin and he_in_cell.end_node:
                    n1, n2 = he_in_cell.origin, he_in_cell.end_node
                    he_in_cell.length = np.sqrt((n2.x - n1.x) ** 2 + (n2.y - n1.y) ** 2)
                    he_in_cell.mid_point = ((n1.x + n2.x) / 2, (n1.y + n2.y) / 2)
                he_in_cell.normal = (0.0, 0.0)
            continue

        # if area_signed < 0: warnings_area_sign += 1 ...
        cell.area = abs(area_signed)
        # ... (形心, z_bed_centroid, 底坡计算不变) ...
        cell.centroid = (sum(node_coords_x) / 3.0, sum(node_coords_y) / 3.0)
        cell.z_bed_centroid = sum(node_z_bed) / 3.0
        denominator = 2.0 * area_signed
        cell.b_slope_x = ((node_coords_y[1] - node_coords_y[2]) * node_z_bed[0] + \
                          (node_coords_y[2] - node_coords_y[0]) * node_z_bed[1] + \
                          (node_coords_y[0] - node_coords_y[1]) * node_z_bed[2]) / denominator
        cell.b_slope_y = ((node_coords_x[2] - node_coords_x[1]) * node_z_bed[0] + \
                          (node_coords_x[0] - node_coords_x[2]) * node_z_bed[1] + \
                          (node_coords_x[1] - node_coords_x[0]) * node_z_bed[2]) / denominator

        for current_he in cell.half_edges_list:  # 使用 cell.half_edges_list
            if current_he.origin is None or current_he.next is None or current_he.next.origin is None:
                print(f"    警告: 预计算时单元 {cell.id} 的半边 {current_he.id} 拓扑不完整。")
                continue

            node1 = current_he.origin
            node2 = current_he.next.origin  # End node

            # 设置中点
            current_he.mid_point = ((node1.x + node2.x) / 2.0, (node1.y + node2.y) / 2.0)

            dx = node2.x - node1.x
            dy = node2.y - node1.y
            length = np.sqrt(dx ** 2 + dy ** 2)
            current_he.length = length

            if length < 1e-12:
                current_he.normal = (0.0, 0.0)
                # validation_issues["edge_length_zero"] +=1
            else:
                nx = dy / length
                ny = -dx / length
                current_he.normal = (nx, ny)
                # ... (验证逻辑不变) ...

    print("  几何属性预计算和验证完成。")
    # ... (打印统计信息不变) ...


def fill_node_adjacency_info(mesh: Mesh):  # 新增函数：填充节点的邻接信息
    """填充 Node.incident_cells 和 Node.incident_half_edges"""
    print("  填充节点邻接信息 (incident_cells, incident_half_edges)...")
    # 1. incident_half_edges
    for he in mesh.half_edges:
        if he.origin:
            if he not in he.origin.incident_half_edges:  # 避免重复
                he.origin.incident_half_edges.append(he)

    # 2. incident_cells
    for cell_obj in mesh.cells:
        if cell_obj and cell_obj.nodes:  # 确保cell和其node列表有效
            for node_obj in cell_obj.nodes:
                if node_obj and cell_obj not in node_obj.incident_cells:  # 避免重复
                    node_obj.incident_cells.append(cell_obj)
    print("  节点邻接信息填充完毕。")


# --- 主加载函数 ---
def load_mesh_data_structure(node_file, cell_file, edge_file, perform_validation=True):
    """
    从 .node, .cell, .edge 文件加载数据并构建完整的半边网格结构 (Mesh 对象)。
    返回构建好的 Mesh 对象，或在失败时返回 None。
    """
    mesh = Mesh()
    print(f"开始加载和设置网格结构:")
    # ... (打印文件路径) ...

    # 1. 加载节点
    print("步骤 1: 加载节点数据...")  # 打印步骤1标题
    mesh.nodes = load_node_data(node_file)  # 加载节点数据
    if mesh.nodes is None:  # 如果加载失败
        print("错误: 节点数据加载失败。无法继续。")  # 打印错误信息
        return None  # 返回 None
    num_nodes = len(mesh.nodes)  # 获取节点数量

    # 2. 加载单元 (得到节点ID列表的列表)
    print("步骤 2: 加载单元数据...")  # 打印步骤2标题
    element_node_ids_list = load_cell_data(cell_file)  # 加载单元数据
    if element_node_ids_list is None:  # 如果加载失败
        print("错误: 单元数据加载失败。无法继续。")  # 打印错误信息
        return None  # 返回 None
    num_cells_from_file = len(element_node_ids_list)  # 获取单元数量

    # 3. 加载边标记
    print("步骤 3: 加载边数据 (用于边界标记)...")  # 打印步骤3标题
    edge_marker_map = load_edge_data(edge_file)  # 加载边标记数据
    # 即使加载失败 (edge_marker_map is None)，也应该继续构建拓扑，只是没有边界标记

    # 4. 构建 Cell 和 HalfEdge 对象 (基础结构)
    print("步骤 4: 构建 Cell 和 HalfEdge 对象...")  # 打印步骤4标题
    mesh.cells = [None] * num_cells_from_file  # 预分配列表
    mesh.half_edges = []  # 初始化半边对象列表
    half_edge_id_counter = 0  # 初始化半边ID计数器
    invalid_cells_during_build = 0

    for cell_idx_from_file, node_ids_for_cell in enumerate(element_node_ids_list):
        # 检查节点ID是否在有效范围内
        if any(nid >= num_nodes or nid < 0 for nid in node_ids_for_cell):
            print(f"  警告: 单元索引 {cell_idx_from_file} 包含无效节点ID {node_ids_for_cell}。跳过。")
            invalid_cells_during_build += 1
            continue

        nodes_in_this_cell = [mesh.nodes[node_id] for node_id in node_ids_for_cell]
        if any(n is None for n in nodes_in_this_cell):
            print(f"  警告: 单元索引 {cell_idx_from_file} 的节点对象之一为 None。跳过。")
            invalid_cells_during_build += 1
            continue

        # 用文件索引作为Cell的ID
        cell = Cell(id=cell_idx_from_file)
        mesh.cells[cell_idx_from_file] = cell  # 存储到预分配的列表中

        hes_in_this_cell = [None] * 3
        valid_hes_created_for_this_cell = True
        for i in range(3):
            he = HalfEdge(id=half_edge_id_counter)
            he.origin = nodes_in_this_cell[i]
            he.cell = cell
            hes_in_this_cell[i] = he
            mesh.half_edges.append(he)
            half_edge_id_counter += 1

        # 设置 next 和 prev (这是你之前问到的地方)
        for i in range(3):
            hes_in_this_cell[i].next = hes_in_this_cell[(i + 1) % 3]
            hes_in_this_cell[i].prev = hes_in_this_cell[(i - 1 + 3) % 3]  # 使用 hes_in_this_cell

        cell.half_edge = hes_in_this_cell[0]
        cell.half_edges_list = list(hes_in_this_cell)  # 直接用列表副本
        cell.nodes = [he.origin for he in hes_in_this_cell]  # 按顺序存储节点

    # 清理在构建过程中被标记为None的单元 (如果预分配列表并直接赋值)
    if invalid_cells_during_build > 0:
        print(f"  在构建基础对象时，有 {invalid_cells_during_build} 个单元因引用问题被跳过。")
    # 过滤掉None的cell对象
    mesh.cells = [c for c in mesh.cells if c is not None]
    # 如果单元ID需要是连续的，需要重新编号，但这会使ID与文件索引不一致
    # 当前实现下，Cell的ID是其在原始cell文件中的行号（如果都有效的话）
    # 如果有无效单元被移除，mesh.cells列表长度会变小，cell.id可能不连续或超出新列表范围。
    # 为了简单和一致性，确保Cell的ID是它在最终 mesh.cells 列表中的索引
    for new_idx, cell_obj in enumerate(mesh.cells):
        cell_obj.id = new_idx

    if not mesh.cells:  # 如果所有单元都无效
        print("错误: 没有有效的单元被构建。请检查输入文件。")
        return None

    print(f"  基础对象构建完成。总半边数: {len(mesh.half_edges)}, 总有效单元数: {len(mesh.cells)}")

    # 为每个节点设置其 half_edge 指针
    # 需要在填充 incident_half_edges 之后或同时进行
    # (此步骤移到 fill_node_adjacency_info 之后，如果 Node.half_edge 由此设置)

    # 5. 填充节点的邻接信息 (新的步骤)
    fill_node_adjacency_info(mesh)

    # 5.b 为Node.half_edge赋值 (现在 incident_half_edges 已填充)
    for node_obj in mesh.nodes:
        if node_obj.incident_half_edges:
            node_obj.half_edge = node_obj.incident_half_edges[0]
        # else:
        # print(f"  警告: 节点 {node_idx} 没有任何出射半边。这可能是一个孤立点或网格问题。") # (可选警告)

    # 6. 设置 twin 关系
    print("步骤 5: 设置半边孪生关系...")  # 打印步骤5标题
    setup_half_edge_structure_optimized(mesh)  # 调用函数设置孪生关系

    # 7. 设置边界半边的标记
    print("步骤 6: 分配边界标记到边界半边...")  # 打印步骤6标题
    assign_boundary_markers_to_halfedges(mesh, edge_marker_map)  # 调用函数分配边界标记

    # 8. 预计算几何量并验证
    print("步骤 7: 预计算几何属性并验证...")  # 打印步骤7标题
    validation_lvl = 1 if perform_validation else 0  # 根据 perform_validation 设置验证级别
    precompute_geometry_and_validate(mesh, validation_level=validation_lvl)  # 调用函数计算几何量并验证

    print("网格加载和设置完成。")  # 打印完成消息
    return mesh  # 返回构建好的 Mesh 对象


# --- 示例使用 ---
if __name__ == "__main__":  # 如果当前脚本是主程序执行
    # 定义输入文件路径
    node_file = "../mesh/test_topo_triangle_lib.node"  # 节点文件
    cell_file = "../mesh/test_topo_triangle_lib.cell"  # 单元文件
    edge_file = "../mesh/test_topo_triangle_lib.edge"  # 边文件

    # 加载网格，并执行验证
    mesh_object = load_mesh_data_structure(node_file, cell_file, edge_file, perform_validation=True)

    if mesh_object:
        print(f"\n--- 网格加载总结 ---")
        # ... (之前的验证和抽样打印逻辑不变) ...
        # 例如，抽样检查新添加的属性：
        if mesh_object.cells:
            c0 = mesh_object.cells[0]
            print(f"\n示例单元 {c0.id} 拓扑:")
            print(f"  组成节点 IDs: {[n.id for n in c0.nodes]}")
            print(f"  组成半边 IDs: {[h.id for h in c0.half_edges_list]}")
            if c0.nodes:
                n0_of_c0 = c0.nodes[0]
                print(f"  单元的第一个节点 {n0_of_c0.id} 的信息:")
                print(f"    从此节点出发的半边 (示例): {n0_of_c0.half_edge.id if n0_of_c0.half_edge else 'None'}")
                print(f"    共享此节点的单元 IDs: {[inc_cell.id for inc_cell in n0_of_c0.incident_cells]}")

        # 更多检查...
    else:
        print("\n网格对象加载失败。")