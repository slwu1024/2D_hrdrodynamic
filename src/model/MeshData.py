# src/model/MeshData.py (或者 src/mesh.py，并调整HydroModel中的导入)
import numpy as np


class Node:  # 节点类
    """节点数据结构类"""

    def __init__(self, id, x, y, z_bed):  # 初始化方法
        self.id = id  # 节点ID (整数)
        self.x = x  # x坐标
        self.y = y  # y坐标
        self.z_bed = z_bed  # 底高程 Z
        self.marker = 0  # 节点边界标记 (整数, 从 .node 文件读取)

        # --- 拓扑与邻接信息 (在初始化后填充) ---
        self.half_edge = None  # 从该节点出发的任意一条半边 (HalfEdge 对象)
        self.incident_cells = []  # list of Cell objects: 共享此节点的单元列表
        self.incident_half_edges = []  # list of HalfEdge objects: 以此节点为起点的半边列表

    def __repr__(self):  # 定义对象的字符串表示，方便调试
        return f"Node(id={self.id}, x={self.x:.2f}, y={self.y:.2f}, z_bed={self.z_bed:.2f})"  # 返回节点信息字符串


class HalfEdge:  # 半边类
    """半边数据结构类"""

    def __init__(self, id):  # 初始化方法
        self.id = id  # 半边ID (整数)

        # --- 基本拓扑 ---
        self.origin = None  # 起始节点 (Node 对象)
        self.twin = None  # 反向孪生半边 (HalfEdge 对象 or None)
        self.next = None  # 同一单元内的下一条半边 (HalfEdge 对象)
        self.prev = None  # 同一单元内的上一条半边 (HalfEdge 对象)
        self.cell = None  # 所属计算单元 (Cell 对象)

        # --- 几何属性 (在初始化后计算) ---
        self.length = 0.0  # 边的长度
        self.normal = (0.0, 0.0)  # 边的单位外法向量 (nx, ny)，指向 cell 的外部
        self.mid_point = (0.0, 0.0)  # 边的中点坐标 (x_mid, y_mid)

        # --- 边界与其它 ---
        self.boundary_marker = 0  # 边界标记 (整数, 0 通常表示内部边)
        # self.edge_physical_id = -1 # (可选) 对应的物理边ID (如果需要追踪)

    @property  # 定义一个属性，使其可以像访问变量一样访问方法
    def end_node(self):  # 获取半边的终点节点
        """Returns the end Node of this half-edge."""
        if self.next:  # 如果存在下一条半边
            return self.next.origin  # 终点是下一条半边的起点
        return None  # 否则返回None

    def __repr__(self):  # 定义对象的字符串表示
        o_id = self.origin.id if self.origin else "N/A"  # 获取起点ID
        e_id = self.end_node.id if self.end_node else "N/A"  # 获取终点ID
        c_id = self.cell.id if self.cell else "N/A (Boundary?)"  # 获取所属单元ID
        t_id = self.twin.id if self.twin else "None"  # 获取孪生半边ID
        return f"HalfEdge(id={self.id}, origin={o_id}, end={e_id}, cell={c_id}, twin={t_id})"  # 返回半边信息字符串


class Cell:  # 单元类
    """计算单元（控制体）数据结构类"""

    def __init__(self, id):  # 初始化方法
        self.id = id  # 单元ID (整数)

        # --- 基本拓扑 (在初始化后填充) ---
        self.half_edge = None  # 指向组成该单元的任意一条半边 (HalfEdge 对象)
        self.nodes = []  # list of Node objects: 组成单元的顶点 (按逆时针顺序)
        self.half_edges_list = []  # list of HalfEdge objects: 组成单元边界的半边 (按逆时针顺序)

        # --- 几何属性 (在初始化后计算) ---
        self.area = 0.0  # 单元面积
        self.centroid = (0.0, 0.0)  # 形心坐标 (x_c, y_c)
        self.z_bed_centroid = 0.0  # 形心处的底高程
        self.b_slope_x = 0.0  # x方向底坡 d(z_bed)/dx (单元平均或基于形心)
        self.b_slope_y = 0.0  # y方向底坡 d(z_bed)/dy (单元平均或基于形心)

        # --- 物理属性 ---
        self.manning_n = 0.025  # 曼宁糙率系数 (可后续从参数或文件设置)

        # --- 水动力状态变量 (在模型运行时更新) ---
        self.U = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # 当前时刻守恒量 [h, hu, hv]
        # self.reconstructed_vars = {} # (可选) 存储重构相关数据，如梯度

    def get_primitive_vars(self):  # 获取原始变量 (h, u, v)
        """Returns primitive variables (h, u, v) from conserved U."""
        h = self.U[0]  # 获取水深
        if h < 1e-9:  # 如果水深过小 (或为0)
            return h, 0.0, 0.0  # 返回水深和零流速
        u = self.U[1] / h  # 计算流速u
        v = self.U[2] / h  # 计算流速v
        return h, u, v  # 返回水深和流速

    def __repr__(self):  # 定义对象的字符串表示
        node_ids = [n.id for n in self.nodes] if self.nodes else "N/A"  # 获取节点ID列表
        h, u, v = self.get_primitive_vars()  # 获取原始变量
        return f"Cell(id={self.id}, nodes={node_ids}, area={self.area:.3f}, h={h:.3f}, hu={self.U[1]:.3f}, hv={self.U[2]:.3f})"  # 返回单元信息字符串


class Mesh:  # 网格容器类
    """网格容器类, 包含所有节点、半边和单元对象"""

    def __init__(self):  # 初始化方法
        self.nodes = []  # list of all Node objects # 所有节点对象的列表
        self.half_edges = []  # list of all HalfEdge objects # 所有半边对象的列表
        self.cells = []  # list of all Cell objects # 所有单元对象的列表

    # 可以在这里添加一些辅助方法，例如通过ID查找对象等
    def get_node_by_id(self, node_id):  # 通过ID获取节点对象
        # 假设节点ID与列表索引一致 (在initialization.py中保证)
        if 0 <= node_id < len(self.nodes):  # 如果ID在范围内
            return self.nodes[node_id]  # 返回对应节点
        return None  # 否则返回None

    def get_cell_by_id(self, cell_id):  # 通过ID获取单元对象
        # 假设单元ID与列表索引一致
        if 0 <= cell_id < len(self.cells):  # 如果ID在范围内
            return self.cells[cell_id]  # 返回对应单元
        return None  # 否则返回None