# hydrodynamic_model/model/initialization.py
import numpy as np

from .mesh import Node, Edge, Cell


def load_mesh(mesh_file):
    """从网格文件加载数据并创建网格对象"""
    nodes = []
    edges = []
    cells = []

    # 假设网格文件格式为自定义格式
    with open(mesh_file, 'r') as f:
        # 解析节点
        for line in f:
            if line.startswith('NODE'):
                parts = line.split()
                node = Node(int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                nodes.append(node)
        # 解析边和单元（根据实际文件格式补充）
        # ...

    return nodes, edges, cells


def precompute_geometry(cells, edges):
    """预计算几何量（面积、质心、法向量等）"""
    for cell in cells:
        x1, y1 = cell.nodes[0].x, cell.nodes[0].y
        x2, y2 = cell.nodes[1].x, cell.nodes[1].y
        x3, y3 = cell.nodes[2].x, cell.nodes[2].y
        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        centroid_x = (x1 + x2 + x3) / 3.0
        centroid_y = (y1 + y2 + y3) / 3.0
        cell.area = area
        cell.centroid = (centroid_x, centroid_y)

    for edge in edges:
        node1, node2 = edge.nodes
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        edge.length = np.sqrt(dx ** 2 + dy ** 2)
        nx = dy / edge.length
        ny = -dx / edge.length
        edge.normal = (nx, ny)


def initialize_model(cells):
    """初始化物理量（水深、流速等）"""
    for cell in cells:
        h_initial = 5.0 - min(cell.z_bed_nodes)  # 假设初始水位为5.0m
        cell.U = np.array([h_initial, 0.0, 0.0])  # [h, hu, hv]