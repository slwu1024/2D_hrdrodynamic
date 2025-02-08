# hydrodynamic_model/model/mesh.py
import numpy as np


class Node:
    def __init__(self, id, x, y, z_bed):
        self.id = id
        self.x = x
        self.y = y
        self.z_bed = z_bed  # 底高程
        self.cells = []  # 关联的单元
        self.edges = []  # 关联的边

class Edge:
    def __init__(self, id, node_start, node_end):
        self.id = id
        self.nodes = (node_start, node_end)  # 边的两个端点
        self.cells = [None, None]  # 左单元和右单元
        self.normal = (0.0, 0.0)   # 外法向量
        self.length = 0.0          # 边长度
        self.type = 'internal'     # 边界类型

class Cell:
    def __init__(self, id, nodes):
        self.id = id
        self.nodes = nodes  # 三个顶点
        self.edges = []      # 三条边
        self.neighbors = []  # 邻接单元
        self.area = 0.0      # 单元面积
        self.centroid = (0.0, 0.0)  # 质心坐标
        self.z_bed_nodes = [n.z_bed for n in nodes]  # 顶点底高程
        self.U = np.array([0.0, 0.0, 0.0])  # 守恒量 [h, hu, hv]
        self.U_old = np.array([0.0, 0.0, 0.0])  # 上一时间步的守恒量