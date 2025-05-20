// src_cpp/include/MeshData_cpp.h
#ifndef MESHDATA_CPP_H
#define MESHDATA_CPP_H

#include <vector> // 包含vector容器
#include <string> // 包含string类
#include <array>  // 包含array容器

namespace HydroCore {

// Node_cpp, HalfEdge_cpp, Cell_cpp 结构体定义不变...
struct Node_cpp { // 节点结构体
    int id;                         // 节点ID
    double x, y, z_bed;             // 坐标和底高程
    int marker;                     // 边界标记
    Node_cpp(int id_ = -1, double x_ = 0, double y_ = 0, double z_bed_ = 0, int marker_ = 0) // 构造函数
        : id(id_), x(x_), y(y_), z_bed(z_bed_), marker(marker_) {} // 初始化列表
}; // 结束结构体定义

struct HalfEdge_cpp { // 半边结构体
    int id;                         // 半边ID
    int origin_node_id = -1;        // 起始节点ID
    int twin_half_edge_id = -1;     // 孪生半边ID (-1表示无)
    int next_half_edge_id = -1;     // 同一单元内下一条半边ID
    int prev_half_edge_id = -1;     // 同一单元内上一条半边ID
    int cell_id = -1;               // 所属计算单元ID

    double length = 0.0;            // 边的长度
    std::array<double, 2> normal = {0.0, 0.0}; // 边的单位外法向量 (nx, ny)
    std::array<double, 2> mid_point = {0.0, 0.0}; // 边的中点坐标 (x_mid, y_mid)
    int boundary_marker = 0;        // 边界标记 (0 通常表示内部边)
    int original_poly_segment_id = -1; // 新增：存储来自 .poly 文件线段的第一个数字 (原始线段序号) // <--- 新增代码

    HalfEdge_cpp(int id_ = -1) : id(id_) {} // 构造函数
}; // 结束结构体定义

struct Cell_cpp { // 单元结构体
    int id;                         // 单元ID
    std::vector<int> node_ids;      // 组成单元的顶点ID (按逆时针顺序)
    std::vector<int> half_edge_ids_list; // 组成单元边界的半边ID (按逆时针顺序)

    double area = 0.0;              // 单元面积
    std::array<double, 2> centroid = {0.0, 0.0}; // 形心坐标
    double z_bed_centroid = 0.0;    // 形心处的底高程
    double b_slope_x = 0.0;         // x方向底坡
    double b_slope_y = 0.0;         // y方向底坡
    double manning_n = 0.025;       // 曼宁糙率

    Cell_cpp(int id_ = -1) : id(id_) {} // 构造函数
}; // 结束结构体定义


class Mesh_cpp { // 网格类
public: // 公有成员
    std::vector<Node_cpp> nodes;         // 所有节点对象的列表
    std::vector<HalfEdge_cpp> half_edges; // 所有半边对象的列表
    std::vector<Cell_cpp> cells;         // 所有单元对象的列表

    Mesh_cpp() = default; // 默认构造函数

    // 从文件直接加载网格数据，并进行预计算
    void load_mesh_from_files(const std::string& node_filepath, // 从文件加载网格数据
                              const std::string& cell_filepath,
                              const std::string& edge_filepath, // 可能为空字符串，表示没有边文件
                              const std::vector<double>& cell_manning_values); // 单元曼宁系数值

    // 保留这些方法，因为 load_mesh_from_files 内部会调用它们
    void load_nodes_from_numpy(const std::vector<double>& flat_node_data, int num_nodes, int num_attrs); // 从NumPy加载节点
    void load_cells_from_numpy(const std::vector<int>& flat_cell_data, int num_cells, int nodes_per_cell, // 从NumPy加载单元
                               const std::vector<double>& cell_manning_values);
    void precompute_geometry_and_topology(const std::vector<int>& flat_edge_data, int num_edges, int num_edge_attrs); // 预计算几何和拓扑

    // 辅助的getters (const版本用于查询，非const可选用于修改)
    const Node_cpp* get_node_by_id(int node_id) const; // 通过ID获取节点(const)
    Node_cpp* get_node_by_id_mutable(int node_id);     // 通过ID获取节点(可修改)
    const Cell_cpp* get_cell_by_id(int cell_id) const; // 通过ID获取单元(const)
    const HalfEdge_cpp* get_half_edge_by_id(int he_id) const; // 通过ID获取半边(const)

private: // 私有成员
    // 私有辅助函数，用于precompute_geometry_and_topology
    void setup_half_edge_structure_optimized_cpp(); // 优化版半边孪生关系设置
    void assign_boundary_markers_to_halfedges_cpp(const std::vector<int>& flat_edge_data, int num_edges, int num_edge_attrs); // 分配边界标记
    void precompute_cell_geometry_cpp(); // 预计算单元几何属性
    void precompute_half_edge_geometry_cpp(); // 预计算半边几何属性

    // 新增：直接从文件读取数据的私有辅助函数 (或者设为静态自由函数)
    bool read_node_file_cpp(const std::string& filepath, std::vector<double>& flat_data, int& num_nodes, int& num_attrs); // 读取节点文件(C++)
    bool read_cell_file_cpp(const std::string& filepath, std::vector<int>& flat_data, int& num_cells, int& num_nodes_per_cell); // 读取单元文件(C++)
    bool read_edge_file_cpp(const std::string& filepath, std::vector<int>& flat_data, int& num_edges, int& num_edge_attrs); // 读取边文件(C++)
}; // 结束类定义

} // namespace HydroCore
#endif //MESHDATA_CPP_H