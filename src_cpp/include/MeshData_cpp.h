// src_cpp/include/MeshData_cpp.h
#ifndef MESHDATA_CPP_H
#define MESHDATA_CPP_H

#include <vector>
#include <string>
#include <array>

namespace HydroCore {

// Node_cpp, HalfEdge_cpp, Cell_cpp 结构体定义 (保持不变，Cell_cpp 已有 region_attribute)
struct Node_cpp {
    int id;
    double x, y, z_bed;
    int marker;
    Node_cpp(int id_ = -1, double x_ = 0, double y_ = 0, double z_bed_ = 0, int marker_ = 0)
        : id(id_), x(x_), y(y_), z_bed(z_bed_), marker(marker_) {}
};

struct HalfEdge_cpp {
    int id;
    int origin_node_id = -1;
    int twin_half_edge_id = -1;
    int next_half_edge_id = -1;
    int prev_half_edge_id = -1;
    int cell_id = -1;
    double length = 0.0;
    std::array<double, 2> normal = {0.0, 0.0};
    std::array<double, 2> mid_point = {0.0, 0.0};
    int boundary_marker = 0;
    int original_poly_segment_id = -1;
    HalfEdge_cpp(int id_ = -1) : id(id_) {}
};

struct Cell_cpp {
    int id;
    std::vector<int> node_ids;
    std::vector<int> half_edge_ids_list;
    double area = 0.0;
    std::array<double, 2> centroid = {0.0, 0.0};
    double z_bed_centroid = 0.0;
    double b_slope_x = 0.0;
    double b_slope_y = 0.0;
    double manning_n = 0.025;
    double region_attribute = 0.0; // 区域属性已存在
    Cell_cpp(int id_ = -1) : id(id_) {}
};

class Mesh_cpp {
public:
    std::vector<Node_cpp> nodes;
    std::vector<HalfEdge_cpp> half_edges;
    std::vector<Cell_cpp> cells;

    Mesh_cpp() = default;

    // 修改1: load_mesh_from_files 也需要接收区域属性的vector，以便传递
    void load_mesh_from_files(const std::string& node_filepath,
                              const std::string& cell_filepath,
                              const std::string& edge_filepath,
                              const std::vector<double>& cell_manning_values); // 4个参数

    void load_nodes_from_numpy(const std::vector<double>& flat_node_data, int num_nodes, int num_attrs);
    // 声明为5个参数 (接收区域属性)
    void load_cells_from_numpy(const std::vector<int>& flat_cell_data, int num_cells, int nodes_per_cell,
                               const std::vector<double>& cell_manning_values,
                               const std::vector<double>& cell_region_attributes); // 5个参数

    void precompute_geometry_and_topology(const std::vector<int>& flat_edge_data, int num_edges, int num_edge_attrs);

    const Node_cpp* get_node_by_id(int node_id) const;
    Node_cpp* get_node_by_id_mutable(int node_id);
    const Cell_cpp* get_cell_by_id(int cell_id) const;
    // 新增一个 getter，以便 Python 端 prepare_initial_conditions 可以访问区域属性
    double get_cell_region_attribute(int cell_id) const;
    const HalfEdge_cpp* get_half_edge_by_id(int he_id) const;

private:
    void setup_half_edge_structure_optimized_cpp();
    void assign_boundary_markers_to_halfedges_cpp(const std::vector<int>& flat_edge_data, int num_edges, int num_edge_attrs);
    void precompute_cell_geometry_cpp();
    void precompute_half_edge_geometry_cpp();

    // 修改2: read_cell_file_cpp 声明需要能输出区域属性
    bool read_node_file_cpp(const std::string& filepath, std::vector<double>& flat_data, int& num_nodes, int& num_attrs);
    bool read_cell_file_cpp(const std::string& filepath,
                            std::vector<int>& flat_cell_data_out,      // 用于节点连接
                            std::vector<double>& cell_attributes_out, // 用于区域属性
                            int& num_cells_out,
                            int& num_nodes_per_cell_out);
    bool read_edge_file_cpp(const std::string& filepath, std::vector<int>& flat_data, int& num_edges, int& num_edge_attrs);
};

}
#endif