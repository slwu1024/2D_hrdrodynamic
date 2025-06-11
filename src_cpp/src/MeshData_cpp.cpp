// src_cpp/src/MeshData_cpp.cpp
#include "MeshData_cpp.h" // 包含对应的头文件
#include <iostream>      // 用于调试打印
#include <fstream>       // 用于文件读取
#include <sstream>       // 用于字符串流处理
#include <string>        // 用于字符串操作
#include <vector>        // 用于动态数组
#include <cmath>         // 用于 std::sqrt, std::abs
#include <algorithm>     // 用于 std::sort, std::find_if 等
#include <map>           // 用于 setup_half_edge_structure_optimized_cpp
#include <stdexcept>     // 用于运行时错误
#include <omp.h>

#ifdef _WIN32 // 仅在Windows平台下引入以下头文件和函数
#define WIN32_LEAN_AND_MEAN // 排除不常用的Windows头文件
#include <windows.h> // 引入Windows API头文件，用于字符转换
#endif
// const double epsilon_mesh_geom = 1e-6; // 定义一个用于几何比较的小量 (避免与全局epsilon冲突)
// ^^^ 在 MeshData_cpp.h 中如果定义了，这里就不需要重复定义，否则取消注释或使用一个统一的epsilon定义

// --- 在文件顶部或合适的位置定义期望的网格特征 ---
// !!! 【重要】请用你实际的 model_river 网格文件中的数值替换下面的占位符 !!!
// const int EXPECTED_FIXED_NODE_COUNT = 66873; // 你的实际值
// const int EXPECTED_FIXED_CELL_COUNT = 132800; // 你的实际值
// --- 结束定义 ---


namespace HydroCore { // HydroCore命名空间开始

#ifdef _WIN32
    // 辅助函数：将UTF-8编码的std::string转换为std::wstring (仅Windows)
    std::wstring utf8_to_wstring_windows(const std::string& utf8_str) { // 定义UTF-8转宽字符函数 (Windows)
        if (utf8_str.empty()) { // 如果输入字符串为空
            return std::wstring(); // 返回空宽字符串
        }
        // 计算转换后需要的宽字符数量
        int wide_char_count = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, NULL, 0); // 调用Windows API计算所需缓冲区大小
        if (wide_char_count == 0) { // 如果计算失败
            return std::wstring(); // 返回空宽字符串
        }
        std::wstring wide_str(wide_char_count -1, 0); // 创建宽字符串，大小为 count-1 (不包括null终止符)
        // 执行转换
        if (MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, &wide_str[0], wide_char_count) == 0) { // 调用Windows API进行转换
            return std::wstring(); // 转换失败则返回空宽字符串
        }
        return wide_str; // 返回转换后的宽字符串
    }
#endif
// --- 文件读取辅助函数的实现 ---
bool Mesh_cpp::read_node_file_cpp(const std::string& filepath_utf8, std::vector<double>& flat_data, int& num_nodes_out, int& num_attrs_out) {
    std::cout << "C++ received UTF-8 filepath for node file: [" << filepath_utf8 << "]" << std::endl;
    std::ifstream file;
#ifdef _WIN32
    std::wstring w_filepath = utf8_to_wstring_windows(filepath_utf8);
    if (!w_filepath.empty()) {
        file.open(w_filepath.c_str());
    } else if (!filepath_utf8.empty()) {
        std::cerr << "Warning: Failed to convert node filepath to wstring, trying with original UTF-8 string." << std::endl;
        file.open(filepath_utf8.c_str());
    } else {
         file.open(filepath_utf8.c_str());
    }
#else
    file.open(filepath_utf8.c_str());
#endif

    if (!file.is_open()) {
        std::cerr << "Error: Could not open node file " << filepath_utf8 << std::endl;
        return false;
    }

    std::string line;
    if (!std::getline(file, line)) { std::cerr << "Error: Node file " << filepath_utf8 << " is empty or unreadable header." << std::endl; file.close(); return false; }
    std::istringstream header_ss(line);
    int dim, point_attrs_count_file, has_marker_file;
    header_ss >> num_nodes_out >> dim >> point_attrs_count_file >> has_marker_file;
    if (header_ss.fail()) { std::cerr << "Error: Invalid node file header format in " << filepath_utf8 << std::endl; file.close(); return false; }
    // 单元数检查，用于
    // if (EXPECTED_FIXED_NODE_COUNT > 0 && num_nodes_out != EXPECTED_FIXED_NODE_COUNT) {
    //     std::cerr << "Error: Node file " << filepath_utf8
    //               << " reports " << num_nodes_out << " nodes, but this model version expects exactly "
    //               << EXPECTED_FIXED_NODE_COUNT << " nodes for the pre-set terrain. Aborting mesh load." << std::endl;
    //     file.close();
    //     return false;
    // } else if (EXPECTED_FIXED_NODE_COUNT == 0) {
    //     std::cout << "Warning: EXPECTED_FIXED_NODE_COUNT is 0. Node count check skipped. Ensure this is intended." << std::endl;
    // }

    num_attrs_out = 4 + (has_marker_file == 1 ? 1 : 0);

    flat_data.clear();
    flat_data.reserve(num_nodes_out * num_attrs_out);

    for (int i = 0; i < num_nodes_out; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Unexpected end of file or read error in node file " << filepath_utf8 << " at node " << i << std::endl;
            file.close(); return false;
        }
        std::istringstream line_ss(line);
        double node_id_d, x, y, z_bed; // node_id_d to avoid shadowing
        double marker_val = 0.0;
        line_ss >> node_id_d >> x >> y;
        for(int attr_idx = 0; attr_idx < point_attrs_count_file; ++attr_idx) {
            double dummy_attr;
            if (!(line_ss >> dummy_attr)) { /* warning or break */ }
        }
        line_ss >> z_bed;

        if (has_marker_file == 1) {
            if (!(line_ss >> marker_val)) { /* warning */ }
        }
        if (line_ss.fail() && !line_ss.eof()) {
             std::cerr << "Error: Invalid data format for node " << i << " in " << filepath_utf8 << ". Line: "<< line << std::endl;
             file.close(); return false;
        }

        flat_data.push_back(node_id_d);
        flat_data.push_back(x);
        flat_data.push_back(y);
        flat_data.push_back(z_bed);
        if (num_attrs_out == 5) flat_data.push_back(marker_val);
    }
    file.close();
    return true;
}

bool Mesh_cpp::read_cell_file_cpp(const std::string& filepath_utf8,
                                  std::vector<int>& flat_cell_data_out,
                                  std::vector<double>& cell_attributes_out,
                                  int& num_cells_out,
                                  int& num_nodes_per_cell_out) {
    std::ifstream file;
#ifdef _WIN32
    std::wstring w_filepath = utf8_to_wstring_windows(filepath_utf8);
    if (!w_filepath.empty()) {
        file.open(w_filepath.c_str());
    } else if (!filepath_utf8.empty()){
        std::cerr << "Warning: Failed to convert cell filepath to wstring, trying with original UTF-8 string." << std::endl;
        file.open(filepath_utf8.c_str());
    } else {
        file.open(filepath_utf8.c_str());
    }
#else
    file.open(filepath_utf8.c_str());
#endif

    if (!file.is_open()) {
        std::cerr << "Error: Could not open cell file " << filepath_utf8 << std::endl;
        return false;
    }

    std::string line;
    if (!std::getline(file, line)) { std::cerr << "Error: Cell file " << filepath_utf8 << " is empty or unreadable header." << std::endl; file.close(); return false; }
    std::istringstream header_ss(line);
    int num_cell_attrs_in_file;
    header_ss >> num_cells_out >> num_nodes_per_cell_out >> num_cell_attrs_in_file;
    if (header_ss.fail()) { std::cerr << "Error: Invalid cell file header format in " << filepath_utf8 << std::endl; file.close(); return false; }

    /*if (EXPECTED_FIXED_CELL_COUNT > 0 && num_cells_out != EXPECTED_FIXED_CELL_COUNT) {
        std::cerr << "Error: Cell file " << filepath_utf8
                  << " reports " << num_cells_out << " cells, but this model version expects exactly "
                  << EXPECTED_FIXED_CELL_COUNT << " cells for the pre-set terrain. Aborting mesh load." << std::endl;
        file.close();
        return false;
    } else if (EXPECTED_FIXED_CELL_COUNT == 0) {
        std::cout << "Warning: EXPECTED_FIXED_CELL_COUNT is 0. Cell count check skipped. Ensure this is intended." << std::endl;
    }*/

    if (num_nodes_per_cell_out != 3) {
        std::cerr << "Error: Cell file " << filepath_utf8 << " indicates " << num_nodes_per_cell_out
                  << " nodes per cell, but C++ MeshData currently only supports 3." << std::endl;
        file.close(); return false;
    }

    flat_cell_data_out.clear();
    flat_cell_data_out.reserve(num_cells_out * (1 + num_nodes_per_cell_out));
    cell_attributes_out.clear();
    cell_attributes_out.reserve(num_cells_out);

    for (int i = 0; i < num_cells_out; ++i) {
        if (!std::getline(file, line)) { std::cerr << "Error: Unexpected end of file or read error in cell file " << filepath_utf8 << " at cell " << i << std::endl; file.close(); return false; }
        std::istringstream line_ss(line);
        std::vector<std::string> parts;
        std::string part;
        while(line_ss >> part) {
            parts.push_back(part);
        }

        if (parts.size() < static_cast<size_t>(1 + num_nodes_per_cell_out)) {
             std::cerr << "Error: Invalid data format for cell " << i << " in " << filepath_utf8 << ". Not enough parts. Line: "<< line << std::endl;
             file.close(); return false;
        }

        int ele_id = std::stoi(parts[0]);
        flat_cell_data_out.push_back(ele_id);
        for (int j = 0; j < num_nodes_per_cell_out; ++j) {
            flat_cell_data_out.push_back(std::stoi(parts[1 + j]));
        }

        if (num_cell_attrs_in_file > 0) {
            if (parts.size() > static_cast<size_t>(num_nodes_per_cell_out + 1)) {
                try {
                    cell_attributes_out.push_back(std::stod(parts[num_nodes_per_cell_out + 1]));
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Warning: Invalid attribute for cell " << ele_id << " ('" << parts[num_nodes_per_cell_out + 1] << "'). Using 0.0. Error: " << ia.what() << std::endl;
                    cell_attributes_out.push_back(0.0);
                } catch (const std::out_of_range& oor) {
                     std::cerr << "Warning: Attribute for cell " << ele_id << " out of range. Using 0.0. Error: " << oor.what() << std::endl;
                    cell_attributes_out.push_back(0.0);
                }
            } else {
                cell_attributes_out.push_back(0.0);
            }
        } else {
            cell_attributes_out.push_back(0.0);
        }
    }
    file.close();
    return true;
}

bool Mesh_cpp::read_edge_file_cpp(const std::string& filepath_utf8, std::vector<int>& flat_data, int& num_edges_out, int& num_edge_attrs_out) {
    if (filepath_utf8.empty()) {
        num_edges_out = 0;
        num_edge_attrs_out = 0;
        flat_data.clear();
        return true;
    }
    std::cout << "C++ received UTF-8 filepath for edge file: [" << filepath_utf8 << "]" << std::endl;
    std::ifstream file;
#ifdef _WIN32
    std::wstring w_filepath = utf8_to_wstring_windows(filepath_utf8);
    if (!w_filepath.empty()) {
        file.open(w_filepath.c_str());
    } else if (!filepath_utf8.empty()){
        std::cerr << "Warning: Failed to convert edge filepath to wstring, trying with original UTF-8 string." << std::endl;
        file.open(filepath_utf8.c_str());
    } else {
        file.open(filepath_utf8.c_str());
    }
#else
    file.open(filepath_utf8.c_str());
#endif
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open edge file " << filepath_utf8 << ". Proceeding without edge data." << std::endl;
        num_edges_out = 0;
        num_edge_attrs_out = 0;
        flat_data.clear();
        return true;
    }
    std::string line;
    if (!std::getline(file, line)) { std::cerr << "Error: Edge file " << filepath_utf8 << " is empty or unreadable header." << std::endl; file.close(); return false; }
    std::istringstream header_ss(line);
    int has_marker_file_dummy; // For edge file, the second header number might be a generic attribute count
    header_ss >> num_edges_out >> has_marker_file_dummy; // Or it might just be a single number (num_edges)
    if (header_ss.fail()) { // If parsing the second number fails, try parsing only one
        header_ss.clear(); // Clear error flags
        header_ss.seekg(0); // Reset stream position
        if (!(header_ss >> num_edges_out)) { // Try parsing only num_edges_out
            std::cerr << "Error: Invalid edge file header format in " << filepath_utf8 << std::endl; file.close(); return false;
        }
        // If we successfully parsed only num_edges_out, assume the file implies the rest of the columns
    }

    num_edge_attrs_out = 4; // n1, n2, type_marker, original_poly_id

    flat_data.clear();
    flat_data.reserve(num_edges_out * num_edge_attrs_out);

    for (int i = 0; i < num_edges_out; ++i) {
        if (!std::getline(file, line)) { std::cerr << "Error: Unexpected end of file reading edge " << i << " in " << filepath_utf8 << std::endl; file.close(); return false; }
        std::istringstream line_ss(line);
        int edge_idx_in_file, node1_id, node2_id, type_marker_val, original_poly_seg_id_val;

        if (!(line_ss >> edge_idx_in_file >> node1_id >> node2_id >> type_marker_val >> original_poly_seg_id_val)) {
            std::cerr << "错误: 边文件 " << filepath_utf8 << " 中第 " << i << " 行数据格式无效。期望5个整数。行: " << line << std::endl;
            file.close(); return false;
        }

        flat_data.push_back(node1_id);
        flat_data.push_back(node2_id);
        flat_data.push_back(type_marker_val);
        flat_data.push_back(original_poly_seg_id_val);
    }
    file.close();
    return true;
}


void Mesh_cpp::load_mesh_from_files(const std::string& node_filepath,
                                  const std::string& cell_filepath,
                                  const std::string& edge_filepath,
                                  const std::vector<double>& cell_manning_values) { // 4参数
    std::cout << "C++ Mesh: Starting to load mesh from files..." << std::endl;
    std::vector<double> flat_nodes_vec;
    int num_nodes_read, node_attrs_read;
    if (!this->read_node_file_cpp(node_filepath, flat_nodes_vec, num_nodes_read, node_attrs_read)) {
        throw std::runtime_error("Failed to read node file: " + node_filepath);
    }
    this->load_nodes_from_numpy(flat_nodes_vec, num_nodes_read, node_attrs_read);
    std::cout << "  Loaded " << num_nodes_read << " nodes with " << node_attrs_read << " attributes each." << std::endl;

    std::vector<int> flat_cells_connectivity_vec;
    std::vector<double> cell_attributes_vec;
    int num_cells_read, nodes_per_cell_read;

    if (!this->read_cell_file_cpp(cell_filepath, flat_cells_connectivity_vec, cell_attributes_vec, num_cells_read, nodes_per_cell_read)) {
        throw std::runtime_error("Failed to read cell file: " + cell_filepath);
    }

    if (!cell_manning_values.empty() && cell_manning_values.size() != static_cast<size_t>(num_cells_read)) {
        std::cerr << "Warning: Provided " << cell_manning_values.size() << " Manning values, but read "
                  << num_cells_read << " cells. Mismatched values might lead to default usage." << std::endl;
    }
     if (cell_attributes_vec.size() != static_cast<size_t>(num_cells_read)) {
         std::cerr << "Warning: Number of read cell attributes (" << cell_attributes_vec.size()
                   << ") does not match number of cells (" << num_cells_read
                   << "). Region attributes might use defaults for some cells." << std::endl;
    }

    this->load_cells_from_numpy(flat_cells_connectivity_vec, num_cells_read, nodes_per_cell_read, cell_manning_values, cell_attributes_vec);
    std::cout << "  Loaded " << num_cells_read << " cells with " << nodes_per_cell_read << " nodes each." << std::endl;

    std::vector<int> flat_edges_vec;
    int num_edges_read = 0, edge_attrs_read = 0;
    if (!edge_filepath.empty()) {
        if (!this->read_edge_file_cpp(edge_filepath, flat_edges_vec, num_edges_read, edge_attrs_read)) {
            if(num_edges_read != 0) {
                throw std::runtime_error("Failed to read edge file: " + edge_filepath);
            }
        }
    }
    if (num_edges_read > 0) {
        std::cout << "  Loaded " << num_edges_read << " edges with " << edge_attrs_read << " attributes each." << std::endl;
    } else {
        std::cout << "  No edge data loaded (or edge file not provided/empty)." << std::endl;
    }

    std::cout << "C++ Mesh: Precomputing geometry and topology..." << std::endl;
    this->precompute_geometry_and_topology(flat_edges_vec, num_edges_read, edge_attrs_read);
    std::cout << "C++ Mesh: Loading and precomputation complete." << std::endl;
}


void Mesh_cpp::load_nodes_from_numpy(const std::vector<double>& flat_node_data, int num_nodes_in, int num_node_attrs) {
    if (num_node_attrs < 4) {
        throw std::runtime_error("Node data must have at least 4 attributes (id, x, y, z_bed).");
    }
    nodes.clear();
    nodes.reserve(num_nodes_in);
    for (int i = 0; i < num_nodes_in; ++i) {
        int base_idx = i * num_node_attrs;
        int id = static_cast<int>(flat_node_data[base_idx + 0]);
        double x = flat_node_data[base_idx + 1];
        double y = flat_node_data[base_idx + 2];
        double z_bed = flat_node_data[base_idx + 3];
        int marker = (num_node_attrs > 4) ? static_cast<int>(flat_node_data[base_idx + 4]) : 0;
        nodes.emplace_back(id, x, y, z_bed, marker);
    }
}

void Mesh_cpp::load_cells_from_numpy(const std::vector<int>& flat_cell_data, int num_cells_in, int nodes_per_cell,
                                   const std::vector<double>& cell_manning_values,
                                   const std::vector<double>& cell_region_attributes) {
    if (nodes_per_cell != 3) {
        throw std::runtime_error("Currently only supports 3 nodes per cell (triangles).");
    }
    bool manning_provided = !cell_manning_values.empty();
    if (manning_provided && cell_manning_values.size() != static_cast<size_t>(num_cells_in) ) {
        std::cerr << "Warning (load_cells_from_numpy): Number of Manning values (" << cell_manning_values.size()
                  << ") does not match number of cells (" << num_cells_in
                  << "). Manning values might be incorrect for some cells." << std::endl;
    }

    bool region_attr_provided = !cell_region_attributes.empty();
    if (region_attr_provided && cell_region_attributes.size() != static_cast<size_t>(num_cells_in)) {
        std::cerr << "Warning (load_cells_from_numpy): Number of region attributes (" << cell_region_attributes.size()
                  << ") does not match number of cells (" << num_cells_in
                  << "). Region attributes will use defaults for some cells if undersupplied." << std::endl;
    }

    cells.clear();
    cells.reserve(num_cells_in);
    for (int i = 0; i < num_cells_in; ++i) {
        int base_idx = i * (1 + nodes_per_cell); // 1 for cell_id itself
        int id = flat_cell_data[base_idx + 0];
        Cell_cpp current_cell(id);
        current_cell.node_ids.reserve(nodes_per_cell);
        for (int j = 0; j < nodes_per_cell; ++j) {
            current_cell.node_ids.push_back(flat_cell_data[base_idx + 1 + j]);
        }

        if (manning_provided) {
            if (static_cast<size_t>(i) < cell_manning_values.size()) {
                current_cell.manning_n = cell_manning_values[i];
            } else {
                std::cerr << "Warning: Manning value for cell " << id << " not provided, using default " << current_cell.manning_n << std::endl;
            }
        } // else it will use the default from Cell_cpp constructor

        if (region_attr_provided) {
            if (static_cast<size_t>(i) < cell_region_attributes.size()) {
                current_cell.region_attribute = cell_region_attributes[i];
            } else {
                std::cerr << "Warning: Region attribute for cell " << id << " not provided, using default " << current_cell.region_attribute << std::endl;
            }
        } // else it will use the default from Cell_cpp constructor (which is 0.0)
        cells.emplace_back(current_cell);
    }
}

void Mesh_cpp::precompute_geometry_and_topology(const std::vector<int>& flat_edge_data, int num_edges, int num_edge_attrs) {
    half_edges.clear();
    int half_edge_id_counter = 0;
    for (size_t cell_idx = 0; cell_idx < cells.size(); ++cell_idx) {
        Cell_cpp& current_cell = cells[cell_idx];
        if (current_cell.node_ids.size() != 3) continue;

        current_cell.half_edge_ids_list.resize(3);
        std::array<int, 3> hes_in_this_cell_ids;

        for (int i = 0; i < 3; ++i) {
            HalfEdge_cpp he(half_edge_id_counter);
            he.origin_node_id = current_cell.node_ids[i];
            he.cell_id = current_cell.id;

            hes_in_this_cell_ids[i] = he.id;
            half_edges.push_back(he);
            half_edge_id_counter++;
        }

        for (int i = 0; i < 3; ++i) {
            half_edges[hes_in_this_cell_ids[i]].next_half_edge_id = hes_in_this_cell_ids[(i + 1) % 3];
            half_edges[hes_in_this_cell_ids[i]].prev_half_edge_id = hes_in_this_cell_ids[(i + 2) % 3];
            current_cell.half_edge_ids_list[i] = hes_in_this_cell_ids[i];
        }
    }

    setup_half_edge_structure_optimized_cpp();
    assign_boundary_markers_to_halfedges_cpp(flat_edge_data, num_edges, num_edge_attrs);
    precompute_cell_geometry_cpp();
    precompute_half_edge_geometry_cpp();
}


void Mesh_cpp::setup_half_edge_structure_optimized_cpp() {
    std::map<std::pair<int, int>, std::vector<int>> edge_map;

    for (const auto& he : half_edges) {
        if (he.origin_node_id == -1 || he.next_half_edge_id == -1) continue;
        const HalfEdge_cpp* next_he = get_half_edge_by_id(he.next_half_edge_id);
        if (!next_he || next_he->origin_node_id == -1) continue;

        int n1 = he.origin_node_id;
        int n2 = next_he->origin_node_id;

        std::pair<int, int> key = (n1 < n2) ? std::make_pair(n1, n2) : std::make_pair(n2, n1);
        edge_map[key].push_back(he.id);
    }

    int twins_found_count = 0;
    int boundary_hes_count = 0;

    for (auto& he : half_edges) {
        if (he.twin_half_edge_id != -1) continue;

        if (he.origin_node_id == -1 || he.next_half_edge_id == -1) continue;
        const HalfEdge_cpp* next_he = get_half_edge_by_id(he.next_half_edge_id);
        if (!next_he || next_he->origin_node_id == -1) continue;

        int n1 = he.origin_node_id;
        int n2 = next_he->origin_node_id;
        std::pair<int, int> key = (n1 < n2) ? std::make_pair(n1, n2) : std::make_pair(n2, n1);

        const auto& candidates = edge_map[key];
        if (candidates.size() == 2) {
            int he1_id = candidates[0];
            int he2_id = candidates[1];
            // Ensure we are not trying to set a twin for an already twinned half-edge
            if (half_edges[he1_id].twin_half_edge_id == -1 && half_edges[he2_id].twin_half_edge_id == -1) {
                 half_edges[he1_id].twin_half_edge_id = he2_id;
                 half_edges[he2_id].twin_half_edge_id = he1_id;
                 twins_found_count++;
            }
        } else if (candidates.size() == 1) {
            boundary_hes_count++; // This HE is a boundary HE
        } else if (candidates.size() > 2) {
             std::cerr << "Warning: Physical edge (" << n1 << "-" << n2
                       << ") corresponds to " << candidates.size() << " half-edges. Mesh problem?" << std::endl;
        }
    }
}

void Mesh_cpp::assign_boundary_markers_to_halfedges_cpp(const std::vector<int>& flat_edge_data, int num_edges_in, int num_edge_attrs) {
    if (num_edges_in == 0 || num_edge_attrs != 4) {
        for (auto& he : half_edges) {
            if (he.twin_half_edge_id == -1) {
                he.boundary_marker = 0; // Default for unspecified boundary
                he.original_poly_segment_id = -1;
            }
        }
        if (num_edges_in > 0 && num_edge_attrs != 4) {
             std::cerr << "警告: assign_boundary_markers_to_halfedges_cpp 期望每条边有 4 个属性, 但得到 " << num_edge_attrs << std::endl;
        }
        return;
    }

    std::map<std::pair<int, int>, std::pair<int, int>> edge_info_map_from_file;
    for (int i = 0; i < num_edges_in; ++i) {
        int base_idx = i * num_edge_attrs;
        int n1_id = flat_edge_data[base_idx + 0];
        int n2_id = flat_edge_data[base_idx + 1];
        int type_marker = flat_edge_data[base_idx + 2];
        int original_id = flat_edge_data[base_idx + 3];
        std::pair<int, int> key = (n1_id < n2_id) ? std::make_pair(n1_id, n2_id) : std::make_pair(n2_id, n1_id);
        edge_info_map_from_file[key] = std::make_pair(type_marker, original_id);
    }

    int assigned_count = 0;
    std::cout << "DEBUG_ASSIGN_BC_MARKERS: Entering assign_boundary_markers_to_halfedges_cpp." << std::endl;
    std::cout << "  Total half_edges to process: " << half_edges.size() << std::endl;
    std::cout << "  Edge info map from .edge file has " << edge_info_map_from_file.size() << " entries." << std::endl;

    for (auto& he : half_edges) {
        if (he.twin_half_edge_id == -1) { // Only process boundary half-edges
            const Node_cpp* node1_ptr = get_node_by_id(he.origin_node_id);
            const HalfEdge_cpp* next_he_ptr = get_half_edge_by_id(he.next_half_edge_id);
            if (!node1_ptr || !next_he_ptr) continue;
            const Node_cpp* node2_ptr = get_node_by_id(next_he_ptr->origin_node_id);
            if (!node2_ptr) continue;

            int n1_he = node1_ptr->id;
            int n2_he = node2_ptr->id;
            std::pair<int, int> key_he = (n1_he < n2_he) ? std::make_pair(n1_he, n2_he) : std::make_pair(n2_he, n1_he);

            auto it = edge_info_map_from_file.find(key_he);
            if (it != edge_info_map_from_file.end()) {
                he.boundary_marker = it->second.first;
                he.original_poly_segment_id = it->second.second;
                assigned_count++;
            } else {
                he.boundary_marker = 1; // Default for unspecified boundary edges from .edge file (e.g. wall)
                he.original_poly_segment_id = -1;
            }
        } else { // Internal half-edge
            he.boundary_marker = 0;
            he.original_poly_segment_id = -1;
        }
    }
    std::cout << "DEBUG_ASSIGN_BC_MARKERS: Assigned markers to " << assigned_count << " boundary half-edges based on .edge file." << std::endl;
}


void Mesh_cpp::precompute_cell_geometry_cpp() {
    for (auto& cell : cells) {
        if (cell.node_ids.size() != 3) continue;

        const Node_cpp* n0 = get_node_by_id(cell.node_ids[0]);
        const Node_cpp* n1 = get_node_by_id(cell.node_ids[1]);
        const Node_cpp* n2 = get_node_by_id(cell.node_ids[2]);

        if (!n0 || !n1 || !n2) {
            std::cerr << "Error: Cell " << cell.id << " has invalid node IDs during geometry precomputation." << std::endl;
            continue;
        }

        double area_signed = 0.5 * (n0->x * (n1->y - n2->y) +
                                   n1->x * (n2->y - n0->y) +
                                   n2->x * (n0->y - n1->y));
        cell.area = std::abs(area_signed);

        if (cell.area < 1e-12) { // A very small area
            cell.centroid = {n0->x, n0->y};
            cell.z_bed_centroid = n0->z_bed;
            cell.b_slope_x = 0.0;
            cell.b_slope_y = 0.0;
            continue;
        }

        cell.centroid[0] = (n0->x + n1->x + n2->x) / 3.0;
        cell.centroid[1] = (n0->y + n1->y + n2->y) / 3.0;
        cell.z_bed_centroid = (n0->z_bed + n1->z_bed + n2->z_bed) / 3.0;

        double denominator = 2.0 * area_signed;
        cell.b_slope_x = ((n1->y - n2->y) * n0->z_bed +
                          (n2->y - n0->y) * n1->z_bed +
                          (n0->y - n1->y) * n2->z_bed) / denominator;
        cell.b_slope_y = ((n2->x - n1->x) * n0->z_bed +
                          (n0->x - n2->x) * n1->z_bed +
                          (n1->x - n0->x) * n2->z_bed) / denominator;
    }
}

void Mesh_cpp::precompute_half_edge_geometry_cpp() {
    for (auto& he : half_edges) {
        if (he.origin_node_id == -1 || he.next_half_edge_id == -1) continue;

        const Node_cpp* n_origin = get_node_by_id(he.origin_node_id);
        const HalfEdge_cpp* he_next = get_half_edge_by_id(he.next_half_edge_id);
        if (!n_origin || !he_next || he_next->origin_node_id == -1) continue;
        const Node_cpp* n_end = get_node_by_id(he_next->origin_node_id);
        if (!n_end) continue;

        double dx = n_end->x - n_origin->x;
        double dy = n_end->y - n_origin->y;
        he.length = std::sqrt(dx * dx + dy * dy);

        he.mid_point[0] = (n_origin->x + n_end->x) / 2.0;
        he.mid_point[1] = (n_origin->y + n_end->y) / 2.0;

        if (he.length < 1e-12) {
            he.normal[0] = 0.0;
            he.normal[1] = 0.0;
        } else {
            he.normal[0] = dy / he.length;
            he.normal[1] = -dx / he.length;
        }
    }
}

// --- 实现缺失的 getter 函数 ---
const Node_cpp* Mesh_cpp::get_node_by_id(int node_id) const {
    // 假设节点ID是它们在nodes向量中的索引 (如果不是，需要更复杂的查找)
    if (node_id >= 0 && static_cast<size_t>(node_id) < nodes.size()) {
        // 进一步验证，以防ID与索引不完全对应 (例如，如果节点ID不从0开始或不连续)
        if (nodes[node_id].id == node_id) { // 确保索引处的节点ID就是我们想要的ID
            return &nodes[node_id];
        }
    }
    // 如果ID不是直接索引，或者上述快速检查失败，则进行线性搜索
    for (const auto& node : nodes) {
        if (node.id == node_id) {
            return &node;
        }
    }
    return nullptr; // 未找到
}

Node_cpp* Mesh_cpp::get_node_by_id_mutable(int node_id) { // 注意：非 const 版本
    if (node_id >= 0 && static_cast<size_t>(node_id) < nodes.size()) {
        if (nodes[node_id].id == node_id) {
            return &nodes[node_id];
        }
    }
    for (auto& node : nodes) { // 注意是 auto&
        if (node.id == node_id) {
            return &node;
        }
    }
    return nullptr;
}

const Cell_cpp* Mesh_cpp::get_cell_by_id(int cell_id) const {
    if (cell_id >= 0 && static_cast<size_t>(cell_id) < cells.size()) {
        if (cells[cell_id].id == cell_id) {
            return &cells[cell_id];
        }
    }
    for (const auto& cell : cells) {
        if (cell.id == cell_id) {
            return &cell;
        }
    }
    return nullptr;
}

const HalfEdge_cpp* Mesh_cpp::get_half_edge_by_id(int he_id) const {
     if (he_id >= 0 && static_cast<size_t>(he_id) < half_edges.size()) {
        if (half_edges[he_id].id == he_id) {
            return &half_edges[he_id];
        }
    }
    for (const auto& he : half_edges) {
        if (he.id == he_id) {
            return &he;
        }
    }
    return nullptr;
}

double Mesh_cpp::get_cell_region_attribute(int cell_id) const {
    const Cell_cpp* cell = get_cell_by_id(cell_id);
    if (cell) {
        return cell->region_attribute;
    }
    // 根据你的错误处理策略，可以抛出异常或返回一个特殊值
    throw std::out_of_range("Mesh_cpp::get_cell_region_attribute: Cell ID " + std::to_string(cell_id) + " not found.");
    // return 0.0; // 或者返回默认值
}

int Mesh_cpp::find_cell_containing_point(double x, double y) const {
    for (const auto& cell : cells) {
        if (cell.node_ids.size() != 3) continue; // 只处理三角形

        const Node_cpp* n0 = get_node_by_id(cell.node_ids[0]);
        const Node_cpp* n1 = get_node_by_id(cell.node_ids[1]);
        const Node_cpp* n2 = get_node_by_id(cell.node_ids[2]);

        if (!n0 || !n1 || !n2) continue;

        // 叉乘法判断点是否在三角形内
        // (P1-P0)x(P-P0), (P2-P1)x(P-P1), (P0-P2)x(P-P2) 的z分量应同号
        double val1 = (n1->x - n0->x) * (y - n0->y) - (n1->y - n0->y) * (x - n0->x);
        double val2 = (n2->x - n1->x) * (y - n1->y) - (n2->y - n1->y) * (x - n1->x);
        double val3 = (n0->x - n2->x) * (y - n2->y) - (n0->y - n2->y) * (x - n2->x);

        // 检查 epsilon 的值，如果它在头文件中定义为全局或类静态成员，这里就不需要 epsilon_mesh_geom
        // 假设有一个全局或易于访问的 epsilon
        double local_epsilon = 1e-9; // 或者使用 this->epsilon 如果定义为成员

        bool has_neg = (val1 < -local_epsilon) || (val2 < -local_epsilon) || (val3 < -local_epsilon);
        bool has_pos = (val1 > local_epsilon) || (val2 > local_epsilon) || (val3 > local_epsilon);

        if (!(has_neg && has_pos)) { // 如果所有符号相同（或为零）
            return cell.id;
        }
    }
    return -1; // 未找到
}


} // namespace HydroCore