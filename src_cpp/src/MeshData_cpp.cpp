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

namespace HydroCore { // HydroCore命名空间开始

// --- 文件读取辅助函数的实现 ---
bool Mesh_cpp::read_node_file_cpp(const std::string& filepath, std::vector<double>& flat_data, int& num_nodes_out, int& num_attrs_out) { // 读取节点文件(C++) 实现
    std::ifstream file(filepath); // 打开文件流
    if (!file.is_open()) { // 如果文件打开失败
        std::cerr << "Error: Could not open node file " << filepath << std::endl; // 打印错误信息
        return false; // 返回失败
    }

    std::string line; // 用于存储读取的每一行
    // 读取头部
    if (!std::getline(file, line)) { std::cerr << "Error: Node file " << filepath << " is empty or unreadable header." << std::endl; return false; } // 读取头部失败
    std::istringstream header_ss(line); // 创建字符串流处理头部
    int dim, point_attrs_count_file, has_marker_file; // 声明头部信息变量
    header_ss >> num_nodes_out >> dim >> point_attrs_count_file >> has_marker_file; // 从字符串流中提取头部信息
    if (header_ss.fail()) { std::cerr << "Error: Invalid node file header format in " << filepath << std::endl; return false; } // 提取失败

    num_attrs_out = 4 + (has_marker_file == 1 ? 1 : 0); // 计算实际属性数量 (id, x, y, z_bed [+ marker])

    flat_data.clear(); // 清空扁平化数据容器
    flat_data.reserve(num_nodes_out * num_attrs_out); // 预分配内存

    for (int i = 0; i < num_nodes_out; ++i) { // 遍历所有节点
        if (!std::getline(file, line)) { // 读取节点数据行
            std::cerr << "Error: Unexpected end of file or read error in node file " << filepath << " at node " << i << std::endl; // 读取失败
            return false; // 返回失败
        }
        std::istringstream line_ss(line); // 创建字符串流处理当前行
        double node_id, x, y, z_bed; // 声明节点数据变量
        double marker_val = 0.0; // 初始化标记值
        line_ss >> node_id >> x >> y; // 提取节点ID, x, y
        // 跳过文件中的额外点属性 (point_attrs_count_file)
        for(int attr_idx = 0; attr_idx < point_attrs_count_file; ++attr_idx) { // 遍历额外属性
            double dummy_attr; // 声明临时变量存储额外属性
            if (!(line_ss >> dummy_attr)) { // 尝试读取
                 // std::cerr << "Warning: Could not read all point attributes for node " << node_id << " in " << filepath << std::endl; // 打印警告
                 // break; // 如果属性不足，可能需要调整策略或报错
            }
        }
        line_ss >> z_bed; // 提取 z_bed (假设z_bed在所有额外属性之后，或者没有额外属性)

        if (has_marker_file == 1) { // 如果文件中有标记
            if (!(line_ss >> marker_val)) { // 尝试提取标记值
                 // std::cerr << "Warning: Could not read marker for node " << node_id << " in " << filepath << std::endl; // 打印警告
            }
        }
        if (line_ss.fail() && !line_ss.eof()) { // 如果提取过程中发生错误（非文件尾导致）
             std::cerr << "Error: Invalid data format for node " << i << " in " << filepath << ". Line: "<< line << std::endl; // 打印错误信息
             // 可能需要更详细的错误定位
             return false; // 返回失败
        }


        flat_data.push_back(node_id); // 添加节点ID
        flat_data.push_back(x);       // 添加x坐标
        flat_data.push_back(y);       // 添加y坐标
        flat_data.push_back(z_bed);   // 添加z_bed
        if (num_attrs_out == 5) flat_data.push_back(marker_val); // 如果有标记则添加标记
    }
    return true; // 读取成功
} // 结束函数

bool Mesh_cpp::read_cell_file_cpp(const std::string& filepath, std::vector<int>& flat_data, int& num_cells_out, int& nodes_per_cell_out) { // 读取单元文件(C++) 实现
    std::ifstream file(filepath); // 打开文件流
    if (!file.is_open()) { // 如果文件打开失败
        std::cerr << "Error: Could not open cell file " << filepath << std::endl; // 打印错误信息
        return false; // 返回失败
    }
    std::string line; // 用于存储读取的每一行
    // 读取头部
    if (!std::getline(file, line)) { std::cerr << "Error: Cell file " << filepath << " is empty or unreadable header." << std::endl; return false; } // 读取头部失败
    std::istringstream header_ss(line); // 创建字符串流处理头部
    int ele_attrs_count_file; // 声明文件中的单元属性数量
    header_ss >> num_cells_out >> nodes_per_cell_out >> ele_attrs_count_file; // 从字符串流中提取头部信息
    if (header_ss.fail()) { std::cerr << "Error: Invalid cell file header format in " << filepath << std::endl; return false; } // 提取失败

    if (nodes_per_cell_out != 3) { // 检查每单元节点数是否为3 (当前仅支持三角形)
        std::cerr << "Error: Cell file " << filepath << " indicates " << nodes_per_cell_out
                  << " nodes per cell, but C++ MeshData currently only supports 3." << std::endl; // 打印错误信息
        return false; // 返回失败
    }

    flat_data.clear(); // 清空扁平化数据容器
    flat_data.reserve(num_cells_out * (1 + nodes_per_cell_out)); // 预分配内存 (1 for cell_id)

    for (int i = 0; i < num_cells_out; ++i) { // 遍历所有单元
        if (!std::getline(file, line)) { // 读取单元数据行
            std::cerr << "Error: Unexpected end of file or read error in cell file " << filepath << " at cell " << i << std::endl; // 读取失败
            return false; // 返回失败
        }
        std::istringstream line_ss(line); // 创建字符串流处理当前行
        int ele_id; // 声明单元ID
        line_ss >> ele_id; // 提取单元ID
        flat_data.push_back(ele_id); // 添加单元ID
        for (int j = 0; j < nodes_per_cell_out; ++j) { // 遍历单元的节点
            int node_idx; // 声明节点索引
            line_ss >> node_idx; // 提取节点索引
            flat_data.push_back(node_idx); // 添加节点索引
        }
        // 跳过文件中的额外单元属性
        for(int attr_idx = 0; attr_idx < ele_attrs_count_file; ++attr_idx) { // 遍历额外属性
            int dummy_attr; // 声明临时变量存储额外属性
            if (!(line_ss >> dummy_attr)) { // 尝试读取
                // std::cerr << "Warning: Could not read all element attributes for element " << ele_id << " in " << filepath << std::endl; // 打印警告
                // break; // 如果属性不足
            }
        }
        if (line_ss.fail() && !line_ss.eof()) { // 如果提取过程中发生错误
            std::cerr << "Error: Invalid data format for cell " << i << " in " << filepath << ". Line: "<< line << std::endl; // 打印错误信息
            return false; // 返回失败
        }
    }
    return true; // 读取成功
} // 结束函数

bool Mesh_cpp::read_edge_file_cpp(const std::string& filepath, std::vector<int>& flat_data, int& num_edges_out, int& num_edge_attrs_out) { // 读取边文件(C++) 实现
    if (filepath.empty()) { // 如果文件路径为空
        num_edges_out = 0; // 边数量为0
        num_edge_attrs_out = 0; // 属性数量为0
        flat_data.clear(); // 清空数据
        return true; // 认为成功 (没有边文件是允许的)
    }
    std::ifstream file(filepath); // 打开文件流
    if (!file.is_open()) { // 如果文件打开失败
        std::cerr << "Warning: Could not open edge file " << filepath << ". Proceeding without edge data." << std::endl; // 打印警告
        num_edges_out = 0; // 边数量为0
        num_edge_attrs_out = 0; // 属性数量为0
        flat_data.clear(); // 清空数据
        return true; // 仍然返回true，表示可以继续，但没有边数据
    }
    std::string line; // 用于存储读取的每一行
    // 读取头部
    if (!std::getline(file, line)) { std::cerr << "Error: Edge file " << filepath << " is empty or unreadable header." << std::endl; return false; } // 读取头部失败
    std::istringstream header_ss(line); // 创建字符串流处理头部
    int has_marker_file; // 声明文件中是否有标记
    header_ss >> num_edges_out >> has_marker_file; // 从字符串流中提取头部信息
    if (header_ss.fail()) { std::cerr << "Error: Invalid edge file header format in " << filepath << std::endl; return false; } // 提取失败

    num_edge_attrs_out = 3 + (has_marker_file == 1 ? 1 : 0); // 计算实际属性数量 (id, n1, n2 [+ marker])

    flat_data.clear(); // 清空扁平化数据容器
    flat_data.reserve(num_edges_out * num_edge_attrs_out); // 预分配内存

    for (int i = 0; i < num_edges_out; ++i) { // 遍历所有边
        if (!std::getline(file, line)) { // 读取边数据行
            std::cerr << "Error: Unexpected end of file or read error in edge file " << filepath << " at edge " << i << std::endl; // 读取失败
            return false; // 返回失败
        }
        std::istringstream line_ss(line); // 创建字符串流处理当前行
        int edge_id, node1_id, node2_id; // 声明边数据变量
        int marker_val = 0; // 初始化标记值
        line_ss >> edge_id >> node1_id >> node2_id; // 提取边ID, 节点1ID, 节点2ID
        if (has_marker_file == 1) { // 如果文件中有标记
            if(!(line_ss >> marker_val)) { // 尝试提取标记值
                // std::cerr << "Warning: Could not read marker for edge " << edge_id << " in " << filepath << std::endl; // 打印警告
            }
        }
        if (line_ss.fail() && !line_ss.eof()) { // 如果提取过程中发生错误
             std::cerr << "Error: Invalid data format for edge " << i << " in " << filepath << ". Line: "<< line << std::endl; // 打印错误信息
             return false; // 返回失败
        }

        flat_data.push_back(edge_id);    // 添加边ID
        flat_data.push_back(node1_id);   // 添加节点1ID
        flat_data.push_back(node2_id);   // 添加节点2ID
        if (num_edge_attrs_out == 4) flat_data.push_back(marker_val); // 如果有标记则添加标记
    }
    return true; // 读取成功
} // 结束函数

void Mesh_cpp::load_mesh_from_files(const std::string& node_filepath, // 从文件加载网格数据实现
                                  const std::string& cell_filepath,
                                  const std::string& edge_filepath,
                                  const std::vector<double>& cell_manning_values) {
    std::cout << "C++ Mesh: Starting to load mesh from files..." << std::endl; // 打印开始加载信息
    std::cout << "  Node file: " << node_filepath << std::endl; // 打印节点文件路径
    std::cout << "  Cell file: " << cell_filepath << std::endl; // 打印单元文件路径
    std::cout << "  Edge file: " << (edge_filepath.empty() ? "Not provided" : edge_filepath) << std::endl; // 打印边文件路径

    std::vector<double> flat_nodes_vec; // 存储扁平化节点数据的vector
    int num_nodes_read, node_attrs_read; // 存储读取的节点数和属性数
    if (!read_node_file_cpp(node_filepath, flat_nodes_vec, num_nodes_read, node_attrs_read)) { // 调用读取节点文件函数
        throw std::runtime_error("Failed to read node file: " + node_filepath); // 抛出运行时错误
    }
    load_nodes_from_numpy(flat_nodes_vec, num_nodes_read, node_attrs_read); // 调用从NumPy加载节点数据的方法 (内部用vector)
    std::cout << "  Loaded " << num_nodes_read << " nodes with " << node_attrs_read << " attributes each." << std::endl; // 打印加载节点信息

    std::vector<int> flat_cells_vec; // 存储扁平化单元数据的vector
    int num_cells_read, nodes_per_cell_read; // 存储读取的单元数和每单元节点数
    if (!read_cell_file_cpp(cell_filepath, flat_cells_vec, num_cells_read, nodes_per_cell_read)) { // 调用读取单元文件函数
        throw std::runtime_error("Failed to read cell file: " + cell_filepath); // 抛出运行时错误
    }
    // 确保曼宁系数值数量与单元数量匹配 (如果提供了)
    if (!cell_manning_values.empty() && cell_manning_values.size() != static_cast<size_t>(num_cells_read)) { // 如果数量不匹配
        std::cerr << "Warning: Provided " << cell_manning_values.size() << " Manning values, but read "
                  << num_cells_read << " cells. Mismatched values might lead to default usage." << std::endl; // 打印警告
    }
    load_cells_from_numpy(flat_cells_vec, num_cells_read, nodes_per_cell_read, cell_manning_values); // 调用从NumPy加载单元数据的方法
    std::cout << "  Loaded " << num_cells_read << " cells with " << nodes_per_cell_read << " nodes each." << std::endl; // 打印加载单元信息

    std::vector<int> flat_edges_vec; // 存储扁平化边数据的vector
    int num_edges_read = 0, edge_attrs_read = 0; // 初始化读取的边数和属性数
    if (!edge_filepath.empty()) { // 如果边文件路径不为空
        if (!read_edge_file_cpp(edge_filepath, flat_edges_vec, num_edges_read, edge_attrs_read)) { // 调用读取边文件函数
            // read_edge_file_cpp 内部会打印警告如果文件不存在，但这里如果是严重错误则抛出
            // 如果只是文件不存在，read_edge_file_cpp 会返回true并将 num_edges_read=0
            if(num_edges_read != 0) { // 如果读取失败且num_edges_read不为0，说明是文件格式问题
                 throw std::runtime_error("Failed to read edge file: " + edge_filepath); // 抛出运行时错误
            }
        }
    }
    if (num_edges_read > 0) { // 如果成功读取到边
        std::cout << "  Loaded " << num_edges_read << " edges with " << edge_attrs_read << " attributes each." << std::endl; // 打印加载边信息
    } else { // 否则
        std::cout << "  No edge data loaded (or edge file not provided/empty)." << std::endl; // 打印无边数据信息
    }


    std::cout << "C++ Mesh: Precomputing geometry and topology..." << std::endl; // 打印预计算信息
    precompute_geometry_and_topology(flat_edges_vec, num_edges_read, edge_attrs_read); // 调用预计算几何和拓扑的方法
    std::cout << "C++ Mesh: Loading and precomputation complete." << std::endl; // 打印加载完成信息
} // 结束函数


// load_nodes_from_numpy, load_cells_from_numpy, precompute_geometry_and_topology,
// setup_half_edge_structure_optimized_cpp, assign_boundary_markers_to_halfedges_cpp,
// precompute_cell_geometry_cpp, precompute_half_edge_geometry_cpp,
// get_node_by_id, get_cell_by_id, get_half_edge_by_id 等方法的实现保持不变。
// ... (此处省略已有的 Mesh_cpp 方法实现，它们现在被 load_mesh_from_files 间接调用) ...
void Mesh_cpp::load_nodes_from_numpy(const std::vector<double>& flat_node_data, int num_nodes_in, int num_node_attrs) { // 从NumPy加载节点实现
    if (num_node_attrs < 4) { // 至少需要 id, x, y, z_bed
        throw std::runtime_error("Node data must have at least 4 attributes (id, x, y, z_bed)."); // 抛出运行时错误
    }
    nodes.clear(); // 清空现有节点
    nodes.reserve(num_nodes_in); // 预分配内存
    for (int i = 0; i < num_nodes_in; ++i) { // 遍历所有节点
        int base_idx = i * num_node_attrs; // 计算基准索引
        int id = static_cast<int>(flat_node_data[base_idx + 0]); // 获取节点ID
        double x = flat_node_data[base_idx + 1]; // 获取x坐标
        double y = flat_node_data[base_idx + 2]; // 获取y坐标
        double z_bed = flat_node_data[base_idx + 3]; // 获取z_bed
        int marker = (num_node_attrs > 4) ? static_cast<int>(flat_node_data[base_idx + 4]) : 0; // 获取标记（如果存在）
        nodes.emplace_back(id, x, y, z_bed, marker); // 创建并添加节点对象
    }
} // 结束函数

void Mesh_cpp::load_cells_from_numpy(const std::vector<int>& flat_cell_data, int num_cells_in, int nodes_per_cell, // 从NumPy加载单元实现
                                   const std::vector<double>& cell_manning_values) {
    if (nodes_per_cell != 3) { // 当前硬编码为三角形
        throw std::runtime_error("Currently only supports 3 nodes per cell (triangles)."); // 抛出运行时错误
    }
    bool manning_provided = !cell_manning_values.empty(); // 检查是否提供了曼宁值
    if (manning_provided && cell_manning_values.size() != static_cast<size_t>(num_cells_in) ) { // 如果提供了但数量不匹配
        std::cerr << "Warning (load_cells_from_numpy): Number of Manning values (" << cell_manning_values.size()
                  << ") does not match number of cells (" << num_cells_in
                  << "). Using default or first value if available for mismatched cells." << std::endl; // 打印警告
    }

    cells.clear(); // 清空现有单元
    cells.reserve(num_cells_in); // 预分配内存
    for (int i = 0; i < num_cells_in; ++i) { // 遍历所有单元
        int base_idx = i * (1 + nodes_per_cell); // 1 for cell_id + nodes_per_cell // 计算基准索引
        int id = flat_cell_data[base_idx + 0]; // 获取单元ID
        Cell_cpp current_cell(id); // 创建单元对象
        current_cell.node_ids.reserve(nodes_per_cell); // 为节点ID列表预分配内存
        for (int j = 0; j < nodes_per_cell; ++j) { // 遍历每单元的节点
            current_cell.node_ids.push_back(flat_cell_data[base_idx + 1 + j]); // 添加节点ID
        }
        if (manning_provided) { // 如果提供了曼宁值
            if (static_cast<size_t>(i) < cell_manning_values.size()) { // 如果当前索引在曼宁值列表范围内
                current_cell.manning_n = cell_manning_values[i]; // 设置曼宁值
            } else { // 如果超出范围 (数量不匹配时发生)
                // current_cell.manning_n = cell_manning_values[0]; // 可以选择用第一个值作为备用，或保持默认
            }
        } // else 使用构造函数中的默认曼宁值
        cells.emplace_back(current_cell); // 添加单元对象
    }
} // 结束函数

void Mesh_cpp::precompute_geometry_and_topology(const std::vector<int>& flat_edge_data, int num_edges, int num_edge_attrs) { // 预计算几何和拓扑实现
    half_edges.clear(); // 清空可能已有的半边
    int half_edge_id_counter = 0; // 初始化半边ID计数器
    for (size_t cell_idx = 0; cell_idx < cells.size(); ++cell_idx) { // 遍历所有单元
        Cell_cpp& current_cell = cells[cell_idx]; // 获取当前单元的引用
        if (current_cell.node_ids.size() != 3) continue; // 只处理三角形

        current_cell.half_edge_ids_list.resize(3); // 调整半边ID列表大小
        std::array<int, 3> hes_in_this_cell_ids; // 存储当前单元的3条半边ID

        for (int i = 0; i < 3; ++i) { // 遍历单元的三个顶点
            HalfEdge_cpp he(half_edge_id_counter); // 创建半边对象
            he.origin_node_id = current_cell.node_ids[i]; // 设置起点ID
            he.cell_id = current_cell.id; // 设置所属单元ID

            hes_in_this_cell_ids[i] = he.id; // 存储半边ID
            half_edges.push_back(he); // 添加到总的半边列表
            half_edge_id_counter++; // 增加半边ID计数器
        }

        for (int i = 0; i < 3; ++i) { // 遍历单元的三个半边
            half_edges[hes_in_this_cell_ids[i]].next_half_edge_id = hes_in_this_cell_ids[(i + 1) % 3]; // 设置下一条半边ID
            half_edges[hes_in_this_cell_ids[i]].prev_half_edge_id = hes_in_this_cell_ids[(i + 2) % 3]; // 设置上一条半边ID
            current_cell.half_edge_ids_list[i] = hes_in_this_cell_ids[i]; // 在单元中记录半边ID
        }
    }

    setup_half_edge_structure_optimized_cpp(); // 设置孪生关系
    assign_boundary_markers_to_halfedges_cpp(flat_edge_data, num_edges, num_edge_attrs); // 分配边界标记
    precompute_cell_geometry_cpp(); // 计算单元几何属性
    precompute_half_edge_geometry_cpp(); // 计算半边几何属性

    // std::cout << "C++ Mesh: Topology and geometry precomputation complete." << std::endl; // 打印完成信息
    // std::cout << "  Total nodes: " << nodes.size() << std::endl; // 打印节点数
    // std::cout << "  Total cells: " << cells.size() << std::endl; // 打印单元数
    // std::cout << "  Total half_edges: " << half_edges.size() << std::endl; // 打印半边数
} // 结束函数


void Mesh_cpp::setup_half_edge_structure_optimized_cpp() { // 优化版半边孪生关系设置实现
    // std::cout << "  Setting up twin relationships in C++..." << std::endl; // 打印开始信息
    std::map<std::pair<int, int>, std::vector<int>> edge_map; // 用于映射物理边到半边ID列表的map

    for (const auto& he : half_edges) { // 遍历所有半边
        if (he.origin_node_id == -1 || he.next_half_edge_id == -1) continue; // 跳过无效半边
        const HalfEdge_cpp* next_he = get_half_edge_by_id(he.next_half_edge_id); // 获取下一条半边
        if (!next_he || next_he->origin_node_id == -1) continue; // 跳过无效的下一条半边

        int n1 = he.origin_node_id; // 获取起点ID
        int n2 = next_he->origin_node_id; // 获取终点ID (即下一条半边的起点)

        std::pair<int, int> key = (n1 < n2) ? std::make_pair(n1, n2) : std::make_pair(n2, n1); // 创建排序后的节点对作为key
        edge_map[key].push_back(he.id); // 将当前半边ID添加到对应物理边的列表中
    }

    int twins_found_count = 0; // 初始化找到的孪生对数量
    int boundary_hes_count = 0; // 初始化边界半边数量

    for (auto& he : half_edges) { // 遍历所有半边 (注意是引用，以便修改)
        if (he.twin_half_edge_id != -1) continue; // 如果已经处理过 (被其孪生半边处理) 则跳过

        if (he.origin_node_id == -1 || he.next_half_edge_id == -1) continue; // 跳过无效半边
        const HalfEdge_cpp* next_he = get_half_edge_by_id(he.next_half_edge_id); // 获取下一条半边
        if (!next_he || next_he->origin_node_id == -1) continue; // 跳过无效的下一条半边

        int n1 = he.origin_node_id; // 获取起点ID
        int n2 = next_he->origin_node_id; // 获取终点ID
        std::pair<int, int> key = (n1 < n2) ? std::make_pair(n1, n2) : std::make_pair(n2, n1); // 创建key

        const auto& candidates = edge_map[key]; // 获取与该物理边对应的所有半边ID
        if (candidates.size() == 2) { // 如果恰好有两条半边对应一个物理边 (内部边)
            int he1_id = candidates[0]; // 获取第一条半边ID
            int he2_id = candidates[1]; // 获取第二条半边ID
            if (half_edges[he1_id].twin_half_edge_id == -1 && half_edges[he2_id].twin_half_edge_id == -1) { // 确保它们都未设置孪生
                 half_edges[he1_id].twin_half_edge_id = he2_id; // 设置孪生关系
                 half_edges[he2_id].twin_half_edge_id = he1_id; // 设置孪生关系
                 twins_found_count++; // 增加孪生对计数
            }
        } else if (candidates.size() == 1) { // 如果只有一条半边对应一个物理边 (边界边)
            boundary_hes_count++; // 增加边界半边计数
        } else if (candidates.size() > 2) { // 如果多于两条半边 (异常情况)
             std::cerr << "Warning: Physical edge (" << n1 << "-" << n2
                       << ") corresponds to " << candidates.size() << " half-edges. Mesh problem?" << std::endl; // 打印警告
        }
    }
    //  std::cout << "  Twin setup: Found " << twins_found_count << " internal physical edges and "
    //           << boundary_hes_count << " boundary physical edges (resulting in " << boundary_hes_count << " boundary half-edges)." << std::endl; // 打印统计信息
} // 结束函数

void Mesh_cpp::assign_boundary_markers_to_halfedges_cpp(const std::vector<int>& flat_edge_data, int num_edges_in, int num_edge_attrs) { // 分配边界标记实现
    if (num_edge_attrs < 3 || num_edges_in == 0) { // 如果属性不足或没有边数据
        // std::cerr << "Warning: Edge data has too few attributes or no edges, cannot assign markers." << std::endl; // 打印警告
        // 将所有边界半边的标记设为默认值0
        for (auto& he : half_edges) { // 遍历所有半边
            if (he.twin_half_edge_id == -1) { // 如果是边界半边
                he.boundary_marker = 0; // 设置默认标记
            }
        }
        return; // 返回
    }
    bool has_marker_in_file = (num_edge_attrs >= 4); // 判断文件中是否有标记
    // std::cout << "  Assigning boundary markers to half-edges in C++ (file has markers: " << std::boolalpha << has_marker_in_file << ")..." << std::endl; // 打印信息

    std::map<std::pair<int, int>, int> edge_marker_map_from_file; // 用于存储从文件读取的边标记
    if (has_marker_in_file) { // 如果文件中有标记
        for (int i = 0; i < num_edges_in; ++i) { // 遍历文件中的所有边
            int base_idx = i * num_edge_attrs; // 计算基准索引
            int n1_id = flat_edge_data[base_idx + 1]; // 获取节点1ID
            int n2_id = flat_edge_data[base_idx + 2]; // 获取节点2ID
            int marker = static_cast<int>(flat_edge_data[base_idx + 3]); // 获取标记
            std::pair<int, int> key = (n1_id < n2_id) ? std::make_pair(n1_id, n2_id) : std::make_pair(n2_id, n1_id); // 创建key
            edge_marker_map_from_file[key] = marker; // 存储标记
        }
    }

    int assigned_count = 0; // 初始化已分配标记的计数
    for (auto& he : half_edges) { // 遍历所有半边 (注意是引用)
        if (he.twin_half_edge_id == -1) { // 如果是边界半边
             if (he.origin_node_id == -1 || he.next_half_edge_id == -1) continue; // 跳过无效半边
             const HalfEdge_cpp* next_he = get_half_edge_by_id(he.next_half_edge_id); // 获取下一条半边
             if (!next_he || next_he->origin_node_id == -1) continue; // 跳过无效的下一条半边

            int n1 = he.origin_node_id; // 获取起点ID
            int n2 = next_he->origin_node_id; // 获取终点ID
            std::pair<int, int> key_he = (n1 < n2) ? std::make_pair(n1, n2) : std::make_pair(n2, n1); // 创建key

            if (has_marker_in_file && edge_marker_map_from_file.count(key_he)) { // 如果文件中有标记且在map中找到
                he.boundary_marker = edge_marker_map_from_file[key_he]; // 设置标记
                assigned_count++; // 增加计数
            } else { // 否则
                he.boundary_marker = 0; // 设置默认标记0
            }
        }
    }
    // std::cout << "  Assigned markers to " << assigned_count << " boundary half-edges (others default to 0)." << std::endl; // 打印统计信息
} // 结束函数

void Mesh_cpp::precompute_cell_geometry_cpp() { // 预计算单元几何属性实现
    // std::cout << "  Precomputing cell geometry in C++..." << std::endl; // 打印开始信息
    for (auto& cell : cells) { // 遍历所有单元 (注意是引用)
        if (cell.node_ids.size() != 3) continue; // 只处理三角形

        const Node_cpp* n0 = get_node_by_id(cell.node_ids[0]); // 获取节点0
        const Node_cpp* n1 = get_node_by_id(cell.node_ids[1]); // 获取节点1
        const Node_cpp* n2 = get_node_by_id(cell.node_ids[2]); // 获取节点2

        if (!n0 || !n1 || !n2) { // 如果任一节点无效
            std::cerr << "Error: Cell " << cell.id << " has invalid node IDs during geometry precomputation." << std::endl; // 打印错误信息
            continue; // 跳过
        }

        double area_signed = 0.5 * (n0->x * (n1->y - n2->y) + // 计算有向面积
                                   n1->x * (n2->y - n0->y) +
                                   n2->x * (n0->y - n1->y));
        cell.area = std::abs(area_signed); // 单元面积取绝对值

        if (cell.area < 1e-12) { // 如果面积过小
            cell.centroid = {n0->x, n0->y}; // 形心简化为第一个点
            cell.z_bed_centroid = n0->z_bed; // 形心底高程简化为第一个点
            cell.b_slope_x = 0.0; // 底坡设为0
            cell.b_slope_y = 0.0; // 底坡设为0
            continue; // 继续下一个单元
        }

        cell.centroid[0] = (n0->x + n1->x + n2->x) / 3.0; // 计算形心x坐标
        cell.centroid[1] = (n0->y + n1->y + n2->y) / 3.0; // 计算形心y坐标
        cell.z_bed_centroid = (n0->z_bed + n1->z_bed + n2->z_bed) / 3.0; // 计算形心底高程

        double denominator = 2.0 * area_signed; // 计算底坡分母
        cell.b_slope_x = ((n1->y - n2->y) * n0->z_bed + // 计算x方向底坡
                          (n2->y - n0->y) * n1->z_bed +
                          (n0->y - n1->y) * n2->z_bed) / denominator;
        cell.b_slope_y = ((n2->x - n1->x) * n0->z_bed + // 计算y方向底坡
                          (n0->x - n2->x) * n1->z_bed +
                          (n1->x - n0->x) * n2->z_bed) / denominator;
    }
} // 结束函数

void Mesh_cpp::precompute_half_edge_geometry_cpp() { // 预计算半边几何属性实现
    // std::cout << "  Precomputing half-edge geometry in C++..." << std::endl; // 打印开始信息
    for (auto& he : half_edges) { // 遍历所有半边 (注意是引用)
        if (he.origin_node_id == -1 || he.next_half_edge_id == -1) continue; // 跳过无效半边

        const Node_cpp* n_origin = get_node_by_id(he.origin_node_id); // 获取起点
        const HalfEdge_cpp* he_next = get_half_edge_by_id(he.next_half_edge_id); // 获取下一条半边
        if (!n_origin || !he_next || he_next->origin_node_id == -1) continue; // 跳过无效的下一条半边
        const Node_cpp* n_end = get_node_by_id(he_next->origin_node_id); // 获取终点 (即下一条半边的起点)
        if (!n_end) continue; // 跳过无效终点

        double dx = n_end->x - n_origin->x; // 计算x方向差值
        double dy = n_end->y - n_origin->y; // 计算y方向差值
        he.length = std::sqrt(dx * dx + dy * dy); // 计算边长

        he.mid_point[0] = (n_origin->x + n_end->x) / 2.0; // 计算中点x坐标
        he.mid_point[1] = (n_origin->y + n_end->y) / 2.0; // 计算中点y坐标

        if (he.length < 1e-12) { // 如果边长过小
            he.normal[0] = 0.0; // 法向量设为0
            he.normal[1] = 0.0; // 法向量设为0
        } else { // 否则
            he.normal[0] = dy / he.length;  // 法向量x分量 (指向单元外部)
            he.normal[1] = -dx / he.length; // 法向量y分量
        }
    }
} // 结束函数

const Node_cpp* Mesh_cpp::get_node_by_id(int node_id) const { // 通过ID获取节点(const)实现
    if (node_id >= 0 && static_cast<size_t>(node_id) < nodes.size() && nodes[node_id].id == node_id) { // 简化假设：ID是连续索引
        return &nodes[node_id]; // 返回对应节点指针
    }
    for (const auto& node : nodes) { // 更鲁棒的查找
        if (node.id == node_id) return &node; // 如果找到则返回
    }
    return nullptr; // 未找到则返回空指针
} // 结束函数
Node_cpp* Mesh_cpp::get_node_by_id_mutable(int node_id) { // 通过ID获取节点(可修改)实现
    if (node_id >= 0 && static_cast<size_t>(node_id) < nodes.size() && nodes[node_id].id == node_id) { // 简化假设
        return &nodes[node_id]; // 返回对应节点指针
    }
    for (auto& node : nodes) { // 更鲁棒的查找 (注意是auto&)
        if (node.id == node_id) return &node; // 如果找到则返回
    }
    return nullptr; // 未找到则返回空指针
} // 结束函数

const Cell_cpp* Mesh_cpp::get_cell_by_id(int cell_id) const { // 通过ID获取单元(const)实现
    if (cell_id >= 0 && static_cast<size_t>(cell_id) < cells.size() && cells[cell_id].id == cell_id) { // 简化假设
        return &cells[cell_id]; // 返回对应单元指针
    }
    for (const auto& cell : cells) { // 更鲁棒的查找
        if (cell.id == cell_id) return &cell; // 如果找到则返回
    }
    return nullptr; // 未找到则返回空指针
} // 结束函数

const HalfEdge_cpp* Mesh_cpp::get_half_edge_by_id(int he_id) const { // 通过ID获取半边(const)实现
     if (he_id >= 0 && static_cast<size_t>(he_id) < half_edges.size() && half_edges[he_id].id == he_id) { // 简化假设
        return &half_edges[he_id]; // 返回对应半边指针
    }
    for (const auto& he : half_edges) { // 更鲁棒的查找
        if (he.id == he_id) return &he; // 如果找到则返回
    }
    return nullptr; // 未找到则返回空指针
} // 结束函数


} // namespace HydroCore