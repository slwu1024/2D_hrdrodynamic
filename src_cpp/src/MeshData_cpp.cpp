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

#ifdef _WIN32 // 仅在Windows平台下引入以下头文件和函数
#define WIN32_LEAN_AND_MEAN // 排除不常用的Windows头文件
#include <windows.h> // 引入Windows API头文件，用于字符转换
#endif

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
            // std::cerr << "Error in MultiByteToWideChar (calculating size): " << GetLastError() << std::endl; // 打印错误信息 (可选)
            return std::wstring(); // 返回空宽字符串
        }
        std::wstring wide_str(wide_char_count -1, 0); // 创建宽字符串，大小为 count-1 (不包括null终止符)
        // 执行转换
        if (MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, &wide_str[0], wide_char_count) == 0) { // 调用Windows API进行转换
            // std::cerr << "Error in MultiByteToWideChar (converting): " << GetLastError() << std::endl; // 打印错误信息 (可选)
            return std::wstring(); // 转换失败则返回空宽字符串
        }
        return wide_str; // 返回转换后的宽字符串
    }
#endif
// --- 文件读取辅助函数的实现 ---
bool Mesh_cpp::read_node_file_cpp(const std::string& filepath_utf8, std::vector<double>& flat_data, int& num_nodes_out, int& num_attrs_out) { // 读取节点文件(C++) 实现
    std::cout << "C++ received UTF-8 filepath for node file: [" << filepath_utf8 << "]" << std::endl; // 打印接收到的UTF-8路径
    std::ifstream file; // 声明文件输入流对象
#ifdef _WIN32 // 如果是Windows平台
    std::wstring w_filepath = utf8_to_wstring_windows(filepath_utf8); // 将UTF-8路径转换为宽字符路径
    if (!w_filepath.empty()) { // 如果转换成功且路径非空
        // std::cout << "C++ attempting to open node file with wstring (Windows): [" << // 调试信息
        //     std::string(w_filepath.begin(), w_filepath.end()) << "]" << std::endl; // 打印宽字符串（可能乱码）
        file.open(w_filepath.c_str()); // 使用宽字符路径打开文件
    } else if (!filepath_utf8.empty()) { // 如果转换失败但原始UTF-8路径非空
        std::cerr << "Warning: Failed to convert node filepath to wstring, trying with original UTF-8 string." << std::endl; // 打印警告
        file.open(filepath_utf8.c_str()); // 尝试用原始UTF-8路径打开
    } else { // 如果原始UTF-8路径也为空
         file.open(filepath_utf8.c_str()); // 尝试用原始UTF-8路径打开（会失败，但保持逻辑一致）
    }
#else
    // 对于非Windows平台，直接使用UTF-8路径
    file.open(filepath_utf8.c_str()); // 直接打开
#endif

    if (!file.is_open()) { // 如果文件打开失败
        std::cerr << "Error: Could not open node file " << filepath_utf8 << std::endl; // 打印错误信息
        return false; // 返回失败
    }

    std::string line; // 用于存储读取的每一行
    // 读取头部
    if (!std::getline(file, line)) { std::cerr << "Error: Node file " << filepath_utf8 << " is empty or unreadable header." << std::endl; file.close(); return false; } // 读取头部失败，关闭文件并返回
    std::istringstream header_ss(line); // 创建字符串流处理头部
    int dim, point_attrs_count_file, has_marker_file; // 声明头部信息变量
    header_ss >> num_nodes_out >> dim >> point_attrs_count_file >> has_marker_file; // 从字符串流中提取头部信息
    if (header_ss.fail()) { std::cerr << "Error: Invalid node file header format in " << filepath_utf8 << std::endl; file.close(); return false; } // 提取失败，关闭文件并返回

    num_attrs_out = 4 + (has_marker_file == 1 ? 1 : 0); // 计算实际属性数量 (id, x, y, z_bed [+ marker])

    flat_data.clear(); // 清空扁平化数据容器
    flat_data.reserve(num_nodes_out * num_attrs_out); // 预分配内存

    for (int i = 0; i < num_nodes_out; ++i) { // 遍历所有节点
        if (!std::getline(file, line)) { // 读取节点数据行
            std::cerr << "Error: Unexpected end of file or read error in node file " << filepath_utf8 << " at node " << i << std::endl; // 读取失败
            file.close(); return false; // 关闭文件并返回失败
        }
        std::istringstream line_ss(line); // 创建字符串流处理当前行
        double node_id, x, y, z_bed; // 声明节点数据变量
        double marker_val = 0.0; // 初始化标记值
        line_ss >> node_id >> x >> y; // 提取节点ID, x, y
        // 跳过文件中的额外点属性 (point_attrs_count_file)
        for(int attr_idx = 0; attr_idx < point_attrs_count_file; ++attr_idx) { // 遍历额外属性
            double dummy_attr; // 声明临时变量存储额外属性
            if (!(line_ss >> dummy_attr)) { // 尝试读取
                 // std::cerr << "Warning: Could not read all point attributes for node " << node_id << " in " << filepath_utf8 << std::endl; // 打印警告
                 // break; // 如果属性不足，可能需要调整策略或报错
            }
        }
        line_ss >> z_bed; // 提取 z_bed (假设z_bed在所有额外属性之后，或者没有额外属性)

        if (has_marker_file == 1) { // 如果文件中有标记
            if (!(line_ss >> marker_val)) { // 尝试提取标记值
                 // std::cerr << "Warning: Could not read marker for node " << node_id << " in " << filepath_utf8 << std::endl; // 打印警告
            }
        }
        if (line_ss.fail() && !line_ss.eof()) { // 如果提取过程中发生错误（非文件尾导致）
             std::cerr << "Error: Invalid data format for node " << i << " in " << filepath_utf8 << ". Line: "<< line << std::endl; // 打印错误信息
             file.close(); return false; // 关闭文件并返回失败
        }


        flat_data.push_back(node_id); // 添加节点ID
        flat_data.push_back(x);       // 添加x坐标
        flat_data.push_back(y);       // 添加y坐标
        flat_data.push_back(z_bed);   // 添加z_bed
        if (num_attrs_out == 5) flat_data.push_back(marker_val); // 如果有标记则添加标记
    }
    file.close(); // 关闭文件
    return true; // 读取成功
} // 结束函数

bool Mesh_cpp::read_cell_file_cpp(const std::string& filepath_utf8,
                                  std::vector<int>& flat_cell_data_out,      // 输出单元节点连接
                                  std::vector<double>& cell_attributes_out, // 输出单元区域属性
                                  int& num_cells_out,
                                  int& num_nodes_per_cell_out) {
    // ... (文件打开逻辑不变) ...
    std::ifstream file;
    // ... (打开文件的代码，包括Windows路径转换) ...
    #ifdef _WIN32
    std::wstring w_filepath = utf8_to_wstring_windows(filepath_utf8);
    if (!w_filepath.empty()) {
        file.open(w_filepath.c_str());
    } else if (!filepath_utf8.empty()) {
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
    if (!std::getline(file, line)) { /* ... error handling ... */ file.close(); return false; }
    std::istringstream header_ss(line);
    int num_cell_attrs_in_file; // 文件中声明的每个单元的属性数量
    header_ss >> num_cells_out >> num_nodes_per_cell_out >> num_cell_attrs_in_file;
    if (header_ss.fail()) { /* ... error handling ... */ file.close(); return false; }

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
        if (!std::getline(file, line)) { /* ... error handling ... */ file.close(); return false; }
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

        // 提取区域属性
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
                // std::cerr << "Warning: Cell " << ele_id << " in " << filepath_utf8
                //           << " declared attributes but line parts are insufficient. Using 0.0 for region attribute." << std::endl;
                cell_attributes_out.push_back(0.0); // 默认值
            }
        } else {
            cell_attributes_out.push_back(0.0); // 文件未声明属性，使用默认值
        }
    }
    file.close();
    return true;
}

// 新增 get_cell_region_attribute 成员函数的定义
double Mesh_cpp::get_cell_region_attribute(int cell_id) const {
    // 调用本类的 get_cell_by_id (可以是 this->get_cell_by_id 或者直接 get_cell_by_id)
    const Cell_cpp* cell = this->get_cell_by_id(cell_id);
    if (cell) {
        return cell->region_attribute;
    }

    // 处理找不到单元的情况 (选择一种)

    // 选项 A: 返回默认值并打印警告 (不推荐在核心库函数中直接打印到cerr，除非用于临时调试)
    // std::cerr << "Warning (Mesh_cpp::get_cell_region_attribute): Cell with ID "
    //           << cell_id << " not found. Returning default attribute 0.0." << std::endl;
    // return 0.0;

    // 选项 B: 抛出异常 (通常更好，让调用者知道出错了)
    throw std::out_of_range("Mesh_cpp::get_cell_region_attribute: Cell ID " + std::to_string(cell_id) + " not found.");

    // 选项 C: 如果你的设计允许单元属性有一个明确的“未找到”或“无效”的值，可以返回那个值
    // return SOME_INVALID_ATTRIBUTE_VALUE;
}

bool Mesh_cpp::read_edge_file_cpp(const std::string& filepath_utf8, std::vector<int>& flat_data, int& num_edges_out, int& num_edge_attrs_out) { // 读取边文件(C++) 实现
    if (filepath_utf8.empty()) { // 如果文件路径为空
        num_edges_out = 0; // 边数量为0
        num_edge_attrs_out = 0; // 属性数量为0
        flat_data.clear(); // 清空数据
        return true; // 认为成功
    }
    std::cout << "C++ received UTF-8 filepath for edge file: [" << filepath_utf8 << "]" << std::endl; // 打印接收到的UTF-8路径
    std::ifstream file; // 声明文件输入流对象
#ifdef _WIN32 // 如果是Windows平台
    std::wstring w_filepath = utf8_to_wstring_windows(filepath_utf8); // 转换路径
    if (!w_filepath.empty()) { // 如果转换成功
        file.open(w_filepath.c_str()); // 使用宽字符路径打开
    } else if (!filepath_utf8.empty()){ // 如果转换失败但原路径非空
        std::cerr << "Warning: Failed to convert edge filepath to wstring, trying with original UTF-8 string." << std::endl; // 打印警告
        file.open(filepath_utf8.c_str()); // 尝试用原始UTF-8路径打开
    } else { // 如果原始UTF-8路径也为空
        file.open(filepath_utf8.c_str()); // 尝试用原始UTF-8路径打开
    }
#else
    file.open(filepath_utf8.c_str()); // 直接打开
#endif
    if (!file.is_open()) { // 如果文件打开失败
        std::cerr << "Warning: Could not open edge file " << filepath_utf8 << ". Proceeding without edge data." << std::endl; // 打印警告
        num_edges_out = 0; // 边数量为0
        num_edge_attrs_out = 0; // 属性数量为0
        flat_data.clear(); // 清空数据
        return true; // 仍然返回true，表示可以继续，但没有边数据
    }
    // ... (文件头部和数据行的读取逻辑保持不变，记得在出错或结束时 file.close())
    std::string line; // 用于存储读取的每一行
    if (!std::getline(file, line)) { std::cerr << "Error: Edge file " << filepath_utf8 << " is empty or unreadable header." << std::endl; file.close(); return false; } // 读取头部失败
    std::istringstream header_ss(line); // 创建字符串流处理头部
    int has_marker_file; // 声明文件中是否有标记
    header_ss >> num_edges_out >> has_marker_file; // 从字符串流中提取头部信息
    if (header_ss.fail()) { std::cerr << "Error: Invalid edge file header format in " << filepath_utf8 << std::endl; file.close(); return false; } // 提取失败

        num_edge_attrs_out = 4; // n1, n2, type_marker, original_poly_id

        flat_data.clear();
        flat_data.reserve(num_edges_out * num_edge_attrs_out);

        for (int i = 0; i < num_edges_out; ++i) {
            if (!std::getline(file, line)) { /* ... error ... */ }
            std::istringstream line_ss(line);
            int edge_idx_in_file, node1_id, node2_id, type_marker_val, original_poly_seg_id_val;

            // 解析: edge_idx_in_file node1_id node2_id type_marker_val original_poly_seg_id_val
            if (!(line_ss >> edge_idx_in_file >> node1_id >> node2_id >> type_marker_val >> original_poly_seg_id_val)) { // <--- 修改此行: 读取所有5个值
                std::cerr << "错误: 边文件 " << filepath_utf8 << " 中第 " << i << " 行数据格式无效。期望5个整数。行: " << line << std::endl; // <--- 修改错误信息
                file.close(); return false;
            }

            // flat_data 存储 n1, n2, marker, original_id
            flat_data.push_back(node1_id);
            flat_data.push_back(node2_id);
            flat_data.push_back(type_marker_val);
            flat_data.push_back(original_poly_seg_id_val);
        }
        file.close();
        return true;
} // 结束函数


void Mesh_cpp::load_mesh_from_files(const std::string& node_filepath,
                                  const std::string& cell_filepath,
                                  const std::string& edge_filepath,
                                  const std::vector<double>& cell_manning_values) { // 4参数
    std::cout << "C++ Mesh: Starting to load mesh from files..." << std::endl;
    // ... (加载节点部分不变) ...
    std::vector<double> flat_nodes_vec;
    int num_nodes_read, node_attrs_read;
    if (!this->read_node_file_cpp(node_filepath, flat_nodes_vec, num_nodes_read, node_attrs_read)) {
        throw std::runtime_error("Failed to read node file: " + node_filepath);
    }
    this->load_nodes_from_numpy(flat_nodes_vec, num_nodes_read, node_attrs_read);
    std::cout << "  Loaded " << num_nodes_read << " nodes with " << node_attrs_read << " attributes each." << std::endl;

    // 为单元数据和区域属性创建vector
    std::vector<int> flat_cells_connectivity_vec;
    std::vector<double> cell_attributes_vec; // <<--- 用于存储从read_cell_file_cpp获取的属性
    int num_cells_read, nodes_per_cell_read;

    // 调用修改后的 read_cell_file_cpp
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
         // 如果数量不匹配，load_cells_from_numpy 内部会处理（例如使用默认值）
    }

    // 调用 load_cells_from_numpy，传递5个参数
    this->load_cells_from_numpy(flat_cells_connectivity_vec, num_cells_read, nodes_per_cell_read, cell_manning_values, cell_attributes_vec);
    std::cout << "  Loaded " << num_cells_read << " cells with " << nodes_per_cell_read << " nodes each." << std::endl;

    // ... (加载边和预计算部分不变) ...
    std::vector<int> flat_edges_vec;
    int num_edges_read = 0, edge_attrs_read = 0;
    if (!edge_filepath.empty()) {
        if (!this->read_edge_file_cpp(edge_filepath, flat_edges_vec, num_edges_read, edge_attrs_read)) {
            if(num_edges_read != 0) { // Only throw if it tried to read but failed format-wise
                throw std::runtime_error("Failed to read edge file: " + edge_filepath);
            }
             // If num_edges_read is 0, it means file not found or empty, which is handled as a warning by read_edge_file_cpp
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
        int base_idx = i * (1 + nodes_per_cell);
        int id = flat_cell_data[base_idx + 0];
        Cell_cpp current_cell(id); // Cell_cpp构造函数会将region_attribute初始化为0.0
        current_cell.node_ids.reserve(nodes_per_cell);
        for (int j = 0; j < nodes_per_cell; ++j) {
            current_cell.node_ids.push_back(flat_cell_data[base_idx + 1 + j]);
        }

        if (manning_provided) {
            if (static_cast<size_t>(i) < cell_manning_values.size()) {
                current_cell.manning_n = cell_manning_values[i];
            } else {
                // 如果曼宁值不足，使用Cell_cpp中定义的默认值
                std::cerr << "Warning: Manning value for cell " << id << " not provided, using default " << current_cell.manning_n << std::endl;
            }
        }

        if (region_attr_provided) {
            if (static_cast<size_t>(i) < cell_region_attributes.size()) {
                current_cell.region_attribute = cell_region_attributes[i];
            } else {
                 // 如果区域属性值不足，使用Cell_cpp中定义的默认值
                std::cerr << "Warning: Region attribute for cell " << id << " not provided, using default " << current_cell.region_attribute << std::endl;
            }
        }
        cells.emplace_back(current_cell);
    }
}

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

void Mesh_cpp::assign_boundary_markers_to_halfedges_cpp(const std::vector<int>& flat_edge_data, int num_edges_in, int num_edge_attrs) {
    // num_edge_attrs 现在应该是 4 (n1, n2, type_marker, original_id)
    if (num_edges_in == 0 || num_edge_attrs != 4) { // <--- 修改条件: 确保 num_edge_attrs 是 4
        // 如果没有边数据或属性数不符，则所有边界半边标记默认为0，原始ID默认为-1
        for (auto& he : half_edges) {
            if (he.twin_half_edge_id == -1) { // 是边界半边
                he.boundary_marker = 0; // 按照您的定义，0是内部，但对于未指定的边界，可能需要一个特定的“未指定边界”标记，或者默认为固壁(1)？
                                        // 这里我们先设为0，意味着它不是一个由.poly定义的外部物理边界。
                he.original_poly_segment_id = -1;
            }
        }
        if (num_edges_in > 0 && num_edge_attrs != 4) {
             std::cerr << "警告: assign_boundary_markers_to_halfedges_cpp 期望每条边有 " << 4 << " 个属性 (n1,n2,marker,orig_id)，但得到 " << num_edge_attrs << "。边界标记和原始ID可能不正确。" << std::endl;
        }
        return;
    }

    // 构建一个从 (排序节点对) 到 (type_marker, original_poly_id) 的映射，以便快速查找
    std::map<std::pair<int, int>, std::pair<int, int>> edge_info_map_from_file;
    for (int i = 0; i < num_edges_in; ++i) {
        int base_idx = i * num_edge_attrs; // num_edge_attrs 现在是 4
        int n1_id = flat_edge_data[base_idx + 0];
        int n2_id = flat_edge_data[base_idx + 1];
        int type_marker = flat_edge_data[base_idx + 2];
        int original_id = flat_edge_data[base_idx + 3];
        std::pair<int, int> key = (n1_id < n2_id) ? std::make_pair(n1_id, n2_id) : std::make_pair(n2_id, n1_id);
        edge_info_map_from_file[key] = std::make_pair(type_marker, original_id);
    }

    int assigned_count = 0;
    // ************************** 新增/修改的调试打印 **************************
    std::cout << "DEBUG_ASSIGN_BC_MARKERS: Entering assign_boundary_markers_to_halfedges_cpp." << std::endl;
    std::cout << "  Total half_edges to process: " << half_edges.size() << std::endl;
    std::cout << "  Edge info map from .edge file has " << edge_info_map_from_file.size() << " entries." << std::endl;
    // ***********************************************************************

    for (auto& he : half_edges) {
        const Node_cpp* node1_ptr = nullptr;
        const Node_cpp* node2_ptr = nullptr;
        const HalfEdge_cpp* next_he_ptr = nullptr;

        if (he.origin_node_id != -1) {
            node1_ptr = get_node_by_id(he.origin_node_id);
        }
        if (he.next_half_edge_id != -1) {
            next_he_ptr = get_half_edge_by_id(he.next_half_edge_id);
            if (next_he_ptr && next_he_ptr->origin_node_id != -1) {
                node2_ptr = get_node_by_id(next_he_ptr->origin_node_id);
            }
        }



        if (he.twin_half_edge_id == -1) { // 只处理边界半边
            std::cout << " (Boundary HE). "; // *标记为边界半边*
            if (!node1_ptr || !next_he_ptr || !node2_ptr) {
                std::cout << "Skipping due to invalid nodes/next_he." << std::endl;
                continue;
            }

            int n1_he = he.origin_node_id;
            int n2_he = next_he_ptr->origin_node_id;
            std::pair<int, int> key_he = (n1_he < n2_he) ? std::make_pair(n1_he, n2_he) : std::make_pair(n2_he, n1_he);

            auto it = edge_info_map_from_file.find(key_he);
            if (it != edge_info_map_from_file.end()) {
                he.boundary_marker = it->second.first;
                he.original_poly_segment_id = it->second.second;
                assigned_count++;

            } else {
                he.boundary_marker = 1;
                he.original_poly_segment_id = -1;

            }
        } else { // 内部半边
            he.boundary_marker = 0;
            he.original_poly_segment_id = -1;

        }
    }
    std::cout << "DEBUG_ASSIGN_BC_MARKERS: Assigned markers to " << assigned_count << " boundary half-edges." << std::endl;
    // std::cout << "  C++: 已为 " << assigned_count << " 个边界半边分配了标记和原始Segment ID。" << std::endl;
}

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
        // if (n_origin && n_end) { // 确保节点有效
        //     // 假设你知道右边界上一个典型的半边，其起点和终点ID
        //     // 例如，如果右边界的节点是1和2 (.poly中的定义)
        //     // 你需要找到代表这条物理边的半边 (可能有多条，取决于连接的单元)
        //     // 一个简单的方法是检查这条边的端点坐标是否在 X=10 附近
        //     if (std::abs(n_origin->x - 10.0) < 1e-3 && std::abs(n_end->x - 10.0) < 1e-3) {
        //         std::cout << "Debug HalfEdge (Right Wall Candidate): id=" << he.id
        //                   << ", cell_id=" << he.cell_id
        //                   << ", origin_node=" << he.origin_node_id << " (" << n_origin->x << "," << n_origin->y << ")"
        //                   << ", end_node=" << he_next->origin_node_id << " (" << n_end->x << "," << n_end->y << ")" // 注意这里用的是 he_next 的起点作为 he 的终点
        //                   << ", dx=" << dx << ", dy=" << dy
        //                   << ", length=" << he.length
        //                   << ", normal=(" << he.normal[0] << "," << he.normal[1] << ")"
        //                   << std::endl;
        //     }
        // }
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