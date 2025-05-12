// src_cpp/src/Reconstruction_cpp.cpp
#include "Reconstruction_cpp.h" // 包含对应的头文件
#include <stdexcept> // 包含标准异常
#include <iostream> // 包含输入输出流 (用于调试)

namespace HydroCore { // 定义HydroCore命名空间

Reconstruction_cpp::Reconstruction_cpp(ReconstructionScheme_cpp scheme, // 构造函数实现
                                       const Mesh_cpp* mesh_ptr,
                                       double gravity,
                                       double min_depth_param)
    : scheme_internal(scheme), mesh(mesh_ptr), g_internal(gravity), // 初始化列表
      min_depth_internal(min_depth_param), epsilon(1e-12) {
    if (!mesh) { // 检查网格指针是否有效
        throw std::invalid_argument("Mesh pointer cannot be null for Reconstruction_cpp."); // 抛出无效参数异常
    }
    // std::cout << "C++ Reconstruction_cpp initialized with scheme: " << static_cast<int>(scheme_internal) << std::endl; // 打印初始化信息
} // 结束构造函数

PrimitiveVars_cpp Reconstruction_cpp::conserved_to_primitive(const std::array<double, 3>& U_cell) const { // 守恒量转原始量实现
    double h = U_cell[0]; // 获取水深
    if (h < min_depth_internal) { // 如果水深小于最小深度
        return {h, 0.0, 0.0}; // 返回水深和零速度
    }
    double h_for_division = std::max(h, epsilon); // 用于除法的安全水深
    return {h, U_cell[1] / h_for_division, U_cell[2] / h_for_division}; // 返回原始变量
} // 结束方法
const std::array<std::array<double, 2>, 3>& Reconstruction_cpp::get_gradient_for_cell(int cell_id) const { // 获取指定单元的梯度实现
    if (gradients_primitive.empty()) { // 如果梯度为空 (例如一阶或未调用prepare_for_step)
        // 返回一个静态的零梯度或者抛出异常
        static const std::array<std::array<double, 2>, 3> zero_gradient = {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}}; // 静态零梯度
        // 或者更严格：
        // throw std::runtime_error("Gradients are not available or not prepared.");
        // std::cerr << "Warning: Accessing gradients when they are not available or not prepared for cell " << cell_id << std::endl; // 打印警告
        return zero_gradient; // 返回零梯度
    }
    if (cell_id < 0 || static_cast<size_t>(cell_id) >= gradients_primitive.size()) { // 检查cell_id是否越界
        throw std::out_of_range("Cell ID is out of bounds for accessing gradients."); // 抛出越界异常
    }
    return gradients_primitive[cell_id]; // 返回对应单元的梯度
}
std::vector<PrimitiveVars_cpp> Reconstruction_cpp::get_all_primitive_states( // 获取所有原始状态实现
    const std::vector<std::array<double, 3>>& U_state_all) const {
    std::vector<PrimitiveVars_cpp> W_state_all(U_state_all.size()); // 初始化原始状态数组
    for (size_t i = 0; i < U_state_all.size(); ++i) { // 遍历所有单元
        W_state_all[i] = conserved_to_primitive(U_state_all[i]); // 转换并存储
    }
    return W_state_all; // 返回所有原始状态
} // 结束方法

void Reconstruction_cpp::prepare_for_step(const std::vector<std::array<double, 3>>& U_state_all) { // 准备步骤实现
    if (scheme_internal == ReconstructionScheme_cpp::FIRST_ORDER) { // 如果是一阶方案
        gradients_primitive.clear(); // 清空梯度 (或设为全零，如果后续逻辑依赖其大小)
        return; // 直接返回
    }
    if (mesh->cells.empty()) return; // 如果没有单元则返回

    // std::cout << "  C++ Reconstruction: Preparing for step (gradients & limiters)..." << std::endl; // 打印准备信息

    size_t num_cells = mesh->cells.size(); // 获取单元数量
    if (U_state_all.size() != num_cells) { // 检查状态数组大小是否与单元数量一致
         throw std::runtime_error("U_state_all size does not match number of cells in mesh."); // 抛出运行时错误
    }

    // 1. 计算所有单元的原始变量状态
    std::vector<PrimitiveVars_cpp> W_state_all = get_all_primitive_states(U_state_all); // 获取所有原始状态

    // 2. 计算无限制的梯度 (Green-Gauss)
    std::vector<std::array<std::array<double, 2>, 3>> unlimited_gradients =
        calculate_gradients_green_gauss(W_state_all); // 计算无限制梯度

    // 3. 计算限制器 phi
    std::vector<std::array<double, 3>> limiters_phi =
        calculate_barth_jespersen_limiters(W_state_all, unlimited_gradients); // 计算限制器

    // 4. 应用限制器得到最终梯度
    gradients_primitive.resize(num_cells); // 调整梯度数组大小并初始化
    for (size_t i = 0; i < num_cells; ++i) { // 遍历所有单元
        for (int var = 0; var < 3; ++var) { // 遍历h, u, v
            gradients_primitive[i][var][0] = limiters_phi[i][var] * unlimited_gradients[i][var][0]; // 应用限制器到x梯度
            gradients_primitive[i][var][1] = limiters_phi[i][var] * unlimited_gradients[i][var][1]; // 应用限制器到y梯度
        }
    }
    // std::cout << "  C++ Reconstruction: Gradients and limiters prepared." << std::endl; // 打印完成信息
} // 结束方法

std::vector<std::array<std::array<double, 2>, 3>> Reconstruction_cpp::calculate_gradients_green_gauss( // Green-Gauss梯度计算实现
    const std::vector<PrimitiveVars_cpp>& W_state_all) const {
    size_t num_cells = mesh->cells.size(); // 获取单元数量
    // 定义一个默认初始化的 value_type (所有 double 成员将为 0.0)
    std::array<std::array<double, 2>, 3> default_gradient_value = {}; // 这是一个 std::array<std::array<double,2>,3> 类型的对象，所有元素为0
    std::vector<std::array<std::array<double, 2>, 3>> gradients_all(num_cells, default_gradient_value); // 用默认值初始化num_cells个元素

    for (size_t i = 0; i < num_cells; ++i) { // 遍历所有单元
        const Cell_cpp& cell_i = mesh->cells[i]; // 获取当前单元
        const PrimitiveVars_cpp& W_i = W_state_all[i]; // 当前单元原始变量

        if (cell_i.area < epsilon) continue; // 跳过面积过小的单元

        std::array<std::array<double, 2>, 3> grad_W_i_sum = {{{0,0},{0,0},{0,0}}}; // 初始化梯度和 (h,u,v 对 x,y)

        for (int he_id : cell_i.half_edge_ids_list) { // 遍历单元的半边ID
            const HalfEdge_cpp* he = mesh->get_half_edge_by_id(he_id); // 获取半边对象
            if (!he) continue; // 跳过无效半边

            PrimitiveVars_cpp W_neighbor = W_i; // 默认使用内部值 (用于边界)
            if (he->twin_half_edge_id != -1) { // 如果是内部边
                const HalfEdge_cpp* twin_he = mesh->get_half_edge_by_id(he->twin_half_edge_id); // 获取孪生半边
                if (twin_he && twin_he->cell_id != -1) { // 如果孪生半边有效且属于某个单元
                    W_neighbor = W_state_all[twin_he->cell_id]; // 获取邻居单元原始变量
                }
            }
            // 界面值近似为两侧单元平均值
            PrimitiveVars_cpp W_face = { // 计算界面值
                0.5 * (W_i.h + W_neighbor.h), // 界面水深
                0.5 * (W_i.u + W_neighbor.u), // 界面u速度
                0.5 * (W_i.v + W_neighbor.v)  // 界面v速度
            }; // 结束界面值计算

            // 累加 Green-Gauss 贡献
            grad_W_i_sum[0][0] += W_face.h * he->normal[0] * he->length; // d(h)/dx
            grad_W_i_sum[0][1] += W_face.h * he->normal[1] * he->length; // d(h)/dy
            grad_W_i_sum[1][0] += W_face.u * he->normal[0] * he->length; // d(u)/dx
            grad_W_i_sum[1][1] += W_face.u * he->normal[1] * he->length; // d(u)/dy
            grad_W_i_sum[2][0] += W_face.v * he->normal[0] * he->length; // d(v)/dx
            grad_W_i_sum[2][1] += W_face.v * he->normal[1] * he->length; // d(v)/dy
        }
        for (int var = 0; var < 3; ++var) { // 遍历h,u,v
            gradients_all[i][var][0] = grad_W_i_sum[var][0] / cell_i.area; // 计算x梯度
            gradients_all[i][var][1] = grad_W_i_sum[var][1] / cell_i.area; // 计算y梯度
        }
    }
    return gradients_all; // 返回所有梯度
} // 结束方法

std::vector<std::array<double, 3>> Reconstruction_cpp::calculate_barth_jespersen_limiters( // Barth-Jespersen限制器计算实现
    const std::vector<PrimitiveVars_cpp>& W_state_all,
    const std::vector<std::array<std::array<double, 2>, 3>>& unlimited_gradients_all) const {
    size_t num_cells = mesh->cells.size(); // 获取单元数量
    std::vector<std::array<double, 3>> limiters_phi_all(num_cells, {1.0, 1.0, 1.0}); // 初始化限制器为1 (无限制)

    for (size_t i = 0; i < num_cells; ++i) { // 遍历所有单元
        const Cell_cpp& cell_i = mesh->cells[i]; // 获取当前单元
        const PrimitiveVars_cpp& W_i = W_state_all[i]; // 当前单元中心值

        // 初始化邻居单元中心值的最大最小值
        PrimitiveVars_cpp W_max_neighbors = W_i; // 初始化为当前单元值
        PrimitiveVars_cpp W_min_neighbors = W_i; // 初始化为当前单元值
        bool has_valid_neighbors = false; // 标记是否有有效邻居

        for (int he_id : cell_i.half_edge_ids_list) { // 遍历半边
            const HalfEdge_cpp* he = mesh->get_half_edge_by_id(he_id); // 获取半边对象
            if (he && he->twin_half_edge_id != -1) { // 如果是内部边
                const HalfEdge_cpp* twin_he = mesh->get_half_edge_by_id(he->twin_half_edge_id); // 获取孪生半边
                if (twin_he && twin_he->cell_id != -1) { // 如果孪生半边有效且属于某个单元
                    const PrimitiveVars_cpp& W_neighbor = W_state_all[twin_he->cell_id]; // 获取邻居值
                    W_max_neighbors.h = std::max(W_max_neighbors.h, W_neighbor.h); // 更新h最大值
                    W_min_neighbors.h = std::min(W_min_neighbors.h, W_neighbor.h); // 更新h最小值
                    W_max_neighbors.u = std::max(W_max_neighbors.u, W_neighbor.u); // 更新u最大值
                    W_min_neighbors.u = std::min(W_min_neighbors.u, W_neighbor.u); // 更新u最小值
                    W_max_neighbors.v = std::max(W_max_neighbors.v, W_neighbor.v); // 更新v最大值
                    W_min_neighbors.v = std::min(W_min_neighbors.v, W_neighbor.v); // 更新v最小值
                    has_valid_neighbors = true; // 标记有有效邻居
                }
            }
        }
        if (!has_valid_neighbors) continue; // 如果没有邻居，phi保持为1

        const std::array<std::array<double, 2>, 3>& grad_W_i = unlimited_gradients_all[i]; // 获取当前单元无限制梯度
        std::array<double, 3> min_phi_for_cell_vars = {1.0, 1.0, 1.0}; // 当前单元各变量的最小phi

        for (int node_id : cell_i.node_ids) { // 遍历单元的顶点ID
            const Node_cpp* node = mesh->get_node_by_id(node_id); // 获取节点对象
            if (!node) continue; // 跳过无效节点

            std::array<double, 2> vec_to_vertex = {node->x - cell_i.centroid[0], node->y - cell_i.centroid[1]}; // 形心到顶点向量

            // 计算顶点处的无限制重构值 W_vertex_unlimited = W_i + grad_W_i * vec_to_vertex
            PrimitiveVars_cpp W_vertex_unlimited = { // 初始化顶点重构值
                W_i.h + (grad_W_i[0][0] * vec_to_vertex[0] + grad_W_i[0][1] * vec_to_vertex[1]), // h
                W_i.u + (grad_W_i[1][0] * vec_to_vertex[0] + grad_W_i[1][1] * vec_to_vertex[1]), // u
                W_i.v + (grad_W_i[2][0] * vec_to_vertex[0] + grad_W_i[2][1] * vec_to_vertex[1])  // v
            }; // 结束初始化

            // 对每个变量 (h, u, v) 检查是否超限
            const PrimitiveVars_cpp* W_ptr_vertex = &W_vertex_unlimited; // 指向顶点重构值
            const PrimitiveVars_cpp* W_ptr_i = &W_i; // 指向单元中心值
            const PrimitiveVars_cpp* W_ptr_max_n = &W_max_neighbors; // 指向邻居最大值
            const PrimitiveVars_cpp* W_ptr_min_n = &W_min_neighbors; // 指向邻居最小值

            for (int var = 0; var < 3; ++var) { // 遍历h, u, v (用指针或switch简化访问)
                double val_vertex = (var == 0) ? W_ptr_vertex->h : ((var == 1) ? W_ptr_vertex->u : W_ptr_vertex->v); // 获取顶点变量值
                double val_i = (var == 0) ? W_ptr_i->h : ((var == 1) ? W_ptr_i->u : W_ptr_i->v); // 获取中心变量值
                double val_max_n = (var == 0) ? W_ptr_max_n->h : ((var == 1) ? W_ptr_max_n->u : W_ptr_max_n->v); // 获取邻居最大值
                double val_min_n = (var == 0) ? W_ptr_min_n->h : ((var == 1) ? W_ptr_min_n->u : W_ptr_min_n->v); // 获取邻居最小值

                double W_diff = val_vertex - val_i; // 计算重构值与中心值的差
                double phi_k = 1.0; // 初始化phi_k

                if (W_diff > epsilon) { // 如果重构值大于中心值
                    double max_allowed_diff = val_max_n - val_i; // 允许的最大差值
                    if (std::abs(W_diff) > epsilon) { // 避免除零
                        phi_k = std::max(0.0, max_allowed_diff / W_diff); // 计算限制因子
                    } else { // W_diff 接近0
                         phi_k = (max_allowed_diff >= -epsilon) ? 1.0 : 0.0; // 如果允许差值也接近0或正，则为1，否则为0
                    }
                } else if (W_diff < -epsilon) { // 如果重构值小于中心值
                    double min_allowed_diff = val_min_n - val_i; // 允许的最小差值 (负数)
                    if (std::abs(W_diff) > epsilon) { // 避免除零
                        phi_k = std::max(0.0, min_allowed_diff / W_diff); // 计算限制因子
                    } else { // W_diff 接近0
                        phi_k = (min_allowed_diff <= epsilon) ? 1.0 : 0.0; // 如果允许差值也接近0或负，则为1，否则为0
                    }
                }
                min_phi_for_cell_vars[var] = std::min(min_phi_for_cell_vars[var], phi_k); // 更新该变量的最小phi
            }
        }
        limiters_phi_all[i] = min_phi_for_cell_vars; // 存储当前单元各变量的限制器
    }
    return limiters_phi_all; // 返回所有限制器
} // 结束方法

std::pair<PrimitiveVars_cpp, PrimitiveVars_cpp> Reconstruction_cpp::get_reconstructed_interface_states( // 重构界面状态获取实现
    const std::vector<std::array<double, 3>>& U_state_all,
    int cell_L_id,
    int cell_R_id, // -1 if boundary
    const HalfEdge_cpp& half_edge_L_to_R, // 从L指向R或外部
    bool is_boundary) const {

    if (cell_L_id < 0 || static_cast<size_t>(cell_L_id) >= U_state_all.size()) { // 检查左单元ID是否有效
        throw std::out_of_range("cell_L_id is out of bounds."); // 抛出越界异常
    }
    const Cell_cpp* cell_L_ptr = mesh->get_cell_by_id(cell_L_id); // 获取左单元指针
    if (!cell_L_ptr) throw std::runtime_error("Could not find cell_L for reconstruction."); // 抛出运行时错误

    PrimitiveVars_cpp W_L_interface = conserved_to_primitive(U_state_all[cell_L_id]); // 获取左单元中心原始变量

    if (scheme_internal != ReconstructionScheme_cpp::FIRST_ORDER) { // 如果不是一阶方案
        if (gradients_primitive.empty() || static_cast<size_t>(cell_L_id) >= gradients_primitive.size()) { // 检查梯度是否已计算且有效
            throw std::runtime_error("Gradients not prepared or cell_L_id out of bounds for gradients."); // 抛出运行时错误
        }
        const auto& grad_W_L = gradients_primitive[cell_L_id]; // 获取左单元梯度
        std::array<double, 2> vec_L_to_face = { // 计算左单元形心到界面中点向量
            half_edge_L_to_R.mid_point[0] - cell_L_ptr->centroid[0], // x分量
            half_edge_L_to_R.mid_point[1] - cell_L_ptr->centroid[1]  // y分量
        }; // 结束计算

        W_L_interface.h += grad_W_L[0][0] * vec_L_to_face[0] + grad_W_L[0][1] * vec_L_to_face[1]; // 更新h
        W_L_interface.u += grad_W_L[1][0] * vec_L_to_face[0] + grad_W_L[1][1] * vec_L_to_face[1]; // 更新u
        W_L_interface.v += grad_W_L[2][0] * vec_L_to_face[0] + grad_W_L[2][1] * vec_L_to_face[1]; // 更新v

        W_L_interface.h = std::max(0.0, W_L_interface.h); // 保证水深非负
        if (W_L_interface.h < min_depth_internal) { // 如果水深过小
            W_L_interface.u = 0.0; // 速度设为0
            W_L_interface.v = 0.0; // 速度设为0
        }
    }

    PrimitiveVars_cpp W_R_interface = {0,0,0}; // 初始化右侧界面状态 (如果边界则无效)
    if (!is_boundary) { // 如果是内部边
        if (cell_R_id < 0 || static_cast<size_t>(cell_R_id) >= U_state_all.size()) { // 检查右单元ID是否有效
            throw std::out_of_range("cell_R_id is out of bounds for internal edge."); // 抛出越界异常
        }
        const Cell_cpp* cell_R_ptr = mesh->get_cell_by_id(cell_R_id); // 获取右单元指针
        if (!cell_R_ptr) throw std::runtime_error("Could not find cell_R for reconstruction."); // 抛出运行时错误

        W_R_interface = conserved_to_primitive(U_state_all[cell_R_id]); // 获取右单元中心原始变量

        if (scheme_internal != ReconstructionScheme_cpp::FIRST_ORDER) { // 如果不是一阶方案
            if (gradients_primitive.empty() || static_cast<size_t>(cell_R_id) >= gradients_primitive.size()) { // 检查梯度是否已计算且有效
                 throw std::runtime_error("Gradients not prepared or cell_R_id out of bounds for gradients."); // 抛出运行时错误
            }
            const auto& grad_W_R = gradients_primitive[cell_R_id]; // 获取右单元梯度
            std::array<double, 2> vec_R_to_face = { // 计算右单元形心到界面中点向量
                half_edge_L_to_R.mid_point[0] - cell_R_ptr->centroid[0], // x分量
                half_edge_L_to_R.mid_point[1] - cell_R_ptr->centroid[1]  // y分量
            }; // 结束计算

            W_R_interface.h += grad_W_R[0][0] * vec_R_to_face[0] + grad_W_R[0][1] * vec_R_to_face[1]; // 更新h
            W_R_interface.u += grad_W_R[1][0] * vec_R_to_face[0] + grad_W_R[1][1] * vec_R_to_face[1]; // 更新u
            W_R_interface.v += grad_W_R[2][0] * vec_R_to_face[0] + grad_W_R[2][1] * vec_R_to_face[1]; // 更新v

            W_R_interface.h = std::max(0.0, W_R_interface.h); // 保证水深非负
            if (W_R_interface.h < min_depth_internal) { // 如果水深过小
                W_R_interface.u = 0.0; // 速度设为0
                W_R_interface.v = 0.0; // 速度设为0
            }
        }
    }
    return {W_L_interface, W_R_interface}; // 返回左右界面状态
} // 结束方法

} // namespace HydroCore