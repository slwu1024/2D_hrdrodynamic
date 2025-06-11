// src_cpp/src/Reconstruction_cpp.cpp
#include "Reconstruction_cpp.h" // 包含对应的头文件
#include "Profiler.h"
#include <stdexcept> // 包含标准异常
#include <iostream> // 包含输入输出流 (用于调试)
#include <omp.h>

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
const std::array<std::array<double, 2>, 3>& Reconstruction_cpp::get_gradient_for_cell(int cell_id) const {
    if (gradients_primitive.empty()) {
        static const std::array<std::array<double, 2>, 3> zero_gradient = { {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}} };
        return zero_gradient;
    }
    if (cell_id < 0 || static_cast<size_t>(cell_id) >= gradients_primitive.size()) {
        throw std::out_of_range("Cell ID is out of bounds for accessing gradients.");
    }
    return gradients_primitive[cell_id];
}
std::vector<PrimitiveVars_cpp> Reconstruction_cpp::get_all_primitive_states(
    const std::vector<std::array<double, 3>>& U_state_all) const {
    PROFILE_FUNCTION();
    std::vector<PrimitiveVars_cpp> W_state_all(U_state_all.size());
    // --- 新增：并行化这个转换循环 ---
#pragma omp parallel for schedule(static)
    for (int i = 0; i < U_state_all.size(); ++i) {
        W_state_all[i] = conserved_to_primitive(U_state_all[i]);
    }
    return W_state_all;
}

void Reconstruction_cpp::prepare_for_step(const std::vector<std::array<double, 3>>& U_state_all) {
    PROFILE_FUNCTION();
    if (scheme_internal == ReconstructionScheme_cpp::FIRST_ORDER) {
        gradients_primitive.clear();
        return;
    }
    if (mesh->cells.empty()) return;

    size_t num_cells = mesh->cells.size();
    if (U_state_all.size() != num_cells) {
        throw std::runtime_error("U_state_all size does not match number of cells in mesh.");
    }

    std::vector<PrimitiveVars_cpp> W_state_all = get_all_primitive_states(U_state_all);

    std::vector<std::array<std::array<double, 2>, 3>> unlimited_gradients =
        calculate_gradients_green_gauss(W_state_all);

    std::vector<std::array<double, 3>> limiters_phi =
        calculate_barth_jespersen_limiters(W_state_all, unlimited_gradients);

    gradients_primitive.resize(num_cells);

    // --- 新增：并行化最终梯度计算的循环 ---
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_cells; ++i) {
        for (int var = 0; var < 3; ++var) {
            gradients_primitive[i][var][0] = limiters_phi[i][var] * unlimited_gradients[i][var][0];
            gradients_primitive[i][var][1] = limiters_phi[i][var] * unlimited_gradients[i][var][1];
        }
    }
}

std::vector<std::array<std::array<double, 2>, 3>> Reconstruction_cpp::calculate_gradients_green_gauss(
    const std::vector<PrimitiveVars_cpp>& W_state_all) const {
    PROFILE_FUNCTION();
    size_t num_cells = mesh->cells.size();
    std::array<std::array<double, 2>, 3> default_gradient_value = {};
    std::vector<std::array<std::array<double, 2>, 3>> gradients_all(num_cells, default_gradient_value);

    // --- 新增：并行化Green-Gauss主循环 ---
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_cells; ++i) {
        const Cell_cpp& cell_i = mesh->cells[i];
        const PrimitiveVars_cpp& W_i = W_state_all[i];

        if (cell_i.area < epsilon) continue; // 在并行循环中使用 continue 是安全的

        std::array<std::array<double, 2>, 3> grad_W_i_sum = { {{0,0},{0,0},{0,0}} };

        for (int he_id : cell_i.half_edge_ids_list) {
            const HalfEdge_cpp* he = mesh->get_half_edge_by_id(he_id);
            if (!he) continue;

            PrimitiveVars_cpp W_neighbor = W_i;
            if (he->twin_half_edge_id != -1) {
                const HalfEdge_cpp* twin_he = mesh->get_half_edge_by_id(he->twin_half_edge_id);
                if (twin_he && twin_he->cell_id != -1) {
                    W_neighbor = W_state_all[twin_he->cell_id];
                }
            }
            PrimitiveVars_cpp W_face = {
                0.5 * (W_i.h + W_neighbor.h),
                0.5 * (W_i.u + W_neighbor.u),
                0.5 * (W_i.v + W_neighbor.v)
            };

            grad_W_i_sum[0][0] += W_face.h * he->normal[0] * he->length;
            grad_W_i_sum[0][1] += W_face.h * he->normal[1] * he->length;
            grad_W_i_sum[1][0] += W_face.u * he->normal[0] * he->length;
            grad_W_i_sum[1][1] += W_face.u * he->normal[1] * he->length;
            grad_W_i_sum[2][0] += W_face.v * he->normal[0] * he->length;
            grad_W_i_sum[2][1] += W_face.v * he->normal[1] * he->length;
        }
        for (int var = 0; var < 3; ++var) {
            gradients_all[i][var][0] = grad_W_i_sum[var][0] / cell_i.area;
            gradients_all[i][var][1] = grad_W_i_sum[var][1] / cell_i.area;
        }
    }
    return gradients_all;
}

std::vector<std::array<double, 3>> Reconstruction_cpp::calculate_barth_jespersen_limiters(
    const std::vector<PrimitiveVars_cpp>& W_state_all,
    const std::vector<std::array<std::array<double, 2>, 3>>& unlimited_gradients_all) const {
    PROFILE_FUNCTION();
    size_t num_cells = mesh->cells.size();
    std::vector<std::array<double, 3>> limiters_phi_all(num_cells, { 1.0, 1.0, 1.0 });

    // --- 新增：并行化限制器计算主循环 ---
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_cells; ++i) {
        const Cell_cpp& cell_i = mesh->cells[i];
        const PrimitiveVars_cpp& W_i = W_state_all[i];

        PrimitiveVars_cpp W_max_neighbors = W_i;
        PrimitiveVars_cpp W_min_neighbors = W_i;
        bool has_valid_neighbors = false;

        for (int he_id : cell_i.half_edge_ids_list) {
            const HalfEdge_cpp* he = mesh->get_half_edge_by_id(he_id);
            if (he && he->twin_half_edge_id != -1) {
                const HalfEdge_cpp* twin_he = mesh->get_half_edge_by_id(he->twin_half_edge_id);
                if (twin_he && twin_he->cell_id != -1) {
                    const PrimitiveVars_cpp& W_neighbor = W_state_all[twin_he->cell_id];
                    W_max_neighbors.h = std::max(W_max_neighbors.h, W_neighbor.h);
                    W_min_neighbors.h = std::min(W_min_neighbors.h, W_neighbor.h);
                    W_max_neighbors.u = std::max(W_max_neighbors.u, W_neighbor.u);
                    W_min_neighbors.u = std::min(W_min_neighbors.u, W_neighbor.u);
                    W_max_neighbors.v = std::max(W_max_neighbors.v, W_neighbor.v);
                    W_min_neighbors.v = std::min(W_min_neighbors.v, W_neighbor.v);
                    has_valid_neighbors = true;
                }
            }
        }
        if (!has_valid_neighbors) continue;

        const std::array<std::array<double, 2>, 3>& grad_W_i = unlimited_gradients_all[i];
        std::array<double, 3> min_phi_for_cell_vars = { 1.0, 1.0, 1.0 };

        for (int node_id : cell_i.node_ids) {
            const Node_cpp* node = mesh->get_node_by_id(node_id);
            if (!node) continue;

            std::array<double, 2> vec_to_vertex = { node->x - cell_i.centroid[0], node->y - cell_i.centroid[1] };

            PrimitiveVars_cpp W_vertex_unlimited = {
                W_i.h + (grad_W_i[0][0] * vec_to_vertex[0] + grad_W_i[0][1] * vec_to_vertex[1]),
                W_i.u + (grad_W_i[1][0] * vec_to_vertex[0] + grad_W_i[1][1] * vec_to_vertex[1]),
                W_i.v + (grad_W_i[2][0] * vec_to_vertex[0] + grad_W_i[2][1] * vec_to_vertex[1])
            };

            const PrimitiveVars_cpp* W_ptr_vertex = &W_vertex_unlimited;
            const PrimitiveVars_cpp* W_ptr_i = &W_i;
            const PrimitiveVars_cpp* W_ptr_max_n = &W_max_neighbors;
            const PrimitiveVars_cpp* W_ptr_min_n = &W_min_neighbors;

            for (int var = 0; var < 3; ++var) {
                double val_vertex = (var == 0) ? W_ptr_vertex->h : ((var == 1) ? W_ptr_vertex->u : W_ptr_vertex->v);
                double val_i = (var == 0) ? W_ptr_i->h : ((var == 1) ? W_ptr_i->u : W_ptr_i->v);
                double val_max_n = (var == 0) ? W_ptr_max_n->h : ((var == 1) ? W_ptr_max_n->u : W_ptr_max_n->v);
                double val_min_n = (var == 0) ? W_ptr_min_n->h : ((var == 1) ? W_ptr_min_n->u : W_ptr_min_n->v);

                double W_diff = val_vertex - val_i;
                double phi_k = 1.0;

                if (W_diff > epsilon) {
                    double max_allowed_diff = val_max_n - val_i;
                    if (std::abs(W_diff) > epsilon) {
                        phi_k = std::max(0.0, max_allowed_diff / W_diff);
                    }
                    else {
                        phi_k = (max_allowed_diff >= -epsilon) ? 1.0 : 0.0;
                    }
                }
                else if (W_diff < -epsilon) {
                    double min_allowed_diff = val_min_n - val_i;
                    if (std::abs(W_diff) > epsilon) {
                        phi_k = std::max(0.0, min_allowed_diff / W_diff);
                    }
                    else {
                        phi_k = (min_allowed_diff <= epsilon) ? 1.0 : 0.0;
                    }
                }
                min_phi_for_cell_vars[var] = std::min(min_phi_for_cell_vars[var], phi_k);
            }
        }
        limiters_phi_all[i] = min_phi_for_cell_vars;
    }
    return limiters_phi_all;
}

std::pair<PrimitiveVars_cpp, PrimitiveVars_cpp> Reconstruction_cpp::get_reconstructed_interface_states(
    const std::vector<std::array<double, 3>>& U_state_all,
    int cell_L_id,
    int cell_R_id, // -1 if boundary
    const HalfEdge_cpp& half_edge_L_to_R,
    bool is_boundary) const {

    // ... (cell_L_id, cell_L_ptr 检查不变) ...
    const Cell_cpp* cell_L_ptr = mesh->get_cell_by_id(cell_L_id); // 获取左单元指针 // 新增：获取左单元指针
    if (!cell_L_ptr) throw std::runtime_error("Could not find cell_L for reconstruction."); // 如果找不到左单元则抛出运行时错误 // 新增：检查左单元指针

    PrimitiveVars_cpp W_L_center = conserved_to_primitive(U_state_all[cell_L_id]);
    PrimitiveVars_cpp W_L_interface = W_L_center;

    // 修改：使用一个更明确的“几乎干”阈值，例如 min_depth_internal 的 2 倍
    // 如果单元水深小于这个阈值，我们就认为它在界面上不应该有流速，水深也应该是其中心值（或0）
    const double almost_dry_h_threshold = min_depth_internal * 2.0; // 定义一个“几乎干”的水深阈值

    if (W_L_center.h < almost_dry_h_threshold) { // 如果左单元中心水深小于“几乎干”阈值
        W_L_interface.h = W_L_center.h; // 界面水深等于中心水深 (可能是0或极小值)
        W_L_interface.u = 0.0;
        W_L_interface.v = 0.0;
    } else if (scheme_internal != ReconstructionScheme_cpp::FIRST_ORDER) { // 如果不是一阶且水深足够
        // ... (高阶重构逻辑不变) ...
         if (gradients_primitive.empty() || static_cast<size_t>(cell_L_id) >= gradients_primitive.size()) { // 检查梯度是否已准备或ID是否越界
             throw std::runtime_error("Gradients not prepared or cell_L_id out of bounds for gradients."); // 抛出运行时错误
         }
         const auto& grad_W_L = gradients_primitive[cell_L_id]; // 获取左单元梯度
         std::array<double, 2> vec_L_to_face = { // 计算左单元形心到界面中点的向量
             half_edge_L_to_R.mid_point[0] - cell_L_ptr->centroid[0], // x分量
             half_edge_L_to_R.mid_point[1] - cell_L_ptr->centroid[1]  // y分量
         };

         W_L_interface.h += grad_W_L[0][0] * vec_L_to_face[0] + grad_W_L[0][1] * vec_L_to_face[1]; // 重构水深
         W_L_interface.u += grad_W_L[1][0] * vec_L_to_face[0] + grad_W_L[1][1] * vec_L_to_face[1]; // 重构u速度
         W_L_interface.v += grad_W_L[2][0] * vec_L_to_face[0] + grad_W_L[2][1] * vec_L_to_face[1]; // 重构v速度

        // 重构后再次检查
        if (W_L_interface.h < almost_dry_h_threshold) { // 如果重构后的界面水深小于“几乎干”阈值
            W_L_interface.h = std::max(0.0, W_L_interface.h); // 确保水深非负
            W_L_interface.u = 0.0; // 清零u速度
            W_L_interface.v = 0.0; // 清零v速度
        }
    }
    // 最终确保 h>=0，并且如果 h < min_depth，则 u,v=0
    W_L_interface.h = std::max(0.0, W_L_interface.h); // 再次确保水深非负
    if (W_L_interface.h < min_depth_internal) { // 如果最终界面水深小于最小水深
        W_L_interface.u = 0.0; // 强制u速度为0
        W_L_interface.v = 0.0; // 强制v速度为0
    }

    PrimitiveVars_cpp W_R_interface = {0,0,0};
    if (!is_boundary) {
        // ... (cell_R_id, cell_R_ptr 检查不变) ...
         if (cell_R_id < 0 || static_cast<size_t>(cell_R_id) >= U_state_all.size()) { // 检查右单元ID是否越界
             throw std::out_of_range("cell_R_id is out of bounds for internal edge."); // 抛出越界异常
         }
         const Cell_cpp* cell_R_ptr = mesh->get_cell_by_id(cell_R_id); // 获取右单元指针
         if (!cell_R_ptr) throw std::runtime_error("Could not find cell_R for reconstruction."); // 如果找不到右单元则抛出运行时错误

        PrimitiveVars_cpp W_R_center = conserved_to_primitive(U_state_all[cell_R_id]);
        W_R_interface = W_R_center;

        if (W_R_center.h < almost_dry_h_threshold) { // 如果右单元中心水深小于“几乎干”阈值
            W_R_interface.h = W_R_center.h; // 界面水深等于中心水深
            W_R_interface.u = 0.0; // 强制界面u速度为0
            W_R_interface.v = 0.0; // 强制界面v速度为0
        } else if (scheme_internal != ReconstructionScheme_cpp::FIRST_ORDER) {
            // ... (高阶重构逻辑不变) ...
             if (gradients_primitive.empty() || static_cast<size_t>(cell_R_id) >= gradients_primitive.size()) { // 检查梯度是否已准备或ID是否越界
                  throw std::runtime_error("Gradients not prepared or cell_R_id out of bounds for gradients."); // 抛出运行时错误
             }
             const auto& grad_W_R = gradients_primitive[cell_R_id]; // 获取右单元梯度
             std::array<double, 2> vec_R_to_face = { // 计算右单元形心到界面中点的向量
                 half_edge_L_to_R.mid_point[0] - cell_R_ptr->centroid[0], // x分量
                 half_edge_L_to_R.mid_point[1] - cell_R_ptr->centroid[1]  // y分量
             };

             W_R_interface.h += grad_W_R[0][0] * vec_R_to_face[0] + grad_W_R[0][1] * vec_R_to_face[1]; // 重构水深
             W_R_interface.u += grad_W_R[1][0] * vec_R_to_face[0] + grad_W_R[1][1] * vec_R_to_face[1]; // 重构u速度
             W_R_interface.v += grad_W_R[2][0] * vec_R_to_face[0] + grad_W_R[2][1] * vec_R_to_face[1]; // 重构v速度

            if (W_R_interface.h < almost_dry_h_threshold) { // 如果重构后的右界面水深小于“几乎干”阈值
                W_R_interface.h = std::max(0.0, W_R_interface.h); // 确保水深非负
                W_R_interface.u = 0.0; // 清零u速度
                W_R_interface.v = 0.0; // 清零v速度
            }
        }
        W_R_interface.h = std::max(0.0, W_R_interface.h); // 再次确保水深非负
        if (W_R_interface.h < min_depth_internal) { // 如果最终右界面水深小于最小水深
            W_R_interface.u = 0.0; // 强制u速度为0
            W_R_interface.v = 0.0; // 强制v速度为0
        }
    }
    return {W_L_interface, W_R_interface};
}

} // namespace HydroCore