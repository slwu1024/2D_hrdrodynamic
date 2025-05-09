# test_vfr_calculator.py
import numpy as np
import sys
import os
import pytest # 导入 pytest，虽然主要靠命名约定，但有时可能需要用到fixture等

# --- 路径设置 (保持不变) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

try:
    from src.model.WettingDrying import VFRCalculator
except ImportError as e:
    print(f"错误: 无法导入 VFRCalculator。请确保路径设置正确。")
    print(f"当前Python路径: {sys.path}")
    print(f"详细错误: {e}")
    pytest.fail("测试环境设置失败，无法导入 VFRCalculator") # 使用pytest标记失败

# --- MockNode 类 (保持不变) ---
class MockNode:
    def __init__(self, x, y, z_bed):
        self.x = x
        self.y = y
        self.z_bed = z_bed

# --- 辅助函数 polygon_area (保持不变) ---
def polygon_area(vertices_coords):
    n = len(vertices_coords)
    if n < 3: return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices_coords[i][0] * vertices_coords[j][1]
        area -= vertices_coords[j][0] * vertices_coords[i][1]
    return abs(area) / 2.0

# === 测试设置 (可以放在函数外部作为全局准备) ===
print("\n--- VFRCalculator 测试准备 ---")
# 定义测试单元
node_coords = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.8)]
node_z_beds_unsorted = [0.1, 0.3, 0.5]
nodes_unsorted = [MockNode(coords[0], coords[1], z) for coords, z in zip(node_coords, node_z_beds_unsorted)]
cell_total_area = polygon_area(node_coords)
print(f"测试单元信息:")
print(f"  顶点坐标: {node_coords}")
print(f"  顶点底高程 (未排序): {node_z_beds_unsorted}")
print(f"  单元总面积: {cell_total_area:.6f}")

# 准备排序后的节点和高程
nodes_with_z_sorted = sorted([(node, node.z_bed) for node in nodes_unsorted], key=lambda item: item[1])
cell_nodes_sorted = [item[0] for item in nodes_with_z_sorted]
b_sorted = [item[1] for item in nodes_with_z_sorted]
b1, b2, b3 = b_sorted[0], b_sorted[1], b_sorted[2]
print(f"排序后的底高程: b1={b1}, b2={b2}, b3={b3}")

# 实例化 VFRCalculator
vfr_calculator = VFRCalculator(min_depth=1e-6)

# 定义测试水位 eta 值
test_etas = [
    b1 - 0.1, b1, b1 + 0.05, (b1 + b2) / 2, b2 - vfr_calculator.epsilon * 0.1,
    b2, b2 + 0.05, (b2 + b3) / 2, b3 - vfr_calculator.epsilon * 0.1,
    b3, b3 + 0.1
]

# 定义容差
tolerance_eta = 1e-5
tolerance_h = 1e-6

# === 将测试逻辑封装到测试函数中 ===

# 使用 pytest.mark.parametrize 可以为每个 eta 值创建一个独立的测试用例
@pytest.mark.parametrize("eta_target", test_etas)
def test_vfr_consistency(eta_target): # 函数名以 test_ 开头
    """
    测试 VFRCalculator 的 get_h_from_eta 和 get_eta_from_h 是否一致。
    对于给定的 eta_target，计算 h_calc，然后反算 eta_recalc，比较 eta_recalc 和 eta_target。
    """
    print(f"\n--- 测试: 目标 eta = {eta_target:.6f} ---") # 打印当前测试信息

    # --- 步骤 a: 从目标 eta 计算 h_avg ---
    try:
        h_calculated = vfr_calculator.get_h_from_eta(
            eta_target, b_sorted, cell_total_area, cell_id_for_debug=f"test_eta_{eta_target:.2f}_a"
        )
        print(f"  步骤 a: get_h_from_eta({eta_target:.6f}) => h_calculated = {h_calculated:.6e}")
    except Exception as e:
        pytest.fail(f"步骤 a: get_h_from_eta 计算失败: {e}") # 使用 pytest.fail 标记失败并停止

    # --- 步骤 b: 从计算出的 h_avg 反算 eta ---
    if h_calculated < vfr_calculator.min_depth / 10: # 干单元情况
        eta_expected_recalc = b1
        print(f"  步骤 b: h_calculated < 阈值，预期反算 eta 结果为 b1 ({b1:.6f})")
        # 断言：检查干单元反算是否符合预期
        if eta_target <= b1 + vfr_calculator.epsilon:
             # 如果目标是干的，反算为b1是正常的
             assert True # 标记通过
             print("  结果: 通过 (目标eta为干，反算eta为b1符合预期)")
        else:
             # 如果目标是湿的，但h算出来是干，反算为b1，则不一致
             assert np.isclose(eta_expected_recalc, eta_target, atol=tolerance_eta), \
                    f"干单元情况，预期eta={eta_expected_recalc:.6f} 与 目标eta={eta_target:.6f} 不符"
             print("  结果: 通过 (h计算为干，反算eta为b1，与目标eta一致)")


    else: # 湿单元情况
        try:
            # 使用目标 eta 作为初始猜测以加速收敛
            eta_recalculated = vfr_calculator.get_eta_from_h(
                h_calculated, b_sorted, cell_nodes_sorted, cell_total_area,
                eta_previous_guess=eta_target, # 提供初始猜测
                cell_id_for_debug=f"test_eta_{eta_target:.2f}_b"
            )
            print(f"  步骤 b: get_eta_from_h({h_calculated:.6e}) => eta_recalculated = {eta_recalculated:.6f}")
        except Exception as e:
            pytest.fail(f"步骤 b: get_eta_from_h 计算失败: {e}")

        # --- 步骤 c: 断言比较反算的 eta 和 目标 eta ---
        assert np.isclose(eta_recalculated, eta_target, atol=tolerance_eta), \
               f"eta_recalculated={eta_recalculated:.6f} 与 目标eta={eta_target:.6f} 不符, 差值={abs(eta_recalculated - eta_target):.3e}"
        print(f"  结果: 通过 (eta_recalculated 与 目标 eta 接近)")

        # --- (可选) 步骤 d: 断言再次验证 h ---
        try:
            h_recalculated_again = vfr_calculator.get_h_from_eta(
                eta_recalculated, b_sorted, cell_total_area, cell_id_for_debug=f"test_eta_{eta_target:.2f}_d"
            )
            print(f"  (可选) 步骤 d: get_h_from_eta({eta_recalculated:.6f}) => h_recalculated_again = {h_recalculated_again:.6e}")
            assert np.isclose(h_recalculated_again, h_calculated, rtol=1e-3, atol=tolerance_h), \
                   f"(可选) h_recalculated_again={h_recalculated_again:.6e} 与 h_calculated={h_calculated:.6e} 不符"
            print(f"  (可选) 步骤 d: 通过 (h_recalculated_again 与 h_calculated 接近)")
        except Exception as e:
             pytest.fail(f"(可选) 步骤 d: get_h_from_eta 计算失败: {e}")

# === 你可以添加更多的测试函数 ===
# def test_specific_edge_case(...):
#     # ...