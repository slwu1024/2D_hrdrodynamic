import meshio
import numpy as np
import os

# 1. 定义点和单元
points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
], dtype=float)

cells_triangle_connectivity = np.array([
    [0, 1, 2],
    [1, 3, 2],
], dtype=int)

# cells 定义：只有一个单元块 "triangle"
cells_info = [("triangle", cells_triangle_connectivity)]
# cells_info = {"triangle": cells_triangle_connectivity} # 字典形式在 meshio 内部也会转为列表

# 2. 定义节点数据 (可选)
point_data_content = {
    "node_values": np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
}

# 3. 定义符合 meshio 期望的 cell_data
# 我们有一个名为 "cell_pressure" 的数据字段。
# 我们的 cells_info 中只有一个单元块 (即 "triangle" 块)。
# 所以，"cell_pressure" 对应的值应该是一个包含一个元素的列表。
# 这个列表中的元素是包含两个三角形压力的 NumPy 数组。
correct_cell_data_content = {
    "cell_pressure": [ # 这是一个列表，对应 cells_info 中的单元块
        np.array([100.0, 200.0], dtype=float) # 第一个 (也是唯一一个) 单元块 ("triangle") 的数据
    ]
    # 如果我们还有其他字段，例如 "cell_temperature":
    # "cell_temperature": [ np.array([25.0, 28.0], dtype=float) ]
}


print("--- DEBUG: Inputs to meshio.Mesh (correct cell_data format) ---")
print(f"points shape: {points.shape}, dtype: {points.dtype}")
print(f"cells_info: {cells_info}")
print(f"point_data_content: {point_data_content}")
print(f"correct_cell_data_content: {correct_cell_data_content}")
print("---------------------------------------------")

try:
    mesh_obj = meshio.Mesh(
        points,
        cells_info,
        point_data=point_data_content,
        cell_data=correct_cell_data_content # 使用修正后的 cell_data 格式
    )

    output_dir = "meshio_test_output"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "test_correct_cell_data.vtk")
    mesh_obj.write(filename)
    print(f"Successfully wrote {filename}")

except Exception as e:
    print(f"Error during meshio operation (correct cell_data): {e}")
    if isinstance(e, KeyError):
        print(f"  KeyError detail: {e.args}")
    import traceback
    traceback.print_exc()