import numpy as np
import pandas as pd

# 定义区域范围和采样密度
x_min, x_max = 0, 8
y_min, y_max = 0, 4
num_x_points = 15  # X方向采样点数
num_y_points = 10  # Y方向采样点数

x_coords = np.linspace(x_min, x_max, num_x_points)  # 生成X坐标点
y_coords = np.linspace(y_min, y_max, num_y_points)  # 生成Y坐标点

points = []  # 初始化点列表


# 定义高程函数
def calculate_elevation(x, y):
    # 主要趋势：从左到右高程降低 (10 -> 2)
    base_elevation = 10 - (x / x_max) * 8

    # Y方向的微小坡度 (可选)
    y_slope_effect = (y / y_max) * 0.5  # Y方向最大影响0.5
    base_elevation += y_slope_effect

    # 局部隆起：在 x=4, y=2 附近有一个高斯隆起
    peak_x, peak_y = 4, 2  # 隆起中心X, Y
    peak_height = 3  # 隆起高度
    peak_std_dev = 1.5  # 隆起范围 (标准差)
    dist_sq = (x - peak_x) ** 2 + (y - peak_y) ** 2  # 计算点到隆起中心的距离平方
    uplift = peak_height * np.exp(-dist_sq / (2 * peak_std_dev ** 2))  # 计算高斯隆起值

    return base_elevation + uplift  # 返回最终高程


# 生成散点数据
for x_val in x_coords:  # 遍历X坐标
    for y_val in y_coords:  # 遍历Y坐标
        # 简单排除岛屿内部的点，虽然插值时这些点不是必须的，但可以使数据更集中在有效区域
        # 岛屿范围: x in [2,4], y in [2,3]
        if not (2.0 < x_val < 4.0 and 2.0 < y_val < 3.0):  # 如果点不在岛屿内部
            z_val = calculate_elevation(x_val, y_val)  # 计算高程
            points.append([x_val, y_val, z_val])  # 添加点到列表

# 额外添加一些边界上的关键点，确保边界有明确定义
# (x, y, z)
boundary_extra_points = [
    [0.0, 0.0, calculate_elevation(0.0, 0.0)],  # 左下角
    [5.0, 0.0, calculate_elevation(5.0, 0.0)],  # 右下角一部分
    [8.0, 1.0, calculate_elevation(8.0, 1.0)],  # 右侧一个点
    [8.0, 4.0, calculate_elevation(8.0, 4.0)],  # 右上角
    [0.0, 4.0, calculate_elevation(0.0, 4.0)],  # 左上角
    # 岛屿边界附近的点 (可选，因为网格生成时孔洞内部无点)
    [1.9, 2.0, calculate_elevation(1.9, 2.0)],
    [4.1, 2.0, calculate_elevation(4.1, 2.0)],
    [1.9, 3.0, calculate_elevation(1.9, 3.0)],
    [4.1, 3.0, calculate_elevation(4.1, 3.0)],
]
for p in boundary_extra_points:  # 遍历额外边界点
    # 避免重复添加与网格点非常接近的点
    is_duplicate = False  # 标记是否重复
    for existing_p in points:  # 遍历已有点
        if np.allclose(existing_p[:2], p[:2], atol=1e-3):  # 如果XY坐标非常接近
            is_duplicate = True  # 标记为重复
            break  # 跳出内层循环
    if not is_duplicate:  # 如果不重复
        points.append(p)  # 添加点

# 创建 DataFrame
df = pd.DataFrame(points, columns=['x', 'y', 'z'])  # 创建Pandas DataFrame

# 保存到 CSV
output_filepath = '../../data/topography.csv'  # 定义输出文件路径
df.to_csv(output_filepath, index=False, float_format='%.4f')  # 保存为CSV文件，不包含索引，浮点数保留4位小数

print(f"已生成散点地形数据到: {output_filepath}")  # 打印完成消息
print(f"共生成 {len(df)} 个散点。")  # 打印生成的散点数量
# 可以在这里加一个简单的3D散点图预览 (可选)
import matplotlib.pyplot as plt  # 导入matplotlib绘图库

fig = plt.figure()  # 创建图形
ax = fig.add_subplot(111, projection='3d')  # 添加3D子图
ax.scatter(df['x'], df['y'], df['z'], c=df['z'], cmap='viridis')  # 绘制3D散点图
ax.set_xlabel('X')  # 设置X轴标签
ax.set_ylabel('Y')  # 设置Y轴标签
ax.set_zlabel('Z (Elevation)')  # 设置Z轴标签
plt.title('Generated Scatter Topography Data')  # 设置标题
plt.show()  # 显示图形