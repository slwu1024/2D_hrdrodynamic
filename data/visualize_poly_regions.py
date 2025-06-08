# visualize_poly_regions.py (位于 data/ 目录下)
import matplotlib.pyplot as plt
import os
import sys  # 导入 sys 模块用于修改 Python 路径

# --- 动态添加 src 目录到 Python 路径 ---
# 获取当前脚本 (visualize_poly_regions.py) 所在的目录 (即 data/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (data/ 的上一级目录)
project_root_dir = os.path.dirname(current_script_dir)
# 构建 src 目录的路径
src_dir_path = os.path.join(project_root_dir, 'src')

# 将 src 目录添加到 sys.path 的开头，以便优先导入
if src_dir_path not in sys.path:
    sys.path.insert(0, src_dir_path)

# 现在可以导入 src 目录下的模块了
try:
    from hydro_model.parse_poly_file import parse_poly_file
except ImportError:
    print(f"错误: 无法从 '{src_dir_path}' 导入 'parse_poly_file'. 请确保该文件存在且路径正确。")
    sys.exit(1)  # 导入失败则退出

# --- 用户配置 ---
# .poly 文件与此脚本在同一 data/ 目录下
# POLY_FILE_PATH = "river_boundary.poly"  # 相对于当前脚本的路径
POLY_FILE_PATH = "river_boundary.poly"  # 相对于当前脚本的路径

# 额外的独立点 (可选，用于标记您想额外关注的位置)
# 格式: [(x1, y1, 'label1', 'color1'), ...]
# 您可以清空这个列表，如果只想看 .poly 文件定义的区域点
EXTRA_POINTS_TO_MARK = [
    (505000.0, 3482000.0, 'Custom Point A', 'magenta'), # 示例自定义点
    (510000.0, 3483500, 'Custom Point B', 'magenta'), # 示例自定义点
]

# 定义边界标记到颜色和标签的映射
BOUNDARY_MARKER_STYLES = {
    0: {'color': 'gray', 'linestyle': '--', 'label': 'Internal Constraint/Region Boundary (Marker 0)',
        'linewidth': 1.0},
    1: {'color': 'black', 'linestyle': '-', 'label': 'Wall Boundary (Marker 1)', 'linewidth': 1.5},
    2: {'color': 'green', 'linestyle': '-', 'label': 'Inflow Boundary (Marker 2)', 'linewidth': 1.5},
    3: {'color': 'purple', 'linestyle': '-', 'label': 'Outflow Boundary (Marker 3)', 'linewidth': 1.5},
}
DEFAULT_SEGMENT_STYLE = {'color': 'cyan', 'linestyle': ':', 'label': 'Unknown Segment Marker', 'linewidth': 1.0}
REGION_SEED_POLY_STYLE = {'marker': 'X', 's': 120, 'edgecolor': 'black', 'linewidth': 1.5, 'alpha': 0.8}  # .poly区域种子点样式
EXTRA_POINT_STYLE = {'marker': 'o', 's': 80, 'edgecolor': 'black', 'linewidth': 1, 'alpha': 0.9}  # 额外点样式


def plot_poly_data(poly_data, extra_points_to_mark=None):
    """
    绘制从 .poly 文件解析出的数据。
    """
    if not poly_data:
        print("无法绘制：poly_data 为空。")
        return

    points = poly_data.get('points')
    segments = poly_data.get('segments')
    segment_markers = poly_data.get('segment_markers')
    regions_from_poly = poly_data.get('regions')  # 从 .poly 文件解析出的区域定义

    if points is None or segments is None:
        print("无法绘制：缺少节点或线段数据。")
        return

    fig, ax = plt.subplots(figsize=(14, 11))  # 调整图形大小

    # 1. 绘制线段，根据标记区分样式
    plotted_labels_seg = set()
    for i, seg_indices in enumerate(segments):
        node1_idx, node2_idx = seg_indices[0], seg_indices[1]
        if not (0 <= node1_idx < len(points) and 0 <= node2_idx < len(points)):
            print(f"警告: 线段 {i} 引用了无效的节点索引({node1_idx}, {node2_idx})。已跳过。")
            continue
        p1, p2 = points[node1_idx], points[node2_idx]
        marker = segment_markers[i] if segment_markers is not None and i < len(segment_markers) else 0
        style = BOUNDARY_MARKER_STYLES.get(marker, DEFAULT_SEGMENT_STYLE)
        current_label = style['label']
        if current_label not in plotted_labels_seg:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **style)  # 使用 **style 解包参数
            plotted_labels_seg.add(current_label)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=style['color'], linestyle=style['linestyle'],
                    linewidth=style['linewidth'])

    plotted_poly_region_legend = False
    if regions_from_poly is not None and len(regions_from_poly) > 0:
        print(f"从 .poly 文件中找到 {len(regions_from_poly)} 个区域定义。")
        for i, region_def in enumerate(regions_from_poly):
            rx, ry, attr, max_area = region_def

            # 检查这个 .poly 区域种子点是否也在 extra_points_to_mark 中定义
            is_managed_by_extra = False
            if extra_points_to_mark:
                for ep_x, ep_y, _, _ in extra_points_to_mark:
                    if abs(rx - ep_x) < 1e-3 and abs(ry - ep_y) < 1e-3:
                        is_managed_by_extra = True
                        break

            if not is_managed_by_extra:  # 如果不由 extra_points 管理，则在这里绘制
                label_text = f"Poly Region {i + 1}\n({rx:.0f}, {ry:.0f})\nArea: {max_area:.1e}"
                point_color = plt.cm.get_cmap('coolwarm', len(regions_from_poly))(i)  # 不同颜色方案

                current_legend_label = "Region Seeds (from .poly)" if not plotted_poly_region_legend else None
                ax.scatter([rx], [ry], color=point_color,
                           marker=REGION_SEED_POLY_STYLE['marker'],
                           s=REGION_SEED_POLY_STYLE['s'],
                           edgecolors=REGION_SEED_POLY_STYLE['edgecolor'],
                           linewidths=REGION_SEED_POLY_STYLE['linewidth'],
                           alpha=REGION_SEED_POLY_STYLE['alpha'],
                           label=current_legend_label,
                           zorder=5)
                if not plotted_poly_region_legend:
                    plotted_poly_region_legend = True

                ax.annotate(label_text, (rx, ry), textcoords="offset points", xytext=(10, -10), ha='left', va='top',
                            fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.2", fc=point_color, ec="gray", lw=0.5, alpha=0.5))

    # 3. 标记 EXTRA_POINTS_TO_MARK 中定义的所有点 (包括那些可能与 .poly 区域种子点重合的点)
    if extra_points_to_mark:
        print(f"标记 {len(extra_points_to_mark)} 个来自 EXTRA_POINTS_TO_MARK 的点。")
        plotted_extra_point_labels = set()  # 用于图例
        for x, y, label, color in extra_points_to_mark:
            unique_legend_label = f"{label}"  # 可以根据需要调整图例标签

            if unique_legend_label not in plotted_extra_point_labels:
                ax.scatter([x], [y], color=color,
                           marker=EXTRA_POINT_STYLE['marker'],
                           s=EXTRA_POINT_STYLE['s'],
                           edgecolors=EXTRA_POINT_STYLE['edgecolor'],
                           linewidths=EXTRA_POINT_STYLE['linewidth'],
                           alpha=EXTRA_POINT_STYLE['alpha'],
                           label=unique_legend_label,
                           zorder=6)  # zorder 更高，会覆盖
                plotted_extra_point_labels.add(unique_legend_label)
            else:
                ax.scatter([x], [y], color=color,
                           marker=EXTRA_POINT_STYLE['marker'],
                           s=EXTRA_POINT_STYLE['s'],
                           edgecolors=EXTRA_POINT_STYLE['edgecolor'],
                           linewidths=EXTRA_POINT_STYLE['linewidth'],
                           alpha=EXTRA_POINT_STYLE['alpha'],
                           zorder=6)

            # 为 EXTRA_POINTS_TO_MARK 中的点添加标注 (通常更突出)
            ax.annotate(label.split('\n')[0], (x, y), textcoords="offset points", xytext=(0, 10), ha='center',
                        va='bottom', fontsize=9, color='black',
                        bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="black", lw=0.5, alpha=0.85))

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"Visualization of .poly File: {os.path.basename(POLY_FILE_PATH)}")
    ax.legend(fontsize='small', loc='best')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试解析 .poly 文件: {POLY_FILE_PATH}")

    # 构建 .poly 文件的绝对路径 (因为脚本在 data/ 目录下)
    abs_poly_file_path = os.path.join(current_script_dir, POLY_FILE_PATH)
    if not os.path.exists(abs_poly_file_path):
        print(f"错误: .poly 文件未找到于 '{abs_poly_file_path}'")
        # 如果 POLY_FILE_PATH 是相对于项目根目录的，可以尝试：
        # abs_poly_file_path = os.path.join(project_root_dir, POLY_FILE_PATH)
        # if not os.path.exists(abs_poly_file_path):
        #     print(f"错误: .poly 文件也未找到于项目根目录下的 '{POLY_FILE_PATH}'")
        #     sys.exit(1)

    parsed_poly_info = parse_poly_file(abs_poly_file_path)  # 使用绝对路径

    if parsed_poly_info:
        print("解析成功，开始绘图...")

        # 准备要传递给绘图函数的 EXTRA_POINTS_TO_MARK
        # 可以选择将 .poly 文件中的区域种子点信息合并到 EXTRA_POINTS_TO_MARK
        # 或者让绘图函数分别处理它们。当前设计是绘图函数会单独处理 .poly 的区域点。

        current_extra_points = list(EXTRA_POINTS_TO_MARK)  # 创建副本

        # 如果希望 EXTRA_POINTS_TO_MARK 中的定义覆盖 .poly 区域点的显示，
        # 并且也想确保 .poly 的区域点被标注（即使 EXTRA_POINTS_TO_MARK 为空），
        # 可以在这里进行一些逻辑处理，或者依赖 plot_poly_data 内部的逻辑。
        # 目前的 plot_poly_data 会绘制所有 .poly 区域点，并用不同颜色，然后绘制 EXTRA_POINTS_TO_MARK。

        plot_poly_data(parsed_poly_info, current_extra_points)
    else:
        print(f"无法解析 .poly 文件: {abs_poly_file_path}")