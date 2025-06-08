# parse_poly_file.py
import numpy as np  # 导入 NumPy 用于数值运算和数组操作
import os  # 导入 os 模块，但在此脚本中未直接使用，可以考虑移除


def parse_poly_file(filepath):
    """
    解析 Triangle 使用的 .poly 文件。

    .poly 文件格式通常包含以下部分：
    1. 节点 (Points/Vertices) 定义：
       <#节点数> <维度(2)> <#节点属性> <#节点边界标记(0或1)>
       <节点ID> <x> <y> [属性值...] [边界标记]
       ...
    2. 线段 (Segments) 定义：
       <#线段数> <#线段边界标记(0或1)>
       <线段ID> <节点1 ID> <节点2 ID> [边界标记]
       ...
    3. 孔洞 (Holes) 定义 (可选)：
       <#孔洞数>
       <孔洞ID> <x> <y> (孔洞内部一点的坐标)
       ...
    4. 区域 (Regions) 定义 (可选)：
       <#区域数>
       <区域ID> <x> <y> <区域属性> <最大三角形面积约束> (区域内部一点的坐标，属性，面积约束)
       ...

    此函数尝试处理这些部分，包括行内注释 (#) 和空行。
    节点ID可以是0基或1基，函数会自动检测并转换为0基索引用于内部存储。

    参数:
        filepath (str): .poly 文件的路径。

    返回:
        dict or None: 包含解析数据的字典，如果文件未找到或解析出错则返回 None。
                      字典结构:
                      {
                          'points': np.array([[x1,y1], [x2,y2], ...]),
                          'point_markers': np.array([m1, m2, ...]), # 如果存在
                          'segments': np.array([[n1_idx,n2_idx], ...]), # 0-based 节点索引
                          'segment_markers': np.array([m1, m2, ...]), # 如果存在
                          'original_segment_ids': np.array([id1, id2, ...]), # .poly 文件中线段的原始ID
                          'holes': np.array([[x1,y1], [x2,y2], ...]), # 如果存在
                          'regions': np.array([[x,y,attr,max_area], ...]), # 如果存在
                          'node_index_base': int, # 检测到的节点索引基准 (0 或 1)
                          'point_attributes_count': int # 每个节点的属性数量 (不含坐标和标记)
                      }
    """
    parsed_data = {  # 初始化用于存储解析数据的字典
        'points': [],  # 存储节点坐标
        'point_markers': [],  # 存储节点标记
        'segments': [],  # 存储线段 (使用0基准节点索引)
        'segment_markers': [],  # 存储线段标记
        'original_segment_ids': [],  # 存储.poly文件中线段的原始ID
        'holes': [],  # 存储孔洞点坐标
        'regions': [],  # 存储区域定义
        'node_index_base': -1,  # 初始化节点索引基准为-1 (待检测)
        'point_attributes_count': 0  # 初始化每个节点的属性数量
    }

    section = None  # 当前正在解析的文件部分 (None, 'nodes', 'segments', 'holes', 'regions')
    expected_items = 0  # 当前部分期望的项目数量
    parsed_items = 0  # 当前部分已解析的项目数量
    current_line_num = 0  # 当前处理的行号 (用于错误报告)

    # 标志位，从文件头读取
    point_marker_present = False  # 节点是否带有边界标记
    segment_marker_present = False  # 线段是否带有边界标记

    print(f"开始解析 .poly 文件: {filepath}")  # 打印开始解析文件的信息

    try:
        with open(filepath, 'r', encoding='utf-8') as f:  # 打开文件进行读取
            for line_raw in f:  # 遍历文件中的每一行
                current_line_num += 1  # 行号加1
                line_content_for_debug = line_raw.strip()  # 去除原始行首尾空白，用于调试

                # 移除行尾注释
                comment_pos = line_raw.find('#')  # 查找注释符号'#'的位置
                if comment_pos != -1:  # 如果找到注释
                    line_to_process = line_raw[:comment_pos].strip()  # 取注释前的内容并去除首尾空白
                else:  # 如果没有注释
                    line_to_process = line_raw.strip()  # 对整行去除首尾空白

                # 调试打印处理前后的行内容 (如果需要更详细的调试，可以取消注释)
                # print(f"DEBUG Line {current_line_num}: repr='{repr(line_content_for_debug)}', processed_line='{line_to_process}'")

                if not line_to_process:  # 如果处理后行为空 (例如，原行是纯注释或纯空白)
                    # print(f"DEBUG Line {current_line_num}: SKIPPED (empty or comment)")
                    continue  # 跳过空行

                parts = line_to_process.split()  # 按空格分割行内容
                if not parts:  # 如果分割后没有部分 (理论上不太可能发生，因为上面检查了 line_to_process)
                    continue  # 跳过

                # --- 状态机：解析文件头和切换节 (section) ---
                if section is None:  # 初始状态，期望节点部分的头部
                    try:
                        num_nodes = int(parts[0])  # 节点数量
                        dim = int(parts[1])  # 维度 (应为2)
                        if dim != 2:
                            raise ValueError(f"维度必须为2，但得到 {dim}")
                        parsed_data['point_attributes_count'] = int(parts[2])  # 节点属性数量
                        point_marker_present = (int(parts[3]) == 1) if len(parts) > 3 else False  # 节点标记是否存在

                        section = 'nodes'  # 切换到节点解析模式
                        expected_items = num_nodes
                        parsed_items = 0
                        print(
                            f"  读取节点部分: {expected_items} 个节点, {parsed_data['point_attributes_count']} 个属性, 点标记: {point_marker_present}")
                        continue  # 处理下一行
                    except (ValueError, IndexError) as e:
                        raise ValueError(
                            f"无效的文件头或节点头格式在行 {current_line_num}: '{line_content_for_debug}'. 错误: {e}")

                elif section == 'nodes' and parsed_items == expected_items:  # 节点解析完毕，期望线段部分的头部
                    try:
                        num_segments = int(parts[0])  # 线段数量
                        segment_marker_present = (int(parts[1]) == 1) if len(parts) > 1 else False  # 线段标记是否存在

                        section = 'segments'  # 切换到线段解析模式
                        expected_items = num_segments
                        parsed_items = 0
                        print(f"  读取线段部分: {expected_items} 条线段, 线段标记: {segment_marker_present}")
                        continue  # 处理下一行
                    except (ValueError, IndexError) as e:
                        raise ValueError(
                            f"无效的线段头格式在行 {current_line_num}: '{line_content_for_debug}'. 错误: {e}")

                elif section == 'segments' and parsed_items == expected_items:  # 线段解析完毕，期望孔洞部分的头部
                    try:
                        num_holes = int(parts[0])  # 孔洞数量

                        section = 'holes'  # 切换到孔洞解析模式
                        expected_items = num_holes
                        parsed_items = 0
                        print(f"  读取孔洞部分: {expected_items} 个孔洞")
                        if expected_items == 0:  # 如果没有孔洞，直接准备读取区域头
                            section = 'regions_header_wait'
                        continue  # 处理下一行
                    except (ValueError, IndexError) as e:
                        raise ValueError(
                            f"无效的孔洞头格式在行 {current_line_num}: '{line_content_for_debug}'. 错误: {e}")

                elif section == 'holes' and parsed_items == expected_items:  # 孔洞解析完毕，期望区域部分的头部
                    section = 'regions_header_wait'  # 进入等待区域头状态
                    # 注意：这里不 continue，因为当前行可能是区域头

                if section == 'regions_header_wait':  # 等待区域头（或文件结束）
                    try:
                        num_regions = int(parts[0])  # 区域数量

                        section = 'regions'  # 切换到区域解析模式
                        expected_items = num_regions
                        parsed_items = 0
                        print(f"  读取区域部分: {expected_items} 个区域")
                        if expected_items == 0:  # 如果没有区域，解析结束
                            print("  没有区域定义，文件解析在此结束。")
                            break
                        continue  # 处理下一行
                    except (ValueError, IndexError):  # 如果无法解析为区域头，说明区域部分不存在或文件结束
                        print(f"  未找到区域部分（可选），解析结束于行 {current_line_num - 1}。")
                        break  # 结束文件解析

                elif section == 'regions' and parsed_items == expected_items:  # 区域解析完毕，文件结束
                    print("  区域部分解析完毕，文件解析结束。")
                    break  # 结束文件解析

                # --- 数据行解析 ---
                if section == 'nodes':
                    try:
                        node_id_poly = int(parts[0])  # .poly文件中的节点ID
                        if parsed_data['node_index_base'] == -1:  # 如果是第一个节点，检测索引基准
                            parsed_data['node_index_base'] = node_id_poly
                            print(f"  检测到节点索引从 {parsed_data['node_index_base']} 开始。")

                        x = float(parts[1])  # x坐标
                        y = float(parts[2])  # y坐标
                        parsed_data['points'].append([x, y])  # 添加到节点列表

                        # 处理节点属性 (跳过，但确保索引正确)
                        idx_after_coords_and_attrs = 3 + parsed_data['point_attributes_count']

                        marker = 0  # 默认节点标记
                        if point_marker_present:
                            if len(parts) > idx_after_coords_and_attrs:
                                try:
                                    marker = int(parts[idx_after_coords_and_attrs])
                                except ValueError:
                                    print(
                                        f"警告: 节点 {node_id_poly} (行 {current_line_num}) 的标记 '{parts[idx_after_coords_and_attrs]}' 无法转为整数。将用0。")
                            else:
                                print(
                                    f"警告: 节点 {node_id_poly} (行 {current_line_num}) 声明有标记但字段不足。将用标记0。")
                        parsed_data['point_markers'].append(marker)
                        parsed_items += 1
                    except (ValueError, IndexError) as e:
                        raise ValueError(
                            f"无效的节点数据格式在行 {current_line_num}: '{line_content_for_debug}'. 错误: {e}")

                elif section == 'segments':
                    try:
                        original_segment_id = int(parts[0])  # .poly文件中的线段ID
                        parsed_data['original_segment_ids'].append(original_segment_id)

                        if parsed_data['node_index_base'] == -1:
                            raise ValueError("在解析线段之前未能确定节点索引基准。")

                        node1_poly_id = int(parts[1])  # 线段起点ID (来自.poly文件)
                        node2_poly_id = int(parts[2])  # 线段终点ID (来自.poly文件)

                        # 转换为0基准索引
                        node1_idx_0based = node1_poly_id - parsed_data['node_index_base']
                        node2_idx_0based = node2_poly_id - parsed_data['node_index_base']

                        # 检查节点索引是否有效 (相对于已解析的节点数)
                        num_parsed_nodes = len(parsed_data['points'])
                        if not (0 <= node1_idx_0based < num_parsed_nodes and \
                                0 <= node2_idx_0based < num_parsed_nodes):
                            raise ValueError(f"线段 {original_segment_id} (行 {current_line_num}) 引用了无效的节点ID "
                                             f"({node1_poly_id} 或 {node2_poly_id}，转换后为 "
                                             f"{node1_idx_0based} 或 {node2_idx_0based})。"
                                             f"已解析节点数: {num_parsed_nodes}")

                        parsed_data['segments'].append([node1_idx_0based, node2_idx_0based])

                        marker = 0  # 默认线段标记
                        if segment_marker_present:
                            if len(parts) > 3:
                                try:
                                    marker = int(parts[3])
                                except ValueError:
                                    print(
                                        f"警告: 线段 {original_segment_id} (行 {current_line_num}) 的标记 '{parts[3]}' 无法转为整数。将用0。")
                            else:
                                print(
                                    f"警告: 线段 {original_segment_id} (行 {current_line_num}) 声明有标记但字段不足。将用标记0。")
                        parsed_data['segment_markers'].append(marker)
                        parsed_items += 1
                    except (ValueError, IndexError) as e:
                        raise ValueError(
                            f"无效的线段数据格式在行 {current_line_num}: '{line_content_for_debug}'. 错误: {e}")

                elif section == 'holes':
                    try:
                        # 孔洞ID 通常是 parts[0]，但我们只关心坐标
                        x = float(parts[1])  # 孔洞内部点x坐标
                        y = float(parts[2])  # 孔洞内部点y坐标
                        parsed_data['holes'].append([x, y])
                        parsed_items += 1
                    except (ValueError, IndexError) as e:
                        raise ValueError(
                            f"无效的孔洞数据格式在行 {current_line_num}: '{line_content_for_debug}'. 错误: {e}")

                elif section == 'regions':
                    try:
                        # 区域ID 通常是 parts[0]
                        x = float(parts[1])  # 区域内部点x坐标
                        y = float(parts[2])  # 区域内部点y坐标
                        attribute = 0.0  # 默认区域属性
                        max_area = -1.0  # 默认最大面积 (triangle中-1.0表示无约束)

                        if len(parts) > 3:  # 区域属性存在
                            attribute = float(parts[3])  # triangle库将其视为float
                        if len(parts) > 4:  # 最大面积约束存在
                            max_area = float(parts[4])

                        parsed_data['regions'].append([x, y, attribute, max_area])
                        parsed_items += 1
                    except (ValueError, IndexError) as e:
                        raise ValueError(
                            f"无效的区域数据格式在行 {current_line_num}: '{line_content_for_debug}'. 错误: {e}")

        # --- 文件读取完毕，将列表转换为NumPy数组 ---
        print(".poly 文件主体解析完成，正在转换数据结构...")
        parsed_data['points'] = np.array(parsed_data['points'], dtype=float)
        if parsed_data['point_markers']:  # 仅当列表非空时转换
            parsed_data['point_markers'] = np.array(parsed_data['point_markers'], dtype=int)
        else:  # 如果为空，创建一个空的正确类型的数组，或删除该键
            del parsed_data['point_markers']  # 或者 parsed_data['point_markers'] = np.array([], dtype=int)

        parsed_data['segments'] = np.array(parsed_data['segments'], dtype=int)
        if parsed_data['segment_markers']:
            parsed_data['segment_markers'] = np.array(parsed_data['segment_markers'], dtype=int)
        else:
            del parsed_data['segment_markers']

        if parsed_data['original_segment_ids']:
            parsed_data['original_segment_ids'] = np.array(parsed_data['original_segment_ids'], dtype=int)
        else:
            del parsed_data['original_segment_ids']

        if parsed_data['holes']:
            parsed_data['holes'] = np.array(parsed_data['holes'], dtype=float)
        else:  # 如果列表为空，创建一个空的正确类型的数组，或删除该键
            del parsed_data['holes']

        if parsed_data['regions']:
            regions_np = np.array(parsed_data['regions'], dtype=float)
            # 区域属性通常是整数，但triangle可能期望整个数组是float，或单独处理属性列
            # Shewchuk的文档指出区域属性是浮点数。如果需要整数，可以在使用时转换。
            # 例如: parsed_data['regions'][:, 2] = parsed_data['regions'][:, 2].astype(int)
            # 但为了通用性，这里保持为float。
            parsed_data['regions'] = regions_np
        else:
            del parsed_data['regions']

        print("数据转换完成。解析得到的统计信息:")
        print(f"  节点数: {len(parsed_data['points'])}")
        if 'point_markers' in parsed_data: print(f"  节点标记数: {len(parsed_data['point_markers'])}")
        print(f"  线段数: {len(parsed_data['segments'])}")
        if 'segment_markers' in parsed_data: print(f"  线段标记数: {len(parsed_data['segment_markers'])}")
        if 'original_segment_ids' in parsed_data: print(f"  原始线段ID数: {len(parsed_data['original_segment_ids'])}")
        if 'holes' in parsed_data: print(f"  孔洞数: {len(parsed_data['holes'])}")
        if 'regions' in parsed_data: print(f"  区域数: {len(parsed_data['regions'])}")

        # 验证解析的项目数量是否与文件头声明的一致 (可选的健全性检查)
        # ...

        return parsed_data

    except FileNotFoundError:
        print(f"错误: .poly 文件 '{filepath}' 未找到。")
        return None
    except ValueError as e:  # 捕获在解析过程中主动抛出的ValueError
        print(f"解析 .poly 文件 '{filepath}' 时出错: {e}")
        return None
    except Exception as e:  # 捕获其他意外错误
        print(f"读取或解析 .poly 文件 '{filepath}' 时发生未知错误: {e}")
        import traceback
        traceback.print_exc()  # 打印详细的堆栈跟踪
        return None


if __name__ == '__main__':
    # --- 测试代码 ---
    # 创建一个示例 .poly 文件用于测试
    test_poly_content_complex = """
# Complex example .poly file for testing parse_poly_file.py
# Nodes: count, dim, attributes, markers
4  2  1  1  # 4 nodes, 2D, 1 attribute, with markers
# Node list: ID, x, y, [attribute], [marker]
1  0.0  0.0  10.0  1 # Node 1, (0,0), attr=10, marker=1
2  1.0  0.0  20.0  1 # Node 2, (1,0), attr=20, marker=1
3  1.0  1.0  30.0  0 # Node 3, (1,1), attr=30, marker=0 (interior)
4  0.0  1.0  40.0  2 # Node 4, (0,1), attr=40, marker=2

# Segments: count, markers
4  1 # 4 segments, with markers
# Segment list: ID, node1, node2, [marker]
1  1  2  101 # Seg 1, N1-N2, marker 101
2  2  3  102 # Seg 2, N2-N3, marker 102
3  3  4  0   # Seg 3, N3-N4, marker 0 (interior constraint)
4  4  1  104 # Seg 4, N4-N1, marker 104

# Holes: count
1 # 1 hole
# Hole list: ID, x, y
1  0.5 0.5 # Hole at (0.5, 0.5)

# Regions: count
2 # 2 regions
# Region list: ID, x, y, attribute, max_area
1  0.2 0.2  1.0  0.01  # Region 1, attr 1, max_area 0.01
2  0.8 0.8  2.0  -1.0  # Region 2, attr 2, no area constraint
    """
    test_poly_filepath = "test_complex.poly"
    with open(test_poly_filepath, "w") as f:
        f.write(test_poly_content_complex)

    print(f"\n--- 测试复杂 .poly 文件: {test_poly_filepath} ---")
    data_complex = parse_poly_file(test_poly_filepath)
    if data_complex:
        print("\n解析结果 (复杂示例):")
        for key, value in data_complex.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                # print(value) # 取消注释以打印数组内容
            else:
                print(f"  {key}: {value}")

        # 验证0基转换
        print("\n  验证0基线段 (应为 [[0,1],[1,2],[2,3],[3,0]] 如果节点从1开始):")
        print(data_complex.get('segments'))

    # 测试一个没有标记和属性，且没有孔洞和区域的文件
    test_poly_content_simple = """
# Simple example .poly file
# Nodes
3 2 0 0
0 0.0 0.0   # 0-indexed nodes
1 1.0 0.0
2 0.5 0.866

# Segments
3 0
0 0 1
1 1 2
2 2 0

# Holes
0

# Regions (optional, could be missing or 0)
0
    """
    test_poly_filepath_simple = "test_simple.poly"
    with open(test_poly_filepath_simple, "w") as f:
        f.write(test_poly_content_simple)

    print(f"\n--- 测试简单 .poly 文件 (0-indexed, no markers/attrs/holes/regions): {test_poly_filepath_simple} ---")
    data_simple = parse_poly_file(test_poly_filepath_simple)
    if data_simple:
        print("\n解析结果 (简单示例):")
        for key, value in data_simple.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")
        print("\n  验证线段 (应为 [[0,1],[1,2],[2,0]] 如果节点从0开始):")
        print(data_simple.get('segments'))
        assert data_simple.get('node_index_base') == 0, "简单示例的node_index_base应为0"

    # 清理测试文件
    os.remove(test_poly_filepath)
    os.remove(test_poly_filepath_simple)
    print("\n测试完成，测试文件已删除。")