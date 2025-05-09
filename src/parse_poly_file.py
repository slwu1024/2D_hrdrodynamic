# parse_poly_file.py
import numpy as np

def parse_poly_file(filepath):
    """
    解析 .poly 文件并返回包含几何和拓扑信息的字典。
    """
    parsed_data = {
        'points': [], 'point_markers': [],
        'segments': [], 'segment_markers': [],
        'holes': [], 'regions': []
    }
    point_attrs_count = 0
    point_marker_present = False
    segment_marker_present = False
    node_index_base = -1 # 初始化为无效值

    section = None # 当前解析的部分
    expected_items = 0
    parsed_items = 0

    print(f"开始解析 .poly 文件: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # 确保使用 UTF-8
            current_line_num = 0
            for line in f:
                current_line_num += 1
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts: continue

                # --- 处理文件头和各部分头信息 ---
                if section is None: # 文件开始，应该是节点头
                    try:
                        num_nodes = int(parts[0])
                        dim = int(parts[1]); assert dim == 2
                        point_attrs_count = int(parts[2])
                        point_marker_present = int(parts[3]) == 1 if len(parts) > 3 else False
                        section = 'nodes'
                        expected_items = num_nodes
                        parsed_items = 0
                        print(f"  读取节点部分: {expected_items} 个节点, {point_attrs_count} 个属性, 边界标记: {point_marker_present}")
                        continue # 头信息处理完毕，继续下一行
                    except (ValueError, IndexError, AssertionError):
                        raise ValueError(f"无效的文件头或节点头格式在行 {current_line_num}")

                elif section == 'nodes' and parsed_items == expected_items: # 节点读完，应是线段头
                    try:
                        num_segments = int(parts[0])
                        segment_marker_present = int(parts[1]) == 1 if len(parts) > 1 else False
                        section = 'segments'
                        expected_items = num_segments
                        parsed_items = 0
                        print(f"  读取线段部分: {expected_items} 条线段, 边界标记: {segment_marker_present}")
                        continue
                    except (ValueError, IndexError):
                        raise ValueError(f"无效的线段头格式在行 {current_line_num}")

                elif section == 'segments' and parsed_items == expected_items: # 线段读完，应是孔洞头
                    try:
                        num_holes = int(parts[0])
                        section = 'holes'
                        expected_items = num_holes
                        parsed_items = 0
                        print(f"  读取孔洞部分: {expected_items} 个孔洞")
                        if expected_items == 0: # 没有孔洞，直接跳到区域头
                            section = 'regions' # *** 直接跳到区域数据解析 ***
                            # 仍然需要读取区域头（如果存在）
                            # 这里逻辑需要更精确的状态转换
                            # 改进：设置一个状态等待区域头
                            section = 'regions_header_wait'
                        continue
                    except (ValueError, IndexError):
                        raise ValueError(f"无效的孔洞头格式在行 {current_line_num}")

                elif section == 'holes' and parsed_items == expected_items: # 孔洞读完，应是区域头
                     section = 'regions_header_wait' # 等待区域头

                elif section == 'regions_header_wait': # 等待区域头状态
                     try:
                        num_regions = int(parts[0])
                        section = 'regions'
                        expected_items = num_regions
                        parsed_items = 0
                        print(f"  读取区域部分: {expected_items} 个区域")
                        if expected_items == 0:
                            break # 没有区域，文件结束
                        continue
                     except (ValueError, IndexError):
                        # 区域部分是可选的，如果这里解析失败，认为文件结束
                        print(f"  未找到区域部分（可选），解析结束于行 {current_line_num-1}。")
                        break

                elif section == 'regions' and parsed_items == expected_items: # 区域读完，文件结束
                     break


                # --- 处理数据行 ---
                if section == 'nodes':
                    try:
                        node_id = int(parts[0])
                        # *** 在处理第一个节点时确定索引基准 ***
                        if node_index_base == -1:
                            node_index_base = node_id
                            print(f"  检测到节点索引从 {node_index_base} 开始。")

                        x = float(parts[1])
                        y = float(parts[2])
                        marker = int(parts[3 + point_attrs_count]) if point_marker_present else 0
                        parsed_data['points'].append([x, y])
                        parsed_data['point_markers'].append(marker)
                        parsed_items += 1
                    except (ValueError, IndexError):
                        raise ValueError(f"无效的节点数据格式在行 {current_line_num}")

                elif section == 'segments':
                    try:
                        # segment_id = int(parts[0])
                        node1_idx = int(parts[1]) - node_index_base # 转换为 0-based
                        node2_idx = int(parts[2]) - node_index_base # 转换为 0-based
                        marker = int(parts[3]) if segment_marker_present else 0
                        # 可以在这里添加检查，确保 node1_idx 和 node2_idx 有效
                        if node1_idx < 0 or node1_idx >= len(parsed_data['points']) or \
                           node2_idx < 0 or node2_idx >= len(parsed_data['points']):
                           raise ValueError(f"线段引用了无效的节点索引 {parts[1]}/{parts[2]}")

                        parsed_data['segments'].append([node1_idx, node2_idx])
                        parsed_data['segment_markers'].append(marker)
                        parsed_items += 1
                    except (ValueError, IndexError):
                        raise ValueError(f"无效的线段数据格式在行 {current_line_num}")

                elif section == 'holes':
                     try:
                        # hole_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        parsed_data['holes'].append([x, y])
                        parsed_items += 1
                     except (ValueError, IndexError):
                         raise ValueError(f"无效的孔洞数据格式在行 {current_line_num}")

                elif section == 'regions':
                     try:
                        # region_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        attribute = int(parts[3])
                        max_area = float(parts[4])
                        parsed_data['regions'].append([x, y, attribute, max_area])
                        parsed_items += 1
                     except (ValueError, IndexError):
                          raise ValueError(f"无效的区域数据格式在行 {current_line_num}")

        print(".poly 文件解析完成。")
        # 转换为 numpy 数组
        parsed_data['points'] = np.array(parsed_data['points'], dtype=float)
        parsed_data['point_markers'] = np.array(parsed_data['point_markers'], dtype=int)
        parsed_data['segments'] = np.array(parsed_data['segments'], dtype=int)
        parsed_data['segment_markers'] = np.array(parsed_data['segment_markers'], dtype=int)
        parsed_data['holes'] = np.array(parsed_data['holes'], dtype=float)
        parsed_data['regions'] = np.array(parsed_data['regions'], dtype=float)
        if len(parsed_data['regions']) > 0:
            parsed_data['regions'][:, 2] = parsed_data['regions'][:, 2].astype(int)

        return parsed_data

    except FileNotFoundError:
        print(f"错误: .poly 文件 {filepath} 未找到。")
        return None
    except ValueError as e:
        print(f"解析 .poly 文件 {filepath} 时出错: {e}")
        return None
    except Exception as e:
        print(f"读取或解析 .poly 文件 {filepath} 时发生未知错误: {e}")
        return None