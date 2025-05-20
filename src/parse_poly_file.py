# parse_poly_file.py
import numpy as np


def parse_poly_file(filepath):
    # ... (parsed_data, point_attrs_count, etc. 初始化不变) ...
    parsed_data = {
        'points': [], 'point_markers': [],
        'segments': [], 'segment_markers': [],
        'original_segment_ids': [],
        'holes': [], 'regions': [],
        'node_index_base': -1
    }
    point_attrs_count = 0
    point_marker_present = False
    segment_marker_present = False

    section = None
    expected_items = 0
    parsed_items = 0

    print(f"开始解析 .poly 文件: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            current_line_num = 0
            for line_raw in f:  # 遍历文件中的每一原始行
                current_line_num += 1

                line_content_for_debug = line_raw.strip()  # 获取原始strip后的行，用于调试打印

                # --- 移除行尾注释并处理行 ---
                comment_pos = line_raw.find('#')  # 查找行内第一个'#'的位置
                if comment_pos != -1:  # 如果找到了注释符号
                    line_to_process = line_raw[:comment_pos].strip()  # 取注释之前的部分，并去除首尾空格
                else:  # 如果没有注释符号
                    line_to_process = line_raw.strip()  # 对整行去除首尾空格
                # --- 行处理结束 ---

                # 调试打印处理前后的行内容
                print(
                    f"DEBUG Line {current_line_num}: repr='{repr(line_content_for_debug)}', processed_line='{line_to_process}'")

                if not line_to_process:  # 如果处理后的行为空 (例如，原行是纯注释或纯空白)
                    print(
                        f"DEBUG Line {current_line_num}: SKIPPED (empty after comment removal or originally empty/comment)")
                    continue  # 跳过这一行，处理下一行

                parts = line_to_process.split()  # 将处理后的行按空格分割成部分
                if not parts:  # 如果分割后没有部分 (理论上如果 line_to_process 非空，parts 也不会空)
                    print(f"DEBUG Line {current_line_num}: SKIPPED (empty parts after split)")
                    continue  # 跳过这一行

                # --- 节(Section)头解析 和 状态切换 ---
                # (这里的逻辑与我上一条回复中建议的修正版本一致，请确保你已经应用了那些修正)
                if section is None:  # 期望节点头
                    try:
                        num_nodes = int(parts[0])
                        dim = int(parts[1]);
                        assert dim == 2
                        point_attrs_count = int(parts[2])
                        point_marker_present = int(parts[3]) == 1 if len(parts) > 3 else False
                        section = 'nodes'
                        expected_items = num_nodes
                        parsed_items = 0
                        print(
                            f"  读取节点部分: {expected_items} 个节点, {point_attrs_count} 个属性, 点标记: {point_marker_present}")
                        continue
                    except (ValueError, IndexError, AssertionError) as e:
                        raise ValueError(f"无效的文件头或节点头格式在行 {current_line_num}: {e} (parts: {parts})")

                elif section == 'nodes' and parsed_items == expected_items:  # 期望线段头
                    try:
                        num_segments = int(parts[0])
                        segment_marker_present = int(parts[1]) == 1 if len(parts) > 1 else False
                        section = 'segments'
                        expected_items = num_segments
                        parsed_items = 0
                        print(f"  读取线段部分: {expected_items} 条线段, 线段标记: {segment_marker_present}")
                        continue
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"无效的线段头格式在行 {current_line_num}: {e} (parts: {parts})")

                elif section == 'segments' and parsed_items == expected_items:  # 期望孔洞头
                    try:
                        num_holes = int(parts[0])
                        expected_items = num_holes
                        parsed_items = 0
                        print(f"  读取孔洞部分: {expected_items} 个孔洞")
                        if expected_items == 0:
                            section = 'regions_header_wait'
                        else:
                            section = 'holes'
                        continue
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"无效的孔洞头格式在行 {current_line_num}: {e} (parts: {parts})")

                elif section == 'holes' and parsed_items == expected_items:
                    section = 'regions_header_wait'
                    # 不 continue，当前行可能是区域头

                if section == 'regions_header_wait':
                    try:
                        num_regions = int(parts[0])
                        section = 'regions'
                        expected_items = num_regions
                        parsed_items = 0
                        print(f"  读取区域部分: {expected_items} 个区域")
                        if expected_items == 0:
                            break
                        continue
                    except (ValueError, IndexError):
                        print(f"  未找到区域部分（可选），解析结束于行 {current_line_num - 1}。")
                        break

                elif section == 'regions' and parsed_items == expected_items:
                    break

                # --- 数据行解析 ---
                if section == 'nodes':
                    try:
                        node_id_poly = int(parts[0])
                        if parsed_data['node_index_base'] == -1:
                            parsed_data['node_index_base'] = node_id_poly
                            print(f"  检测到节点索引从 {parsed_data['node_index_base']} 开始。")
                        x = float(parts[1])
                        y = float(parts[2])
                        marker = 0
                        if point_marker_present:
                            idx_of_marker_value = 3 + point_attrs_count
                            if len(parts) > idx_of_marker_value:
                                try:
                                    marker = int(parts[idx_of_marker_value])
                                except ValueError:
                                    print(
                                        f"警告: 节点 {node_id_poly} (行 {current_line_num}) 的标记 '{parts[idx_of_marker_value]}' 无法转为整数。将用0。")
                                    marker = 0
                            else:
                                print(
                                    f"警告: 节点 {node_id_poly} (行 {current_line_num}) 声明有标记但字段不足。将用标记0。")
                                marker = 0
                        parsed_data['points'].append([x, y])
                        parsed_data['point_markers'].append(marker)
                        parsed_items += 1
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"无效的节点数据格式在行 {current_line_num}: {e} (parts: {parts})")

                # ... (elif section == 'segments', elif section == 'holes', elif section == 'regions' 的数据解析逻辑保持不变，
                #      确保它们使用的是 parts 变量，并且错误处理也包含了 parts) ...
                elif section == 'segments':
                    try:
                        original_segment_id = int(parts[0])
                        if parsed_data['node_index_base'] == -1: raise ValueError("未确定节点索引基准。")
                        node1_poly_id = int(parts[1]);
                        node2_poly_id = int(parts[2])
                        node1_idx_0based = node1_poly_id - parsed_data['node_index_base']
                        node2_idx_0based = node2_poly_id - parsed_data['node_index_base']
                        type_marker = 0
                        if segment_marker_present:
                            if len(parts) > 3:
                                type_marker = int(parts[3])
                            else:
                                print(f"警告: 线段 {original_segment_id} (行 {current_line_num}) 声明有标记但字段不足。")
                        num_pts = len(parsed_data['points'])
                        if not (0 <= node1_idx_0based < num_pts and 0 <= node2_idx_0based < num_pts):
                            raise ValueError(f"线段 {original_segment_id} (行 {current_line_num}) 引用无效节点ID。")
                        parsed_data['segments'].append([node1_idx_0based, node2_idx_0based])
                        parsed_data['segment_markers'].append(type_marker)
                        parsed_data['original_segment_ids'].append(original_segment_id)
                        parsed_items += 1
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"无效的线段数据格式在行 {current_line_num}: {e} (parts: {parts})")

                elif section == 'holes':
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        parsed_data['holes'].append([x, y])
                        parsed_items += 1
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"无效的孔洞数据格式在行 {current_line_num}: {e} (parts: {parts})")

                elif section == 'regions':
                    try:
                        x = float(parts[1]);
                        y = float(parts[2])
                        attribute = int(parts[3]);
                        max_area = float(parts[4])
                        parsed_data['regions'].append([x, y, attribute, max_area])
                        parsed_items += 1
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"无效的区域数据格式在行 {current_line_num}: {e} (parts: {parts})")

            # ... (函数末尾的Numpy转换和返回逻辑不变) ...
            print(".poly 文件解析完成。")
            parsed_data['points'] = np.array(parsed_data['points'], dtype=float)
            parsed_data['point_markers'] = np.array(parsed_data['point_markers'], dtype=int)
            parsed_data['segments'] = np.array(parsed_data['segments'], dtype=int)
            parsed_data['segment_markers'] = np.array(parsed_data['segment_markers'], dtype=int)
            parsed_data['original_segment_ids'] = np.array(parsed_data['original_segment_ids'], dtype=int)
            parsed_data['holes'] = np.array(parsed_data['holes'], dtype=float)
            if len(parsed_data['regions']) > 0:
                parsed_data['regions'] = np.array(parsed_data['regions'], dtype=float)
                if parsed_data['regions'].size > 0:
                    parsed_data['regions'][:, 2] = parsed_data['regions'][:, 2].astype(int)

            print(f"  解析得到节点数: {len(parsed_data['points'])}")
            print(f"  解析得到线段数: {len(parsed_data['segments'])}")
            if len(parsed_data['segments']) > 0:
                print(
                    f"  示例线段 (0-based节点): {parsed_data['segments'][0] if len(parsed_data['segments']) > 0 else 'N/A'}")
                print(
                    f"  示例线段类型标记: {parsed_data['segment_markers'][0] if len(parsed_data['segment_markers']) > 0 else 'N/A'}")
                print(
                    f"  示例原始线段ID: {parsed_data['original_segment_ids'][0] if len(parsed_data['original_segment_ids']) > 0 else 'N/A'}")
            return parsed_data
    # ... (except块不变) ...
    except FileNotFoundError:
        print(f"错误: .poly 文件 {filepath} 未找到。")
        return None
    except ValueError as e:
        print(f"解析 .poly 文件 {filepath} 时出错: {e}")
        return None
    except Exception as e:
        print(f"读取或解析 .poly 文件 {filepath} 时发生未知错误: {e}")
        return None