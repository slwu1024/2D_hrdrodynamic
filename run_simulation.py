# run_simulation.py
import yaml
import os
import sys
import traceback
import cProfile # 导入 cProfile
import pstats   # 导入 pstats 用于分析结果
import io       # 导入 io 用于将结果输出到字符串

try:
    from src.initialization import load_mesh_data_structure
    from src.model.HydroModel import HydroModel
    # 确认导入路径指向 src/model/MeshData.py
    from src.model.MeshData import Mesh, Node, Cell, HalfEdge  # 可能还需要导入其他类
except ImportError as e:
    print(f"错误：无法导入必要的模块。请检查文件路径和类定义是否正确。") # 修改错误提示信息
    print(f"详细错误: {e}")
    # 打印更详细的路径信息帮助调试
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python搜索路径 (sys.path): {sys.path}")
    sys.exit(1)

def main() -> None:  # 主函数，明确表示无返回值
    """主程序，用于加载配置、初始化模型并运行模拟。"""

    config_filepath = 'config.yaml'  # 配置文件名 (假设与此脚本同级)
    project_root = os.path.dirname(os.path.abspath(__file__))  # 获取此脚本所在的绝对路径 (即项目根目录)

    # --- 1. 加载配置 ---
    print("-" * 30)
    print(f"[*] 步骤 1: 加载配置")
    print(f"    配置文件: {os.path.abspath(config_filepath)}")
    try:
        with open(config_filepath, 'r', encoding='utf-8') as f:  # 使用utf-8编码打开
            config = yaml.safe_load(f)
        if config is None:  # 如果文件为空或无效YAML
            print(f"错误: 配置文件 '{config_filepath}' 为空或格式无效。")
            sys.exit(1)
        print(f"[+] 配置加载成功.")
    except FileNotFoundError:
        print(f"错误: 配置文件 '{config_filepath}' 未找到。请确保它在项目根目录下。")
        sys.exit(1)
    except yaml.YAMLError as e:  # 捕获YAML解析错误
        print(f"错误: 解析配置文件 '{config_filepath}' 失败。请检查YAML语法。")
        print(f"详细错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载配置文件时发生未知错误: {e}")
        traceback.print_exc()  # 打印完整的错误堆栈
        sys.exit(1)

    # --- 2. 构建必要的输入文件路径 ---
    print("\n[*] 步骤 2: 构建输入文件路径")
    try:
        node_f = os.path.join(project_root, config['file_paths']['node_file'])
        cell_f = os.path.join(project_root, config['file_paths']['cell_file'])
        edge_f = os.path.join(project_root, config['file_paths']['edge_file'])
        # 边界条件文件路径 (允许它们不存在，在Handler中处理)
        elev_file = os.path.join(project_root, config['file_paths'].get('boundary_timeseries_elevation_file', ''))
        flux_file = os.path.join(project_root, config['file_paths'].get('boundary_timeseries_discharge_file', ''))

        print(f"    节点文件: {node_f}")
        print(f"    单元文件: {cell_f}")
        print(f"    边界文件: {edge_f}")
        if elev_file and os.path.exists(elev_file): print(f"    水位边界文件: {elev_file}")
        if flux_file and os.path.exists(flux_file): print(f"    流量边界文件: {flux_file}")

        # 检查核心网格文件是否存在
        for fpath in [node_f, cell_f, edge_f]:  # 检查node, cell, edge文件
            if not os.path.exists(fpath):  # 如果文件不存在
                print(f"错误: 输入网格文件 '{fpath}' 未找到。请检查 config.yaml 中的 file_paths。")
                sys.exit(1)

    except KeyError as e:
        print(f"错误: 在 'config.yaml' 的 'file_paths' 部分缺少必要的键: {e}。请检查配置文件。")
        sys.exit(1)
    except TypeError as e:  # 如果 config['file_paths'] 不是字典
        print(f"错误: 'config.yaml' 中的 'file_paths' 部分格式不正确，应为字典。错误: {e}")
        sys.exit(1)

    # --- 3. 加载网格数据结构 ---
    print("\n[*] 步骤 3: 加载网格数据结构")
    try:
        # 假设 perform_validation 控制是否执行内部几何验证
        validation_flag = config.get('simulation_control', {}).get('perform_mesh_validation', True)
        mesh_obj: Mesh | None = load_mesh_data_structure(node_f, cell_f, edge_f, perform_validation=validation_flag)

        if mesh_obj is None:
            print("错误: 加载网格数据结构失败。请检查 initialization.py 的输出。")
            sys.exit(1)
        print(f"[+] 网格加载成功 (包含 {len(mesh_obj.nodes)} 个节点, {len(mesh_obj.cells)} 个单元)。")
    except Exception as e:
        print(f"错误: 加载网格时发生异常: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 4. 准备模型参数并初始化 HydroModel ---
    print("\n[*] 步骤 4: 初始化水动力模型")
    try:
        # 将所有相关配置合并到一个参数字典中传递给模型
        model_params = {
            **config.get('simulation_control', {}),
            **config.get('physical_parameters', {}),
            **config.get('numerical_schemes', {}),
            **config.get('model_parameters', {}),
            **config.get('initial_conditions', {}),
            'boundary_conditions': config.get('boundary_conditions', {}),
            'elev_timeseries_filepath': elev_file,  # 传递绝对路径或确认相对路径基于根
            'discharge_timeseries_filepath': flux_file,
            # 输出目录：在 HydroModel 内部根据配置创建或确认
            'output_directory': config.get('file_paths', {}).get('output_directory', 'output_default')
        }

        hydro_model = HydroModel(mesh=mesh_obj, parameters=config)
        print(f"[+] 水动力模型初始化成功。")

    except KeyError as e:  # 如果配置字典缺少必要键
        print(f"错误: 初始化模型时配置参数缺失，键: {e}。请检查 config.yaml。")
        sys.exit(1)
    except ValueError as e:  # 捕获模型初始化中可能抛出的值错误
        print(f"错误: 初始化模型时参数值无效: {e}。请检查 config.yaml。")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 初始化 HydroModel 时发生异常: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- **性能分析开始** ---
    print("\n[*] 步骤 5: 开始运行模拟 (带性能分析)")
    profiler = cProfile.Profile()  # 创建 Profile 对象
    profiler.enable()  # 开始性能分析

    # **运行核心的模拟函数**
    hydro_model.run_simulation()

    profiler.disable()  # 结束性能分析
    # --- **性能分析结束** ---

    print("\n[*] 步骤 6: 分析性能数据")
    # 创建一个字符串流来捕获 pstats 的输出
    s = io.StringIO()
    # sortby 可以是 'cumulative', 'tottime', 'calls', 'pcalls', 'filename', 'lineno', 'module', 'name', 'nfl', 'stdname'
    # 'cumulative' (累计耗时) 和 'tottime' (函数自身耗时) 是最常用的
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')  # 按累计耗时排序
    ps.print_stats(30)  # 打印耗时最多的前 30 个函数
    # ps.print_callers(.5, 'render') # 打印调用 render 函数的调用者信息，如果 render 耗时超过总时间的 50%
    # ps.print_callees() # 打印每个函数调用的其他函数信息

    print(s.getvalue())  # 将捕获的性能统计信息打印到控制台

    # # --- 5. 运行模拟 ---
    # print("\n[*] 步骤 5: 开始运行模拟")
    # try:
    #     hydro_model.run_simulation()  # 调用模型的运行方法
    #     print("\n[+] 模拟运行完成。")
    # except KeyboardInterrupt:  # 允许用户手动中断 (Ctrl+C)
    #     print("\n[!] 用户中断了模拟。")
    #     sys.exit(0)
    # except Exception as e:
    #     print(f"\n错误: 模拟运行时发生异常: {e}")
    #     traceback.print_exc()  # 打印详细错误信息
    #     sys.exit(1)


if __name__ == "__main__":
    main()  # 执行主函数