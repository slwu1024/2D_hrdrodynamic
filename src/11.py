# run_simulation.py (或者你的主执行脚本)

import гидродинамика # 假设你的模型代码在 'гидродинамика' 包或模块中
import cProfile # 导入 cProfile
import pstats   # 导入 pstats 用于分析结果
import io       # 导入 io 用于将结果输出到字符串

# ... 其他导入和函数定义，例如 load_config, setup_logging 等 ...

def main():
    # 1. 加载配置 (假设你有一个函数 load_config)
    print("[*] 步骤 1: 加载配置")
    config = load_config('../config.yaml') # 请确保路径正确
    if not config:
        return
    # pprint.pprint(config) # 如果需要再次打印配置，取消注释

    # 2. 构建输入文件路径 (假设你有一个函数或直接在 main 中处理)
    print("\n[*] 步骤 2: 构建输入文件路径")
    # ... 从 config 中获取 node_file, cell_file, edge_file 等路径 ...
    # print(f"    节点文件: {node_file}")
    # ...

    # 3. 加载网格数据结构
    print("\n[*] 步骤 3: 加载网格数据结构")
    # 假设你的网格加载函数在 гидродинамика.initialization 中
    # (你需要根据你的实际项目结构调整这里的导入和调用)
    # from гидродинамика.initialization import load_mesh_data_structure
    # mesh_obj = load_mesh_data_structure(node_file, cell_file, edge_file)
    # 替换为你实际的网格加载逻辑，例如：
    from src.initialization import load_mesh_data_structure # 假设在 src.initialization
    from src.model.MeshData import Mesh # 确保 Mesh 类被正确导入
    node_file = config['file_paths']['node_file']
    cell_file = config['file_paths']['cell_file']
    edge_file = config['file_paths']['edge_file']
    mesh_obj = load_mesh_data_structure(node_file, cell_file, edge_file)
    if not mesh_obj:
        print("[-] 网格加载失败，程序终止。")
        return
    print(f"[+] 网格加载成功 (包含 {len(mesh_obj.nodes)} 个节点, {len(mesh_obj.cells)} 个单元)。")


    # 4. 初始化水动力模型
    print("\n[*] 步骤 4: 初始化水动力模型")
    # from гидродинамика.model import HydroModel # 假设 HydroModel 在此
    # 替换为你实际的 HydroModel 导入和初始化，例如：
    from src.model.HydroModel import HydroModel # 假设在 src.model.HydroModel
    hydro_model = HydroModel(mesh=mesh_obj, parameters=config) # 将完整配置传递给模型
    print(f"[+] 水动力模型初始化成功。")

    # --- **性能分析开始** ---
    print("\n[*] 步骤 5: 开始运行模拟 (带性能分析)")
    profiler = cProfile.Profile() # 创建 Profile 对象
    profiler.enable() # 开始性能分析

    # **运行核心的模拟函数**
    hydro_model.run_simulation()

    profiler.disable() # 结束性能分析
    # --- **性能分析结束** ---

    print("\n[*] 步骤 6: 分析性能数据")
    # 创建一个字符串流来捕获 pstats 的输出
    s = io.StringIO()
    # sortby 可以是 'cumulative', 'tottime', 'calls', 'pcalls', 'filename', 'lineno', 'module', 'name', 'nfl', 'stdname'
    # 'cumulative' (累计耗时) 和 'tottime' (函数自身耗时) 是最常用的
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative') # 按累计耗时排序
    ps.print_stats(30) # 打印耗时最多的前 30 个函数
    # ps.print_callers(.5, 'render') # 打印调用 render 函数的调用者信息，如果 render 耗时超过总时间的 50%
    # ps.print_callees() # 打印每个函数调用的其他函数信息

    print(s.getvalue()) # 将捕获的性能统计信息打印到控制台

    # (可选) 将性能分析结果保存到文件，以便使用可视化工具 (如 snakeviz) 查看
    # profiler.dump_stats('simulation_profile.prof')
    # print("\n性能分析结果已保存到 simulation_profile.prof")
    # print("你可以使用 'snakeviz simulation_profile.prof' 来可视化结果 (需要先 pip install snakeviz)")

    print("\n[+] 模拟流程结束。")

if __name__ == "__main__":
    main()