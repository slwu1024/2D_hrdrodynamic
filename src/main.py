import logging
from src.model.HydroModel import HydroModel
from src.data_processing.DataReader import DataReader
from src.analysis.Visualizer import Visualizer
from src.tools.tools import setup_logging

def main():
    # ������־��¼
    setup_logging()

    # ��ʼ�����ݶ�ȡ��
    data_reader = DataReader('path/to/data')
    data = data_reader.read_data()

    # ��ʼ��ˮ����ģ��
    hydro_model = HydroModel(parameters={'param1': 'value1', 'param2': 'value2'})
    hydro_model.setup()

    # ����ģ��
    hydro_model.run_simulation()

    # �����Ϳ��ӻ����
    visualizer = Visualizer()
    visualizer.plot_results(hydro_model.results)

    # ������
    hydro_model.save_results('path/to/output')

if __name__ == "__main__":
    main()
