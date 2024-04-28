import numpy as np
import pandas as pd
import time as python_time
from collections import Counter

from PSO import PSOForFS
from GA import GAForFS
from DE import DEForFS
from CS import CSForFS
from WOA import WOAForFS
from GWO import GWOForFS
from HRO import HROForFS
from IHRO import IHROForFS
from CC_IHRO import CCIHROForFS


if __name__ == '__main__':
    # 设置实验参数
    # 进行测试的算法
    optimizers = ['GA', 'PSO', 'DE', 'CS', 'GWO', 'HRO', 'IHRO', 'CC-IHRO M=2', 'CC-IHRO M=3', 'CC-IHRO M=4', 'CC-IHRO M=5']
    # 进行测试的数据集
    datasets = ['NSL-KDD']
    # datasets = ['Hill-Valley', 'Ionosphere', 'Libras', 'MUSK1', 'Sonar', 'Urban', 'LSVT', 'WDBC']
    # datasets = ['mfeat-fou']
    # 进行测试的分类器
    classifiers = ['XGB']
    # 每个算法运行次数
    times = 1
    # 最大迭代次数
    max_iter = 50
    # 搜索域
    pos_max = 10
    pos_min = -10

    # 为了方便审阅数据建立专用索引
    data_index = []
    for dataset in datasets:
        for classifier in classifiers:
            for time in range(times):
                data_index.append(str(dataset) + '-' + str(classifier) + '-' + str(time+1))
            data_index.append('')
        data_index.append('')

    # 创建DataFrame储存运行结果
    results_data = pd.DataFrame(index=data_index, columns=optimizers)

    # 为储存结果的文件生成时间戳
    ts = python_time.strftime("%Y%m%d%H%M%S")
    results_file_name = 'results' + '_' + ts + '.xlsx'
    records_file_name = 'records' + '_' + ts + '.xlsx'

    # 创建操作Excel文件的句柄
    results_writer = pd.ExcelWriter(results_file_name)
    records_writer = pd.ExcelWriter(records_file_name)

    # 开始测试
    for opt in optimizers:
        print('==============================')
        print('开始测试算法：', opt)
        print('==============================')

        # 创建DataFrame储存收敛过程
        records_data = pd.DataFrame(index=np.arange(max_iter + 1), columns=data_index)

        for dataset in datasets:
            # 输出当前的测试集
            print('----当前测试的数据集为：', dataset)

            # 从CSV文件中读取样本数据
            if dataset == 'NSL-KDD':
                train_set = pd.read_csv('input/KDDTrain+.csv', header=None)
                test_set = pd.read_csv('input/KDDTest+.csv', header=None)
                # 获取特征数量
                attr_size = train_set.shape[1] - 1
                raw_data = [train_set, test_set]
            else:
                raw_data = pd.read_csv('input/' + dataset + '.csv', header=None)
                # 获取特征数量
                attr_size = raw_data.shape[1] - 1

            for classifier in classifiers:
                # 输出当前的分类器
                print('--------当前使用的分类器为：', classifier)

                # 创建优化器对象
                optimizer = None

                # 确定当前所使用的算法及其参数
                if opt == 'PSO':
                    optimizer = PSOForFS(ps_size=6, dim=attr_size, max_iter=max_iter, pos_max=pos_max, pos_min=pos_min,
                                         raw_data=raw_data, classifier=classifier,
                                         w_max=0.9, w_min=0.4, c1=2, c2=2, vel_max=1, vel_min=-1)

                if opt == 'GA':
                    optimizer = GAForFS(size=6, dim=attr_size, max_iter=max_iter, pos_max=pos_max, pos_min=pos_min,
                                        raw_data=raw_data, classifier=classifier,
                                        select_type='rws', cross_type='spc', cross_rate=0.8, mutation_type='rm', mutation_rate=0.05, keep_elite=3)

                if opt == 'DE':
                    optimizer = DEForFS(size=6, dim=attr_size, max_iter=max_iter, pos_max=pos_max, pos_min=pos_min,
                                        raw_data=raw_data, classifier=classifier,
                                        F=1, CR=0.5)

                if opt == 'CS':
                    optimizer = CSForFS(size=6, dim=attr_size, max_iter=max_iter, pos_max=pos_max, pos_min=pos_min,
                                        raw_data=raw_data, classifier=classifier,
                                        pa=0.25, beta=1.5, step_scaling=0.1)

                # if opt == 'WOA':
                #     optimizer = WOAForFS(size=10, dim=attr_size, max_iter=max_iter, pos_max=pos_max, pos_min=pos_min,
                #                          raw_data=raw_data, classifier=classifier,
                #                          a=2, b=1)

                if opt == 'GWO':
                    optimizer = GWOForFS(size=6, dim=attr_size, max_iter=max_iter, pos_max=pos_max, pos_min=pos_min,
                                         raw_data=raw_data, classifier=classifier,
                                         a=2)

                if opt == 'HRO':
                    optimizer = HROForFS(size=6, dim=attr_size, max_iter=max_iter, r_max=pos_max, r_min=pos_min,
                                         raw_data=raw_data, classifier=classifier,
                                         max_self=10)

                if opt == 'IHRO':
                    optimizer = IHROForFS(size=6, dim=attr_size, max_iter=max_iter, r_max=pos_max, r_min=pos_min,
                                          raw_data=raw_data, classifier=classifier,
                                          max_self=10, beta=1.5, step_scaling=0.1)

                if opt == 'CC-IHRO M=2':
                    optimizer = CCIHROForFS(size=6, dim=attr_size, max_iter=max_iter, r_max=pos_max, r_min=pos_min,
                                            raw_data=raw_data, classifier=classifier,
                                            max_self=10, beta=1.5, step_scaling=0.1, M=2)

                if opt == 'CC-IHRO M=3':
                    optimizer = CCIHROForFS(size=6, dim=attr_size, max_iter=max_iter, r_max=pos_max, r_min=pos_min,
                                            raw_data=raw_data, classifier=classifier,
                                            max_self=10, beta=1.5, step_scaling=0.1, M=3)

                if opt == 'CC-IHRO M=4':
                    optimizer = CCIHROForFS(size=6, dim=attr_size, max_iter=max_iter, r_max=pos_max, r_min=pos_min,
                                            raw_data=raw_data, classifier=classifier,
                                            max_self=10, beta=1.5, step_scaling=0.1, M=4)

                if opt == 'CC-IHRO M=5':
                    optimizer = CCIHROForFS(size=6, dim=attr_size, max_iter=max_iter, r_max=pos_max, r_min=pos_min,
                                            raw_data=raw_data, classifier=classifier,
                                            max_self=10, beta=1.5, step_scaling=0.1, M=5)

                # 多次运行
                for time in range(times):
                    # 若算法是CC-IHRO 需要进行子种群的划分
                    if opt == 'CC-IHRO M=2' or opt == 'CC-IHRO M=3' or opt == 'CC-IHRO M=4' or opt == 'CC-IHRO M=5':
                        optimizer.subspace_subswarm()
                    # 初始化
                    optimizer.initial()
                    # 开始迭代
                    optimizer.optimal()
                    # 收敛结果
                    print('--------第', time + 1, '次收敛结果为：', optimizer.final_result)
                    # 运行结果的索引
                    result_index = datasets.index(dataset) * (len(classifiers) * (times + 1) + 1) + classifiers.index(classifier) * (times + 1) + time
                    # 储存运行结果
                    results_data[opt].iloc[result_index] = optimizer.final_result
                    # results_data[opt].iloc[result_index] = Counter(optimizer.g_best_binary_pos)[1.0]
                    # 储存收敛过程
                    records_data.iloc[:, result_index] = optimizer.fit_record

                    # 因为NSL-KDD每次运行时间较长且总运行次数较少，因此考虑每运行一次都储存一次结果
                    if dataset == 'NSL-KDD':
                        # 为储存创建文件名
                        file_name = 'result' + '_' + ts + '_' + opt + '_' + str(time) + '.csv'
                        # 拼接要储存的数据
                        fit_record = pd.DataFrame(optimizer.fit_record)
                        selected_attr = pd.DataFrame(optimizer.g_best_binary_pos)
                        data = pd.concat([fit_record, selected_attr])
                        # 转存到CSV文件
                        data.to_csv(file_name, header=[opt])
                    print(python_time.perf_counter())

        # 将当前算法的收敛过程存入Excel文件
        records_data.to_excel(records_writer, sheet_name=opt)

        print('==============================')
        print(opt, '测试结束')
        print('==============================')

    # 测试结束
    # 将所有算法的运行结果存入Excel文件
    results_data.to_excel(results_writer)
    # 关闭操作Excel文件的句柄
    results_writer.close()
    records_writer.close()

    # 输出程序运行时间
    print(python_time.perf_counter())





















