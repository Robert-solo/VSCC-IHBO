import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
import sklearn.metrics

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# 编程期间辅助设置
# 显示所有列
# pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


# 支持特征选择的UCI分类
def fs_classify(binary_pos, raw_data, classifier):
    # 根据二进制位置信息进行特征选择
    selected_attr = []
    for i in range(binary_pos.size):
        # 1代表选择该特征，0代表不选
        if binary_pos[i] == 1:
            selected_attr.append(i)

    # 判断当前所测试的数据集
    if isinstance(raw_data, list):
        # 若是入侵检测数据集
        train_set = raw_data[0]
        test_set = raw_data[1]
        train_set_x = train_set.iloc[:, selected_attr]
        train_set_y = train_set.iloc[:, -1]
        test_set_x = test_set.iloc[:, selected_attr]
        test_set_y = test_set.iloc[:, -1]
    else:
        # 若是UCI数据集
        sample_data = raw_data.iloc[:, selected_attr]
        target_data = raw_data.iloc[:, -1]
        # 划分训练集和测试集
        train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(sample_data, target_data, test_size=0.5)

    # 预测结果
    predict_y = None

    # 根据参数判断使用哪一种分类器
    if classifier == 'XGB':
        # 设置XGBoost参数
        xgb = XGBClassifier(tree_method='gpu_hist')
        # 利用XGBoost进行训练
        xgb.fit(train_set_x, train_set_y)
        # 利用XGBoost进行测试
        predict_y = xgb.predict(test_set_x)

    if classifier == 'KNN':
        # 设置KNN参数
        knn = KNeighborsClassifier()
        # 利用KNN进行训练
        knn.fit(train_set_x, train_set_y)
        # 利用KNN进行分类
        predict_y = knn.predict(test_set_x)

    if classifier == 'SVM':
        # 设置SVC参数
        scv = SVC()
        # 利用KNN进行训练
        scv.fit(train_set_x, train_set_y)
        # 利用KNN进行分类
        predict_y = scv.predict(test_set_x)

    # 获取本次分类的各项指标
    accuracy_score = sklearn.metrics.accuracy_score(test_set_y, predict_y)

    # Classification Report
    # print('XGBoost Classification Report:\n')
    # print(sklearn.metrics.classification_report(test_set_y, predict_y))
    # print('\n')

    # precision_score_all = sklearn.metrics.precision_score(test_set_y, predict_y, average=None)
    # recall_score_all = sklearn.metrics.recall_score(test_set_y, predict_y, average=None)
    # f1_score_all = sklearn.metrics.f1_score(test_set_y, predict_y, average=None)
    # precision_score_macro = sklearn.metrics.precision_score(test_set_y, predict_y, average='macro')
    # recall_score_macro = sklearn.metrics.recall_score(test_set_y, predict_y, average='macro')
    # f1_score_macro = sklearn.metrics.f1_score(test_set_y, predict_y, average='macro')
    #
    # detail_data = []
    # for score in precision_score_all:
    #     detail_data.append(score)
    # detail_data.append('')
    # for score in recall_score_all:
    #     detail_data.append(score)
    # detail_data.append('')
    # for score in f1_score_all:
    #     detail_data.append(score)
    # detail_data.append('')
    # detail_data.append(precision_score_macro)
    # detail_data.append(recall_score_macro)
    # detail_data.append(f1_score_macro)
    #
    # return detail_data

    return accuracy_score


# 基本的UCI分类
def basic_classify(raw_data, classifier):
    # 判断当前所测试的数据集
    if isinstance(raw_data, list):
        # 若是入侵检测数据集
        train_set = raw_data[0]
        test_set = raw_data[1]
        # 获取特征数量
        attr_size = train_set.shape[1] - 1
        # 划分训练集和测试集
        train_set_x = train_set.iloc[:, :attr_size]
        train_set_y = train_set.iloc[:, -1]
        test_set_x = test_set.iloc[:, :attr_size]
        test_set_y = test_set.iloc[:, -1]
    else:
        # 若是UCI数据集
        # 获取特征数量
        attr_size = raw_data.shape[1] - 1
        sample_data = raw_data.iloc[:, :attr_size]
        target_data = raw_data.iloc[:, -1]
        # 划分训练集和测试集
        train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(sample_data, target_data, test_size=0.2)

    # 预测结果
    predict_y = None

    # 根据参数判断使用哪一种分类器
    if classifier == 'XGB':
        # 设置XGBoost参数
        xgb = XGBClassifier(tree_method='gpu_hist')
        # 利用XGBoost进行训练
        xgb.fit(train_set_x, train_set_y)
        # 利用XGBoost进行测试
        predict_y = xgb.predict(test_set_x)

    if classifier == 'KNN':
        # 设置KNN参数
        knn = KNeighborsClassifier()
        # 利用KNN进行训练
        knn.fit(train_set_x, train_set_y)
        # 利用KNN进行分类
        predict_y = knn.predict(test_set_x)

    if classifier == 'SVM':
        # 设置SVC参数
        scv = SVC()
        # 利用KNN进行训练
        scv.fit(train_set_x, train_set_y)
        # 利用KNN进行分类
        predict_y = scv.predict(test_set_x)

    # 获取本次分类的各项指标
    macro_precision = sklearn.metrics.precision_score(test_set_y, predict_y, average='macro')
    macro_recall = sklearn.metrics.recall_score(test_set_y, predict_y, average='macro')
    macro_f1_score = sklearn.metrics.f1_score(test_set_y, predict_y, average='macro')

    accuracy_score = sklearn.metrics.accuracy_score(test_set_y, predict_y)
    print(accuracy_score)

    # Classification Report
    print('XGBoost Classification Report:\n')
    print(sklearn.metrics.classification_report(test_set_y, predict_y))
    print('\n')


# 获取特征选择后的具体数据
def detailData(raw_data, classifier):
    records = pd.read_csv('input/Selected_Features.csv', header=None)
    detail_data_all = []
    for index, row in records.iterrows():
        binary_pos = np.array(row)
        detail_data = fs_classify(binary_pos, raw_data, classifier)
        detail_data_all.append(detail_data)

    # 为储存创建文件名
    file_name = 'detail_data.csv'
    # 拼接要储存的数据
    data = pd.DataFrame(detail_data_all)
    # 转存到CSV文件
    header = ['precision of normal', 'precision of dos', 'precision of probe', 'precision of u2r', 'precision of r2l', '',
              'recall of normal', 'recall of dos', 'recall of probe', 'recall of u2r', 'recall of r2l', '',
              'f1score of normal', 'f1score of dos', 'f1score of probe', 'f1score of u2r', 'f1score of r2l', '',
              'marco precision', 'marco recall', 'marco f1score']
    data.to_csv(file_name, header=header)


if __name__ == '__main__':

    # 要测试的数据集
    # datasets = ['Hill-Valley', 'Ionosphere', 'Libras', 'MUSK1', 'Sonar', 'Urban', 'LSVT', 'WDBC']
    datasets = ['NSL-KDD']

    for dataset in datasets:
        # 从CSV文件中读取样本数据
        # raw_data = pd.read_csv('input/' + dataset + '.csv', header=None)

        # NSL-KDD
        train_set = pd.read_csv('input/KDDTrain+.csv', header=None)
        test_set = pd.read_csv('input/KDDTest+.csv', header=None)
        # 获取特征数量
        attr_size = train_set.shape[1] - 1
        raw_data = [train_set, test_set]

        # # 已经选好的最优特征子集
        # raw_pos = ''
        # raw_pos = raw_pos.replace('\t', '')
        # pos = []
        # for binary in raw_pos:
        #     pos.append(int(binary))
        # binary_pos = np.array(pos)

        for t in range(1):
            # basic_classify(raw_data, 'XGB')
            detailData(raw_data, 'XGB')

    print(time.perf_counter())






















