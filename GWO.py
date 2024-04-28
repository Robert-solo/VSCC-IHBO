import numpy as np
import pandas as pd
import random
import time
import math


import UCI_Classify
import common_tool


# 创建用于特征选择的GWO类
class GWOForFS:
    # 构造函数
    def __init__(self, size, dim, pos_max, pos_min, max_iter, a, raw_data, classifier):
        # 种群大小
        self.size = size
        # 维度
        self.dim = dim
        # 搜索域上限
        self.pos_max = pos_max
        self.pos_min = pos_min
        # 迭代次数
        self.max_iter = max_iter
        # 线性递减参数a
        self.a = a
        # 个体位置信息
        self.pos = None
        # 个体适应度信息
        self.fit = None
        # Alpha Beta Delta 的位置和适应度值
        self.alpha_pos = None
        self.alpha_fit = None
        self.beta_pos = None
        self.beta_fit = None
        self.delta_pos = None
        self.delta_fit = None
        # 记录每一代的最优适应度值
        self.fit_record = None
        # 最后的收敛结果
        self.final_result = None

        # --- 特征选择所需二进制变量 --- #
        # 全局最优适应度值
        self.g_best_fit = None
        # 转换为二进制的位置
        self.binary_pos = None
        # 全局最优二进制位置
        self.g_best_binary_pos = None
        # 进行测试的数据集
        self.raw_data = raw_data
        # 测试使用的分类器
        self.classifier = classifier

    # 种群初始化
    def initial(self):
        # 将部分关键变量重置
        # 个体位置信息
        self.pos = np.zeros((self.size, self.dim))
        # 个体适应度信息
        self.fit = np.zeros(self.size)
        # Alpha Beta Delta 的位置和适应度值
        self.alpha_pos = None
        self.alpha_fit = None
        self.beta_pos = None
        self.beta_fit = None
        self.delta_pos = None
        self.delta_fit = None
        # 记录每一代的最优适应度值
        self.fit_record = np.zeros(self.max_iter + 1)
        # 最后的收敛结果
        self.final_result = None

        # --- 特征选择所需二进制变量 --- #
        # 全局最优适应度值
        self.g_best_fit = None
        # 粒子二进制的位置
        self.binary_pos = np.zeros((self.size, self.dim))
        # 全局最优二进制位置
        self.g_best_binary_pos = None

        # 随机生成每个个体的初始位置
        for i in range(self.size):
            for j in range(self.dim):
                self.pos[i, j] = random.uniform(self.pos_min, self.pos_max)
                # 将粒子位置通过sigmoid函数进行二进制转换
                self.binary_pos[i, j] = common_tool.binary_func(self.pos[i, j])
            # 利用二进制位置进行特征选择得到适应度值
            self.fit[i] = UCI_Classify.fs_classify(self.binary_pos[i], self.raw_data, self.classifier)

        # 明确Alpha Beta Delta个体
        alpha_index = np.argsort(-self.fit)[0]
        beta_index = np.argsort(-self.fit)[1]
        delta_index = np.argsort(-self.fit)[2]
        # Alpha
        self.alpha_pos = self.pos[alpha_index].copy()    # deep copy
        self.alpha_fit = self.fit[alpha_index]
        # Beta
        self.beta_pos = self.pos[beta_index].copy()    # deep copy
        self.beta_fit = self.fit[beta_index]
        # Delta
        self.delta_pos = self.pos[delta_index].copy()    # deep copy
        self.delta_fit = self.fit[delta_index]
        # 记录初始全局最优
        self.g_best_fit = self.alpha_fit
        self.g_best_binary_pos = self.binary_pos[alpha_index].copy()    # deep copy
        self.fit_record[0] = self.g_best_fit

        # print('初始最优适应度值为：')
        # print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
        # 开始迭代
        for iter_count in range(self.max_iter):
            # 更新线性递减参数a
            a = self.a - iter_count * (self.a / self.max_iter)

            # 更新种群位置
            for i in range(self.size):
                # 更新个体位置
                for j in range(self.dim):
                    # X1
                    r1 = random.random()
                    r2 = random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.pos[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    # X2
                    r1 = random.random()
                    r2 = random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.pos[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    # X3
                    r1 = random.random()
                    r2 = random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.pos[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    # 更新位置
                    self.pos[i, j] = (X1 + X2 + X3) / 3
                    # 判断更新后的位置是否越界并进行二进制转换
                    self.pos[i, j] = common_tool.bound_check(self.pos[i, j], self.pos_max, self.pos_min)
                    self.binary_pos[i, j] = common_tool.binary_func(self.pos[i, j])

                # 计算个体位置更新后的适应度值
                self.fit[i] = UCI_Classify.fs_classify(self.binary_pos[i], self.raw_data, self.classifier)

            # 更新Alpha Beta Delta个体
            alpha_index = np.argsort(-self.fit)[0]
            beta_index = np.argsort(-self.fit)[1]
            delta_index = np.argsort(-self.fit)[2]
            # Alpha
            if self.alpha_fit < self.fit[alpha_index]:
                self.alpha_pos = self.pos[alpha_index].copy()    # deep copy
                self.alpha_fit = self.fit[alpha_index]
            # Beta
            if self.beta_fit < self.fit[beta_index]:
                self.beta_pos = self.pos[beta_index].copy()    # deep copy
                self.beta_fit = self.fit[beta_index]
            # Delta
            if self.delta_fit < self.fit[delta_index]:
                self.delta_pos = self.pos[delta_index].copy()    # deep copy
                self.delta_fit = self.fit[delta_index]

            # 判断当前种群最优适应度值是否优于全局最优
            if self.alpha_fit > self.g_best_fit:
                # 若优于，则更新全局最优位置和适应度值
                self.g_best_binary_pos = self.binary_pos[alpha_index].copy()    # deep copy
                self.g_best_fit = self.alpha_fit

            # 输出本次迭代的全局最优位置和适应度值
            # print('当前迭代次数：', iter_count + 1)
            # print(self.g_best_fit)
            # 记录本次迭代的最优适应度值
            self.fit_record[iter_count + 1] = self.g_best_fit
            # 本次迭代结束，判断是否提前收敛
            if self.g_best_fit == 1:
                # 若准确率为100%则可以提前结束实验
                # print('--------本次迭代提前结束--------')
                # 将fit_record剩余部分全部置为1
                self.fit_record[self.fit_record == 0] = 1
                break

        # 迭代寻优结束，记录最终结果
        self.final_result = self.fit_record[-1]


# # 从CSV文件中读取样本数据
# raw_data = pd.read_csv('input/Ionosphere.csv', header=None)
# # 获取特征数量
# attr_size = raw_data.shape[1] - 1
#
#
# gwo = GWOForFS(size=30, dim=attr_size, pos_max=10, pos_min=-10, max_iter=10, a=2, raw_data=raw_data, classifier='KNN')
#
# for t in range(10):
#     gwo.initial()
#     gwo.optimal()
#     print(gwo.alpha_fit)
#     print(gwo.g_best_binary_pos)
#
# print(time.perf_counter())











