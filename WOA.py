import numpy as np
import pandas as pd
import random
import time
import math


import UCI_Classify
import common_tool


# 创建用于特征选择的WOA类
class WOAForFS:
    # 构造函数
    def __init__(self, size, dim, pos_max, pos_min, max_iter, a, b, raw_data, classifier):
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
        # 螺旋更新参数b
        self.b = b
        # 个体位置信息
        self.pos = None
        # 个体适应度信息
        self.fit = None
        # 全局最优位置
        self.g_best_pos = None
        # 全局最优适应度值
        self.g_best_fit = None
        # 记录每一代的最优适应度值
        self.fit_record = None
        # 最后的收敛结果
        self.final_result = None

        # --- 特征选择所需二进制变量 --- #
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
        # 全局最优位置
        self.g_best_pos = None
        # 全局最优适应度值
        self.g_best_fit = None
        # 记录每一代的最优适应度值
        self.fit_record = np.zeros(self.max_iter + 1)
        # 最后的收敛结果
        self.final_result = None

        # --- 特征选择所需二进制变量 --- #
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

        # 记录初始全局最优下标、位置和适应度值
        max_index = np.argsort(-self.fit)[0]
        self.g_best_pos = self.pos[max_index].copy()    # deep copy
        self.g_best_binary_pos = self.binary_pos[max_index].copy()    # deep copy
        self.g_best_fit = self.fit[max_index]
        self.fit_record[0] = self.g_best_fit

        # print('初始最优适应度值为：')
        # print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
        # 开始迭代
        for iter_count in range(self.max_iter):
            # 确定线性递减参数a1 a2
            a1 = self.a - iter_count * (self.a / self.max_iter)
            a2 = -1 + iter_count * (-1 / self.max_iter)

            # 利用每个个体开始寻优
            for i in range(self.size):
                # 计算当前个体的参数A
                A = 2 * a1 * random.uniform(0, 1) - a1
                # 计算当前个体的参数C
                C = 2 * random.uniform(0, 1)
                # 生成随机数p
                p = random.uniform(0, 1)

                # 判断当前所要进行的操作
                if p < 0.5:
                    # Encircling Prey 或 Search for Prey
                    if abs(A) < 1:
                        # Encircling Prey
                        # 对个体中的每个位置进行操作
                        for j in range(self.dim):
                            # 计算参数D
                            D = abs(C * self.g_best_pos[j] - self.pos[i, j])
                            # 更新后的位置
                            self.pos[i, j] = self.g_best_pos[j] - A * D
                    else:
                        # Search for Prey
                        # 随机选择一个个体
                        rand_index = random.randint(0, self.size - 1)
                        # 对个体中的每个位置进行操作
                        for j in range(self.dim):
                            # 计算参数D
                            D = abs(C * self.pos[rand_index, j] - self.pos[i, j])
                            # 更新后的位置
                            self.pos[i, j] = self.pos[rand_index, j] - A * D
                else:
                    # Attacking
                    # 生成随机数l
                    l = (a2 - 1) * random.uniform(0, 1) + 1
                    # 对个体中的每个位置进行操作
                    for j in range(self.dim):
                        # 计算参数D
                        D = abs(self.g_best_pos[j] - self.pos[i, j])
                        # 更新后的位置
                        self.pos[i, j] = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.g_best_pos[j]

                # 判断新生成的位置是否越界并进行二进制转换
                for j in range(self.dim):
                    self.pos[i, j] = common_tool.bound_check(self.pos[i, j], self.pos_max, self.pos_min)
                    self.binary_pos[i, j] = common_tool.binary_func(self.pos[i, j])

                # 计算当前个体的适应度值
                # 鉴于WOA容易越界的问题，需要特殊处理防止一个特征都不选
                if self.binary_pos[i].any() is True:
                    # 起码选取一个特征才进行分类
                    curr_fit = UCI_Classify.fs_classify(self.binary_pos[i], self.raw_data, self.classifier)
                    # 如果当前个体的适应度值优于全局最优适应度值
                    if curr_fit > self.g_best_fit:
                        # 替换全局最优位置和最优适应度值
                        self.g_best_pos = self.pos[i].copy()    # deep copy
                        self.g_best_binary_pos = self.binary_pos[i].copy()    # deep copy
                        self.g_best_fit = curr_fit

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
# woa = WOAForFS(size=60, dim=attr_size, pos_max=10, pos_min=-10, max_iter=100, a=2, b=1)
#
# for t in range(10):
#     woa.initial()
#     woa.optimal()
#     print(woa.g_best_fit)
#     print(woa.g_best_binary_pos)
#
# print(time.perf_counter())
















