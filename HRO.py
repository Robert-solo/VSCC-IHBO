import numpy as np
import pandas as pd
import random
import time
import math


import UCI_Classify
import common_tool


# 创建用于特征选择的HRO类
class HROForFS:
    # 构造方法
    def __init__(self, size, dim, max_self, r_max, r_min, max_iter, raw_data, classifier):
        # 种群大小
        self.size = size
        # 维度
        self.dim = dim
        # 自交阈值
        self.max_self = max_self
        # 搜索域上限
        self.r_max = r_max
        # 搜索域下限
        self.r_min = r_min
        # 最大迭代次数
        self.max_iter = max_iter
        # 个体位置、自交次数和适应度值
        self.pos_time_bpos_fit = None
        # 全局最优位置
        self.g_best_pos = None
        # 全局最优适应度值
        self.g_best_fit = None
        # 记录每一代的最优适应度值
        self.fit_record = None
        # 最后的收敛结果
        self.final_result = None

        # --- 特征选择所需二进制变量 --- #
        # 全局最优二进制位置
        self.g_best_binary_pos = None
        # 进行测试的数据集
        self.raw_data = raw_data
        # 测试使用的分类器
        self.classifier = classifier

    # 种群初始化
    def initial(self):
        # 将部分关键变量重置
        # 个体位置、自交次数和适应度值
        self.pos_time_bpos_fit = np.zeros((self.size, self.dim * 2 + 2))
        '''
            说明：
            self.pos_time_bpos_fit为储存个体位置、自交次数、二进制位置和适应度值的二维数组
            根据数组下标，具体储存安排如下
            0 ~ dim-1       --> 问题的解
            dim             --> 自交次数
            dim+1 ~ dim*2   --> 二进制位置
            -1              --> 适应度值
        '''
        # 全局最优位置
        self.g_best_pos = None
        # 全局最优适应度值
        self.g_best_fit = None
        # 记录每一代的最优适应度值
        self.fit_record = np.zeros(self.max_iter + 1)
        # 最后的收敛结果
        self.final_result = None

        # --- 特征选择所需二进制变量 --- #
        # 全局最优二进制位置
        self.g_best_binary_pos = None

        # 随机生成每个个体的初始位置
        for i in range(self.size):
            for j in range(self.dim):
                self.pos_time_bpos_fit[i, j] = random.uniform(self.r_min, self.r_max)
                # 将粒子位置通过sigmoid函数进行二进制转换
                self.pos_time_bpos_fit[i, self.dim+1+j] = common_tool.binary_func(self.pos_time_bpos_fit[i, j])
            # 记录个体的初始适应度值
            self.pos_time_bpos_fit[i, -1] = UCI_Classify.fs_classify(self.pos_time_bpos_fit[i, self.dim+1:self.dim*2+1], self.raw_data, self.classifier)

        # 根据适应度值对种群进行排序
        self.pos_time_bpos_fit = self.pos_time_bpos_fit[np.argsort(-self.pos_time_bpos_fit[:, -1])]
        # 记录全局初始最优位置和适应度值
        self.g_best_fit = self.pos_time_bpos_fit[0, -1]
        self.g_best_pos = self.pos_time_bpos_fit[0, :self.dim].copy()    # deep copy
        self.g_best_binary_pos = self.pos_time_bpos_fit[0, self.dim+1:self.dim*2+1].copy()    # deep copy
        self.fit_record[0] = self.g_best_fit

        # print('初始最优适应度值为：')
        # print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
        # 开始迭代
        for iter_count in range(self.max_iter):
            # 创建保持系 恢复系 不育系索引
            maintainer_index = np.arange(0, int(self.size/3))
            restorer_index = np.arange(int(self.size/3), int(self.size/3)*2)
            sterile_index = np.arange(int(self.size/3)*2, self.size)

            # 保持系与不育系进行杂交
            for index in sterile_index:
                # 初始化杂交后产生的新不育个体
                new_sterile = np.zeros(self.dim * 2 + 2)
                # 随机选择一个保持系个体
                selected_maintainer = self.pos_time_bpos_fit[random.choice(maintainer_index)]
                # 随机选择一个不育系个体
                selected_sterile = self.pos_time_bpos_fit[random.choice(sterile_index)]
                # 开始杂交过程
                for i in range(self.dim):
                    # 生成随机数r1 r2
                    r1 = random.uniform(-1, 1)    # 原始
                    # r1 = random.uniform(0, 1)    # 改进
                    r2 = random.uniform(-1, 1)
                    # 根据所定义的公式进行杂交
                    new_sterile[i] = (r1 * selected_maintainer[i] + r2 * selected_sterile[i]) / (r1 + r2)    # 原始
                    # new_sterile[i] = (r1 * selected_maintainer[i] + (1 - r1) * selected_sterile[i])   # 改进1
                    # 判断个体位置是否会越界
                    new_sterile[i] = common_tool.bound_check(new_sterile[i], self.r_max, self.r_min)
                    # 进行二进制转换
                    new_sterile[self.dim+1+i] = common_tool.binary_func(new_sterile[i])
                # 计算新个体的适应度值
                new_sterile[-1] = UCI_Classify.fs_classify(new_sterile[self.dim+1:self.dim*2+1], self.raw_data, self.classifier)
                # 如果新个体的适应度值优于当前不育系个体，则替换之
                if new_sterile[-1] > self.pos_time_bpos_fit[index, -1]:
                    self.pos_time_bpos_fit[index] = new_sterile

            # 恢复系自交或重置
            for index in restorer_index:
                # 判断当前个体自交次数是否已达上限
                if self.pos_time_bpos_fit[index, self.dim] < self.max_self:
                    # 若自交次数未达上限
                    # 初始化自交后产生的新恢复个体
                    new_restorer = np.zeros(self.dim * 2 + 2)
                    # 开始自交过程
                    for i in range(self.dim):
                        # 随机选择一个恢复系个体（与当前个体不重复）
                        selected_restorer = self.pos_time_bpos_fit[random.choice(restorer_index[restorer_index != index])]
                        # 生成随机数r3
                        r3 = random.uniform(0, 1)
                        # 根据所定义的公式进行自交
                        new_restorer[i] = r3 * (self.g_best_pos[i] - selected_restorer[i]) + self.pos_time_bpos_fit[index, i]
                        # 判断个体位置是否会越界
                        new_restorer[i] = common_tool.bound_check(new_restorer[i], self.r_max, self.r_min)
                        # 进行二进制转换
                        new_restorer[self.dim + 1 + i] = common_tool.binary_func(new_restorer[i])
                    # 计算新个体的适应度值
                    new_restorer[-1] = UCI_Classify.fs_classify(new_restorer[self.dim+1:self.dim*2+1], self.raw_data, self.classifier)
                    # 判断新生成的个体适应度值是否优于之前的个体
                    if new_restorer[-1] > self.pos_time_bpos_fit[index, -1]:
                        # 如若优于，则替换之
                        self.pos_time_bpos_fit[index] = new_restorer
                        # 同时该个体自交次数置0
                        self.pos_time_bpos_fit[index, self.dim] = 0
                    else:
                        # 如若未优于，则个体自交次数+1
                        self.pos_time_bpos_fit[index, self.dim] = self.pos_time_bpos_fit[index, self.dim] + 1
                else:
                    # 若自交次数已达上限
                    # 进行重置操作
                    for i in range(self.dim):
                        # 随机生成新个体并进行二进制转换
                        self.pos_time_bpos_fit[index, i] = random.uniform(self.r_min, self.r_max)
                        self.pos_time_bpos_fit[index, self.dim+1+i] = common_tool.binary_func(self.pos_time_bpos_fit[index, i])
                    # 重新计算该个体的适应度值
                    self.pos_time_bpos_fit[index, -1] = UCI_Classify.fs_classify(self.pos_time_bpos_fit[index, self.dim+1:self.dim*2+1], self.raw_data, self.classifier)
                    # 将该个体自交次数置0
                    self.pos_time_bpos_fit[index, self.dim] = 0

            # 当前迭代完成，根据适应度值对种群重新排序
            self.pos_time_bpos_fit = self.pos_time_bpos_fit[np.argsort(-self.pos_time_bpos_fit[:, -1])]
            # 判断是否需要更新全局最优
            if self.pos_time_bpos_fit[0, -1] > self.g_best_fit:
                # 更新全局最优位置和适应度值
                self.g_best_fit = self.pos_time_bpos_fit[0, -1]
                self.g_best_pos = self.pos_time_bpos_fit[0, :self.dim].copy()    # deep copy
                self.g_best_binary_pos = self.pos_time_bpos_fit[0, self.dim + 1:self.dim * 2 + 1].copy()    # deep copy

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


# # 设置HRO的各项参数
# hro = HROForFS(size=30, dim=attr_size, max_self=10, r_max=10, r_min=-10, max_iter=100)
#
# for t in range(100):
#     hro.initial()
#     if hro.g_best_fit > 0.96:
#         continue
#     hro.optimal()
#     print(hro.g_best_fit)
#     print(hro.g_best_binary_pos)
#     print(hro.fit_record)
#
# print(time.perf_counter())
























