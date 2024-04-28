import numpy as np
import pandas as pd
import random
import time
import math


import UCI_Classify
import common_tool


# 创建用于特征选择的GA类
class GAForFS:
    # 构造方法
    def __init__(self, size, dim, pos_max, pos_min, max_iter, raw_data, classifier,
                 select_type, cross_type, cross_rate, mutation_type, mutation_rate, keep_elite):
        # 种群大小
        self.size = size
        # 维度
        self.dim = dim
        # 搜索域上限
        self.pos_max = pos_max
        self.pos_min = pos_min
        # 迭代次数
        self.max_iter = max_iter
        # 选择类型
        self.select_type = select_type
        # 交叉类型
        self.cross_type = cross_type
        # 交叉概率
        self.cross_rate = cross_rate
        # 变异类型
        self.mutation_type = mutation_type
        # 变异概率
        self.mutation_rate = mutation_rate
        # 保留的精英数量
        self.keep_elite = keep_elite
        # 保留的精英个体下标
        self.keep_elite_index = None
        # 个体位置信息
        self.pos = None
        # 个体适应度信息
        self.fit = None
        # 全局最优下标
        self.g_best_index = None
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
        # 保留的精英个体下标
        self.keep_elite_index = np.zeros(self.keep_elite)
        # 个体位置信息
        self.pos = np.zeros((self.size, self.dim))
        # 个体适应度信息
        self.fit = np.zeros(self.size)
        # 全局最优下标
        self.g_best_index = None
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
        self.g_best_index = max_index
        self.g_best_pos = self.pos[max_index].copy()    # deep copy
        self.g_best_binary_pos = self.binary_pos[max_index].copy()    # deep copy
        self.g_best_fit = self.fit[max_index]
        self.fit_record[0] = self.g_best_fit
        # 记录初始保留精英及其下标
        self.keep_elite_index = np.argsort(-self.fit)[0:self.keep_elite]

        # print('初始最优适应度值为：')
        # print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
        # 开始迭代
        for iter_count in range(self.max_iter):
            # 执行选择操作
            if self.select_type == 'rws':
                # 轮盘赌选择
                self.roulette_wheel_selection()

            # 执行交叉操作
            if self.cross_type == 'spc':
                # 单点交叉
                self.single_point_crossover()

            # 执行变异操作
            if self.mutation_type == 'rm':
                # 单点变异
                self.random_mutation()

            # 将粒子位置通过sigmoid函数进行二进制转换
            for i in range(self.size):
                # 精英个体不进行二进制转换
                if i in self.keep_elite_index:
                    continue
                else:
                    for j in range(self.dim):
                        self.binary_pos[i, j] = common_tool.binary_func(self.pos[i, j])

            # 重新计算适应度值
            for i in range(self.size):
                self.fit[i] = UCI_Classify.fs_classify(self.binary_pos[i], self.raw_data, self.classifier)

            # 获取当前最优下标和适应度值
            max_index = np.argsort(-self.fit)[0]
            best_fit = self.fit[max_index]
            # 若当前最优适应度值优于全局最优
            if best_fit > self.g_best_fit:
                self.g_best_pos = self.pos[max_index].copy()    # deep copy
                self.g_best_binary_pos = self.binary_pos[max_index].copy()    # deep copy
                self.g_best_fit = best_fit
            # 更新需要保留的精英个体下标
            self.keep_elite_index = np.argsort(-self.fit)[0:self.keep_elite]

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

    # 轮盘赌选择
    def roulette_wheel_selection(self):
        # 计算每个个体的选择概率
        select_prob = self.fit / np.sum(self.fit)
        # 创建数组储存新被选中的个体
        selected_pos = np.zeros((self.size, self.dim))
        # 开始轮盘赌选择循环
        for i in range(self.size):
            # 判断当前个体是否是需要保留的精英个体
            if i in self.keep_elite_index:
                # 若当前个体是需要保留的精英个体
                # 直接将该个体添加到被选中个体的数组中
                selected_pos[i] = self.pos[i]
            else:
                # 若当前个体不是最优个体
                # 随机一个0到1之间的随机数
                random_num = random.uniform(0, 1)
                for j in range(self.size):
                    # 通过累计概率判断随机数落到了哪个区间
                    add_prob = np.sum(select_prob[:j + 1])
                    # 如果随机数小于当前累计概率，则说明随机数落在了当前区间
                    if random_num < add_prob:
                        # 添加新选中个体
                        selected_pos[i] = self.pos[j]
                        # 跳出当前循环
                        break
        # 选择过程结束后，用新位置信息数组替换原位置信息数组
        self.pos = selected_pos.copy()  # deep copy

    # 单点交叉
    def single_point_crossover(self):
        # 创建数组储存尚未参与杂交个体的下标
        uncross_index = list(range(0, self.size))
        # 为保留精英个体，移除精英个体的下标
        for index in self.keep_elite_index:
            uncross_index.remove(index)
        # 开始单点交叉循环
        while len(uncross_index) > 1:
            # 随机选择两个尚未参与杂交的个体
            chosen = random.sample(uncross_index, 2)
            # 将选中的个体移除出尚未参与杂交的数组
            uncross_index.remove(chosen[0])
            uncross_index.remove(chosen[1])
            # 根据交叉概率判断本次是否进行交叉
            cross_prob = random.uniform(0, 1)
            if cross_prob < self.cross_rate:
                # 随机要交叉的单点下标
                cross_index = random.randint(0, self.dim - 1)
                # 执行单点交叉
                self.pos[chosen[0], cross_index], self.pos[chosen[1], cross_index] = \
                self.pos[chosen[1], cross_index], self.pos[chosen[0], cross_index]

    # 单点变异
    def random_mutation(self):
        # 开始单点变异循环
        for i in range(self.size):
            # 需要保留的精英个体不参与变异
            if i in self.keep_elite_index:
                continue
            else:
                # 根据变异概率判断本个体是否进行变异
                mutation_prob = random.uniform(0, 1)
                if mutation_prob < self.mutation_rate:
                    # 随机要变异的单点下标
                    mutation_index = random.randint(0, self.dim - 1)
                    # 执行单点变异
                    self.pos[i, mutation_index] = random.uniform(self.pos_min, self.pos_max)


# # 从CSV文件中读取样本数据
# raw_data = pd.read_csv('input/Ionosphere.csv', header=None)
# # 获取特征数量
# attr_size = raw_data.shape[1] - 1
#
#
# # 设置GA的各项参数
# ga = GAForFS(size=30, dim=attr_size, pos_max=10, pos_min=-10, max_iter=10,
#              select_type='rws', cross_type='spc', cross_rate=0.8, mutation_type='rm', mutation_rate=0.05, keep_elite=3)
#
# for t in range(1):
#     ga.initial()
#     ga.optimal()
#     print(ga.g_best_fit)
#     print(ga.g_best_binary_pos)
#
# print(time.perf_counter())












