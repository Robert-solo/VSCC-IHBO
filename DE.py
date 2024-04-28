import numpy as np
import pandas as pd
import random
import time
import math


import UCI_Classify
import common_tool


# 创建用于特征选择的DE类
class DEForFS:
    # 构造方法
    def __init__(self, size, dim, pos_max, pos_min, max_iter, F, CR, raw_data, classifier):
        # 种群大小
        self.size = size
        # 维度
        self.dim = dim
        # 搜索域上限
        self.pos_max = pos_max
        self.pos_min = pos_min
        # 迭代次数
        self.max_iter = max_iter
        # 缩放比例
        self.F = F
        # 交叉概率
        self.CR = CR
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
            # 对每个个体进行变异、交叉、选择操作
            for i in range(self.size):
                # Mutation
                # 记录target_vector
                target_vector = self.pos[i].copy()    # deep copy
                # 随机选取r1,r2,r3，需与i不同
                r_list = []
                # 循环选取
                while len(r_list) < 3:
                    # 随机选取一个数
                    r_temp = random.randint(0, self.size-1)
                    # 若该数不与i相同
                    if r_temp != i:
                        # 则将该数添加进被选数组
                        r_list.append(r_temp)
                # r1,r2,r3
                r1 = r_list[0]
                r2 = r_list[1]
                r3 = r_list[2]
                # 生成mutant_vector
                mutant_vector = self.pos[r1] + self.F * (self.pos[r2] - self.pos[r3])

                # Crossover
                # 创建trial_vector和trial_vector_binary
                trial_vector = np.zeros(self.dim)
                trial_vector_binary = np.zeros(self.dim)
                # 随机生成 rnbr
                rnbr = random.randint(0, self.dim-1)
                # 开始交叉过程
                for j in range(self.dim):
                    # 生成决定是否交叉的随机数
                    randb = random.uniform(0, 1)
                    # 判断是否进行交叉操作
                    if randb <= self.CR or j == rnbr:
                        # 进行交叉操作
                        trial_vector[j] = mutant_vector[j]
                    else:
                        # 不进行交叉操作
                        trial_vector[j] = target_vector[j]
                    # 进行边界检查和二进制转换
                    trial_vector[j] = common_tool.bound_check(trial_vector[j], self.pos_max, self.pos_min)
                    trial_vector_binary[j] = common_tool.binary_func(trial_vector[j])

                # Selection
                # 记录target_vector的适应度值
                target_vector_fit = self.fit[i]
                # 计算trial_vector的适应度值
                trial_vector_fit = UCI_Classify.fs_classify(trial_vector_binary, self.raw_data, self.classifier)
                # 比较双方的适应度值
                if trial_vector_fit > target_vector_fit:
                    # 若trial_vector适应度值优于target_vector，则替换之
                    self.pos[i] = trial_vector.copy()    # deep copy
                    self.binary_pos[i] = trial_vector_binary.copy()    # deep copy
                    # 并同时替换适应度值
                    self.fit[i] = trial_vector_fit
                    # 比较更新的适应度值与全局最优适应度值
                    if trial_vector_fit > self.g_best_fit:
                        # 更新全局最优适应度值及位置
                        self.g_best_fit = trial_vector_fit
                        self.g_best_pos = trial_vector.copy()    # deep copy
                        self.g_best_binary_pos = trial_vector_binary.copy()    # deep copy

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
# de = DEForFS(size=30, dim=attr_size, pos_max=10, pos_min=-10, max_iter=10, F=1, CR=0.5)
#
# for t in range(10):
#     de.initial()
#     de.optimal()
#     print(de.g_best_fit)
#     print(de.g_best_binary_pos)
#
# print(time.perf_counter())





















