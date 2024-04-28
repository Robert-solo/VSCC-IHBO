import numpy as np
import pandas as pd
import random
import time
import math


import UCI_Classify
import common_tool


# 创建用于特征选择的CS类
class CSForFS:
    # 构造方法
    def __init__(self, size, dim, pos_max, pos_min, max_iter, pa, beta, step_scaling, raw_data, classifier):
        # 种群大小
        self.size = size
        # 维度
        self.dim = dim
        # 搜索域
        self.pos_max = pos_max
        self.pos_min = pos_min
        # 迭代次数
        self.max_iter = max_iter
        # 鸟蛋被发现概率
        self.pa = pa
        # beta参数
        self.beta = beta
        # 步长缩放系数
        self.step_scaling = step_scaling
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
            # 更新种群位置
            # 使用列维飞行生成步长
            step_length = common_tool.levy_flight(self.size, self.dim, self.beta)
            # 随机生成飞行方向
            step_direction = np.random.rand(self.size, self.dim)
            # 逐个更新个体位置
            for i in range(self.size):
                # 根据步长和方向更新位置
                new_pos = self.pos[i] + self.step_scaling * step_length[i] * step_direction[i] * (self.pos[i] - self.g_best_pos)
                # 创建二进制的新生成个体
                new_binary_pos = np.zeros(self.dim)
                # 对新生成的个体进行边界检查和二进制转换
                for j in range(self.dim):
                    new_pos[j] = common_tool.bound_check(new_pos[j], self.pos_max, self.pos_min)
                    new_binary_pos[j] = common_tool.binary_func(new_pos[j])
                # 计算新个体的适应度值
                new_fit = UCI_Classify.fs_classify(new_binary_pos, self.raw_data, self.classifier)
                # 若新个体优于老个体则进行替换
                if new_fit > self.fit[i]:
                    self.pos[i] = new_pos
                    self.fit[i] = new_fit
                    self.binary_pos[i] = new_binary_pos

            # 抛弃部分个体
            for i in range(self.size):
                # 判断是否要抛弃当前个体
                if np.random.rand() < self.pa:
                    # 为原个体加上一个随机步长
                    self.pos[i] += np.random.rand() * (self.pos[np.random.randint(0, self.size)] - self.pos[np.random.randint(0, self.size)])
                    # 进行边界检查和二进制转换
                    for j in range(self.dim):
                        self.pos[i][j] = common_tool.bound_check(self.pos[i][j], self.pos_max, self.pos_min)
                        self.binary_pos[i][j] = common_tool.binary_func(self.pos[i][j])
                    # 计算适应度值
                    self.fit[i] = UCI_Classify.fs_classify(self.binary_pos[i], self.raw_data, self.classifier)

            # 判断当前种群最优适应度值是否优于全局最优
            if np.max(self.fit) > self.g_best_fit:
                # 若优于，则更新全局最优位置和适应度值
                max_index = np.argmax(self.fit)
                self.g_best_pos = self.pos[max_index].copy()    # deep copy
                self.g_best_binary_pos = self.binary_pos[max_index].copy()    # deep copy
                self.g_best_fit = self.fit[max_index]

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
# cs = CSForFS(size=30, dim=attr_size, pos_max=10, pos_min=-10, max_iter=10, pa=0.25, beta=1.5, step_scaling=0.1)
#
# for t in range(10):
#     cs.initial()
#     cs.optimal()
#     print(cs.g_best_fit)
#     print(cs.g_best_binary_pos)
#
# print(time.perf_counter())




















