import numpy as np
import pandas as pd
import random
import time
import math


import UCI_Classify
import common_tool


# 创建用于特征选择的PSO类
class PSOForFS:
    # 构造方法
    def __init__(self, w_max, w_min, c1, c2, pos_max, pos_min, vel_max, vel_min, max_iter, ps_size, dim, raw_data, classifier):
        # 惯性权重
        self.w_max = w_max
        self.w_min = w_min
        # 加速度因子
        self.c1 = c1
        self.c2 = c2
        # 搜索域
        self.pos_max = pos_max
        self.pos_min = pos_min
        # 速度域
        self.vel_max = vel_max
        self.vel_min = vel_min
        # 迭代次数
        self.max_iter = max_iter
        # 种群规模
        self.ps_size = ps_size
        # 维度
        self.dim = dim
        # 粒子群位置和速度
        self.p_pos = None
        self.p_vel = None
        # 粒子最优位置
        self.p_best_pos = None
        # 全局最优位置
        self.g_best_pos = None
        # 粒子最优适应度值
        self.p_best_fit = None
        # 全局最优适应度值
        self.g_best_fit = None
        # 记录每一代的最优适应度值
        self.fit_record = None
        # 最后的收敛结果
        self.final_result = None

        # --- 特征选择所需二进制变量 --- #
        # 转换为二进制的位置
        self.p_binary_pos = None
        # 全局最优二进制位置
        self.g_best_binary_pos = None
        # 进行测试的数据集
        self.raw_data = raw_data
        # 测试使用的分类器
        self.classifier = classifier

    # 种群初始化
    def initial(self):
        # 将部分关键变量重置
        # 粒子群位置和速度
        self.p_pos = np.zeros((self.ps_size, self.dim))
        self.p_vel = np.zeros((self.ps_size, self.dim))
        # 粒子最优位置
        self.p_best_pos = np.zeros((self.ps_size, self.dim))
        # 全局最优位置
        self.g_best_pos = None
        # 粒子最优适应度值
        self.p_best_fit = np.zeros(self.ps_size)
        # 全局最优适应度值
        self.g_best_fit = None
        # 记录每一代的最优适应度值
        self.fit_record = np.zeros(self.max_iter + 1)
        # 最后的收敛结果
        self.final_result = None

        # --- 特征选择所需二进制变量 --- #
        # 粒子二进制的位置
        self.p_binary_pos = np.zeros((self.ps_size, self.dim))
        # 全局最优二进制位置
        self.g_best_binary_pos = None

        # 随机生成每个粒子的初始位置和初始速度
        for i in range(self.ps_size):
            for j in range(self.dim):
                self.p_pos[i, j] = random.uniform(self.pos_min, self.pos_max)
                self.p_vel[i, j] = random.uniform(self.vel_min, self.vel_max)
                # 将粒子位置通过sigmoid函数进行二进制转换
                self.p_binary_pos[i, j] = common_tool.binary_func(self.p_pos[i, j])
            # 利用二进制位置进行特征选择得到适应度值
            self.p_best_fit[i] = UCI_Classify.fs_classify(self.p_binary_pos[i], self.raw_data, self.classifier)
            # 记录粒子的初始位置信息
            self.p_best_pos[i] = self.p_pos[i]
        # 记录全局初始最优位置和适应度值
        max_index = np.where(self.p_best_fit == np.max(self.p_best_fit))[0][0]
        self.g_best_fit = np.max(self.p_best_fit)
        self.g_best_pos = self.p_pos[max_index].copy()    # deep copy
        self.g_best_binary_pos = self.p_binary_pos[max_index].copy()    # deep copy
        self.fit_record[0] = self.g_best_fit
        
        # print('初始最优适应度值为：')
        # print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
        # 开始迭代
        for iter_count in range(self.max_iter):
            # 计算当前惯性权重w的值
            # w = (self.w_max + (self.max_iter - iter_count) * (self.w_max - self.w_min)) / self.max_iter
            w = random.uniform(self.w_min, self.w_max)

            # 更新粒子位置和速度
            for i in range(self.ps_size):
                # 粒子速度更新
                self.p_vel[i] = w * self.p_vel[i] + \
                                self.c1 * random.uniform(0, 1) * (self.p_best_pos[i] - self.p_pos[i]) + \
                                self.c2 * random.uniform(0, 1) * (self.g_best_pos - self.p_pos[i])
                # 判断粒子速度是否超过边界
                for j in range(self.dim):
                    self.p_vel[i, j] = common_tool.bound_check(self.p_vel[i, j], self.vel_max, self.vel_min)
                # 粒子位置更新
                self.p_pos[i] = self.p_pos[i] + self.p_vel[i]
                # 判断粒子位置是否超过边界
                for j in range(self.dim):
                    self.p_pos[i, j] = common_tool.bound_check(self.p_pos[i, j], self.pos_max, self.pos_min)

                # 将粒子位置通过sigmoid函数进行二进制转换
                for j in range(self.dim):
                    self.p_binary_pos[i, j] = common_tool.binary_func(self.p_pos[i, j])
                # 计算当前粒子的适应度值
                curr_fit = UCI_Classify.fs_classify(self.p_binary_pos[i], self.raw_data, self.classifier)
                # 根据粒子适应度值判断是否更新粒子以及全局的最优位置和适应度值
                if curr_fit > self.p_best_fit[i]:
                    # 更新粒子最优位置和适应度值
                    self.p_best_fit[i] = curr_fit
                    self.p_best_pos[i] = self.p_pos[i].copy()    # deep copy
                    if self.p_best_fit[i] > self.g_best_fit:
                        # 更新全局最优位置和适应度值
                        self.g_best_fit = self.p_best_fit[i]
                        self.g_best_pos = self.p_best_pos[i].copy()    # deep copy
                        self.g_best_binary_pos = self.p_binary_pos[i].copy()    # deep copy

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
# raw_data = pd.read_csv('input/WDBC.csv', header=None)
# # 获取特征数量
# attr_size = raw_data.shape[1] - 1
#
#
# # # 设置PSO的各项参数
# pso = PSOForFS(ps_size=10, dim=attr_size, max_iter=10, pos_max=10, pos_min=-10, raw_data=raw_data, classifier='SVM',
#                w_max=0.9, w_min=0.4, c1=2, c2=2,  vel_max=1, vel_min=-1)
#
# for t in range(1):
#     pso.initial()
#     pso.optimal()
#     print(pso.g_best_fit)
#     print(pso.g_best_binary_pos)
#
# print(time.perf_counter())






































