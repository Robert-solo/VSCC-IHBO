import numpy as np
import pandas as pd
import random
import time
import math


import UCI_Classify
import common_tool


# 创建用于特征选择的IHRO类
class IHROForFS:
    # 构造方法
    def __init__(self, size, dim, max_self, r_max, r_min, max_iter, beta, step_scaling, raw_data, classifier):
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
        # 列维飞行参数beta
        self.beta = beta
        # 列维飞行缩放参数
        self.step_scaling = step_scaling
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

        # 创建原始种群和反向种群
        origin_pos_time_bpos_fit = np.zeros((self.size, self.dim * 2 + 2))
        oppo_pos_time_bpos_fit = np.zeros((self.size, self.dim * 2 + 2))
        # 初始化原始种群和反向种群
        origin_pos_time_bpos_fit[:, :self.dim] = np.random.uniform(self.r_min, self.r_max, (self.size, self.dim))
        oppo_pos_time_bpos_fit[:, :self.dim] = self.r_min + self.r_max - origin_pos_time_bpos_fit[:, :self.dim]
        # 对两个种群进行二进制转换
        for i in range(self.size):
            for j in range(self.dim):
                origin_pos_time_bpos_fit[i, self.dim+1+j] = common_tool.binary_func(origin_pos_time_bpos_fit[i, j])
                oppo_pos_time_bpos_fit[i, self.dim+1+j] = common_tool.binary_func(oppo_pos_time_bpos_fit[i, j])

        # 将两个种群合并
        temp_pos_time_bpos_fit = np.concatenate((origin_pos_time_bpos_fit, oppo_pos_time_bpos_fit), axis=0)
        # 记录个体的初始适应度值
        for pos in temp_pos_time_bpos_fit:
            pos[-1] = UCI_Classify.fs_classify(pos[self.dim+1:self.dim*2+1], self.raw_data, self.classifier)
        # 根据适应度值对种群进行排序
        temp_pos_time_bpos_fit = temp_pos_time_bpos_fit[np.argsort(-temp_pos_time_bpos_fit[:, -1])]
        # 根据定义的种群大小取对应数量的个体进行后续迭代
        self.pos_time_bpos_fit = temp_pos_time_bpos_fit[:self.size]

        # 记录全局初始最优位置和适应度
        self.g_best_fit = self.pos_time_bpos_fit[0, -1]
        self.g_best_pos = self.pos_time_bpos_fit[0, :self.dim].copy()  # deep copy
        self.g_best_binary_pos = self.pos_time_bpos_fit[0, self.dim + 1:self.dim * 2 + 1].copy()  # deep copy
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

            # -------------------- #
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
                    new_sterile[self.dim + 1 + i] = common_tool.binary_func(new_sterile[i])
                # 计算新个体的适应度值
                new_sterile[-1] = UCI_Classify.fs_classify(new_sterile[self.dim+1:self.dim*2+1], self.raw_data, self.classifier)
                # 如果新个体的适应度值优于当前不育系个体，则替换之
                if new_sterile[-1] > self.pos_time_bpos_fit[index, -1]:
                    self.pos_time_bpos_fit[index] = new_sterile
            # -------------------- #

            # -------------------- #
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
                    # 利用精英反向学习策略进行重置
                    # 确定当前种群中的最大最小边界值
                    search_space_min = self.pos_time_bpos_fit[:, :self.dim].min()
                    search_space_max = self.pos_time_bpos_fit[:, :self.dim].max()
                    # 确定当前个体的反向位置
                    new_restorer = np.zeros(self.dim * 2 + 2)
                    new_restorer[:self.dim] = search_space_min + search_space_max - self.pos_time_bpos_fit[index, :self.dim]
                    # 进行二进制转换
                    for i in range(self.dim):
                        new_restorer[self.dim + 1 + i] = common_tool.binary_func(new_restorer[i])
                    # 计算新个体的适应度值
                    new_restorer[-1] = UCI_Classify.fs_classify(new_restorer[self.dim+1:self.dim*2+1], self.raw_data, self.classifier)
                    # 利用新个体替换老个体
                    self.pos_time_bpos_fit[index] = new_restorer
            # -------------------- #

            # -------------------- #
            # 保持系进行列维飞行
            for index in maintainer_index:
                # 初始化列维飞行后的新保持系个体
                new_maintainer = np.zeros(self.dim * 2 + 2)
                # 使用列维飞行生成步长
                step_length = common_tool.levy_flight(1, self.dim, self.beta)[0]
                # 随机生成飞行方向
                step_direction = np.random.rand(self.dim)
                # 更新位置
                new_pos = self.pos_time_bpos_fit[index, :self.dim] + self.step_scaling * step_length * step_direction * (self.pos_time_bpos_fit[index, :self.dim] - self.g_best_pos)
                # 对新生成的个体进行边界检查和二级制转换
                for i in range(self.dim):
                    new_maintainer[i] = common_tool.bound_check(new_pos[i], self.r_max, self.r_min)
                    new_maintainer[self.dim + 1 + i] = common_tool.binary_func(new_maintainer[i])
                # 计算新个体的适应度值
                new_maintainer[-1] = UCI_Classify.fs_classify(new_maintainer[self.dim+1:self.dim*2+1], self.raw_data, self.classifier)
                # 判断是否替换之前的保持系个体
                if new_maintainer[-1] > self.pos_time_bpos_fit[index, -1]:
                    # 如若优于，则替换之
                    self.pos_time_bpos_fit[index] = new_maintainer
            # -------------------- #

            # 当前迭代完成，根据适应度值对种群重新排序
            self.pos_time_bpos_fit = self.pos_time_bpos_fit[np.argsort(-self.pos_time_bpos_fit[:, -1])]
            # 判断是否需要更新全局最优
            if self.pos_time_bpos_fit[0, -1] > self.g_best_fit:
                # 更新全局最优位置和适应度值
                self.g_best_fit = self.pos_time_bpos_fit[0, -1]
                self.g_best_pos = self.pos_time_bpos_fit[0, :self.dim].copy()  # deep copy
                self.g_best_binary_pos = self.pos_time_bpos_fit[0, self.dim + 1:self.dim * 2 + 1].copy()  # deep copy

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
# # 设置HRO的各项参数
# ihro = IHROForFS(size=30, dim=attr_size, max_self=10, r_max=10, r_min=-10, max_iter=10, beta=1.5, step_scaling=0.1)
#
# for t in range(10):
#     ihro.initial()
#     ihro.optimal()
#     print(ihro.g_best_fit)
#     print(ihro.g_best_binary_pos)
#     print(ihro.fit_record)
#
# print(time.perf_counter())
























