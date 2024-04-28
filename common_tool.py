import numpy as np
import pandas as pd
import math
import random
import scipy.special

from collections import Counter
from itertools import groupby
from operator import itemgetter


# 对位置信息利用sigmoid函数进行二进制转换
def binary_func(pos):
    # 将连续的位置信息进行sigmoid转换
    sigmoid_num = 1 / (1 + math.exp(-pos))
    # 生成一个随机数
    random_num = random.random()
    # 若生成的随机数小于经过sigmoid转换的数，则返回1，反之返回0
    if random_num < sigmoid_num:
        return 1
    else:
        return 0


# Boundary Check 越界检查
def bound_check(item, upper_bound, lower_bound):
    if item > upper_bound:
        item = upper_bound
    if item < lower_bound:
        item = lower_bound
    return item


# Levy Flights 列维飞行
def levy_flight(size, dim, beta):
    # 根据论文 Nature-Inspired Metaheuristic Algorithms 所提供的公式生成符合种群大小和问题维度要求的步长数组
    sigma_u = (scipy.special.gamma(1+beta)*np.sin(np.pi*beta/2)/(scipy.special.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    sigma_v = 1
    u = np.random.normal(0, sigma_u, (size, dim))
    v = np.random.normal(0, sigma_v, (size, dim))
    steps = u/((np.abs(v))**(1/beta))
    return steps


# elite combination strategy 精英组合策略建立完整的位置信息
def elite_combination(cur_pos, best_pos, features):
    pos = 0
    # 循环所有特征
    for i in features:
        # 将当前位置信息整合进全局最优位置中
        best_pos[i] = cur_pos[pos]
        pos += 1
    # 返回组合后的位置信息
    return best_pos






# 熵
def entropy(x):
    """Calculate the entropy (H(X)) of an array.
    Parameters
    ----------
    x : array-like, shape (n,)
        The array.
    Returns
    -------
    float : H(X) value
    """
    return math.log(len(x)) - math.fsum(v * math.log(v) for v in Counter(x).values()) / len(x)


# 条件熵
def conditional_entropy(x_j, y):
    """Calculate the conditional entropy (H(Y|X)) between two arrays.
    Parameters
    ----------
    x_j : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.
    Returns
    -------
    float : H(Y|X) value
    """
    buf = [[e[1] for e in g] for _, g in groupby(sorted(zip(x_j, y)), itemgetter(0))]
    return math.fsum(entropy(group) * len(group) for group in buf) / len(x_j)


# 对称不确定性
def su_measure(x, y):
    """SU is a correlation measure between the features and the class
    calculated via formula SU(X,Y) = 2 * I(X|Y) / (H(X) + H(Y)). Bigger values
    mean more important features. This measure works best with discrete
    features due to being based on information theory.
    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    Returns
    -------
    array-like, shape (n_features,) : feature scores
    See Also
    --------
    https://pdfs.semanticscholar.org/9964/c7b42e6ab311f88e493b3fc552515e0c764a.pdf
    """
    def __SU(feature):
        entropy_x = entropy(feature)
        return 2 * (entropy_x - conditional_entropy(y, feature)) / (entropy_x + entropy_y)

    entropy_y = entropy(y)
    return np.apply_along_axis(__SU, 0, x)





















