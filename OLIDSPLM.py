# -*- coding = utf-8 -*-
# @Time : 2023/10/7 下午9:17
# @Author : 柳佳乐
# @File : OLIDSPLM.py
# @Software : PyCharm

from math import log, e, pi
import numpy as np

class OnlineGaussianNaiveBayes(object):
    def __init__(self, D, epsilon, priors=False, w=30):
        # priors : 用来指定P(Y)的先验概率 在这个里面就是要不要算上其先验概率
        #
        self.epsilon = epsilon
        self.D_ = np.int32(D)
        assert self.D_ > 0
        self.priors_ = priors  # 是否运用先验概率
        self.n_ = {}  # 每一类中的个数统计
        self.mean_ = {}  # 每一类中每个特征单独的均值
        self.M2_ = {}  # /n-1 则为var_ ais的平方  /n 则为样本标准差 sigma的平方
        self.var_ = {}  # 每个类中每个特征的总体标准差

        self.w = w
        self.wd = {}
        self.W = {}
        self.W_mean = {}
        self.W_M2 = {}
        self.W_var_ = {}

        self.total_ = np.int64(0)
        self.min_var_ = np.inf


        self.probabilitylist = []

        self.Drift_point = []

    def fit2(self, x, c):
        y2 = self.predict(x)
        if c == None:  # 缺失标签的情况
            y1 = self.predict1(x)
            if y1 == y2 and y1 != None:
                self.fit(x, y2)
                return y1
            else:
                if y1 != None and y2 == None:
                    return y1
                else:
                    return y2  # 或者y2
        else:
            self.fit(x, c)
            return y2

    def fit(self, x, c):
        # self.N_ += 1
        assert len(x) == self.D_
        self.total_ += 1
        if c not in self.n_:
            D = self.D_
            self.n_[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.mean_[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.M2_[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.var_[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))

            self.W[c] = np.ndarray((self.w, self.D_), dtype=np.float64)
            self.wd[c] = 0
            self.W_mean[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.W_M2[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.W_var_[c] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))


        for i in range(self.D_):
            if x[i] == None:
                continue
            self.n_[c][i] += 1
            n = self.n_[c][i]
            delta = x[i] - self.mean_[c][i]
            self.mean_[c][i] += delta / n
            self.M2_[c][i] += delta * (x[i] - self.mean_[c][i])

            if n < 2:
                self.var_[c][i] = np.float64(0)
            else:
                self.var_[c][i] = self.M2_[c][i] / (n - 1)

            if self.var_[c][i] < self.min_var_ and self.var_[c][i] != 0:
                self.min_var_ = self.var_[c][i]

            # delta1 = x[i] - self.W_mean[c][i]
            # self.W_mean[c][i] += delta1 / n1
            # self.W_M2[c][i] += delta1 * (x[i] - self.W_mean[c][i])




            self.W[c][self.wd[c] % self.w][i] = (x[i])

        self.wd[c] += 1

        if self.wd[c] % self.w == 0:
            f=0
            for i in range(self.D_):
                # print(self.w)
                x = list(self.W[c][:, i])
                if None in x:
                    x.remove(None)
                self.W_mean[c][i] = np.mean(x)
                self.W_var_[c][i] = np.std(x, ddof=1)
                if self.W_var_[c][i]!=0 and np.sqrt(self.var_[c][i])!=0:
                    K_L1 = K_L(self.W_mean[c][i], self.W_var_[c][i], self.mean_[c][i], np.sqrt(self.var_[c][i]))
                    K_L2 = K_L(self.mean_[c][i], np.sqrt(self.var_[c][i]), self.W_mean[c][i], self.W_var_[c][i])
                    f+=(K_L1 + K_L2) / 2
                    # if f<(K_L1 + K_L2) / 2:
                    #     f=(K_L1 + K_L2) / 2


                    # if (K_L1 + K_L2) / 2 > self.epsilon:
                    #     self.reset()
                    #     break

            if f > self.epsilon:
                self.reset()
            self.probabilitylist.append(f)


        if len(self.probabilitylist)==50:
            self.Calculate_epsilon()

            # self.epsilon=np.mean(self.probabilitylist[50:])+np.mean(self.probabilitylist[50:])*0.5
            # self.epsilon=max(self.probabilitylist)+max(self.probabilitylist)*0.1


    def _pdf(self, x, m, v):
        # 将所有特征中方差最小的以某个比例添加到估计的方差中， 在Sklearn包中是找的方差最大的
        v += self.min_var_ / 1000  # 平滑处理
        return (-0.5 * log(2 * pi * v) - ((x - m) ** 2 / (2 * v)))

    def predict(self, x):
        best_score_class_pair = (-np.inf, None)  # 等于一个最小值和None
        # z=self.weight()
        z = self.W2()
        for c in self.mean_:
            score = 0
            for i in range(self.D_):
                if x[i] == None or np.isnan(x[i]):
                    continue
                score += (self._pdf(x[i], self.mean_[c][i], self.var_[c][i]))
            # if self.priors_:
            #     score += log(self.n_[c]) - log(self.total_)  # 计算后验概率
            best_score_class_pair = max(best_score_class_pair, (score, c), key=lambda item: item[0])
        if best_score_class_pair[1] == None:
            return 0
        else:
            return best_score_class_pair[1]

    def predict1(self, x):
        best_score_class_pair = (np.inf, None)
        # z = self.W2()
        for j in self.mean_:
            sum_c = 0
            for i in range(self.D_):
                if x[i] == None:
                    continue
                sum_c += abs((self.mean_[j][i] - float(x[i])) / self.mean_[j][i])  # 除以特征的均值，防止某个特征值过大影响判断。 乘以权重
            best_score_class_pair = min(best_score_class_pair, (sum_c, j), key=lambda item: item[0])
        if best_score_class_pair[1] == None:
            return 0
        else:
            return best_score_class_pair[1]

    def reset(self):
        self.Drift_point.append(self.total_)
        # priors : 用来指定P(Y)的先验概率 在这个里面就是要不要算上其先验概率
        self.n_ = {}  # 每一类中的个数统计
        self.mean_ = self.W_mean  # 每一类中每个特征单独的均值
        self.M2_ = self.W_M2  # /n-1 则为var_ ais的平方  /n 则为样本标准差 sigma的平方
        self.var_ = self.W_var_  # 每个类中每个特征的总体标准差
        # #
        # self.mean_ = {}
        # self.M2_ = {}
        # self.var_ = {}

        self.wd = {}
        self.W = {}
        self.W_mean = {}
        self.W_M2 = {}
        self.W_var_ = {}

        self.probabilitylist = []

    # def reward(self,reward):
    #     if reward:
    #         warning_status, drift_status= self.mode.run(reward)
    #     else:
    #         warning_status, drift_status= self.mode.run(reward)
    #     if drift_status:
    #         self.reset()
    #         self.mode.reset()
    # def reward (self,reward,true_class):
    #
    #     pass

    def W2(self):
        KL_array = np.ndarray(self.D_, dtype=np.float64, buffer=np.ones(self.D_))
        for i in range(self.D_):
            KL_ = 0
            le = list(self.mean_.keys())
            for c in le:
                le.remove(c)
                if len(le) > 1:
                    for c_ in le:
                        v1 = np.sqrt(self.var_[c][i])
                        v2 = np.sqrt(self.var_[c_][i])
                        m1 = self.mean_[c][i]
                        m2 = self.mean_[c_][i]
                        if v1 != 0 and v2 != 0:
                            # print("第{}个特征，第{}类和第{}类".format(i,c,c_))
                                KL_ = (K_L(m1, v1, m2, v2) + K_L(m2, v2, m1, v1)) / 2 + KL_
                        else:
                            # print("权重都是1")
                            return np.ndarray(self.D_, dtype=np.float64, buffer=np.ones(self.D_))
            KL_array[i] = KL_

        KL_array[np.isnan(KL_array)] = 0
        if not (np.any(KL_array)):
            return np.ndarray(self.D_, dtype=np.float64, buffer=np.ones(self.D_))
        else:
            KL_array = KL_array / sum(KL_array)
        return KL_array  # KL 3

    def Drift_sequence(self):
        return self.Drift_point


    # 计算阈值epsilon
    def Calculate_epsilon(self):
        b= self.probabilitylist[30:]

        # b.sort()
        #
        # pmax=b[-1]
        # pmin=b[0]
        #
        # pmax_list= b[39:49]
        # pmin_list = b[1:11]
        # a=pmax_list+pmin_list


        # self.epsilon = (pmax-pmin)+np.mean(a)
        self.epsilon =np.mean(b)+3*np.std(b)


def K_L(m1, v1, m2, v2):
    kl = np.log(v2 / v1) + ((v1) ** 2 + (m1 - m2) ** 2) / (2 * v2 ** 2) - 1 / 2
    return kl

