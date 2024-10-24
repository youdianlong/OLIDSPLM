from OLIDSPLM import OnlineGaussianNaiveBayes
import numpy as np
import random
import copy
import preprocess
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")


np.random.seed(preprocess.random_seed)
random.seed(preprocess.random_seed)


preprocess_l = [
preprocess.readcsv("./dataset/Data_Cortex_Nuclear.csv"),

preprocess.readcsv("./dataset/Occupancy_Estimation.csv"),
]

dataset_n = [
"Data_Cortex_Nuclear.csv",
"Occupancy_Estimation.csv",
]

# precess = [
#     preprocess.readIsolet(),
#     preprocess.readSpambaseNormalized(),
# ]
# dataset_name = [
#     "Isolet",
#     "Spambase",
# ]


len_l =len(preprocess_l)
# 缺失特征的，
# len_l = len(dataset_name)
# dataset_n = dataset_name

precess=preprocess_l


epsilon_list=[np.inf]

rmlable_list=[0]
rmfeature_list=[0]
windows_list = [30]
max_acc = 0
for windows in windows_list:
    for rmlable in rmlable_list:
        for rmfeature in rmfeature_list:
            for i in range(len_l):
                for epsilon in epsilon_list:
                    print('rmlable:',rmlable,'rmfeature:',rmfeature,'dataset_n:',dataset_n[i],'epsilon:',epsilon)
                    data_chu =precess[i]
                    data = copy.deepcopy(data_chu)

                    log_list = []
                    log_list.append(dataset_n[i])
                    log_list.append("特征数:" + str(len(data[0][0])))
                    n = 0
                    bingo = 0
                    bad_done = 0

                    c = []
                    for k, g in data:
                        c.append(g)
                    data,data_0 = preprocess.removefeature(data, rmfeature)
                    data = preprocess.removelable(data, rmlable)
                    clf = OnlineGaussianNaiveBayes(len(data[0][0]), epsilon,False, windows)
                    a = []
                    a1 = []
                    start = time.time()
                    for r, t in tqdm(zip(data, c)):
                        X = r[0]
                        y = r[1]
                        c = t
                        n += 1

                        y1 = clf.fit2(X, y)
                        a1.append(c)
                        a.append(y1)
                        if y1 == c:
                            bingo += 1
                        else:
                            bad_done += 1
                    end = time.time()
                    log_list.append("正确数：" + str(bingo))
                    log_list.append("错误数：" + str(bad_done))
                    # p+=1
                    log_list.append("总数:" + str(n))
                    log_list.append("准确率：" + str(round(bingo / n, 10)))
                    log_list.append("时间：" + str((end - start)))
                    log_list.append("窗口大小：" + str((windows)))
                    log_list.append("标签缺失率：" + str(rmlable))
                    log_list.append("特征缺失率：" + str(rmfeature))
                    log_list.append("seed:" + str(preprocess.random_seed))
                    log_list.append("漂移点："+str(clf.Drift_sequence()))
                    print(log_list)

