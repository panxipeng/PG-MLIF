import csv

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random



# 加载数据集
# 使用 Pandas 读取 CSV 文件
data = pd.read_csv("/home/yons/An/paper/20220318-PathomicFusion/20230529PO-Fusion/data/TCGA_GBMLGG/pnas_splits.csv",usecols=[0])
# 获取第一列的所有值,获取所有唯一的 ID
unique_ids = data['ID'].unique()

# 计算测试集大小
num_splits = 15
test_size = int(0.2 * len(unique_ids))
train_list = []
test_list = []
random.seed(789)
# 执行循环，并对每个 fold 进行划分
for i in range(num_splits):
    np.random.shuffle(unique_ids)  # 对 ID 进行随机排列

    # 将当前 fold 中的数据进一步划分为训练集和测试集
    # train, test = train_test_split(unique_ids, test_size=test_size,random_state=22)
    train, test = train_test_split(unique_ids, test_size=test_size)

    train_list.append(train)
    test_list.append(test)
    # for i in unique_ids:
    #     if i in test:

    # 将数组转化为 Pandas 数据框
    df = pd.DataFrame(unique_ids)
ids_list = unique_ids
train_tmp = []

for i in range(num_splits):
    train_t = []
    for id in ids_list:
        if id in train_list[i]:
            train_t.append("Train")
        else:
            train_t.append("Test")
    train_tmp.append(train_t)
# for i in range(15):
#     print(len(train_tmp[i]))
# print(len(ids_list))
dataframe = pd.DataFrame({'':ids_list, 'Randomization - 1':train_tmp[0],'Randomization - 2':train_tmp[1],'Randomization - 3':train_tmp[2],
                          'Randomization - 4':train_tmp[3],'Randomization - 5':train_tmp[4],'Randomization - 6':train_tmp[5],'Randomization - 7':train_tmp[6],
                          'Randomization - 8':train_tmp[7],'Randomization - 9':train_tmp[8],'Randomization - 10':train_tmp[9],'Randomization - 11':train_tmp[10],
                          'Randomization - 12':train_tmp[11],'Randomization - 13':train_tmp[12],'Randomization - 14':train_tmp[13],'Randomization - 15':train_tmp[14]})
dataframe.to_csv(r"/home/yons/An/paper/20220318-PathomicFusion/20230529PO-Fusion/2splits_15/2splits_15.csv",sep=',')
