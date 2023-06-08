from math import sqrt
from DataProcessing import loadTrainSet, loadTestSet, process_trainset
from Model import SVDMode
import random
import os


# 计算均方根误差
def calculateRMSE(pred, oracle):
    sum = 0.0
    for i in range(len(pred)):
        sum += (oracle[i] - pred[i]) ** 2
    return sqrt(sum / len(pred))


# 划分训练集、验证集
def split_train_and_val(dataset, train_ratio, val_ratio):
    assert train_ratio + val_ratio == 100
    random.seed(1)
    trainset = []
    valset = []
    for i in range(len(dataset)):
        rand = random.randint(0, 100)
        if rand == val_ratio:
            valset.append(dataset[i])
        else:
            trainset.append(dataset[i])
    return trainset, valset


def train():
    dataset = loadTrainSet('./Data/train.txt')
    print("开始划分训练集、验证集...")
    trainset, valset = split_train_and_val(dataset, 99, 1)
    print("划分完毕！")
    trainset = process_trainset(trainset)
    SVD = SVDMode()
    print("开始训练...")
    SVD.SVDtrain(trainset)
    print("训练完成！")
    val = []
    oracle = []
    # 真实值
    for (userid, itemid, score) in valset:
        val.append([userid, itemid])
        oracle.append(score)
    # 预测值
    pred = SVD.SVDpredict(val)
    # 计算RMSE
    RMSE = calculateRMSE(pred, oracle)
    print('val_RMSE:', RMSE)
    # 加载测试集
    testset = loadTestSet('./Data/test.txt')
    # 统计每个用户的待评分数量
    rating_counts = {}
    for user, _ in testset:
        rating_counts[user] = rating_counts.get(user, 0) + 1
    print("开始预测...")
    pred = SVD.SVDpredict(testset)
    print("预测完成！")
    # 将预测的评分结果保存到文件
    print("保存结果...")
    result_dir = './Result'
    os.makedirs(result_dir, exist_ok=True)
    with open('./Result/result.txt', 'w+') as f:
        now_user = ""
        for i in range(len(testset)):
            user, item = testset[i]
            score = pred[i]
            if user != now_user:
                now_user = user
                f.write(user + '|' + str(rating_counts[user]) + '\n')
                f.write(item + '  ' + str(score) + '\n')
            else:
                f.write(item + '  ' + str(score) + '\n')
    print("保存完毕！")
