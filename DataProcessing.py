import time
import random
import numpy as np
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans


class TrainSet(object):
    """
        user_ratings: 每个用户的评分情况
        num_users: 用户数量
        num_items: 商品数量
        num_ratings: 评分数量
        map_users_id: 原始用户id映射到矩阵对应的id
        map_items_id: 原始商品id映射到矩阵对应的id
        mean_rating: 评分均值
    """

    def __init__(self, user_ratings, num_users, num_items, num_ratings, map_users_id, map_items_id):
        self.user_ratings = user_ratings
        self.num_users = num_users
        self.num_items = num_items
        self.num_ratings = num_ratings
        self.map_users_id = map_users_id
        self.map_items_id = map_items_id
        self.mean_rating = None

    def get_all_ratings(self):
        for uid, ratings in self.user_ratings.items():
            for iid, rating in ratings:
                yield uid, iid, rating

    def calculate_mean_rating(self):
        if self.mean_rating is None:
            self.mean_rating = np.mean([rating for (_, _, rating) in self.get_all_ratings()])
        return self.mean_rating

    def exist_user(self, userid):
        return userid in self.map_users_id

    def exist_item(self, itemid):
        return itemid in self.map_items_id

    def get_map_userid(self, userid):
        return self.map_users_id[userid]

    def get_map_itemid(self, itemid):
        return self.map_items_id[itemid]


def cluster_items(filename):
    """
    商品聚类
    """

    items = []
    x1 = []
    x2 = []
    index = 0
    print('开始加载itemAttribute.txt...')
    with open(filename, 'r') as f:
        lines = f.readlines()
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        line = line.replace('None', '0')
        item, attribution_1, attribution_2 = line.split('|')
        items.append(item)
        x1.append(int(attribution_1))
        x2.append(int(attribution_2))
    print("加载完毕！")
    x = np.transpose([x1, x2])
    cluster = MiniBatchKMeans(n_clusters=250, batch_size=100000, n_init=15)
    print("开始聚类...")
    start = time.time()
    labels = cluster.fit_predict(x)
    end = time.time()
    print('聚类完成,共花费', end - start, '秒')
    return items, labels


def find_similar_items(target, labels):
    index = []
    for i in range(len(labels)):
        if labels[i] == target:
            index.append(i)
    return index


def loadTrainSet(filename):
    items, labels = cluster_items('./Data/itemAttribute.txt')
    print('开始加载train.txt...')
    start = time.time()
    with open(filename, 'r') as f:
        lines = f.readlines()
    rating_list = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if '|' not in line:
            continue
        userid, num_ratings = line.split('|')
        num_ratings = int(num_ratings)
        num = num_ratings
        for i in range(num_ratings):
            item_rating = lines[index + i].strip()
            itemid, rating = item_rating.split('  ')
            rating_list.append((userid, itemid, float(rating)))
            # 若该用户的评分数量小于15，利用对商品的聚类结果填充矩阵
            if num < 15:
                if itemid in items:
                    similar_item_indexs = find_similar_items(labels[items.index(itemid)], labels)
                    rand = random.randint(0, len(similar_item_indexs) - 1)
                    selected = similar_item_indexs[rand]
                    rating_list.append((userid, items[selected], float(rating)))
                    num += 1
        index += num_ratings
    end = time.time()
    print('加载完毕,共花费', end - start, '秒')
    return rating_list


def loadTestSet(filename):
    print('开始加载test.txt...')
    with open(filename, 'r') as f:
        lines = f.readlines()
    testset = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if '|' not in line:
            continue
        userid, num_ratings = line.split('|')
        num_ratings = int(num_ratings)
        for i in range(num_ratings):
            itemid = lines[index + i].strip()
            testset.append((userid, itemid))
    print('加载完毕！')
    return testset


def process_trainset(dataset):
    users = set()
    items = set()
    map_users_id = {}
    map_items_id = {}
    user_ratings = defaultdict(list)
    user_index = 0
    item_index = 0
    for ruid, riid, rating in dataset:
        if ruid in users:
            uid = map_users_id[ruid]
        else:
            users.add(ruid)
            uid = user_index
            user_index += 1
            map_users_id[ruid] = uid
        if riid in items:
            iid = map_items_id[riid]
        else:
            items.add(riid)
            iid = item_index
            item_index += 1
            map_items_id[riid] = iid
        user_ratings[uid].append([iid, rating])
    num_users = len(users)
    num_items = len(items)
    num_ratings = len(dataset)
    trainset = TrainSet(user_ratings, num_users, num_items, num_ratings, map_users_id, map_items_id)
    return trainset
