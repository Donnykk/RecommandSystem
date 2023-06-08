user_set = set()
item_set = set()
score_num = 0
min_score = 100
max_score = 0
score_sum = 0.0

with open('./Data/train.txt', 'r') as f:
    lines = f.readlines()

index = 0
while index < len(lines):
    line = lines[index].strip()
    index += 1

    if '|' not in line:
        continue

    # 获取用户id和评分数量
    userid, num_ratings = line.split('|')
    user_set.add(userid)
    number = int(num_ratings)
    score_num += number

    # 获取商品id以及用户对各个商品的评分
    for i in range(number):
        line = lines[index + i].strip()
        itemid, score = line.split('  ')
        item_set.add(itemid)
        score = float(score)
        score_sum += score

        if score > max_score:
            max_score = score

        if score < min_score:
            min_score = score

    index += number


def AnalyzeData():
    print('用户数量:', len(user_set))
    print('商品数量:', len(item_set))
    print('评分数量:', score_num)
    print('最高评分:', max_score)
    print('最低评分:', min_score)
    print('评分均值:', score_sum / score_num)
