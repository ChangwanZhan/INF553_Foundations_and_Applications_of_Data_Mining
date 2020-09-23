# hybrid CF
import xgboost as xgb
from pyspark import SparkContext
import sys
import json
import os
import numpy as np
from itertools import islice


def item_based_pred_rate(user_business):
    user, business = user_business
    if business in business_user_rate_dict:
        if user in business_user_rate_dict[business]:
            return user_business, business_user_rate_dict[business][user]
        rated_users = business_user_rate_dict[business].keys()
        business_avg = avg_business[business]
    else:
        return user_business, avg_rate

    if user not in user_set:
        return user_business, business_avg

    rated_business = user_business_dict[user]

    weight = []

    for rb in rated_business:
        inter_users = set(rated_users) & set(business_user_rate_dict[rb].keys())
        if not inter_users:
            continue
        rb_avg = avg_business[rb]
        normalized_rb_rate = np.zeros(len(inter_users))
        normalized_business_rate = np.zeros(len(inter_users))
        for i, inter_user in enumerate(inter_users):
            normalized_rb_rate[i] = business_user_rate_dict[rb][inter_user]-rb_avg
            normalized_business_rate[i] = business_user_rate_dict[business][inter_user]-business_avg

        w = sum(normalized_rb_rate * normalized_business_rate) / ((sum(normalized_rb_rate ** 2)) ** 0.5 * (sum(normalized_business_rate ** 2)) ** 0.5)
        weight.append((w, business_user_rate_dict[rb][user]))

    weight = list(filter(lambda x: x[0] > 3.0, weight))
    w_abs = sum([abs(w[0]) for w in weight])
    if w_abs == 0:
        return user_business, business_avg
    else:
        rate = sum([w[0] * w[1] for w in weight]) / w_abs
        return user_business, rate


def gen_idx_and_reverse(l):
    assert l
    idx = {}
    reverse = {}
    for i, item in enumerate(l):
        idx[item] = i
        reverse[i] = item
    return idx, reverse


def gen_label_matrix(label, label_num, idx):
    mat = np.zeros(label_num)
    for user_business, rate in label:
        id = idx[user_business]
        mat[id] = rate
    return mat


def main():
    global avg_rate, business_user_rate_dict, user_set, user_business_dict, avg_business
    folder_path, test_file, output_file = sys.argv[1:4]
    sc = SparkContext('local[*]', 'task2_1')

    train_file = os.path.join(folder_path, 'yelp_train.csv')
    train_rdd = sc.textFile(train_file).mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it).map(lambda line: line.split(',')).cache()

    # item_based
    rate = train_rdd.map(lambda x: (float(x[2]), 1)).reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))
    avg_rate = rate[0]/rate[1]
    print(avg_rate)

    user_set = set(train_rdd.map(lambda x: x[0]).distinct().collect())
    business_user = train_rdd.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).cache()
    business_rate = train_rdd.map(lambda x: (x[1], [float(x[2])])).reduceByKey(lambda x, y: x + y).cache()
    avg_business = business_rate.map(lambda x: (x[0], sum(x[1])/len(x[1]))).collectAsMap()
    user_business = train_rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y).cache()
    business_user_rate = business_user.join(business_rate).map(lambda x: (x[0], dict(zip(x[1][0], x[1][1]))))
    user_business_dict = user_business.collectAsMap()
    business_user_rate_dict = business_user_rate.collectAsMap()

    test_rdd = sc.textFile(test_file).mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it).map(lambda line: line.split(','))
    item_based_pred = test_rdd.map(lambda x: (x[0], x[1])).distinct().partitionBy(2 ** 4).map(item_based_pred_rate).collectAsMap()

    # model_based
    train_label = train_rdd.map(lambda x: ((x[0], x[1]), x[2])).collect()
    train_user_business_list = train_rdd.map(lambda x: (x[0], x[1])).distinct().collect()
    assert train_user_business_list
    train_user_business_idx = {}
    train_business_idx = {}
    train_user_idx = {}
    for i, user_business in enumerate(train_user_business_list):
        train_user_business_idx[user_business] = i
        if train_business_idx.get(user_business[1]):
            train_business_idx[user_business[1]].append(i)
        else:
            train_business_idx[user_business[1]] = [i]
        if train_user_idx.get(user_business[0]):
            train_user_idx[user_business[0]].append(i)
        else:
            train_user_idx[user_business[0]] = [i]
    train_user_business_num = i + 1
    y_train = gen_label_matrix(train_label, train_user_business_num, train_user_business_idx)

    user_json_path = os.path.join(folder_path, 'user.json')
    user_feature12_rdd = sc.textFile(user_json_path).map(json.loads).map(lambda x: (x["user_id"], (x["review_count"], x["average_stars"]))).cache()
    user_feature12 = user_feature12_rdd.collectAsMap()
    user_feature12_avg = user_feature12_rdd.map(lambda x: (1, (x[1][0], x[1][1], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2])).mapValues(lambda x: (x[0] / x[2], x[1] / x[2])).collect()
    user_feature1_avg, user_feature2_avg = user_feature12_avg[0][1]
    checkin_json_path = os.path.join(folder_path, 'checkin.json')
    user_feature3_rdd = sc.textFile(checkin_json_path).map(json.loads).map(lambda x: (x["business_id"], len(x["time"]))).reduceByKey(lambda x, y: x + y).cache()
    user_feature3 = user_feature3_rdd.collectAsMap()
    user_feature3_avg = user_feature3_rdd.map(lambda x: (1, (x[1], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda x: x[0] / x[1]).collect()[0][1]
    business_json_path = os.path.join(folder_path, 'business.json')
    user_feature4_rdd = sc.textFile(business_json_path).map(json.loads).map(lambda x: (x["business_id"], (x["stars"], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda x: x[0] / x[1]).cache()
    user_feature4 = user_feature4_rdd.collectAsMap()
    user_feature4_avg = user_feature4_rdd.map(lambda x: (1, (x[1], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda x: x[0] / x[1]).collect()[0][1]
    x_train = np.zeros((train_user_business_num, 4))
    for user_business in train_user_business_list:
        user, business = user_business
        u_ids = train_user_idx[user]
        b_ids = train_business_idx[business]
        for u_id in u_ids:
            if user_feature12.get(user):
                x_train[u_id][0] = user_feature12[user][0]
                x_train[u_id][1] = user_feature12[user][1]
            else:
                x_train[u_id][0] = user_feature1_avg
                x_train[u_id][1] = user_feature2_avg
        for b_id in b_ids:
            if user_feature3.get(business):
                x_train[b_id][2] = user_feature3[business]
            else:
                x_train[b_id][2] = user_feature3_avg
            if user_feature4.get(business):
                x_train[b_id][3] = user_feature4[business]
            else:
                x_train[b_id][3] = user_feature4_avg

    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)

    test_user_business_list = test_rdd.map(lambda x: (x[0], x[1])).distinct().collect()
    assert test_user_business_list
    test_user_business_idx = {}
    test_idx_user_business = {}
    test_business_idx = {}
    test_user_idx = {}
    for i, user_business in enumerate(test_user_business_list):
        test_user_business_idx[user_business] = i
        test_idx_user_business[i] = user_business
        if test_business_idx.get(user_business[1]):
            test_business_idx[user_business[1]].append(i)
        else:
            test_business_idx[user_business[1]] = [i]
        if test_user_idx.get(user_business[0]):
            test_user_idx[user_business[0]].append(i)
        else:
            test_user_idx[user_business[0]] = [i]
    test_user_business_num = i + 1

    x_test = np.zeros((test_user_business_num, 4))
    for user_business in test_user_business_list:
        user, business = user_business
        u_ids = test_user_idx[user]
        b_ids = test_business_idx[business]
        for u_id in u_ids:
            if user_feature12.get(user):
                x_test[u_id][0] = user_feature12[user][0]
                x_test[u_id][1] = user_feature12[user][1]
            else:
                x_test[u_id][0] = user_feature1_avg
                x_test[u_id][1] = user_feature2_avg
        for b_id in b_ids:
            if user_feature3.get(business):
                x_test[b_id][2] = user_feature3[business]
            else:
                x_test[b_id][2] = user_feature3_avg
            if user_feature4.get(business):
                x_test[b_id][3] = user_feature4[business]
            else:
                x_test[b_id][3] = user_feature4_avg

    pred = model.predict(data=x_test)

    # hybrid
    hybrid_rate = {}
    review_train_path = os.path.join(folder_path, 'review_train.json')
    test_business_review_num = sc.textFile(review_train_path).map(json.loads).map(lambda x: (x["business_id"], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()

    for i in range(len(pred)):
        user_id, business_id, model_based_rate = test_idx_user_business[i][0], test_idx_user_business[i][1], pred[i]
        item_based_rate = item_based_pred[(user_id, business_id)]
        if test_business_review_num[business_id] >= 500:
            hybrid_rate[(user_id, business_id)] = 0.8 * item_based_rate + 0.2 * model_based_rate
        elif test_business_review_num[business_id] >= 250:
            hybrid_rate[(user_id, business_id)] = 0.7 * item_based_rate + 0.3 * model_based_rate
        elif test_business_review_num[business_id] >= 100:
            hybrid_rate[(user_id, business_id)] = 0.6 * item_based_rate + 0.4 * model_based_rate
        else:
            hybrid_rate[(user_id, business_id)] = 0.2 * item_based_rate + 0.8 * model_based_rate

    with open(output_file, "w") as f:
            f.write("user_id, business_id, prediction")
            for ub, rate in hybrid_rate.items():
                user, business = ub
                f.write("\n" + user + "," + business + "," + str(rate))


if __name__ == '__main__':
    main()