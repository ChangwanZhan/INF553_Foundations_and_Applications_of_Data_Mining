# model based CF
import xgboost as xgb
from pyspark import SparkContext
import sys
import json
import os
import numpy as np
from itertools import islice
import time


def gen_idx(l):
    assert l
    idx = {}
    for i, item in enumerate(l):
        idx[item] = i
    return idx, i+1


def gen_label_matrix(label, label_num, idx):
    mat = np.zeros(label_num)
    for user_business, rate in label:
        id = idx[user_business]
        mat[id] = rate
    return mat


def main():
    start = time.time()
    folder_path, test_file, output_file = sys.argv[1:4]
    sc = SparkContext('local[*]', 'task2_2')
    
    # train
    train_label_path = os.path.join(folder_path, 'yelp_val.csv')
    train_label_rdd = sc.textFile(train_label_path).mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it).map(lambda line: line.split(',')).cache()
    train_label = train_label_rdd.map(lambda x: ((x[0], x[1]), x[2])).collect()

    # all
    # rdd = sc.union([train_label_rdd, test_rdd]).cache()
    train_user_business_list = train_label_rdd.map(lambda x: (x[0], x[1])).distinct().collect()
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
    train_user_business_num = i+1
    y_train = gen_label_matrix(train_label, train_user_business_num, train_user_business_idx)

    # business_list = rdd.map(lambda x: x[1]).distinct().collect()
    user_json_path = os.path.join(folder_path, 'user.json')
    user_feature12_rdd = sc.textFile(user_json_path).map(json.loads).map(lambda x: (x["user_id"], (x["review_count"], x["average_stars"]))).cache()
    user_feature12 = user_feature12_rdd.collectAsMap()
    user_feature12_avg = user_feature12_rdd.map(lambda x: (1, (x[1][0], x[1][1], 1))).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2])).mapValues(lambda x: (x[0]/x[2], x[1]/x[2])).collect()
    user_feature1_avg, user_feature2_avg = user_feature12_avg[0][1]
    checkin_json_path = os.path.join(folder_path, 'checkin.json')
    user_feature3_rdd = sc.textFile(checkin_json_path).map(json.loads).map(lambda x: (x["business_id"], len(x["time"]))).reduceByKey(lambda x, y: x+y).cache()
    user_feature3 = user_feature3_rdd.collectAsMap()
    user_feature3_avg = user_feature3_rdd.map(lambda x: (1, (x[1], 1))).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])).mapValues(lambda x: x[0]/x[1]).collect()[0][1]
    business_json_path = os.path.join(folder_path, 'business.json')
    user_feature4_rdd = sc.textFile(business_json_path).map(json.loads).map(lambda x: (x["business_id"], (x["stars"], 1))).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])).mapValues(lambda x: x[0]/x[1]).cache()
    user_feature4 = user_feature4_rdd.collectAsMap()
    user_feature4_avg = user_feature4_rdd.map(lambda x: (1, (x[1], 1))).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])).mapValues(lambda x: x[0]/x[1]).collect()[0][1]
    x_train = np.zeros((train_user_business_num, 4))
    for user_business in train_user_business_list:
        user, business = user_business
        u_ids = train_user_idx[user]
        b_ids = train_business_idx[business]
        # ub_id = train_user_train_business_idx
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

    # model
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)

    # test
    test_rdd = sc.textFile(test_file).mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it).map(lambda line: line.split(',')).cache()
    # test_label = test_rdd.map(lambda x: ((x[0], x[1]), x[2])).collect()
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
    test_user_business_num = i+1
    # y_test = gen_label_matrix(test_label, test_user_business_num, test_user_business_idx)

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

    with open(output_file, "w") as f:
        f.write("user_id, business_id, prediction")
        for i in range(len(pred)):
            f.write("\n" + test_idx_user_business[i][0] + "," + test_idx_user_business[i][1] + "," + str(pred[i]))

    # RMSE
    # rmse = ((1/len(pred)) * sum((pred - y_test) * (pred - y_test))) ** 0.5
    # print(rmse)



if __name__ == '__main__':
    main()