# item based CF
from pyspark import SparkContext
import sys
import numpy as np
from itertools import islice


def pred_rate(user_business):
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
        rate = min(max(1.0, rate), 5.0)
        return user_business, rate


def main():
    global avg_rate, business_user_rate_dict, user_set, user_business_dict, avg_business
    train_file, test_file, output_file = sys.argv[1:4]
    sc = SparkContext('local[*]', 'task2_1')

    train_rdd = sc.textFile(train_file).mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it).map(lambda line: line.split(',')).cache()

    rate = train_rdd.map(lambda x: (float(x[2]), 1)).reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))
    avg_rate = rate[0]/rate[1]

    user_set = set(train_rdd.map(lambda x: x[0]).distinct().collect())
    business_user = train_rdd.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).cache()
    business_rate = train_rdd.map(lambda x: (x[1], [float(x[2])])).reduceByKey(lambda x, y: x + y).cache()
    avg_business = business_rate.map(lambda x: (x[0], sum(x[1])/len(x[1]))).collectAsMap()
    user_business = train_rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y).cache()
    business_user_rate = business_user.join(business_rate).map(lambda x: (x[0], dict(zip(x[1][0], x[1][1]))))
    user_business_dict = user_business.collectAsMap()
    business_user_rate_dict = business_user_rate.collectAsMap()

    test_rdd = sc.textFile(test_file).mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it).map(lambda line: line.split(','))
    pred_res = test_rdd.map(lambda x: (x[0], x[1])).distinct().partitionBy(2 ** 4).map(pred_rate).collect()

    # test_business_rate = test_rdd.map(lambda x: (x[1], [float(x[2])])).reduceByKey(lambda x, y: x + y).cache()
    # test_business_user = test_rdd.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).cache()
    # test_business_user_rate = test_business_user.join(test_business_rate).map(lambda x: (x[0], dict(zip(x[1][0], x[1][1])))).collectAsMap()
    #
    # count = 0
    # rmse = 0
    # for pred in pred_res:
    #     count += 1
    #     user = pred[0][0]
    #     business = pred[0][1]
    #     rate = pred[1]
    #     t_rate = test_business_user_rate[business][user]
    #     rmse += (rate - t_rate) ** 2
    # rmse = (rmse/count) ** 0.5
    # print(rmse)

    with open(output_file, "w") as f:
            f.write("user_id, business_id, prediction")
            for pred in pred_res:
                f.write("\n" + pred[0][0] + "," + pred[0][1] + "," + str(pred[1]))


if __name__ == '__main__':
    main()