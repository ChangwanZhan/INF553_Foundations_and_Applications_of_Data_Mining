from pyspark import SparkContext
import sys
from itertools import islice
from itertools import combinations
import random


def gen_hash_function():
    res = set()
    for i in range(hash_function_num):
        x = random.randint(1, random_max)
        while x in res:
            x = random.randint(1, random_max)
        res.add(x)
    res = list(res)
    return res


def gen_signature(business_user):
    business, users = business_user[0], business_user[1]
    sig = [random_max for _ in range(hash_function_num)]
    for idx, user in enumerate(user_list):
        if user in users:
            for i in range(hash_function_num):
                sig[i] = min(sig[i], (idx*hash_function[0][i]+hash_function[1][i]) % random_max)
    sig_bands = []
    i = 0
    while i < hash_function_num:
        band = tuple(sig[i:i+r])
        sig_bands.append((band, [business]))
        i = i+r
    return sig_bands


def gen_combination(candidates):
    pairs = list(combinations(candidates, 2))
    for i in range(len(pairs)):
        pairs[i] = tuple(sorted(pairs[i]))
    return pairs


def jaccard_similarity(candidate_pair):
    user_1 = business_user_dict[candidate_pair[0]]
    user_2 = business_user_dict[candidate_pair[1]]
    jaccard_similarity = float(len(user_1.intersection(user_2))/len(user_1.union(user_2)))
    return sorted(candidate_pair), jaccard_similarity


def main():
    global user_list, business_user_dict, hash_function_num, hash_function, b, r, random_max
    input_file, output_file = sys.argv[1], sys.argv[2]

    partition_num = 2**4
    b, r = 20, 3

    random_max = 2**16
    hash_function_num = b * r
    hash_function = [gen_hash_function(), gen_hash_function()]

    sc = SparkContext('local[*]', 'task1')
    rdd = sc.textFile(input_file).mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it).map(lambda line: line.split(',')).cache()
    user_list = rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y).keys().collect()
    business_user = rdd.map(lambda x: (x[1], x[0])).partitionBy(partition_num).groupByKey().mapValues(set).cache()
    business_user_dict = business_user.collectAsMap()
    pairs = business_user.flatMap(gen_signature).reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) > 1)\
        .values().flatMap(gen_combination).distinct().map(jaccard_similarity).filter(lambda x: x[1] >= 0.5).sortByKey().collect()
    with open(output_file, "w") as f:
        f.write("business_id_1, business_id_2, similarity")
        for pair, similarity in pairs:
            res = pair + [str(similarity)]
            f.write("\n" + ",".join(res))


if __name__ == "__main__":
    main()