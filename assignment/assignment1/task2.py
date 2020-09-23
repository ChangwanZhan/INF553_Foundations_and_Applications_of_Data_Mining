from pyspark import SparkContext
import json
import sys
import time


def main():
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]
    n_partition = int(sys.argv[3])

    sc = SparkContext('local[*]', 'task2')
    res = {}
    res['default'] = {}
    res['customized'] = {}

    # default
    reviews_RDD = sc.textFile(review_filepath).map(json.loads).map(lambda line: (line["business_id"], 1)).cache()
    default_reviews_RDD = reviews_RDD.repartition(2**4).cache()
    default_n_items = default_reviews_RDD.glom().map(len).collect()
    start = time.time()
    default_reviews_RDD.reduceByKey(lambda a, b: a+b).takeOrdered(10, key=lambda businesses: (-businesses[1], businesses[0]))
    default_exe_time = time.time()-start

    # customized
    customized_reviews_RDD = reviews_RDD.partitionBy(n_partition, lambda business: hash(business)%n_partition).cache()
    customized_n_items = customized_reviews_RDD.glom().map(len).collect()
    start = time.time()
    customized_reviews_RDD.reduceByKey(lambda a, b: a+b).takeOrdered(10, key=lambda businesses: (-businesses[1], businesses[0]))
    customized_exe_time = time.time()-start

    res['default']['n_partition'] = 2**4
    res['default']['n_items'] = default_n_items
    res['default']['exe_time'] = default_exe_time

    res['customized']['n_partition'] = n_partition
    res['customized']['n_items'] = customized_n_items
    res['customized']['exe_time'] = customized_exe_time

    with open(output_filepath, 'w') as json_file:
        json.dump(res, json_file, indent=4)


if __name__ == "__main__":
    main()