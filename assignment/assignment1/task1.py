from pyspark import SparkContext
import sys
import json


def main():
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    sc = SparkContext('local[*]', 'task1')
    review_RDD = sc.textFile(review_filepath).coalesce(2**4).map(json.loads).map(lambda line: (line["date"].split()[0].split('-')[0], line["user_id"], line["business_id"])).cache()
    res = {}

    # A. the total number of reviews
    n_review = review_RDD.count()
    res["n_review"] = n_review

    # B. the number of reviews in 2018
    n_review_2018 = review_RDD.filter(lambda line: "2018" == line[0]).count()
    res["n_review_2018"] = n_review_2018

    # C. the number of distinct users who wrote reviews
    user_map = review_RDD.map(lambda line: (line[1], 1)).reduceByKey(lambda a, b: a+b).cache()
    n_user = user_map.count()
    res["n_user"] = n_user

    # D. the top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
    top10_user = user_map.takeOrdered(10, key=lambda users: (-users[1], users[0]))
    res["top10_user"] = top10_user
    #user_map.unpersist()

    # E. the number of distinct business that have been reviewed
    business_map = review_RDD.map(lambda line: (line[2], 1)).reduceByKey(lambda a, b: a+b).cache()
    n_business = business_map.count()
    res["n_business"] = n_business

    # F. the top 10 businesses that had the largest numbers of reviews and the number of reviews they had
    top10_business = business_map.takeOrdered(10, key=lambda businesses: (-businesses[1], businesses[0]))
    res["top10_business"] = top10_business

    with open(output_filepath, 'w') as json_file:
        json.dump(res, json_file, indent=4)


if __name__ == "__main__":
    main()