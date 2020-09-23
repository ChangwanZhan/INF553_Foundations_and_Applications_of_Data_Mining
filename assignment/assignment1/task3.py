from pyspark import SparkContext
import sys
import json
import time

def main():
    review_filepath = sys.argv[1]
    business_filepath = sys.argv[2]
    output_filepath_question_a = sys.argv[3]
    output_filepath_question_b = sys.argv[4]

    sc = SparkContext('local[*]', 'task2')
    review_RDD = sc.textFile(review_filepath).map(lambda line: (json.loads(line)['business_id'], json.loads(line)['stars'])).partitionBy(4)
    business_RDD = sc.textFile(business_filepath).map(lambda line: (json.loads(line)['business_id'], json.loads(line)['city'])).partitionBy(4)
    city_star_RDD = review_RDD.join(business_RDD).map(lambda line: (line[1][1], (line[1][0], 1)))

    # A. What is the average stars for each city?
    city_star_ave = city_star_RDD.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])).mapValues(lambda x: x[0]/x[1]).cache()
    city_star_ave_list_A = city_star_ave.sortBy(lambda x: (-x[1], x[0])).collect()

    # B. print top 10 cities with highest stars
    # Method1: Collect all the data, sort in python, and then print the first 10 cities
    start = time.time()
    city_star_ave_list = city_star_ave.collect()
    city_star_ave_list.sort(key=lambda x: (-x[1], x[0]))
    # city_star_ave_list = sorted(city_star_ave_list, key=lambda x: (-x[1], x[0]))
    city_star_ave_list = city_star_ave_list[:10]
    print(city_star_ave_list)
    m1 = time.time() - start

    # Method2: Sort in Spark, take the first 10 cities, and then print these 10 cities
    start = time.time()
    city_star_ave_list = city_star_ave.takeOrdered(10, lambda x: (-x[1], x[0]))
    # city_star_ave_list = city_star_ave.sortBy(lambda x: (-x[1], x[0])).take(10)
    print(city_star_ave_list)
    m2 = time.time() - start
    res_b = {"m1": m1, "m2": m2}

    with open(output_filepath_question_a, 'w') as f:
        f.write('city,stars\n')
        for city, ave_star in city_star_ave_list_A:
            f.write('{city},{ave_star}\n'.format(city=city.encode('utf-8'), ave_star=str(ave_star)))

    with open(output_filepath_question_b, 'w') as json_file:
        json.dump(res_b, json_file, indent=4)


if __name__ == "__main__":
    main()