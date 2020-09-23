from pyspark import SparkContext
import sys
from itertools import islice
from itertools import combinations
import time


def make_combinations(candidates, pair_len, candidates_pruned):
    res = set()
    for i in range(len(candidates)-1):
        for j in range(i+1, len(candidates)):
            candidate = sorted(list(set(candidates[i]+candidates[j])))
            if len(candidate) == pair_len and not (set(combinations(candidate, pair_len-1)) & candidates_pruned):
                res.add(tuple(candidate))
    return res


def a_prior(chunks, t):
    count = {}
    baskets = []
    for chunk in chunks:
        baskets.append(set(chunk))
        for item in chunk:
            if item not in count:
                count[item] = 1
            else:
                count[item] = count[item] + 1

    candidates = []
    for item, n in count.items():
        if n >= t:
            candidates.append((item,))
    candidates.sort()
    yield 1, candidates
    candidates = set(candidates)
    candidates_pruned = set()
    pair_len = 2

    while candidates:
        # gen new candidate pairs
        candidates = make_combinations(list(candidates), pair_len, candidates_pruned)
        count = {}
        for candidate in candidates:
            for basket in baskets:
                if set(candidate).issubset(basket):
                    if candidate not in count:
                        count[candidate] = 1
                    else:
                        count[candidate] = count[candidate] + 1
        candidates_pruned = candidates
        candidates = set()
        for item, n in count.items():
            if n >= t:
                candidates.add(item)
        candidates_pruned = candidates_pruned - candidates
        if not candidates:
            continue
        candidates = list(candidates)
        candidates.sort()
        yield pair_len, candidates
        candidates = set(candidates)
        pair_len = pair_len + 1


def count_candidates(basket, candidates_all):
    count = {}
    for chunk in basket:
        for candidates in candidates_all:
            for candidate in candidates[1]:
                if set(candidate).issubset(set(chunk)):
                    if candidate not in count:
                        count[candidate] = 1
                    else:
                        count[candidate] = count[candidate] + 1
    for key, value in count.items():
        yield key, value



def main():
    case_number = sys.argv[1]
    support = int(sys.argv[2])
    input_filepath = sys.argv[3]
    output_filepath = sys.argv[4]

    chunk_num = 2
    p = 0.8
    t = p * support

    start = time.time()

    sc = SparkContext('local[*]', 'task1')
    rdd = sc.textFile(input_filepath).mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it)
    if case_number == "1":
        basket = rdd.distinct().map(lambda line: (line.split(',')[0], line.split(',')[1])).partitionBy(chunk_num).groupByKey().map(lambda x: list(x[1])).cache()
    elif case_number == "2":
        basket = rdd.distinct().map(lambda line: (line.split(',')[1], line.split(',')[0])).partitionBy(chunk_num).groupByKey().map(lambda x: list(x[1])).cache()
    else:
        raise Exception("Invalid Case Number {}".format(case_number))
    phase_1 = basket.mapPartitions(lambda chunk: a_prior(chunk, t/chunk_num)).reduceByKey(lambda x, y: set(x).union(set(y))).mapValues(lambda x: sorted(x))
    candidates = phase_1.collect()
    candidates.sort()

    phase_2 = basket.mapPartitions(lambda chunk: count_candidates(chunk, candidates)).reduceByKey(lambda x, y: x+y).sortBy(lambda x: (len(x[0]), x[0])).filter(lambda x: x[1]>=support)
    frequent_items = phase_2.collect()

    duration = time.time()-start
    print("duration")
    print(duration)

    with open(output_filepath, "w") as f:
        f.write("Candidates:\n")
        for pair_len, candidate in candidates:
            if pair_len == 1:
                line = ",".join([str(c).replace(",", "") for c in candidate])+"\n"
                f.write(line)
                f.write("\n")
            else:
                f.write(",".join([str(c) for c in candidate]) + "\n")
                f.write("\n")

        f.write("Frequent Itemsets:\n")
        prev_len = 1
        line = ""
        for item in frequent_items:
            item = item[0]
            cur_len = len(item)
            if cur_len > prev_len:
                line = line[:-1] + "\n\n"
                f.write(line)
                line = ""
                prev_len = cur_len
            if cur_len == 1:
                line = line + str(item).replace(",", "") + ","
            else:
                line = line + str(item) + ","
        f.write(line[:-1])


if __name__ == "__main__":
    main()