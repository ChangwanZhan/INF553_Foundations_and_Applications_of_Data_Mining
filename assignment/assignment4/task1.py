from pyspark import SparkContext, sql
import sys
import os
from graphframes import *

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")


def main():
    input_file, output_file = sys.argv[1], sys.argv[2]
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("ERROR")
    rdd = sc.textFile(input_file).map(lambda x: x.split()).cache()

    vertices_list = list(set(rdd.flatMap(set).collect()))
    edges_list = rdd.map(tuple).distinct().collect()
    edges_list_undirected = set()

    for i in range(len(vertices_list)):
        vertices_list[i] = tuple([vertices_list[i]])

    for edge in edges_list:
        if edge not in edges_list_undirected:
            edges_list_undirected.add(edge)
        edge2 = (edge[1], edge[0])
        if edge2 not in edges_list_undirected:
            edges_list_undirected.add(edge2)

    sqlContext = sql.SQLContext(sc)
    vertices = sqlContext.createDataFrame(vertices_list, ["id"])
    edges = sqlContext.createDataFrame(edges_list_undirected, ["src", "dst"])
    g = GraphFrame(vertices, edges)
    result = g.labelPropagation(maxIter=5)

    community_list = result.select("id", "label").collect()
    communities = {}
    for community in community_list:
        if community.label not in communities:
            communities[community.label] = []
        communities[community.label].append(community.id)

    communities_res = {}
    for c, ids in communities.items():
        if len(ids) not in communities_res:
            communities_res[len(ids)] = []
        ids_str = "', '".join(sorted(ids))
        ids_str = "'{}'".format(ids_str)
        communities_res[len(ids)].append(ids_str)

    with open(output_file, "w") as f:
        for k in sorted(communities_res):
            for id in sorted(communities_res[k]):
                f.write(id+"\n")


if __name__ == '__main__':
    main()