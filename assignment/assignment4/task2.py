from pyspark import SparkContext
import sys
import time
from copy import deepcopy


def gen_betweenness(vertices, graph):
    def bfs(root):
        visited = {root}
        queue = [(root, 1)]
        res = [(root, None)]
        level_cur = 0
        d_s_cur = {}
        visited_cur = set()
        for d, level in queue:
            if level != level_cur:
                level_cur = level
                visited = visited.union(visited_cur)
                visited_cur = set()
                for d_cur, s_cur in d_s_cur.items():
                    res.append((d_cur, tuple(s_cur)))
                d_s_cur = {}
            for v in graph[d]:
                if v in visited:
                    continue
                visited_cur.add(v)
                queue.append((v, level+1))
                if not d_s_cur.get(v):
                    d_s_cur[v] = set()
                d_s_cur[v].add(d)
        for d_cur, s_cur in d_s_cur.items():
            res.append((d_cur, tuple(s_cur)))
        return res

    betweenness = {}
    for v in vertices:
        bfs_tree = bfs(v)
        path_count = {}
        for child, parent in bfs_tree:
            if not parent:
                path_count[child] = 1.0
            else:
                path_count[child] = 0.0
                for p in parent:
                    path_count[child] += path_count[p]
        vertices_sum = {}
        for d, s in reversed(bfs_tree):
            if not s:
                continue
            if not vertices_sum.get(d):
                vertices_sum[d] = 1.0
            for p in s:
                if not vertices_sum.get(p):
                    vertices_sum[p] = 1.0
                vertices_sum[p] += float(vertices_sum[d] * path_count[p]/path_count[d])
                if not betweenness.get(tuple(sorted([d, p]))):
                    betweenness[tuple(sorted([d, p]))] = 0
                betweenness[tuple(sorted([d, p]))] += float(vertices_sum[d] * path_count[p]/path_count[d])/2
    return betweenness


def gen_communities(vertices, graph, betweenness):
    m = len(betweenness)
    tmp_betweennes, tmp_graph = betweenness.copy(), deepcopy(graph)
    max_modularity, res = -1, []
    while tmp_betweennes:
        tmp_v = vertices.copy()
        communities = []
        while tmp_v:
            queue = [tmp_v.pop()]
            community = set()
            for q in queue:
                community.add(q)
                for d in tmp_graph[q]:
                    if d not in community:
                        queue.append(d)
            communities.append(sorted(list(community)))
            tmp_v = tmp_v - community
        modularity = 0.0
        for community in communities:
            for i in community:
                for j in community:
                    k_i = len(graph[i])
                    k_j = len(graph[j])
                    a_ij = 1.0 if j in graph[i] else 0.0
                    modularity += a_ij - ((k_i*k_j)/(2*m))

        modularity = modularity/(2*m)
        if max_modularity < modularity:
            res = communities
            max_modularity = modularity
        max_betweenness = max(tmp_betweennes.values())
        for e, b in tmp_betweennes.items():
            if b == max_betweenness:
                tmp_graph[e[0]] -= {e[1]}
                tmp_graph[e[1]] -= {e[0]}
        tmp_betweennes = gen_betweenness(vertices, tmp_graph)
    return res


def main():
    start = time.time()
    input_file, betweenness_output_file, community_output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel("ERROR")

    rdd = sc.textFile(input_file).map(lambda x: x.split()).cache()
    vertices_list = set(sorted(rdd.flatMap(set).collect()))
    edges_list = sorted(rdd.map(lambda x: tuple(sorted(x))).distinct().collect())

    graph = {}
    for edge in edges_list:
        if not graph.get(edge[0]):
            graph[edge[0]] = set()
        if not graph.get(edge[1]):
            graph[edge[1]] = set()
        graph[edge[0]].add(edge[1])
        graph[edge[1]].add(edge[0])

    # betweenness
    betweenness = gen_betweenness(vertices_list, graph)

    communities = gen_communities(vertices_list, graph, betweenness)
    communities.sort(key=lambda x: (len(x), x[0]))

    with open(betweenness_output_file, "w") as f:
        for key, val in sorted(betweenness.items(), key=lambda x: (-x[1], x[0])):
            f.write("{}, {}\n".format(key, val))

    with open(community_output_file, "w") as f:
        for community in communities:
            f.write("'"+"', '".join(community)+"'\n")

    duration = time.time() - start
    print("Duration: "+str(duration))


if __name__ == '__main__':
    main()
