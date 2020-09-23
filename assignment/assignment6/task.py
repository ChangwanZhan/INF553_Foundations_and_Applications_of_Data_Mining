import sys
import random
import numpy as np
from pyspark import SparkContext
from sklearn.cluster import KMeans


class DS:
    def __init__(self, features):
        self.N = len(features)
        self.SUM = np.sum(features, axis=0)
        self.SUMSQ = np.sum(features**2, axis=0)
        self.sig = np.sqrt((self.SUMSQ / self.N) - (self.SUM / self.N)**2)


class CS:
    def __init__(self, features, pids):
        self.N = len(features)
        self.SUM = np.sum(features, axis=0)
        self.SUMSQ = np.sum(features**2, axis=0)
        self.sig = np.sqrt((self.SUMSQ / self.N) - (self.SUM / self.N) ** 2)
        self.pids = pids


def gen_mahalanobis_dis(p, s):
    return np.sqrt(np.sum(((s.SUM / s.N - p) / s.sig) ** 2))


def update_param(c, idx, p):
    c[idx].N += 1
    c[idx].SUM += p
    c[idx].SUMSQ += p ** 2
    c[idx].sig = np.sqrt((c[idx].SUMSQ / c[idx].N) - (c[idx].SUM / c[idx].N) ** 2)


def main():
    input_file, output_file = sys.argv[1], sys.argv[3]
    cluster_num = int(sys.argv[2])
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("ERROR")
    data = sc.textFile(input_file).map(lambda x: x.split(",")).collect()

    with open(output_file, "w") as f:
        f.write("The intermediate results:\n")

    # divide data into 5 parts(20%)
    random.shuffle(data)
    partitions = [data[i::5] for i in range(5)]
    pid = {}
    gt, pred = {}, {}
    features = []
    for partition in partitions:
        feature = []
        for p in partition:
            t_pid, p_cluster, t_feature = int(p[0]), int(p[1]), p[2:]
            gt[t_pid] = p_cluster
            feature.append(t_feature)
            pid[tuple(t_feature)] = t_pid
        features.append(feature)
    dim = len(features[0][0])
    K = cluster_num * 8

    # DS initialization
    rs, ds, cs = [], [], []
    ds_point, cs_point = 0, 0
    cs_cluster = 0
    feature = np.array(features[0])
    labels = KMeans(n_clusters=K, random_state=0).fit_predict(np.array(feature, dtype=np.float64))

    for label in range(K):
        label_id = np.argwhere(labels == label)
        if len(label_id) == 1:
            label_id = label_id.reshape(label_id.shape[0])
            rs += list(feature[label_id])
            labels = np.delete(labels, label_id)
            feature = np.delete(feature, label_id, 0)
    labels = KMeans(n_clusters=cluster_num, random_state=0).fit_predict(np.array(feature, dtype=np.float64))

    for label in range(cluster_num):
        label_id = np.argwhere(labels == label)
        label_id = label_id.reshape(label_id.shape[0])
        label_feature = feature[label_id]
        for lf in label_feature:
            pred[pid[tuple(lf)]] = label
            ds_point += 1
        ds.append(DS(np.array(label_feature, dtype=np.float64)))

    rs = np.array(rs)
    if rs.shape[0] > K:
        rs_labels = KMeans(n_clusters=K, random_state=0).fit_predict(np.array(rs, dtype=np.float64))
        for label in range(K):
            label_id = np.argwhere(rs_labels==label)
            if len(label_id) > 1:
                label_id = label_id.reshape(-1)
                label_feature = rs[label_id]
                labeled_pids = []
                for lf in label_feature:
                    labeled_pids.append(pid[tuple(lf)])
                cs.append(CS(np.array(label_feature, dtype=np.float64), labeled_pids))
                cs_cluster += 1
                cs_point += len(label_id)
                rs = np.delete(rs, label_id, 0)
                rs_labels = np.delete(rs_labels, label_id)

    with open(output_file, "a") as f:
        f.write("Round 1: {},{},{},{}\n".format(ds_point, cs_cluster, cs_point, rs.shape[0]))

    for i, feature in enumerate(features[1:]):
        feature = np.array(feature)
        for j in range(feature.shape[0]):
            p = np.array(feature[j], dtype=np.float64)
            min_dis, assigned_cluster = 2 * (dim ** 0.5), -1
            for c_id, cluster in enumerate(ds):
                mhlb_dis = gen_mahalanobis_dis(p, cluster)
                if mhlb_dis < min_dis:
                    min_dis, assigned_cluster = mhlb_dis, c_id
            if assigned_cluster != -1:
                update_param(ds, assigned_cluster, p)
                pred[pid[tuple(feature[j])]] = assigned_cluster
                ds_point += 1
                continue

            min_dis, assigned_cluster = 2 * (dim ** 0.5), -1
            for c_id, cluster in enumerate(cs):
                mhlb_dis = gen_mahalanobis_dis(p, cluster)
                if mhlb_dis < min_dis:
                    min_dis, assigned_cluster = mhlb_dis, c_id
            if assigned_cluster != -1:
                update_param(cs, assigned_cluster, p)
                cs_point += 1
                continue
            # print(rs, feature[j])
            rs = np.append(rs, [feature[j]], axis=0)

        if rs.shape[0] > K:
            rs_labels = KMeans(n_clusters=K, random_state=0).fit_predict(np.array(rs, dtype=np.float64))
            for label in range(K):
                label_id = np.argwhere(rs_labels == label)
                if len(label_id) > 1:
                    label_id = label_id.reshape(-1)
                    label_feature = rs[label_id]
                    labeled_pids = []
                    for lf in label_feature:
                        labeled_pids.append(pid[tuple(lf)])
                    cs.append(CS(np.array(label_feature, dtype=np.float64), labeled_pids))
                    cs_cluster += 1
                    cs_point += len(label_id)
                    rs = np.delete(rs, label_id, 0)
                    rs_labels = np.delete(rs_labels, label_id)

        for j in range(cs_cluster-1):
            k = j+1
            while k < cs_cluster:
                if gen_mahalanobis_dis(cs[k].SUM/cs[k].N, cs[j]) < 2 * (dim ** 0.5):
                    cs[j].N += cs[k].N
                    cs[j].SUM += cs[k].SUM
                    cs[j].SUMSQ += cs[k].SUMSQ
                    cs[j].sig = np.sqrt((cs[j].SUMSQ / cs[j].N) - (cs[j].SUM / cs[j].N)**2)
                    cs.pop(k)
                    cs_cluster -= 1
                else:
                    k += 1

        if i == 4:  # the last round
            j = 0
            while j < cs_cluster:
                min_dis, assigned_cluster = 2 * (dim ** 0.5), -1
                for k in range(cluster_num):
                    mhlb_dis = gen_mahalanobis_dis(cs[j].SUM/cs[j].N, ds[k])
                    if mhlb_dis < min_dis:
                        min_dis = mhlb_dis
                        assigned_cluster = k
                if assigned_cluster != -1:
                    for pid in cs[j].pids:
                        pred[pid] = assigned_cluster
                    ds[assigned_cluster].N += cs[j].N
                    ds[assigned_cluster].SUM += cs[j].SUM
                    ds[assigned_cluster].SUMSQ += cs[j].SUMSQ
                    ds[assigned_cluster].sig = np.sqrt((ds[assigned_cluster].SUMSQ / ds[assigned_cluster].N) - (ds[assigned_cluster].SUM / ds[assigned_cluster].N) ** 2)
                    ds_point += cs[j].N
                    cs_cluster -= 1
                    cs_point -= cs[j].N
                    cs.pop(j)
                else:
                    j += 1

        with open(output_file, "a") as f:
            f.write("Round {}: {},{},{},{}\n".format(i+2, ds_point, cs_cluster, cs_point, rs.shape[0]))

    with open(output_file, "a") as f:
        f.write("\nThe clustering results:\n")
        for i in sorted(pred.keys()):
            f.write("{},{}\n".format(i, pred[i]))


if __name__ == '__main__':
    main()