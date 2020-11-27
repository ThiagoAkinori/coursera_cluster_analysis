import pandas as pd 
import numpy as np
from sklearn.metrics import  normalized_mutual_info_score

def jaccard_similarity(ground_truth, cluster):
    fp = 0
    fn = 0
    tn = 0
    tp = 0

    df = ground_truth.set_index(0).join(cluster.set_index(0), rsuffix='_cluster')
    df.columns=['true_label', 'cluster_label']

    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i != j:
                if row_i['true_label'] == row_j['true_label'] and row_i['cluster_label'] == row_j['cluster_label']:
                    tp = tp+1
                elif row_i['true_label'] == row_j['true_label'] and row_i['cluster_label'] != row_j['cluster_label']:
                    fn = fn+1
                elif row_i['true_label'] != row_j['true_label'] and row_i['cluster_label'] == row_j['cluster_label']:
                    fp = fp+1
                elif row_i['true_label'] != row_j['true_label'] and row_i['cluster_label'] != row_j['cluster_label']:
                    tn = tn+1
    print(tn)
    return tp/(tp+fn+fp)
    
ground_truth = pd.read_csv('data/partitions.txt', sep=' ', header=None)

cluster_1 = pd.read_csv('data/clustering_1.txt', sep=' ', header=None)
cluster_2 = pd.read_csv('data/clustering_2.txt', sep=' ', header=None)
cluster_3 = pd.read_csv('data/clustering_3.txt', sep=' ', header=None)
cluster_4 = pd.read_csv('data/clustering_4.txt', sep=' ', header=None)
cluster_5 = pd.read_csv('data/clustering_5.txt', sep=' ', header=None)

clusters = [cluster_1, cluster_2, cluster_3, cluster_4, cluster_5]

jaccard = list()
NMI = list()

for cluster in clusters:
    jac_score = jaccard_similarity(ground_truth, cluster)
    norm_mutual_info = normalized_mutual_info_score(ground_truth[1], cluster[1], average_method='geometric')
    jaccard.append(jac_score)
    NMI.append(norm_mutual_info)

df = pd.DataFrame(
    {
        '0':NMI,
        '1':jaccard
    }
    )

df.to_csv('scores.txt', sep=' ', header=None, index= None)
