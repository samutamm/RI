
import sys
sys.path.append('..')

import pandas as pd
import numpy as np

from Evaluation import IRList
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import SpectralClustering


def map_cluster(data, clusters, by):
    ranks = pd.DataFrame(
        np.array([(cl, by(data[data['cluster'] == cl]['rank'])) for cl in clusters]),
        columns=['cluster', 'result']
    )
    return ranks

class ClusteringDiversifier:

    def __init__(self, index, cluster_order_by='rank'):
        self.index = index
        self.cluster_order_by = cluster_order_by

    def diversify(self, ir_list,
                  by_top_n = 100,
                  n_clusters=5):
        rank = ir_list.document_rank[:by_top_n, :]
        data = pd.DataFrame({'text': [self.index.getStrDoc(i) for i in rank[:, 0]],
                             'id': rank[:, 0],
                             'rank': np.arange(1, rank.shape[0] + 1)})

        data.text = data.text.apply(str.lower)

        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(data.text)

        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        prediction = model.fit_predict(bow_matrix)

        data['cluster'] = prediction

        data = data.drop(columns='text')

        clusters = np.arange(n_clusters)
        if self.cluster_order_by == 'rank':
            min_ranks = map_cluster(data, clusters, lambda d : d.min())
            cluster_order = min_ranks.sort_values(by='result')['cluster']
        elif self.cluster_order_by == 'size_ascending':
            cluster_sizes = map_cluster(data, clusters, lambda d: d.count())
            cluster_order = cluster_sizes.sort_values(by='result', ascending=True)['cluster']
        elif self.cluster_order_by == 'size_descending':
            cluster_sizes = map_cluster(data, clusters, lambda d: d.count())
            cluster_order = cluster_sizes.sort_values(by='result', ascending=False)['cluster']
        #TODO similarite docs representant

        ids_by_clusters = {}
        for cl in clusters:
            ids_by_clusters[cl] = data[data['cluster'] == cl]['id'].values.tolist()

        diversified = []
        i = 0
        while len(diversified) < data.shape[0]:
            current_cluster = cluster_order[i % n_clusters]
            ids = ids_by_clusters[current_cluster]
            if len(ids) > 0:
                diversified.append(ids.pop(0))
            i += 1

        # https://stackoverflow.com/questions/23482668/sorting-by-a-custom-list-in-pandas
        data.id = data.id.astype("category")
        data.id.cat.set_categories(diversified, inplace=True)
        data = data.sort_values(by="id", ascending=)
        return data, cluster_order, diversified

