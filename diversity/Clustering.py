
import sys
sys.path.append('..')

import pandas as pd

from Evaluation import IRList
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import SpectralClustering

class ClusteringDiversifier:

    def __init__(self, index):
        self.index = index

    def diversify(self, ir_list,
                  by_top_n = 100,
                  n_clusters=5):
        rank = ir_list.document_rank[:by_top_n, :]
        data = pd.DataFrame({'text': [self.index.getStrDoc(i) for i in rank[:, 0]],
                             'id': rank[:, 0],
                             'prec': rank[:, 1]})

        data.text = data.text.apply(str.lower)

        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(data.text)

        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        prediction = model.fit_predict(bow_matrix)

        data['cluster'] = prediction
        return data.drop(columns='text')
