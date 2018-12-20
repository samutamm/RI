
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GreedyRanking:

    def __init__(self, index):
        self.index = index
        self.sim1 = cosine_similarity
        self.sim2 = cosine_similarity

        self.all_doc_ids = list(index.getDocIds())
        self.doc_vectors, self.identity2index, self.vectorizer = calculate_vector_presentation(self.index, self.all_doc_ids)

        self.query_vector_cache = {}

    def set_value_function(self, alpha):
        self.value_function = make_mmr(self.sim1, alpha = alpha)

    def diversified_rank(self, document_list, query_Q, k = 20):
        print("Query length : " + str(len(query_Q)))
        pivot = []
        pivot_index = []

        doc_vectors = self.doc_vectors[[self.identity2index[doc_id] for doc_id in document_list]]

        if query_Q not in self.query_vector_cache:
            self.query_vector_cache[query_Q] = self.vectorizer.transform([query_Q])[0] # text --> sparse vector

        query_vec = self.query_vector_cache[query_Q]

        similarity_matrix = self.sim2(doc_vectors, doc_vectors)
        for i in range(k):
            all_values_min = 0 if i == 0 else np.array(all_values).min()
            # take 0 for first iteration, otherwise the minimum from last iteration
            all_values = []
            for doc_vectors_index, doc_ident in enumerate(document_list):

                if doc_ident not in pivot:
                    doc_u_vecs = doc_vectors[doc_vectors_index]
                    value = self.value_function(query_vec, doc_u_vecs, doc_vectors_index, pivot_index, similarity_matrix)
                else:
                    value = all_values_min
                all_values.append(value)


            d_idx = np.array(all_values).argmax()

            pivot.append(document_list[d_idx])
            pivot_index.append(d_idx)

        return pivot

    def diversify(self, document_rank, query,
                  by_top_n=100,
                  order_n=5):

        top_n_doc_ids = document_rank[:by_top_n, 0]

        data = pd.DataFrame({'text': [self.index.getStrDoc(i) for i in top_n_doc_ids],
                             'rank': np.arange(1, top_n_doc_ids.shape[0] + 1)},
                            index=top_n_doc_ids)
        diversified = self.diversified_rank(top_n_doc_ids, query, k=order_n)
        return data.loc[diversified].reset_index()


def calculate_vector_presentation(index, document_list):
    identity2index = {}

    docs = []
    for i, doc_i in enumerate(document_list):
        identity2index[doc_i] = i
        docs.append(index.getStrDoc(doc_i))

    vectorizer = CountVectorizer()
    doc_vectors = vectorizer.fit_transform(docs)
    return doc_vectors, identity2index, vectorizer

def make_mmr(sim1, alpha = 0.5):

    def mmr(query_vec, doc_u_vecs, doc_index, pivot_index, similarity_matrix):
        max_distance_to_pivot = 0 if len(pivot_index) == 0 else similarity_matrix[doc_index, pivot_index].max()
        value = alpha * sim1(doc_u_vecs, query_vec) - (1 - alpha) * max_distance_to_pivot
        return value

    return mmr
