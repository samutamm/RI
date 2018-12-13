
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GreedyRanking:

    def __init__(self, index):
        self.index = index
        self.sim1 = cosine_similarity
        self.sim2 = cosine_similarity
        self.value_function = make_mmr(self.sim1)

        self.vector_cache = {} # add vectors to cache to avoid recalculating them

    def diversified_rank(self, document_list, query_Q, K = 20):
        pivot = []
        pivot_index = []
        if query_Q not in self.vector_cache:
            doc_vectors, identity2index, query_vec = calculate_vector_presentation(self.index, document_list, query_Q)
            self.vector_cache[query_Q] = (doc_vectors, identity2index, query_vec)
        else:
            doc_vectors, identity2index, query_vec = self.vector_cache[query_Q]

        similarity_matrix = self.sim2(doc_vectors, doc_vectors)
        for i in range(K):

            all_values_min = 0 if i == 0 else np.array(all_values).min()
            # take 0 for first iteration, otherwise the minimum from last iteration
            all_values = []
            for doc_ident in document_list:

                if doc_ident not in pivot:
                    #doc_u_text = self.index.getStrDoc(doc_ident)
                    doc_u_vecs = doc_vectors[identity2index[doc_ident]]
                    doc_index = identity2index[doc_ident]
                    value = self.value_function(query_vec, doc_u_vecs, doc_index, pivot_index, similarity_matrix)
                else:
                    value = all_values_min
                all_values.append(value)


            d_idx = np.array(all_values).argmax()

            pivot.append(document_list[d_idx])
            pivot_index.append(d_idx)

        return pivot

def calculate_vector_presentation(index, document_list, query_Q):
    identity2index = {}

    docs = []
    for i, doc_i in enumerate(document_list):
        identity2index[doc_i] = i
        docs.append(index.getStrDoc(doc_i))

    vectorizer = CountVectorizer()
    doc_vectors = vectorizer.fit_transform(docs)
    query_vec = vectorizer.transform([query_Q])[0]
    return doc_vectors, identity2index, query_vec

def make_mmr(sim1, alpha = 0.5):

    def mmr(query_vec, doc_u_vecs, doc_index, pivot_index, similarity_matrix):
        max_distance_to_pivot = 0 if len(pivot_index) == 0 else similarity_matrix[doc_index, pivot_index].max()
        value = alpha * sim1(doc_u_vecs, query_vec) - (1 - alpha) * max_distance_to_pivot
        return value

    return mmr
