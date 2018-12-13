
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# vectorize
def cosinus_similarity(a, b):
    vectorizer = CountVectorizer()
    vecs = vectorizer.fit_transform([a, b])
    return cosine_similarity(vecs[0], vecs[1])[0][0]


class GreedyRanking():

    def __init__(self, index):
        self.index = index
        self.sim1 = cosinus_similarity
        self.sim2 = cosinus_similarity
        self.value_function = make_mmr(self.sim1)

    def diversified_rank(self, document_list, requete_Q, K = 20):
        pivot = []
        pivot_index = []
        similarity_matrix, identity2index = calculate_similarity_matrix(self.index, document_list, self.sim2)

        for i in range(K):

            all_values = []
            for doc_ident in document_list:

                if doc_ident not in pivot:
                    doc_u_text = self.index.getStrDoc(doc_ident)
                    value = self.value_function(requete_Q, doc_u_text, doc_ident, pivot_index, similarity_matrix)
                else:
                    value = np.array(all_values).min()
                all_values.append(value)


            d_idx = np.array(all_values).argmax()

            pivot.append(document_list[d_idx])
            pivot_index.append(d_idx)
            #del document_list[d_i]

        return pivot

def calculate_similarity_matrix(index, document_list, sim2):
    similary_matrix = np.zeros((len(document_list), len(document_list)))

    identity2index = {}

    for i, doc_i in enumerate(document_list):
        identity2index[doc_i] = i
        for j, doc_j in enumerate(document_list):
            similary_matrix[i, j] = sim2(index.getStrDoc(doc_i), index.getStrDoc(doc_j))

    return similary_matrix, identity2index

def make_mmr(sim1, alpha = 0):

    def mmr(requete_Q, doc_u_text, doc_ident, pivot_index, similarity_matrix):
        value = alpha * sim1(doc_u_text, requete_Q) - (1 - alpha) * similarity_matrix[doc_ident, pivot_index].max()
        return value

    return mmr
