import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class GreedyRankingMMR:
    def __init__(self, index, sim1='score'):
        self.index = index
        self.vectorizer, self.doc_vector_dic = self.vectorize(index)
        self.sim1 = self.sim1_score if sim1=='score' else self.sim1_cos
        # mis à jour à chaque appel de sim1_cos
        self.query_vector_dic = {}
        # mis à jour à chaque appel de sim2
        self.docs_similarity_dic = {}
        # changé à chaque appel de diversify :
        self.docs_score = None
        self.alpha = None
    def vectorize(self, index):
        docs_list = list(index.getDocIds())
        doc_strings = []
        for idx, doc_id in enumerate(docs_list):
            doc_strings.append(index.getStrDoc(doc_id))
        vectorizer = CountVectorizer()
        doc_vectors = vectorizer.fit_transform(doc_strings)
        return vectorizer, dict(zip(docs_list, doc_vectors))

    def diversify(self, query, document_rank, doc_limit=100, order_n=20, alpha=0.85):
        '''
        Diversify the rank given in argument,
        by reordering the first 'order_n' slots
        using the first 'doc_limit' docs.
        '''
        self.docs_score = {doc_rank[0]:doc_rank[1] for doc_rank in document_rank}
        self.alpha = alpha
        top_docs = document_rank[:doc_limit, 0]
        new_docs_list = self.diversified_rank(top_docs, query, order_n)
        new_docs_rank = np.array([[doc_id, self.docs_score[doc_id]] for doc_id in new_docs_list])
        return new_docs_rank
    def diversified_rank(self, docs_list, query, k = 20):
        print("Query length :", len(query))
        pivot = []
        for i in range(k):
            best_doc_id = docs_list[0]
            best_doc_value = -99999
            for doc_id in docs_list:
                if doc_id in pivot:
                    continue
                doc_value = self.value_function(doc_id, pivot, query)
                if doc_value > best_doc_value:
                    best_doc_id = doc_id
                    best_doc_value = doc_value
            pivot.append(best_doc_id)
        return pivot
    def value_function(self, doc_id, pivot, query=None):
        return self.alpha * self.sim1(doc_id, query) - (1-self.alpha)*self.sim_psi(doc_id, pivot)
    def sim1_score(self, doc_id, query=None):
        '''
        Similarité entre le doc et la requête
        Renvoie le score calculé par la baseline, normalisé
        '''
        return self.docs_score[doc_id]
    def sim1_cos(self, doc_id, query):
        '''
        Similarité entre le doc et la requête
        Renvoie la similarité cosine
        '''
        if query not in self.query_vector_dic:
            # text --> sparse vector
            self.query_vector_dic[query] = self.vectorizer.transform([query])[0]
        query_vector = self.query_vector_dic[query]
        doc_vector = self.doc_vector_dic[doc_id]
        return cosine_similarity(doc_vector, query_vector)

    def sim_psi(self, doc_id, pivot):
        '''
        Similarité entre le doc et le pivot
        Renvoie la similarité maximale entre le doc et chaque doc du pivot
        '''
        if len(pivot) == 0:
            return 0
        else:
            return max([self.sim2(doc_id, doc2_id) for doc2_id in pivot])

    def sim2(self, doc1_id, doc2_id):
        '''
        Similarité entre les deux docs
        Renvoie la similarité cosinus
        '''
        if doc1_id not in self.docs_similarity_dic:
            self.docs_similarity_dic[doc1_id] = {}
            #self.compute_cosine(doc1_id)
        if doc2_id not in self.docs_similarity_dic:
            self.docs_similarity_dic[doc2_id] = {}
            #self.compute_cosine(doc2_id)
        if doc2_id not in self.docs_similarity_dic[doc1_id]:
            doc1_vector = self.doc_vector_dic[doc1_id]
            doc2_vector = self.doc_vector_dic[doc2_id]
            sim = cosine_similarity(doc1_vector, doc2_vector)
            self.docs_similarity_dic[doc1_id][doc2_id] = sim
            self.docs_similarity_dic[doc2_id][doc1_id] = sim
        return self.docs_similarity_dic[doc1_id][doc2_id]
    def compute_cosine(self, doc_id):
        '''
        Calcule la similarité cosinus entre le doc et
        les docs préenregistrés dans docs_similarity_dic
        Met à jour docs_similarity_dic
        '''
        doc_vector = self.doc_vector_dic[doc_id]
        doc_similarity_dic = {}
        for doc2_id, doc2_similarity_dic in self.docs_similarity_dic.items():
            doc2_vector = self.doc_vector_dic[doc2_id]
            sim = cosine_similarity(doc_vector, doc2_vector)
            doc_similarity_dic[doc2_id] = sim
            doc2_similarity_dic[doc_id] = sim 
        self.docs_similarity_dic[doc_id] = doc_similarity_dic

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
