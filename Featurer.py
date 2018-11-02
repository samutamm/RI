
import numpy as np
import pandas as pd
from TextRepresenter import PorterStemmer
from Weighter import WeighterVector
from IRModel import BM25Model

def idf(N, num_documents):
    return np.log(N / (num_documents + 1))


class Featurer:

    def __init__(self, index, models=None):
        self.index = index
        print("precalcul features")
        self.doc_N, self.corpus_features = self.pre_calcul_features()
        self.stemmer = PorterStemmer()
        # for each query, stock thing calculated in calcul_features_query and calcul_features_query_doc
        self.query_feature_cache = {}
        self.ranking_feature_cache = {}
        self.models = models if models is not None else [BM25Model(WeighterVector(index))]

    def getFeatures(self, doc_id, query):

        query_array = self.stemmer.getTextRepresentation(query)
        # calcul features for query
        query_idf, query_w_count, query_char_count = self.calculate_features_query(query_array, query)

        # calcul features for query and corpus

        pass


    def pre_calcul_features(self):
        """
        Let's precalcule features that are independent from the query.

        -longueur du document
        -nombre de termes
        -somme des idf des termes du document
        -pagerank

        """
        doc_N, doc_features = self.index.getDocFeatures()
        # add pagerank
        df = pd.DataFrame(doc_features, columns=['id', 'len', 'stems', 'idfs'])
        df['pagerank'] = np.zeros(df.shape[0])
        return doc_N, df


    def calculate_features_query(self, query_array, query):
        """
        For already calculated request, check cache.
        For new query, calculates
        -sum of idfs
        -sum of words
        -length of query

        :param query_array: contains all words in query
        :param query as plain text
        :return:
        """
        if query not in self.query_feature_cache:
            idfs = np.array([idf(self.doc_N, len(self.index.getTfsForStem(w).keys())) for w in query_array])
            self.query_feature_cache[query] = (idfs.sum(), idfs.shape[0], len(query))
        return self.query_feature_cache[query]

    def calculate_features_query_doc(self, doc_id, query):
        """
            For new request, calculates the ranking according the models.
            For existing request, check cache.
        :param doc_id:
        :param query:
        :return: Ranking of given doc_id for given query
        """
        if query not in self.ranking_feature_cache:
            rankings = {}
            for model in self.models:
                model_name = type(model).__name__
                ranking = model.getRanking(query)
                ranking_array = np.array(ranking, dtype=object)
                doc_index, rank = ranking_array.T # split columns
                df = pd.DataFrame({'ranking': rank}, index=doc_index)
                rankings[model_name] = df
            self.ranking_feature_cache[query] = rankings

        query_rankings = self.ranking_feature_cache[query]
        return {model_name:query_rankings[model_name].loc[doc_id].values for model_name in query_rankings.keys()}



