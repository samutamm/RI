import pdb

import numpy as np
import pandas as pd
import pickle
import random

from ParserQuery import QueryParser, RandomQueryParser

class IRModel:
    def __init__(self, weighter):
        self.weighter = weighter
        if weighter is not None:
            self.docNorms = weighter.getDocNorms()
    
    def getScores(self, query):
        pass
    
    def getRanking(self, query):
        pass
    
    def _count_ranking(self, scores, minimum=0):
        scores_unranked = np.array([(key, scores[key]) for key in scores.keys()])
        scores_ranked = scores_unranked[scores_unranked[:,1].argsort()[::-1]].tolist()
        # let's add all documents that does not contain query words
        all_doc_ids = self.weighter.index.getDocIds()
        for doc_id in all_doc_ids:
            if int(doc_id) not in scores.keys():
                scores_ranked.append([str(doc_id), minimum])
                
        rank = scores_ranked
        return [[int(rank[i][0]), float(rank[i][1])] for i in range(len(rank))]
    
class Vectoriel(IRModel):
    
    def getScores(self, query, normalized=False):
        wtq = self.weighter.getWeightsForQuery(query)    
        s = {}
        
        for term in query.keys():
            term_docs = self.weighter.getDocWeightsForStem(term)
            
            term_norm = np.sqrt((np.array(list(term_docs.values())) ** 2).sum())
            for doc_id in term_docs.keys():
                doc_norm = self.docNorms[str(doc_id)]
                if doc_id not in s:
                    s[doc_id] = 0
                norm = term_norm * doc_norm if normalized else 1
                s[int(doc_id)] += wtq[term] * term_docs[doc_id] / norm
        
        return s
    
    def getRanking(self, query, normalized=True):
        stemmer = self.weighter.stemmer
        query_vector = stemmer.getTextRepresentation(query)
        scores = self.getScores(query_vector, normalized=normalized)
        return self._count_ranking(scores)
        
            
# Modele de langue probabiliste, utilisant l'index inversé
# jamais de boucle sur l'ensemble des docs :
#for i in docs
  #for …
# mais faire
#for t in query
  #…
# term frequency n'est pas réellement une fréquence, mais c'était le terme utilisé dans les premiers papiers.
# attention les scores sont négatifs.
# [] pénalité globable
# for t in query
#  l = getdocweightsforstem(t) 
#  for d in l:
#   st = w
# L(d) somme des tf
class LanguageModel(IRModel):
    def __init__(self, weighter):
        super().__init__(weighter)
        self.l_docs_ = {int(idx):sum(self.weighter.getDocWeightsForDoc(idx).values()) for idx in self.weighter.index.getDocIds()}    
    def getScores(self, query, lambd=1):
        scores = {} 
        tw4q = self.weighter.getWeightsForQuery(query)
        l_c = sum(self.l_docs_.values())
        score_absents = 0
        for stem in query.keys():
            dw4s = self.weighter.getDocWeightsForStem(stem)
            tf_t_c = sum(dw4s.values())
            score_absents += tw4q[stem] * np.log((1-lambd)*(tf_t_c/l_c))
            for d in dw4s.keys():
                #print(dw4s[d]/self.l_docs_[d])
                if d not in scores:
                    scores[d] = 0
                scores[d] += tw4q[stem] * np.log( lambd*(dw4s[d]/self.l_docs_[d]) + (1-lambd)*(tf_t_c/l_c) )
        #print("Score minimal :", score_absents)
        return scores, score_absents
    def getRanking(self, query, lambd=1):
        stemmer = self.weighter.stemmer
        query_vector = stemmer.getTextRepresentation(query)
        scores, score_absents = self.getScores(query_vector, lambd=lambd)
        ranking = self._count_ranking(scores, minimum=score_absents)
        return ranking


# BM25 vectoriel

def idf_prime(term, df_term, N):
    dft = len(df_term)
    idf = np.log((N - dft + 0.5) / (dft + 0.5))
    return np.max([0, idf])

class BM25Model(IRModel):
    
    def __init__(self, weighter):
        super().__init__(weighter)
        self.N = len(weighter.index.getDocIds())
        self.L = {int(idx):sum(self.weighter.getDocWeightsForDoc(idx).values()) for idx in self.weighter.index.getDocIds()}    

    def getScores(self, query, k1=1, b=0.75):
        L_mean = self.weighter.doc_mean_length
        scores = {}
        tw4q = self.weighter.getWeightsForQuery(query)
        for stem in query.keys():
            dw41 = self.weighter.getDocWeightsForStem(stem)
            idf = idf_prime(stem, dw41, self.N)
            for doc_id in dw41.keys():
                tf_td = dw41[doc_id]
                nominator = (k1 + 1) * tf_td
                denominator = k1 * ((1 - b) + b*self.L[doc_id] / L_mean) + tf_td
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += idf * (nominator / denominator)
        #print(scores)
        return scores
    
    def getRanking(self, query, k1=1, b=0.75):
        stemmer = self.weighter.stemmer
        query_vector = stemmer.getTextRepresentation(query)
        scores = self.getScores(query_vector, k1=k1, b=b) #  k1=1, b=0.75
        ranking = self._count_ranking(scores)
        return ranking


def loss(relevants, irrelevants, scores, lambda_, thetas):
    norm = lambda_ * np.sqrt((thetas ** 2).sum())
    cross_product = scores[relevants].to_frame("d")\
                    .assign(foo=1)\
                    .merge(scores[irrelevants].to_frame("d_prime").assign(foo=1), on='foo')\
                    .drop('foo', 1)

    diff = 1 - cross_product["d"] + cross_product["d_prime"]
    diff[diff < 0] = 0
    return (diff + norm).mean()


class MetaModel(IRModel):
    def __init__(self, featurers_list):
        super().__init__(None)
        self.featurers_list = featurers_list

class LinearMetaModel(MetaModel):
    def __init__(self,
                 featurers_list,
                 filename_queries="cacm/cacm.qry",
                 filename_jugements="cacm/cacm.rel"):
        super().__init__(featurers_list)
        
        self.filename_queries = filename_queries
        self.filename_jugements = filename_jugements
        self.initialize_weights()

        query_parser = QueryParser()
        query_parser.initFile(self.filename_queries, self.filename_jugements)
        self.query_minmax = query_parser.get_query_min_max(self.featurers_list.index)
        self.doc_ids = np.array(list(featurers_list.index.getDocIds())).astype(int)

        self.feature_cache = {} # save features here to not to recalcul in gradient descent
        self.theta_filename="models/linear_metamodel_thetas"
        
    def initialize_weights(self):
        keys, example_features = self.featurers_list.get_features(1, "test")
        self.attribute_names = keys
        self.thetas = np.random.randn(len(example_features)) / 10 

    def train(self, max_iter, epsilon, lambda_, loss_interval = 5):
        self.initialize_weights()
        query_parser = RandomQueryParser()        
        query_parser.initFile(self.filename_queries, self.filename_jugements)
        queries = pd.Series([query_parser.queries_[k] for k in query_parser.query_keys_])
        # train
        print(self.thetas)
        print("Queries : " + str(queries.shape))
        losses = []
        docs = self.doc_ids
        for i in range(max_iter):
            query = queries[np.random.choice(queries.shape[0])]

            relevants = np.array(query.relevants_).astype(int) 
            irrelevants = docs[~np.isin(docs, relevants)]

            d = random.choice(relevants)
            dp = random.choice(irrelevants)
             
            scores = self.getScores(query.text_)
            _, scores_d = self.featurers_list.get_features(d, query.text_)
            _, scores_dp = self.featurers_list.get_features(dp, query.text_)
            scores_d = np.array(scores_d)
            scores_dp = np.array(scores_dp)
            score_d = scores.get(int(d))
            score_dp = scores.get(int(dp))
            if 1 - score_d + score_dp > 0:
                self.thetas += epsilon*(scores_d - scores_dp)
                self.thetas = (1 - 2*epsilon*lambda_)*self.thetas

            if i % loss_interval == 0:
                print("Iteration {}".format(str(i)))
                query_losses = 0
                for query in queries:
                    relevants = np.array(query.relevants_).astype(int)
                    irrelevants = docs[~np.isin(docs, relevants)]
                    scores = self.getScores(query.text_) # depends of thetas
                    query_losses += loss(relevants, irrelevants, scores, lambda_, self.thetas)
                losses.append((i, query_losses))
        return losses

    def save_weights(self):
        """
            Save weights to file.
        """
        np.save(open(self.theta_filename, 'wb'), self.thetas)
    
    def load_weights(self):
        """
            Load weights from file
        """
        self.thetas = np.load(open(self.theta_filename, 'rb'))

    def getScores(self, query):
        """
        :param plain text query:
        :return: pd.Series of scores indexed by doc_id. Access by .get(int_id)
        """
        if 'scores' not in self.feature_cache: # check cache before calculating again
            feature_scores = []

            keys = None
            for doc_id in self.doc_ids:
                keys, features = self.featurers_list.get_features(int(doc_id), query)
                feature_scores.append(features)

            feature_scores = np.array(feature_scores)
            feature_scores = pd.DataFrame(feature_scores, columns=keys)
            # NORMALIZATION
            normalized_scores = []
            for column_name in feature_scores:
                column = feature_scores[column_name]
                min = column.min()
                max = column.max()
                if column.name == 'query_idf':
                    min = self.query_minmax['query_idf_min']
                    max = self.query_minmax['query_idf_max']
                if column.name == 'query_char_count':
                    min = self.query_minmax['query_len_min']
                    max = self.query_minmax['query_len_max']
                if min == max:
                    print("Columns {} min and max are equal {}".format(column_name, min))
                    continue
                normalized = (column - min) / (max - min)
                normalized_scores.append(normalized)

            normalized_scores = np.array(normalized_scores).T
            self.feature_cache['scores'] = normalized_scores

        normalized_scores = self.feature_cache['scores']
        score = normalized_scores.dot(self.thetas)
        return pd.Series(score, index=self.doc_ids)
    
    def getRanking(self, query):
        scores = self.getScores(query)
        ranking = self._count_ranking(scores)
        return ranking
