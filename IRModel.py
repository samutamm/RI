import numpy as np
import pickle

from ParserQuery import QueryParser

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

class MetaModel(IRModel):
    def __init__(self, featurers_list):
        super().__init__(None)
        self.featurers_list = featurers_list
    
class LinearMetaModel(MetaModel):
    def __init__(self, featurers_list):
        super().__init__(featurers_list)
        self.thetas = {model_name:0, for model_name in featurers_list.keys()}

    def train(max_iter, filename_queries="cacm/cacm.qry", 
    filename_jugements="cacm/cacm.rel"):
        query_parser = QueryParser()        
        query_parser.initFile(filename_queries, filename_jugements)

    def getScores(self, query):
        scores = {}
        for doc_id in self.featurers_list.index.getDocIds():
            features = self.featurers_list.getFeatures(doc_id,query)
            score =  0
            for model_name, theta in self.thetas.items()
                score += theta*features[model_name]
            scores[doc_id] = score
            return scores
    
    def getRanking(self, query):
        scores = self.getScores(query)
        ranking = self._count_ranking(scores)
        return ranking
