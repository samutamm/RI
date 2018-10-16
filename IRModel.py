import numpy as np
import pickle

class IRModel:
    
    def __init__(self, weighter):
        self.weighter = weighter
        self.docNorms = weighter.getDocNorms() 
    
    def getScores(self, query):
        pass
    
    def getRanking(self, query):
        pass
    
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
    
    def getRanking(self, query, normalized=False):
        stemmer = self.weighter.stemmer
        query_vector = stemmer.getTextRepresentation(query)
        scores = self.getScores(query_vector, normalized=normalized)
        
        scores_unranked = np.array([(str(key),float(scores[key])) for key in scores.keys()], 
                                   dtype=(np.character,np.float))
        scores_ranked = scores_unranked[scores_unranked[:,1].argsort()[::-1]].tolist()
        
        # let's add all documents that does not contain query words
        all_doc_ids = self.weighter.index.getDocIds()
        for doc_id in all_doc_ids:
            if int(doc_id) not in scores.keys():
                scores_ranked.append([str(doc_id), 0])
                
        rank = scores_ranked
        return [[str(rank[i][0]), float(rank[i][1])] for i in range(len(rank))]
            
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
      self.l_docs_ = {idx:sum(self.weighter.getDocWeightsForDoc(idx).values()) for idx in self.weighter.index.keys()}    
    def getScores(self, query, lambd=1):
        s = {} 
        tw4q = self.weighter.getWeightsForQuery(query)
        for t in query:
          dw4t = self.weighter.getDocWeightsForStem()
          tf_t_c = sum(dw4t.values())
          l_c = sum(self.l_docs_.values())
          for d in dw4t:
            s['d'] += tw4q[t] * np.log( lambd*(dw4t[d]/self.l_docs_[d]) + (1-lambd)*(tf_t_c/l_c) )
        return s 
    def getRanking(self, query):
        pass


# BM25 vectoriel
