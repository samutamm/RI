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
            if doc_id not in scores:
                scores_ranked.append([str(doc_id), 0])
                
        rank = scores_ranked
        return [[str(rank[i][0]), float(rank[i][1])] for i in range(len(rank))]
            
