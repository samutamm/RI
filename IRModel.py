import numpy as np
import cPickle

class IRModel:
    
    def __init__(self, weighter):
        self.weighter = weighter
        self.norm_file = "indexes/document_norm_" # TODO move to weighter
        self.docNorms = self.getDocNorms() 
    
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
            
            term_norm = np.sqrt((np.array(term_docs.values()) ** 2).sum())
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
            
    # TODO move to weighter! ca depend du weighter
    def calculeNorms(self):
        class_name = self.__class__.__name__.lower()
        norm_filename = self.norm_file + class_name
        
        
        index_name = self.weighter.index.name
        index_places_doc_name = self.weighter.index.index_places_doc
        with open(r"indexes/" + index_name + index_places_doc_name) as index_places_doc_file:
            unpickler = cPickle.Unpickler(index_places_doc_file)
            index_places_doc = unpickler.load()
        
        norms = {} # doc id => norm
        with open(r"indexes/" + self.weighter.index.name + "_index", "rb") as doc_file:
            for doc_id in index_places_doc.keys():
                doc_file.seek(index_places_doc[doc_id])
                tfs = cPickle.Unpickler(doc_file).load()
                
                vector = np.array(tfs.values())
                
                norms[doc_id] = np.sqrt((vector ** 2).sum()) # NORM CALCULS
                
        with open(norm_filename, 'wb') as f:
            cPickle.dump(norms, f)
            
    def getDocNorms(self):
        class_name = self.__class__.__name__.lower()
        norm_filename = self.norm_file + class_name
        with open(norm_filename, 'rb') as f:
            return cPickle.Unpickler(f).load()
        #return norms[str(doc_id)]