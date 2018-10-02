import numpy as np
import cPickle

class IRModel:
    
    def __init__(self, weighter):
        self.weighter = weighter
        self.norm_file = "indexes/document_norm_"
    
    def getScores(self, query):
        pass
    
    def getRanking(self, query):
        pass
    
class Vectoriel(IRModel):
    
    def getScores(self, query, normalized=False):
        wtq = self.weighter.getWeightsForQuery(query)    
        s = 0
        for term in query.keys():
            s += (wtq[term] * np.array(self.weighter.getDocWeightsForStem(term).values())).sum()
        return s
    
    def getRanking(self, query):
        pass
    
    def calculeNorms(self):
        """
            Precalcule les norms : Je suis pas si c'est comme ca mais reponse à prochaine:
            
            Attention: les normes des vecteurs des documents ne doivent pas être
            recalculées à chaque nouvelle requête.
        """
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
            
    def getNormDocs(self, docs):
        class_name = self.__class__.__name__.lower()
        norm_filename = self.norm_file + class_name
        with open(norm_filename, 'rb') as f:
            norms = cPickle.Unpickler(f).load()
        #import pdb; pdb.set_trace()
        result = []
        for d in docs:
            result.append(norms[d])
        return result