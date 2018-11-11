from TextRepresenter import PorterStemmer
import sys

import pickle
import numpy as np
    
class Weighter:
    def __init__(self, index):
        self.index = index
        self.stemmer = PorterStemmer()
        with open(r"indexes/dictionary", "rb") as file:
            self.terms = pickle.load(file)
        self.norm_file = "indexes/document_norm_" + self.__class__.__name__
        self.doc_mean_length = np.array(
            [len(index.getTfsForDoc(doc_id)) for doc_id in index.getDocIds()]
        ).mean()
    
    def getDocWeightsForDoc(self, idDoc):
        pass
    
    def getDocWeightsForStem(self, stem):
        pass
    
    def getWeightsForQuery(self, query):
        pass
    
class WeighterBoolean(Weighter):
    '''
        w_t,d = tf_t,d
        w_t,q = 1       si t appartient à q
        w_t,q = 0       sinon
    '''
    def getDocWeightsForDoc(self, idDoc):
        return self.index.getTfsForDoc(idDoc)
    
    def getDocWeightsForStem(self, stem):
        return self.index.getTfsForStem(stem)
    
    def getWeightsForQuery(self, query):
        new_dico = {}
        for key in query.keys():
            new_dico[key] = 1
        return new_dico
        
        
class WeighterVector(Weighter):
    '''
        w_t,d = tf_t,d
        w_t,q = tf_t,q
    '''
    def getDocWeightsForDoc(self, idDoc):
        return self.index.getTfsForDoc(idDoc)
    
    def getDocWeightsForStem(self, stem):
        return self.index.getTfsForStem(stem)
    
    def getWeightsForQuery(self, query):
        return query
    
    def calculeNorms(self):        
        index_name = self.index.name
        index_places_doc_name = self.index.index_places_doc
        with open(r"indexes/" + index_name + index_places_doc_name, "rb") as index_places_doc_file:
            unpickler = pickle.Unpickler(index_places_doc_file)
            index_places_doc = unpickler.load()
        
        norms = {} # doc id => norm
        with open(r"indexes/" + index_name + "_index", "rb") as doc_file:
            for doc_id in index_places_doc.keys():
                doc_file.seek(index_places_doc[doc_id])
                tfs = pickle.Unpickler(doc_file).load()['stems']
                
                vector = np.array(list(tfs.values()))
                norms[doc_id] = np.sqrt((vector ** 2).sum()) # NORM CALCULS
                
        with open(self.norm_file, 'wb') as f:
            pickle.dump(norms, f)
            
    def getDocNorms(self):
        norm_filename = self.norm_file
        with open(norm_filename, 'rb') as f:
            return pickle.Unpickler(f).load()

class WeighterSchema3(Weighter):
    '''
        w_t,d = tf_t,d
        w_t,q = idf_t   si t appartient à q,
        w_t,q = 0       sinon
    '''
    def getDocWeightsForDoc(self, idDoc):
        return self.index.getTfsForDoc(idDoc)
    
    def getDocWeightsForStem(self, stem):
        return self.index.getTfsForStem(stem)
    
    def getWeightsForQuery(self, query):
        n_D = len(self.index.getDocIds())
        w_q  = {key:n_D/len(self.index.getTfsForStem(key)) if len(self.index.getTfsForStem(key)) > 0 else 0
        for key in query}
        return w_q 
    def calculeNorms(self):        
        index_name = self.index.name
        index_places_doc_name = self.index.index_places_doc
        with open(r"indexes/" + index_name + index_places_doc_name, "rb") as index_places_doc_file:
            unpickler = pickle.Unpickler(index_places_doc_file)
            index_places_doc = unpickler.load()
        
        norms = {} # doc id => norm
        with open(r"indexes/" + index_name + "_index", "rb") as doc_file:
            for doc_id in index_places_doc.keys():
                doc_file.seek(index_places_doc[doc_id])
                tfs = pickle.Unpickler(doc_file).load()['stems']
                vector = np.array(list(tfs.values()))
                norms[doc_id] = np.sqrt((vector ** 2).sum()) # NORM CALCULS
                
        with open(self.norm_file, 'wb') as f:
            pickle.dump(norms, f)
    def getDocNorms(self):
        # même norme que pour le Schéma 2.
        norm_filename = self.norm_file
        with open(norm_filename, 'rb') as f:
            return pickle.Unpickler(f).load()

   
