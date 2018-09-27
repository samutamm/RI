
from TextRepresenter import PorterStemmer
import sys

import cPickle

    
class Weighter:
    
    
    def __init__(self, index):
        self.index = index
        self.stemmer = PorterStemmer()
        with open(r"indexes/dictionary", "rb") as file:
            self.terms = cPickle.load(file)
    
    def getDocWeightsForDoc(self, idDoc):
        pass
    
    def getDocWeightsForStem(self, stem):
        pass
    
    def getWeightsForQuery(self, query):
        pass
    
class WeighterBoolean(Weighter):
    
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
    
    def getDocWeightsForDoc(self, idDoc):
        return self.index.getTfsForDoc(idDoc)
    
    def getDocWeightsForStem(self, stem):
        return self.index.getTfsForStem(stem)
    
    def getWeightsForQuery(self, query):
        return query