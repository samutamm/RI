import numpy as np

class IRModel:
    
    def __init__(self, weighter):
        self.weighter = weighter
    
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
    