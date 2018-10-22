import numpy as np

from ParserQuery import QueryParser
from Evaluation import IRList, PrecisionRecallEval, PrecisionMeanEval

class EvalIRModel:
    
    def __init__(self, 
                 filename_queries="cacm/cacm.qry", 
                 filename_jugements="cacm/cacm.rel"):
        self.filename_queries = filename_queries
        self.filename_jugements = filename_jugements
        self.precision_recall = PrecisionRecallEval()
        self.precision_mean = PrecisionMeanEval()
        
    def initParser(self):
        self.query_parser = QueryParser()
        self.query_parser.initFile(self.filename_queries, self.filename_jugements)
    
    def evalModel(self, model, ranking_call = lambda m, text: m.getRanking(text)):
        self.initParser()
        precision_recalls = []
        precision_means = []
        query = self.query_parser.nextQuery()
        while query:
            rank = ranking_call(model, query.text_)#model.getRanking(query.text_)
            irlist = IRList(query, rank)
            
            x, y = self.precision_recall.eval(irlist)
            precision_recalls.append(y) # TODO fix nulls before
            precision_means.append(self.precision_mean.eval(irlist))
            
            query = self.query_parser.nextQuery()
        
        precision_recalls = np.array(precision_recalls).mean(axis=0)
        precision_means = np.array(precision_means)
        output = {
            'precision_recall':precision_recalls,
            'precision_mean':precision_means.mean(),
            'precision_mean_std':precision_means.std()
        }
        return output
       

        