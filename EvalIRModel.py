import numpy as np

from ParserQuery import QueryParser
from Evaluation import IRList, PrecisionRecallEval, PrecisionMeanEval

class EvalIRModel:
    
    def __init__(self, 
                 filename_queries="cacm/cacm.qry", 
                 filename_jugements="cacm/cacm.rel"):
        self.query_parser = QueryParser()
        self.query_parser.initFile(filename_queries, filename_jugements)
        self.precision_recall = PrecisionRecallEval()
        self.precision_mean = PrecisionMeanEval()
    
    def evalModel(self, model):
        precision_recalls = []
        precision_means = []
        query = self.query_parser.nextQuery()
        while query:
            rank = model.getRanking(query.text_)
            irlist = IRList(query, rank)
            
            #precision_recalls.append(self.precision_recall.eval(irlist)) # TODO fix nulls before
            precision_means.append(self.precision_mean.eval(irlist))
            
            query = self.query_parser.nextQuery()

        precision_means = np.array(precision_means)
        return precision_means.mean(), precision_means.std()
       

        