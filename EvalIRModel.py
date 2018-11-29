import numpy as np

from ParserQuery import SplitQueryParser
from Evaluation import IRList, PrecisionRecallEval, PrecisionMeanEval, PrecisionAtN, ClusterRecallAtN

class EvalIRModel:
    
    def __init__(self, 
                 filename_queries="../cacm/cacm.qry", 
                 filename_jugements="../cacm/cacm.rel"):
        self.filename_queries = filename_queries
        self.filename_jugements = filename_jugements
        self.precision_recall = PrecisionRecallEval()
        self.precision_mean = PrecisionMeanEval()
        self.precision_20 = PrecisionAtN()
        self.cluster_recall_20 = ClusterRecallAtN()
        
    def initParser(self, train_prop=1, seed=42):
        self.query_parser = SplitQueryParser()
        self.query_parser.initFile(self.filename_queries, self.filename_jugements, train_prop, seed)
    
    def evalModel(self, model, ranking_call = lambda m, text: m.getRanking(text), train_prop=1, seed=42, mode = 'train'):
        self.initParser(train_prop=train_prop, seed=seed)
        precision_recalls = []
        precision_means = []
        precision_20 = []
        cluster_recall_20 = []
        if mode == 'train':
            query = self.query_parser.next_train_query()
        else:
            query = self.query_parser.next_test_query()
        while query:
            rank = ranking_call(model, query.text_)#model.getRanking(query.text_)
            irlist = IRList(query, rank)
            
            x, y = self.precision_recall.eval(irlist)
            precision_recalls.append(y) # TODO fix nulls before ??????
            precision_means.append(self.precision_mean.eval(irlist))
            precision_20.append(self.precision_20.eval(irlist, 20))
            cluster_recall_20.append(self.cluster_recall_20.eval(irlist, 20))
            
            if mode == 'train':
                query = self.query_parser.next_train_query()
            else:
                query = self.query_parser.next_test_query()
        
        precision_recalls = np.array(precision_recalls)#.mean(axis=0)
        precision_means = np.array(precision_means)
        output = {
            'precision_recall':precision_recalls.mean(axis=0),
            'precision_recall_std':precision_recalls.std(axis=0),
            'precision_mean':precision_means.mean(),
            'precision_mean_std':precision_means.std(),
            'precision_at_20':precision_20.mean(),
            'cluster_recall_at_20':cluster_recall_20.mean()
        }
        return output
       

        
