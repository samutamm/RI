import numpy as np
import pandas as pd

class EvalMeasure:
    def __init__(self):
        pass
    
    def eval(self, ir_list):
        pass
    
    def calcule_stats(self, ir_list):
        """
            Returns pandas.DataFrame table that contains columns :
            -rank : 
            -pertinent : boolean, if list element is relevant
        """
        relevants = ir_list.query.relevants_
        relevants = np.array(relevants, dtype=int)
        relevants_N = len(relevants)
        
        document_rank = np.array(ir_list.document_rank)
        doc_id = document_rank[:, 0]
        scores = document_rank[:,1]
        
        doc_id_column = pd.to_numeric(doc_id, errors='coerce').astype(np.int64)
        score_column = pd.to_numeric(scores, errors='coerce').astype(np.float64)
        # l'air le méme que exemple dans le slide 109
        table = pd.DataFrame({'doc_id':doc_id_column, 'score':score_column})
        
        table['pertinent'] = table.doc_id.isin(relevants)
        #table['pertinent_retrieved'] = np.logical_and(table.pertinent, table.score > 0)
        
        # slide 108 & 115
        table['true_positive'] = table.pertinent.cumsum()
        growing_index = pd.Series(np.arange(1, table.shape[0] + 1)) # 1,2,..n
        table['precision'] = table['true_positive'] / growing_index
        table['rappel'] = table['true_positive'] / relevants_N
        
        return table, relevants_N
    
class PrecisionRecallEval(EvalMeasure):
    def eval(self, ir_list, nbLevels = 20):      
        table,_ = self.calcule_stats(ir_list) # call parent
        
        k_parametres = np.linspace(0,1,nbLevels)
        interpolated_precisions = []
        for k in k_parametres:
            idx = table.rappel >= k
            max_prec = table.precision[idx].max()
            interpolated_precisions.append(max_prec)
        return k_parametres, interpolated_precisions

class PrecisionMeanEval(EvalMeasure):
    def eval(self, ir_list):
        table, relevants_N = self.calcule_stats(ir_list) # call parent
        filtered_table = table[table.pertinent == True]
        return filtered_table.precision.sum() / relevants_N
        
    
class IRList:
    def __init__(self, query, document_rank):
        self.query = query
        self.document_rank = document_rank
