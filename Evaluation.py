import numpy as np
import pandas as pd

def remove_b(x):
    if x[0] != 'b':
        return x
    return x[2:-1]

class EvalMeasure:
    
    def __init__(self):
        pass
    
    def eval(self, ir_list):
        pass
    
class PrecisionRappelEval(EvalMeasure):
    
    def eval(self, ir_list, nbLevels=3):      
        relevants = ir_list.query.relevants_
        relevants_N = len(relevants)
        rank = np.array(ir_list.document_rank)[:, 0]
        rank = np.vectorize(remove_b)(rank)
        
        pertinent = [(r in relevants) for r in rank]
        # l'air le méme que exemple dans le slide 109
        table = pd.DataFrame({'rank':rank, 'pertinent':pertinent})
        print(table)
        # slide 108 & 115
        k_parametres = np.linspace(0,1,nbLevels)
        
        table['true_positive'] = table.pertinent.cumsum()
        growing_index = pd.Series(np.arange(1, table.shape[0] + 1)) # 1,2,..n
        table['precision'] = table['true_positive'] / growing_index
        table['rappel'] = table['true_positive'] / relevants_N
        
        interpolated_precisions = []
        for k in k_parametres:
            idx = table.rappel >= k
            max_prec = table.precision[idx].max()
            interpolated_precisions.append(max_prec)
        return interpolated_precisions, k_parametres
                    
class IRList:
    
    def __init__(self, query, document_rank):
        self.query = query
        self.document_rank = document_rank