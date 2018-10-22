
from EvalIRModel import EvalIRModel
import numpy as np

def create_2_var_grid(a, b):
    al = len(a)
    bl = len(b)
    
    first_var_length = bl
    second_var_length = max(al, bl)
    
    return list(zip(np.repeat(a, first_var_length), b * second_var_length))

class GridSearch:
    
    def __init__(self, param_a_name=None, param_a_values=[], param_b_name=None, param_b_values=[]):
        self.evaluator = EvalIRModel()
        assert param_a_name is not None, "Please give param name and values."
        self.param_a = param_a_name
        self.param_b = param_b_name
        self.only_one = len(param_b_values) == 0
        self.param_values = param_a_values if self.only_one else create_2_var_grid(param_a_values, 
                                                                                  param_b_values)
        
    def search(self, model):
        results = []
        for values in self.param_values:
            if self.only_one:
                kwargs = {self.param_a: values}
            else:
                kwargs = {self.param_a: values[0], self.param_b: values[1]}
            ranking_call = lambda m, text: m.getRanking(text, **kwargs)
            
            res = self.evaluator.evalModel(model, ranking_call=ranking_call)
            results.append((kwargs, res["precision_mean"]))
        return results