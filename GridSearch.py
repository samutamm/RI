
from EvalIRModel import EvalIRModel


class GridSearch:
    
    def __init__(self, params):
        self.evaluator = EvalIRModel()
        self.params = param