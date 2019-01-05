import numpy as np

class RandomDiversifier:
    def __init__(self, seed):
        np.random.seed(seed)

    def diversify(self, document_rank, doc_limit=100, order_n=20):
        top_docs = document_rank[:doc_limit]
        np.random.shuffle(top_docs)
        return top_docs
        
