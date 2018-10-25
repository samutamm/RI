
import numpy as np
import pandas as pd

class Featurer:

    def __init__(self, index):
        self.index = index
        self.corpus_features = self.preCalculFeatures()

    def getFeatures(self, doc_id, query):

        # calcul features for query

        # calcul features for query and corpus

        pass


    def preCalculFeatures(self):
        """
        Let's precalcule features that are independent from the query.

        -longueur du document
        -nombre de termes
        -somme des idf des termes du document
        -pagerank

        """
        doc_features = self.index.getDocFeatures()
        # add pagerank
        df = pd.DataFrame(doc_features, columns=['id','len','stems','idfs'])
        df['pagerank'] = np.zeros(df.shape[0])
        return df



