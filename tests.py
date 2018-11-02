import pickle
import numpy as np
import pandas as pd

#from Document import Document
#from ParserCACM import ParserCACM
#from ParserQuery import QueryParser
#from porter import stem
#from TextRepresenter import PorterStemmer
#from EvalIRModel import EvalIRModel
from Index import Index, InvertedIndexPlaces
from Weighter import WeighterBoolean, WeighterVector
from IRModel import Vectoriel, LanguageModel, BM25Model, LinearMetaModel
from RandomWalk import *
from Featurer import Featurer

index = Index("cacm", "cacm/cacm.txt")
weighter = WeighterVector(index)
#weighter.calculeNorms()
#vectoriel_model = Vectoriel(weighter)
#bm25_model = BM25Model(weighter)
#language_model = LanguageModel(weighter)
''' PageRank '''
'''
#EvalIRModel().evalModel(LanguageModel)
#language_model.getRanking('computer sciences', lambd=1)
ranking = bm25_model.getRanking('computer sciences')
S = np.array(ranking)[:10, 0]
graph = whole_graph(index)
graphq, Vq = sub_graph(graph, S, 3)
pr2 = PageRank2(graphq)
mus = pr2.compute_mus(10)
ranking = np.array(ranking, dtype=object)
Vq = np.array(Vq, dtype=object)
pr_ranking = np.concatenate((Vq.reshape(-1,1), mus.reshape(-1,1)), axis=1)
pr_ranking = pr_ranking[pr_ranking[:, 1].argsort(-1)[::-1]]

print("BM25 ranking:", ranking)
print("PR ranking:", pr_ranking)
'''

''' LinearMetaModel '''
fl = Featurer(index)
lmm = LinearMetaModel(fl)

