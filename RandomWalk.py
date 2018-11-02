import random
import numpy as np
from scipy.sparse import csc_matrix

import pdb

def create_dico_graph(index):
    graph = {}
    doc_ids = index.getDocIds()
    for doc_id in doc_ids:
        try:
            graph[int(doc_id)] = [int(l) for l in index.getLinksForDoc(doc_id)]
        except:
            import pdb; pdb.set_trace()
    return graph

def whole_graph(index):
    ''' Arguments:
        index
        Returns:
        graph: the graph as a sparse matrix of outgoing links
    '''
    I = []
    J = []
    doc_ids = index.getDocIds()
    for i in doc_ids:
        for j in index.getLinksForDoc(i):
            I.append(int(i) - 1) # link sortant, index commence d'un
            J.append(int(j) - 1) # link entrant
    I = np.array(I)
    J = np.array(J)
    V = np.ones(I.shape[0])
    n = I.max() + 1
    m = n
    graph = csc_matrix((V,(I,J)), shape=(n,m))
    return graph

def sub_graph(graph, seeds, number_of_backlinks):
    ''' Arguments:
        graph : global sparse matrix of outgoing links
        seeds : list or set S
        number_of_backlinks : k
        Returns:
        sub_graph: the graph as a sparse matrix of outgoing links
        Vq
    '''
    # à faire : s'arranger pour avoir des index consistants (tous int ou tous string, pas de float…)
    seeds = np.array(seeds).astype(float).astype(int)
    #Vq = seeds.copy()
    Vq = set(seeds)
    for doc in seeds:
        doc = int(float(doc))
        #Vq.extend(seeds[doc, :].nonzero()[1])
        #Vq.extend(random.sample(list(seeds[:, doc].nonzero()[0]), number_of_backlinks))
        Vq.update(graph[doc, :].nonzero()[1])
        backlinks = list(graph[:, doc].nonzero()[0])
        Vq.update(random.sample(backlinks, min(number_of_backlinks, len(backlinks))))
    Vq = list(Vq)
    sub_graph = graph[Vq, :][:, Vq]
    return sub_graph, Vq


class RandomWalk:
    def __init__(self, graphe):
        ''' le graphe donné en paramètre est un dictionnaire avec pour clés les id des documents,
            pour valeur une liste contenant les id des documents vers lesquels pointent les liens contenus dans le document-clé
            et une liste contenant les id des documents qui possèdent des liens pointant vers le document-clé
        '''
        self.graph = graphe
        self.mus = [1/len(graphe)]*len(graphe)
    def compute_mus(self, d, n_iter):
        pass

class PageRank(RandomWalk):
    def compute_mus(self, d, n_iter):
        courant = random.choice(list(self.graph.keys()))
        l_j =  len(self.graph[courant][0])
        for i in range(len(self.mus)):
            self.mus[i] = (1-d)/len(self.graph) + d**self.mus[i]
        
    def document_suivant(self, courant, d):
        if random.random() < d:
            #suivant = self.graph[courant][random.randint(0, len(self.graph[courant]-1)]
            suivant = random.choice(self.graph[courant][0])
        else:
            #suivant = list(self.graph.keys())[random.randint(0, len(self.graph)-1)]
            suivant = random.choice(list(self.graph.keys()))
        return suivant

    def update_mus(self, courant):
        pass

#A = csc_matrix(G,dtype=np.float)
class PageRank2(RandomWalk):

    def __init__(self, graph):
        '''
            Prend une matrice sparse en entrée
        '''
        self.graph = graph

    def compute_mus(self, n_iter, d = .85):
        n = self.graph.shape[0]

        A = self.graph
        rsums = np.array(A.sum(1))[:, 0]
        ri, ci = A.nonzero()
        A.data /= rsums[ri]

        sink = rsums == 0

        ro, r = np.zeros(n), np.ones(n)

        for _ in range(n_iter): # TODO replace with converge criterion
            ro = r.copy()

            for i in range(n):
                Ai = np.array(A[:, i].todense())[:, 0]

                Di = sink / float(n)

                Ei = np.ones(n) / float(n)

                r[i] = ro.dot(Ai * d + Di * d + Ei * (1 - d))

        return r / float(sum(r))

