import random

def create_dico_graph(index):
    graph = {}
    doc_ids = index.getDocIds()
    for doc_id in doc_ids:
        try:
            graph[int(doc_id)] = [int(l) for l in index.getLinksForDoc(doc_id)]
        except:
            import pdb; pdb.set_trace()
    return graph

class RandomWalk:
    def __init__(self, graphe):
        ''' le graphe donnée en paramètre est un dictionnaire avec pour clés les id des documents,
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

