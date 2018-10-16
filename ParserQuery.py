from ParserCACM import ParserCACM 
from TextRepresenter import PorterStemmer

class QueryParser():
    
    def initFile(self, filename_queries, filename_jugements):
        self.queries_ = {}
        parser = ParserCACM()
        parser.initFile(filename_queries)
        doc = parser.nextDocument()
        while doc:
            self.queries_[int(doc.getId())] = Query(doc.getId(), doc.getText().replace('\n', ''))
            doc = parser.nextDocument()
        with open(filename_jugements, 'r') as f:
            for line in f:
                line = line.split(' ') 
                #import pdb; pdb.set_trace()
                self.queries_[int(line[0])].add_relevant(line[1])
        self.index_ = 0
        self.query_keys_ = list(self.queries_.keys())
        
    def nextQuery(self):
        if self.index_ >= len(self.query_keys_):
            return None
        i = self.index_
        self.index_ += 1
        return self.queries_[self.query_keys_[i]]


class Query:
    def __init__(self, id_, text_, relevants=[]):
        self.id_ = id_
        self.text_ = text_
        self.relevants_ = relevants

    def add_relevant(self, relevant):
        if relevant not in self.relevants_:
            self.relevants_.append(relevant)