from random import shuffle

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
                self.queries_[int(line[0])].add_relevant(line[1])
        self.index_ = 0
        self.query_keys_ = list(self.queries_.keys())
        
    def nextQuery(self):
        if self.index_ >= len(self.query_keys_):
            return None
        i = self.index_
        self.index_ += 1
        return self.queries_[self.query_keys_[i]]

class RandomQueryParser(QueryParser):
    def trainTestSplit(propTrain = 0.8):
        shuffle(self.query_keys_) 
        index = int(len(self.query_keys_)*propTrain)
        self.query_keys_train_ = self.query_keys_[:index]
        self.query_keys = self.query_keys_[index:]
    def nextRandomTrainQuery()
        i = random.choice(self.query_keys_train_) 
        return self.queries_[self.query_keys_[i]]

class Query:
    def __init__(self, id_, text_, relevants=[]):
        self.id_ = id_
        self.text_ = text_
        self.relevants_ = relevants

    def add_relevant(self, relevant):
        if relevant not in self.relevants_:
            self.relevants_.append(relevant)
