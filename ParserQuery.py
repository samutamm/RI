import numpy as np
import random

from ParserCACM import ParserCACM
from TextRepresenter import PorterStemmer

class QueryParser():
    def initFile(self, filename_queries, filename_jugements):
        self.queries_ = {}
        self.subtopic_names = {}
        parser = ParserCACM()
        parser.initFile(filename_queries)
        doc = parser.nextDocument()
        while doc:
            self.queries_[doc.getId()] = Query(doc.getId(), doc.getText().replace('\n', ''), {})
            doc = parser.nextDocument()
        with open(filename_jugements, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                line = line.split(' ')
                if line[0] == '#':
                    if line[2] != 'Description':
                        query_id = line[1][1:]
                        topic_local_id = line[2]  # this is not unique, depends of query
                        topic_name = ' '.join(line[3:])
                        if query_id not in self.subtopic_names:
                            self.subtopic_names[query_id] = {}
                        else:
                            self.subtopic_names[query_id][topic_local_id] = topic_name
                    continue
                id_query = int(line[0])
                id_doc = int(line[1])
                subtopic = int(line[3])
                self.queries_[id_query].add_relevant(id_doc, subtopic)
        self.index_ = 0
        self.query_keys_ = list(self.queries_.keys())
        
    def nextQuery(self):
        if self.index_ >= len(self.query_keys_):
            return None
        i = self.index_
        self.index_ += 1
        return self.queries_[self.query_keys_[i]]

    def get_query_min_max(self, doc_index):
        if not hasattr(self, 'queries_'):
            print("Call initFile before get_query_min_max")
            return None
        stemmer = PorterStemmer()
        lengths = []
        idfs = []
        query = self.nextQuery()

        while query:
            text = query.text_
            stems = stemmer.getTextRepresentation(text)
            idf = np.sum([len(doc_index.getTfsForStem(w).keys()) for w in stems])

            lengths.append(len(text))
            idfs.append(idf)

            query = self.nextQuery()
        lengths = np.array(lengths)
        idfs = np.array(idfs)
        return {'query_len_min': lengths.min(), 'query_len_max': lengths.max(),
                'query_idf_min': idfs.min(), 'query_idf_max': idfs.max()}


class SplitQueryParser(QueryParser):
    def initFile(self, filename_queries, filename_jugements, train_prop=0.8, seed=42):
        super().initFile(filename_queries, filename_jugements)
        random.seed(seed)
        random.shuffle(self.query_keys_) 
        index = int(len(self.query_keys_)*train_prop)
        self.train_index_ = 0
        self.query_keys_train_ = self.query_keys_[:index]
        self.test_index_ = 0
        self.query_keys_test_ = self.query_keys_[index:]

    def next_train_query(self):
        if self.train_index_ >= len(self.query_keys_train_):
           return None
        i = self.train_index_
        self.train_index_ += 1
        return self.queries_[self.query_keys_train_[i]]
    def next_test_query(self):
        if self.test_index_ >= len(self.query_keys_test_):
           return None
        i = self.test_index_
        self.test_index_ += 1
        return self.queries_[self.query_keys_test_[i]]
      
    def random_train_query(self):
        return self.queries_[random.choice(self.query_keys_train_)]

    def random_test_query(self):
        return self.queries_[random.choice(self.query_keys_test_)]

class Query:
    def __init__(self, id_, text_, relevants):
        self.id_ = id_
        self.text_ = text_
        self.relevants_ = relevants

    def add_relevant(self, relevant, topic):
        if relevant not in self.relevants_:
            self.relevants_[relevant] = topic

