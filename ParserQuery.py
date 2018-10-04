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
    with open(filename_jugements, 'rb') as f:
      for line in f:
        line = line.split(' ') 
        self.queries_[int(line[0])].add_relevant(line[1])
    self.index_ = 0
  def nextQuery():
    self.index_ += 1
    return self.queries_[self.index-1]
  
class Query:
  def __init__(self, id_, text_, relevants=[]):
    self.id_ = id_
    self.text_ = text_
    self.relevants_ = relevants
  def add_relevant(self, relevant):
    if type(relevant) == list:
      self.relevants_.extend(relevant)
    else:
      self.relevants_.append(relevant)
