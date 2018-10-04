from ParserCACM import ParserCACM 

class QueryParser():
  def initFile(self, filename_queries, filename_jugements):
    self.dict_queries = {}
    parser = ParserCACM()
    parser.initFile(filename_queries)
    while True:
      doc = parser.nextDocument()
      if doc is None:
        break
      print(doc.getId())
      self.dict_queries[doc.getId()] = Query(doc.getId(), doc.getText())
    return self.dict_queries
    '''with open(filename_jugements, 'rb') as f:
      for line in f:
        line = line.split(' ') 
        self.dict_queries[line[0]].add_relevant(line[1])
    self.index_ = 0'''
  def nextQuery():
    self.index_ += 1
    return self.dict_queries[self.index-1]
  
class Query:
  def __init__(self, id_, text_, relevants=[]):
    self.id_ = id_
    self.text_ = text_
    self.relevants_ = relevants
  def add_relevant(relevant):
    if type(relevant) == list:
      self.relevants_.extend(relevant)
    else:
      self.relevants_.append(relevant)
