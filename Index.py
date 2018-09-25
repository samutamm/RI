
from ParserCACM import ParserCACM
from TextRepresenter import PorterStemmer
import cPickle
import numpy as np


def get_dictionary_length(count):
    """ 
        Returns list of lengths of dictionaries in string format.
        For exemple length of empty dictionary is 2, dictonary with
        one (fixed length) element has 16, then 32, 48 etc..
    """
    if count == 0:
        return 2
    return count * 16

class InvertedIndexPlaces:
    
    def __init__(self, dico):
        self.word2fileplace = {}
        self.word_lengths = np.zeros(len(dico), dtype='int64') # length of each dictionary
        self.word_start_indexes = np.zeros(len(dico), dtype='int64') # length of each dictionary
        self.word_file_count = np.zeros(len(dico))
        for i,w in enumerate(dico):
            self.word2fileplace[w] = i
            
    def add_file_to_word(self,filename, words):
        for word in words:
            # just count the files per word
            index_file_place = self.word2fileplace[word]
            self.word_file_count[index_file_place] += 1
            
    def count_word_fileplaces(self):
        #import pdb; pdb.set_trace()
        for i, count in enumerate(self.word_file_count):
            c = int(count)
            self.word_lengths[i] = get_dictionary_length(c)
        for i, place in enumerate(self.word_lengths):
            self.word_start_indexes[i:] += place
        print(self.word_start_indexes)
            
    def get_place_for_word(self, word):
        idx = self.word2fileplace[word]
        return int(self.fileplaces[idx]),int(self.word_lengths[idx]) 

class Index:
    
    def __init__(self, name, filename):
        self.name = name
        self.index_places_doc = "_index_places_doc"
        self.index_places_stem = "_index_places_stem"
        self.filename = filename
        # parser, textRepresenter
        # docs, stems, docFrom
    
    def normalIndexation(self):
        self.parser = ParserCACM()
        self.parser.initFile(self.filename)
        
        stemmer = PorterStemmer()
        dictionary = set()
        f = open(r"indexes/" + str(self.name) + "_index", "wb")
        pickler = cPickle.Pickler(f)
        doc_start_indexes = {}

        doc = self.parser.nextDocument()
        while(doc):
            text = doc.getText()
            doc_id = doc.getId()

            #save place
            doc_start_indexes[doc_id] = f.tell()

            #save to file
            stems = stemmer.getTextRepresentation(text)
            pickler.dump(stems)

            # dictionary can fit to memory
            dictionary.update(stems.keys())
            
            #iterate
            doc = self.parser.nextDocument()

        f.close()

        with open(r"indexes/" + self.name + self.index_places_doc, "wb") as output_file:
            cPickle.dump(doc_start_indexes, output_file)
        with open(r"indexes/dictionary", "wb") as output_file:
            cPickle.dump(dictionary, output_file)
        return dictionary
    
    def inversedIndexation2(self, dictionary):
        with open(r"indexes/" + self.name + self.index_places_doc) as index_places_doc_file:
            unpickler = cPickle.Unpickler(index_places_doc_file)
            index_places_doc = unpickler.load()
            
        with open(r"indexes/" + self.name + "_index", "rb") as doc_file:
            stem_file = open(r"indexes/" + str(self.name) + "_inversed", "wb")
            pickler = cPickle.Pickler(stem_file)
            stem_start_indexes = {}
            n = len(dictionary)
            
            for i, word in enumerate(dictionary):
                files = {}

                if i % 100 == 0:
                    print(i / n * 100)
                #save place
                stem_start_indexes[word] = stem_file.tell()
                
                # calcul frequencies in different files
                for doc_id in index_places_doc.keys():
                    doc_file.seek(index_places_doc[doc_id])
                    tfs = cPickle.Unpickler(doc_file).load()
                    if word in tfs:
                        files[doc_id] = tfs[word]
                
                # save to file   
                pickler.dump(files)
         
        stem_file.close()
        with open(r"indexes/" + self.name + self.index_places_stem, "wb") as output_file:
            cPickle.dump(stem_start_indexes, output_file)
            
    def inversedIndexation(self, dictionary):
        with open(r"indexes/" + self.name + self.index_places_doc) as index_places_doc_file:
            unpickler = cPickle.Unpickler(index_places_doc_file)
            index_places_doc = unpickler.load()
            
        iip = InvertedIndexPlaces(dictionary)
        
        with open(r"indexes/" + self.name + "_index", "rb") as doc_file:
            for doc_id in index_places_doc.keys():
                doc_file.seek(index_places_doc[doc_id])
                tfs = cPickle.Unpickler(doc_file).load()

                iip.add_file_to_word(doc_id, tfs.keys())
            iip.count_word_fileplaces()
            
            for doc_id in index_places_doc.keys():
                doc_file.seek(index_places_doc[doc_id])
                tfs = cPickle.Unpickler(doc_file).load()
                
                for word in tfs.keys():
                    place, length = iip.get_place_for_word(word)
                    # update this place
                    # TODO

           
    def indexation(self):
        self.dico = self.normalIndexation()
        self.inversedIndexation(self.dico)
    
    def getTfsForDoc(self, doc_id):
        with open(r"indexes/" + self.name + self.index_places_doc) as index_places_file:
            unpickler = cPickle.Unpickler(index_places_file)
            index_places = unpickler.load()
        with open(r"indexes/" + self.name + "_index", "rb") as f:
            f.seek(index_places[doc_id])
            unpickler = cPickle.Unpickler(f)
            tfs = unpickler.load()
            return tfs
    
    def getTfsForStem(self):
        pass
    
    def getStrDoc(self, doc_id):
        self.parser = ParserCACM()
        self.parser.initFile(self.filename)
        
        doc = self.parser.nextDocument()
        while(doc):
            if doc.getId() == str(doc_id):
                return doc.getText()
         
            #iterate
            doc = self.parser.nextDocument()
        return "No doc with id " + str(doc_id)