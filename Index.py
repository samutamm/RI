# -*- coding: utf-8 -*-
from ParserCACM import ParserCACM
from TextRepresenter import PorterStemmer
import pickle
import numpy as np
from ast import literal_eval
     



def get_dictionary_length(count):
    """ 
        Returns list of lengths of dictionaries in string format.
        For exemple length of empty dictionary is 2, dictonary with
        one (fixed length) element has 16, then 32, 48 etc..
    """
    if count == 0:
        return 0
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
            self.word_start_indexes[(i+1):] += place
            
    def get_place_for_word(self, word):
        idx = self.word2fileplace[word]
        start, length = int(self.word_start_indexes[idx]),int(self.word_lengths[idx])
        next_start = int(self.word_start_indexes[idx + 1]) if idx < len(self.word_start_indexes) -1 else start + length + 1
        return start, length, next_start
    
    def get_indexfile_total_length(self):
        return self.word_start_indexes[-1] + self.word_lengths[-1] # the last word and it's length
    
    def save_places_to_file(self, filename):
        with open(filename, "wb") as f:
            pickler = pickle.Pickler(f)
            pickler.dump({'word2fileplace':self.word2fileplace, 
                          'word_start_indexes':self.word_start_indexes, 
                          'word_lengths':self.word_lengths
                         })

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
        #f = open(r"indexes/" + str(self.name) + "_index", "w") # ASCII as STR mode
        f = open(r"indexes/" + str(self.name) + "_index", "wb") # ASCII as BYTES mode 
        #pickler = pickle.Pickler(f, protocol=0) # protocol 0 : saves in plain ASCII txt
        doc_start_indexes = {}

        doc = self.parser.nextDocument()
        while(doc):
            text = doc.getText()
            doc_id = doc.getId()

            #save place
            doc_start_indexes[doc_id] = f.tell()

            #save to file
            stems = stemmer.getTextRepresentation(text)
            #pickled = pickle.dumps(stems, protocol=0).decode() # dumps : return ASCII as BYTES # decode : binary -> str
            #f.write(pickled)
            pickle.dump(stems, f, protocol=-1) # -1 = HIGHEST_PROTOCOL (binary)
            # dictionary can fit to memory
            dictionary.update(stems.keys())
            
            #iterate
            doc = self.parser.nextDocument()

        f.close()

        with open(r"indexes/" + self.name + self.index_places_doc, "wb") as output_file:
            pickle.dump(doc_start_indexes, output_file, protocol=-1)
        with open(r"indexes/dictionary", "wb") as output_file:
            pickle.dump(dictionary, output_file, protocol=-1)
        return dictionary
    
    def inversedIndexation(self, dictionary):
        with open("indexes/" + self.name + self.index_places_doc, 'rb') as index_places_doc_file:
            unpickler = pickle.Unpickler(index_places_doc_file)
            index_places_doc = unpickler.load()
            
        iip = InvertedIndexPlaces(dictionary)
        
        with open(r"indexes/" + str(self.name) + "_inversed", "wb") as inversed_writer:
            with open(r"indexes/" + str(self.name) + "_inversed", "rb") as inversed_reader:
                with open(r"indexes/" + self.name + "_index", "rb") as doc_file:
                    for doc_id in index_places_doc.keys():
                        doc_file.seek(index_places_doc[doc_id])
                        tfs = pickle.load(doc_file)
                        #tfs = pickle.Unpickler(doc_file).load()

                        iip.add_file_to_word(doc_id, tfs.keys())
                    iip.count_word_fileplaces()
                    iip.save_places_to_file(r"indexes/" + self.index_places_stem)
                    # write inversed index full of space
                    file_total_length = iip.get_indexfile_total_length()
                    inversed_writer.seek(0)
                    inversed_writer.write(str.encode(' ' * file_total_length)) # ASCII str to bytes
                    
                    N = len(index_places_doc.keys())
                    print("in total " + str(N) + " documents")
                    for doc_count,doc_id in enumerate(index_places_doc.keys()):
                        if doc_count % 100 == 0:
                            print(str(doc_count))
                        doc_file.seek(index_places_doc[doc_id])
                        tfs = pickle.Unpickler(doc_file).load()
                        
                        for word in tfs.keys():
                            place, length, next_place = iip.get_place_for_word(word)
                            
                            #standardize the length of filename
                            filename = ' ' * (4 - len(doc_id)) + doc_id
                            
                            # check old value
                            inversed_reader.seek(place)
                            old_string_value = inversed_reader.read(length).strip()
                            
                            if len(old_string_value) == 0: # no hits yet
                                old_dico = {}
                            else:
                                old_dico = literal_eval(old_string_value.decode()) # ASCII as bytes to str
           
                            
                            old_dico[filename] = format(tfs[word],'04d')
                                                      
                            # write result back
                            updated_dico = str(old_dico)
                            
                            correct_length_updated_dico = updated_dico + ' ' * (length - len(updated_dico))
                            
                            inversed_writer.seek(place)
                            inversed_writer.write(str.encode(correct_length_updated_dico))
                                            
           
    def indexation(self):
        self.dico = self.normalIndexation()
        self.inversedIndexation(self.dico)
    
    def getTfsForDoc(self, doc_id):
        with open(r"indexes/" + self.name + self.index_places_doc, 'rb') as index_places_file:
            unpickler = pickle.Unpickler(index_places_file)
            index_places = unpickler.load()
        with open(r"indexes/" + self.name + "_index", "rb") as f:
            f.seek(index_places[doc_id])
            unpickler = pickle.Unpickler(f)
            tfs = unpickler.load()
            return tfs
    
    def getTfsForStem(self, word):
        with open(r"indexes/" + self.index_places_stem, 'rb') as index_places_f:
            unpickler = pickle.Unpickler(index_places_f)
            index_places = unpickler.load()
        word_start_indexes = index_places['word_start_indexes']
        word2fileplace = index_places['word2fileplace']
        word_lengths = index_places['word_lengths']
        
        idx = word2fileplace[word]
        place = word_start_indexes[idx]
        length = word_lengths[idx]
        
        with open(r"indexes/" + str(self.name) + "_inversed", "rb") as inversed_reader:
            inversed_reader.seek(place)
            dico_str = inversed_reader.read(length)
        dico = literal_eval(dico_str.decode())
        return {int(k.strip()):int(dico[k]) for k in dico.keys()}    
    
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
    
    def getDocIds(self):
        with open(r"indexes/" + self.name + self.index_places_doc, "rb") as doc_file:
            return pickle.load(doc_file).keys()
