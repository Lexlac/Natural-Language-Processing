
# coding: utf-8

# In[ ]:

from __future__ import division
import argparse
import pandas as pd

# useful stuff

import sys
reload(sys) 
sys.setdefaultencoding('UTF8')

import numpy as np 
from scipy.special import expit
from sklearn.preprocessing import normalize

from six import iteritems
import math

import time
import itertools

from six import iteritems, itervalues, string_types

try:
    from queue import Queue
except ImportError:
    from Queue import Queue
    
import threading

import scipy.sparse

from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32, uint32, seterr 
from numpy import array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum

import nltk
from   nltk.tokenize import TweetTokenizer 
from   nltk.stem.snowball import SnowballStemmer 
from   nltk.corpus import stopwords

__authors__ = ['Rafaelle Aygalenq','Sarah Lina Hammoutene','Dora Linda Kocsis','Noemie Quere']
__emails__  = ['B00724587@essec.edu','B00712035@essec.edu','B00714326@essec.edu','B00719656@essec.edu']

def zeros_aligned(shape, dtype, order='C', align=128):
    nbytes = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
    buffer = np.zeros(nbytes + align, dtype=np.uint8)  
    start_index = -buffer.ctypes.data % align
    return buffer[start_index : start_index + nbytes].view(dtype).reshape(shape, order=order)

def unitvec(vec):
    
    blas = lambda name, ndarray: scipy.linalg.get_blas_funcs((name,), (ndarray,))[0]
    blas_nrm2 = blas('nrm2', np.array([], dtype=float))
    blas_scal = blas('scal', np.array([], dtype=float))

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        veclen = blas_nrm2(vec)
        if veclen > 0.0:
            return blas_scal(1.0 / veclen, vec)
        else:
            return vec

    try:
        first = next(iter(vec)) 
    except:
        return vec



def group(iterable, chunksize, as_numpy=False):

    it = iter(iterable)
    while True:
        if as_numpy:
            # convert each document to a 2d numpy array 
            wrapped_chunk = [[numpy.array(doc) for doc in itertools.islice(it, int(chunksize))]]
        else:
            wrapped_chunk = [list(itertools.islice(it, int(chunksize)))]
        if not wrapped_chunk[0]:
            break
        yield wrapped_chunk.pop()



def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    print("Text to Sentences Conversion")
    stemmer = SnowballStemmer('english')
    stopwords_ = nltk.corpus.stopwords.words('english')
    punctuation_=[]
    for i in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
        punctuation_.append(i)
    punctuation_.append('``')
    punctuation_.append("''")
    
    with open(path) as f:
        
        # Tokenizing
        for l in f:
            sentences.append(nltk.word_tokenize(l.encode("utf8").decode('utf-8').lower()))
        for i in range(len(sentences)):
            
            # stemming : we keep only the root of every word 
            #The following line should be commented if we want to take the whole word not only its stem 
            sentences[i]=map(lambda k: stemmer.stem(sentences[i][k]) , range(len(sentences[i])))
            
            # removing stopwords
            sentences[i]=[w for w in sentences[i] if w not in stopwords_]
            
            # removing punctuation
            sentences[i]=[w for w in sentences[i] if w not in punctuation_]

        return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    
    #The three following lines MUST be commented if the words in the csv are already stemmed
    stemmer=SnowballStemmer('english')
    data['word1']=map(lambda k: stemmer.stem(data['word1'][k]) , range(len(data['word1'])))
    data['word2']=map(lambda k: stemmer.stem(data['word2'][k]) , range(len(data['word2'])))

    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

class mSkipGram:
    def __init__(self,sentences=None, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        #- self: context 
        #- sentences: 
        #- nEmbed: is the dimensionality of the features vector
        #- negativeRate: the int for negative specifies how many "noise words" should be drawn 
        #- winSize: is the maximum distance between the current and predicted word within a sentence
        #- minCount: ignore all words with total frequency lower than this
        
       
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount= minCount 
        self.vocabulary = {}
        self.idxtoword = []
        self.table = None    # for negative sampling 
        self.layer1_size = nEmbed
        
        self.alpha=0.025 #initial learning rate (will linearly drop to zero as training progresses)
        self.min_alpha =0.0001   #Set the minimum learning rate


        self.workers= 1   # Can be changed according to the how many worker threads we want to train the model

        
        if sentences is not None:
            print("Vocabulary construction")
            #Build the vocabulary:
            total_words = 0   
            vocab = {}      
            # First step is to compute the number of occurences of each word
            for sentence in sentences:
                for word in sentence:
                    total_words += 1
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            
            #Assign a unique index to each word and save both number of occurences and index for each word
            for word, occ in iteritems(vocab):
                # Select only the words that appear more than 5 times (minCount value)
                if occ >= self.minCount:
                    self.idxtoword.append(word)
                    self.vocabulary[word] = [occ, len(self.vocabulary)]
                    

            #Make a table for drawing random words for negative sampling
            print("Negative Sampling Table Construction")
            vocab_size = len(self.idxtoword)
            self.table = np.zeros(vocab_size)
            power=0.75  #By default we set this value at 3/4
            
            if vocab_size:  #Test if the vocabulary is not empty
                
                #Compute sum of all powers 
                total_words_pow = float(sum([self.vocabulary[word][0]**power for word in self.vocabulary]))
         
                idx1 = 0
                p = self.vocabulary[self.idxtoword[idx1]][0]**power / total_words_pow

                #Go through the whole table and fill it up with the word indexes proportional to a word's count^power
                for idx2 in xrange(vocab_size):
                    self.table[idx2] = idx1
                    if (idx2 / vocab_size) > p:
                        idx1 += 1
                        p += self.vocabulary[self.idxtoword[idx2]][0]**power / total_words_pow
                    if (idx2 / vocab_size):
                        idx2 = vocab_size - 1
                        
            #Reset all projection weights to an initial (untrained) state
            np.random.seed(1)
            self.syn0 = np.empty((len(self.vocabulary), self.layer1_size), dtype=np.float32)
            for i in xrange(len(self.vocabulary)):
                self.syn0[i] = (np.random.rand(self.layer1_size) - 0.5) / self.layer1_size
            self.syn1neg = np.zeros((len(self.vocabulary), self.layer1_size), dtype=np.float32)
            self.syn0norm = None
            self.syn1neg = np.array(self.syn1neg)
            
            #We call the train function
            #self.train(sentences, 100)
            #print(self.vocabulary)
            
    
    def train(self, stepsize, epochs):
        print("Training in Process")

        start = time.time() # Time Start
        word_count = [0]
        
        #The total number of words =sum of all the occurances
        total_words = int(sum(v[0] for v in itervalues(self.vocabulary)))   
        jobs = Queue(maxsize=2 * self.workers) # buffer that contains the jobs.
       
        def worker_train():
            work = np.zeros(self.layer1_size, dtype=np.float32)  # each thread must have its own work memory
            neu1 = zeros_aligned(self.layer1_size, dtype=np.float32) 
            
            while True:
                job = jobs.get()
                if job is None:  # if data finished
                    break
                    
                # We update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                
                job_words=0
                for sentence in job:
                    s=  train_sentence(sentence, alpha, work)   #Update skip-gram model by training on a single sentence
                    if not s is None: 
                        job_words += s                         #Update the number of words processed
                word_count[0] += job_words
            
        
        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  
            thread.start()
            
        def train_sentence(sentence, alpha, work=None):
            labels = np.zeros(self.negativeRate + 1)
            labels[0] = 1.0
            
            # We go through every sentence to train our model
            for pos, word in enumerate(sentence):
                
                # If there is a "word" which is equal to None, due to the fact that our pre-processing function
                # text2sentences is not perfect, we ignore it and continue
                if word is None:
                    continue
                    
                #We check that the word is in the vocabulary (because some have been removed in the occ < min_occ)
                if  word in self.vocabulary:                     
                    reduced_window = np.random.randint(self.winSize)       
                    start = max(0, pos - self.winSize + reduced_window) 
                    
                    # We go over all words from the (reduced) window predicting each one of them at the time
                    for pos2, word2 in enumerate(sentence[start : pos + self.winSize + 1 - reduced_window], start):
                        #If it's not the 'word' itself and if the word is also in the vocabulary
                        if word2 and not (pos2 == pos) and (word2 in self.vocabulary):
                           
                            l1 = self.syn0[self.vocabulary[word2][1]]
                            neu1e = np.zeros(l1.shape)
                           
                            # We use this word (label = 1) + 'negative' other random words not from this sentence (label = 0)
                            word_indices = [int(self.vocabulary[word][1])]
                            while len(word_indices) < self.negativeRate + 1:
                                w = self.table[np.random.randint(self.table.shape[0])]
                                if w != self.vocabulary[word][1]:
                                    word_indices.append(int(w))
                            
                            # Propagate hidden layer to the output                            
                            self.syn1neg = np.array(self.syn1neg)             
                            l2b = self.syn1neg[word_indices] 
                            fb = 1. / (1. + np.exp(-dot(l1,  np.transpose(l2b) ))) 
                                                        
                            # Vector of error gradients multiplied by the learning rate alpha
                            gb = (labels - fb) * alpha 
                            # learn hidden to the output
                            self.syn1neg[word_indices] += outer(gb, l1) 
                            # We save the error
                            neu1e += dot(gb, l2b) 
                            # Learn input to the hidden layer                         
                            self.syn0[ int(self.vocabulary[word2][1])] += neu1e
                                                        
                    #We return the number of words in the sentence that have been processed (that are in the vocabulary)        
                    if ( len([word for word in sentence if word in self.vocabulary]) ) is None : 
                        return 0
                    else :
                        return len([word for word in sentence if word in self.vocabulary])          
                        
        for job_no, job in enumerate(group(sentences, epochs)):
            jobs.put(job)
       
        for _ in xrange(self.workers):
            jobs.put(None)  

        for thread in workers:
            thread.join()
        
        print("End Of Training")                                               
        #The training time
        elapsed = time.time() - start
        print "Training time: ", elapsed
    
        return word_count[0]         
        
        
        
    
    def save(self,path):
        
        # We save the model, the input-hidden weight matrix in the path given as argument
        
        assert (len(self.vocabulary), self.layer1_size) == self.syn0.shape
        with open(path,'wb') as fout:
            fout.write(("%s %s\n" % self.syn0.shape).encode('utf8'))
            
            # Words are stored in descending order (most frequent words are stored first)
            for word, vocab in sorted(iteritems(self.vocabulary), key=lambda item: -item[1][0]):
                row = self.syn0[vocab[1]]
                fout.write(("%s %s\n" % (word, ' '.join("%f" % val for val in row))).encode('utf8'))
                

          
    def similarity(self,word1,word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        # We compute here the cosine similarity which is the dot product between the two words turned into vectors
        # and we rescale the value between 0 and 1 as it is asked (the cosine similarity produces values between
        # -1 and 1)
        
        if(word1 in self.vocabulary) and (word2 in self.vocabulary) :
            return  (dot(unitvec(self[word1]), unitvec(self[word2]))+1)/2 
        else:
            return 'NA'
            


    
    def __getitem__(self, word):
        
        #Return a word's representation in vector space, as a 1D numpy array
        
        if not word in self.vocabulary : 
            raise RuntimeError ('The word is  not in the vocabulary')
        return self.syn0[self.vocabulary[word][1]]
    
    @staticmethod
    def load(path):    

        # We are loading the model, the input-hidden weight matrix, in the same way it was saved 
        counts = None

        with open(path) as fin:
            header = unicode(fin.readline(), encoding='utf8', errors='strict')
            vocab_size, layer1_size = map(int, header.split()) 
            
            result = mSkipGram(nEmbed=layer1_size)
            
            result.syn0 = zeros((vocab_size, layer1_size))
            
            # We scan line by line and save the weights
            for line_no, line in enumerate(fin):
                parts = line.split()
                if len(parts) != layer1_size + 1:
                    raise ValueError("invalid vector on line %s" % (line_no))
                    
                word, weights = parts[0], map(float32, parts[1:])
                
                if counts is None:
                    result.vocabulary[word] = [vocab_size - line_no,line_no]
                elif word in counts:
                    result.vocabulary[word] = [counts[word],line_no]
                else:
                    result.vocabulary[word] = [None,line_no]
                    
                result.idxtoword.append(word)
                result.syn0[line_no] = weights
 
        return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='text.txt', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mSkipGram(sentences)
        sg.train(sentences, 100)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        sg = mSkipGram.load(opts.model)
        for a,b,_ in pairs:
            print sg.similarity(a,b)
            

