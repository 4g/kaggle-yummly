import argparse
import scipy
import scipy.sparse
from tdec import timeit
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, datasets, metrics
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import random

uniq_elems = {}
global_weight = {}
stop = stopwords.words('english')
stemmer = PorterStemmer()

def main(train, test, out, mode):
  
  train_f , train_l , train_i = load_vectors(train)
  
  if mode == 'test':
    train_f, test_f, train_l, test_l = train_test_split(train_f, 
                                                        train_l,
                                                        test_size = 0.2,
                                                        random_state  = 0)
  """train the model"""
  model = cmodel(train_f, train_l)
  
  """test the model"""
  if mode == 'test':
    print get_accuracy(train_f, train_l, test_f, test_l, model)
  
  else:
    test_f , test_l , test_i = load_vectors(test, create_f = False)
    print test_f.shape , len(test_l) , len(test_i)
    out = open(out, 'wb')
    out.write("id, cuisine\n")
    for features, ide in zip(test_f, test_i):
      label = model.predict(features)
      string = str(ide) + "," + uniq_elems[label[0]] + "\n"
      out.write(string)
              
def get_accuracy(train_f, train_l, test_f, test_l, model):
  false , true = 0 , 0
  for features,label in zip(test_f,test_l):
    if model.predict(features) == label:
      true += 1
    else:
      false += 1
  return true*1.0/(false + true)

@timeit
def cmodel(f, l):
  classifier = linear_model.LogisticRegression(multi_class = 'ovr', 
                                               solver = 'liblinear', 
                                               max_iter = 1000)
  classifier.fit(f, l)
  return classifier

@timeit
def load_vectors(fname, create_f = True):
  import simplejson as json
  data = json.loads(open(fname).read())
  features = []
  labels = []
  ids = [] 
  for elem in data:
    i =  elem.get('ingredients',None)
    i = tokenize(i)
    c = elem.get('cuisine',None)
    ide = elem.get('id')
    ids += [ide]
    if create_f:
      for x in i + [c]:
        if x not in uniq_elems:
          uniq_elems[x] = len(uniq_elems)
          uniq_elems[uniq_elems[x]] = x
    i = filter(lambda x:x in uniq_elems,i)
    features += [[uniq_elems[a] for a in i]]
    if c != None:
      labels += [uniq_elems[c]]
  
  if create_f:
    calculate_global_weights(features, labels)
  
  features = to_sparse_matrix(features)
  return features , labels, ids

def tokenize(i):
  clean = lambda x:re.sub('[^A-Za-z]', ' ', x)
  i = [word_tokenize(clean(e).lower()) for e in i]
  i = reduce(lambda x,y:x + y,i)
  i = map(stemmer.stem, i)
  i = filter(lambda x:x not in stop,i)
  return i

@timeit
def calculate_global_weights(features, labels):
  total_cuisines = len(set(labels))
  word_cf = {}
  
  for f,l in zip(features,labels):
    for elem in set(f):
      word_cf[elem] = word_cf.get(elem,[]) + [l]
   
  for elem in word_cf:
#    print map(uniq_elems.get, word_cf[elem]) , uniq_elems[elem]
    word_cf[elem] = len(set(word_cf[elem]))
    
  for elem in word_cf:
    global_weight[elem] = math.log(1.0 + total_cuisines/word_cf[elem])

#  for elem in global_weight:
#    print uniq_elems[elem] , global_weight[elem]
  
#  for elem in word_df:
#    print uniq_elems[elem] , word_df[elem] , global_weight[elem]

@timeit
def to_sparse_matrix(ll):
  w , h = len(ll) , len(uniq_elems)
  mat = scipy.sparse.dok_matrix((w,h))
  for i in range(w):
    for j in ll[i]:
      mat[i,j] += global_weight[j]
  return mat

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Generate model")
  parser.add_argument('--train',  required=True, help="training file")
  parser.add_argument('--test',  required=True, help="test file")
  parser.add_argument('--out',  required=True, help="output file")
  parser.add_argument('--mode',  required=True, help="testmode or not")
  args = parser.parse_args()
  main(args.train, args.test, args.out, args.mode)
