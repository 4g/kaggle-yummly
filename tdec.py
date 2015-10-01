"""
@author:apurva.gupta@bloomreach.com
"""

import time  
import inspect 
import os.path
import cPickle as pickle
import hashlib

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        print method.__name__
        result = method(*args, **kw)
        te = time.time()
        print '%r %r %2.2f sec' % \
              (method.__name__, len(kw), te-ts)
        return result
    return timed

def offline(method):
  location = "/tmp/"
  extension = ".offline.tmp"
  def saved(*args, **kw):
    code_hash = method.func_code.__hash__()
    input_hash = hashlib.sha1(str(args)).hexdigest()
    complete_hash = str(code_hash) + str(input_hash)
    fname = location + complete_hash + extension 
    print fname
    if os.path.isfile(fname):
      with open(fname, 'rb') as fhandle:
        result =  pickle.load(fhandle)
    else:
      result = method(*args,**kw)
      with open(fname, 'wb') as fhandle:
        pickle.dump(result,fhandle)
    return result
  return saved

@timeit
@offline
def test_offline(x):
  for i in range(1,50000000):
      x = x + 4
  return x*3 + x*2

if __name__=="__main__":
  print test_offline(10)
