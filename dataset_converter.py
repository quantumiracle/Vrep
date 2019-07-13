import cPickle
import gzip
import bz2
import time
import pickle
import os

N_FILES = len(os.listdir('dataset'))
for i in range(N_FILES):
	filename = "dataset/replay_"+str(i).zfill(2)+".gz"
	newfilename = "replay_"+str(i).zfill(2)+".gz"
	dataset = None
	print('loading file',filename)
	with gzip.open(filename,'r') as f:
		dataset = cPickle.load(f)
	print('loading completed',filename)

	print('starting pickle saving')
	start = time.time()
	with gzip.GzipFile(newfilename, 'wb', compresslevel=3) as f:
		pickle.dump(dataset,f,protocol=cPickle.HIGHEST_PROTOCOL)
	end = time.time()
	print('pickle done in',(end-start))