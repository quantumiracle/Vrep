import cPickle
import gzip
import bz2
import time
import pickle

print('loading file')
with open('dataset/replay_00.cPickle','r') as f:
	img = cPickle.load(f)
print('loading completed')

print('starting cPickle saving')
start = time.time()
with open('dataset/derp.cPickle','wb') as f:
	cPickle.dump(img,f,protocol=cPickle.HIGHEST_PROTOCOL)
end = time.time()
print('cPickle done in',(end-start))

#print('starting bz2 saving')
#start = time.time()
#with bz2.BZ2File('dataset/derp.pbz2','wb') as f:
#	cPickle.dump(img,f,protocol=cPickle.HIGHEST_PROTOCOL)
#end = time.time()
#print('bz2 done in',(end-start))

for i in range(9):
	print('starting gzip saving level',i)
	start = time.time()
	filename = 'dataset/derp'+str(i)+'.pgz'
	with gzip.GzipFile(filename,'wb',compresslevel=i) as f:
		cPickle.dump(img,f,protocol=cPickle.HIGHEST_PROTOCOL)
	end = time.time()
	print('gzip compress level',i,'done in',(end-start))
