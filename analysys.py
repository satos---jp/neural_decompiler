#! /usr/bin/python3

from matplotlib import pyplot as plt

import pickle

ds = pickle.load(open('dist_dist.pickle','rb'))


maxlen = 20

ds = list(filter(lambda x: x[0]<maxlen,ds))


lens = list(map(lambda x: x[0],ds))
rates = list(map(lambda x: x[1],ds))

#plt.hist(lens,bins=20)
plt.hist2d(lens,rates,bins=[maxlen,100],range=[[0,maxlen],[0.0,1.0]])
plt.show()



