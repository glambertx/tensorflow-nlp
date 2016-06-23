import numpy as np
import pandas as pd
import re
import random



# gather 3 dumped VAERS data
with open('data/serious.txt', 'r') as infile:
    serious1 = infile.readlines()

with open('data/nonserious.txt', 'r') as infile:
    nonserious1 = infile.readlines()

with open('data/serious2.txt', 'r') as infile:
    serious2 = infile.readlines()

with open('data/nonserious2.txt', 'r') as infile:
    nonserious2 = infile.readlines()

with open('data/serious3.txt', 'r') as infile:
    serious3 = infile.readlines()

with open('data/nonserious3.txt', 'r') as infile:
    nonserious3 = infile.readlines()


serious1 = np.array(serious1)
serious2 = np.array(serious2)
serious3 = np.array(serious3)
nonserious1 = np.array(nonserious1)
nonserious2 = np.array(nonserious2)
nonserious3 = np.array(nonserious3)

serious = np.concatenate((serious1,serious2), axis=0)
nonserious = np.concatenate((nonserious1,nonserious2), axis=0)
serious = np.concatenate((serious,serious3), axis=0)
nonserious = np.concatenate((nonserious,nonserious3), axis=0)
#serious = serious1
#nonserious = nonserious1
"""
# balance labels 50/50
serious_size =  len(serious)
print "LENGTH", serious_size

nonserious = random.sample(nonserious, serious_size)
negative_size = len(nonserious)
print "LENGTH", negative_size
"""
#np.savetxt('data/serious', serious, fmt="%s")
#np.savetxt('data/nonserious', nonserious,fmt="%s")
#use 1 for positive sentiment, 0 for negative


y = np.concatenate((np.ones(len(serious)), np.zeros(len(nonserious))))
y = y.astype(int)
np.savetxt('data/labels',y ,delimiter=',')
print np.bincount(y)
print y[0]

#serious_size =  min(np.unique(y, return_counts=True)[1])
#nonserious = random.sample(nonserious, serious_size*2)
