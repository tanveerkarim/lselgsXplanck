"""This script boostraps ELG photo-z dndz estimation
by training Random Forest over 32 regions and testing 
over 8 regions."""

import os
os.environ["OMP_NUM_THREADS"] = "64"

from multiprocessing import Pool
import argparse
import pandas as pd 
import numpy as np 
from time import time 
import pickle 

from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#parser for bash arguments
parser = argparse.ArgumentParser()
parser.add_argument("--JID", "-jid", type = int, help = "")
args = parser.parse_args()
JOBID = args.JID

##GLOBALS
ndist = 100  # number of distributions to be calculated in this single node
nparallel = 10  # number of parallel processes in this node
outp_dir = "/pscratch/sd/t/tanveerk/photo-z/"

#read in cleaned up sample 
df = pd.read_pickle("/pscratch/sd/t/tanveerk/final_data_products/elgXplanck/fuji-elg-single-tomo.pkl")

##--standardize dataset--##
#split into features and classes
X, y = df.drop(["bin_label"], axis=1), df["bin_label"].values

##--scale only features of interest-##
#these are MAG_*, g-r, r-z, *fib
sc = StandardScaler()
X_transformed = sc.fit_transform(X.iloc[:, 7:-1])

# concatenate identifying information to be used for train-test split
X_transformed = np.hstack((np.array(X.iloc[:, :7]), X_transformed))

# define rosette indices range and size of the sub-samples
rosetteIDs = np.arange(20*2)
tst_sz = 4*2
trn_sz = 16*2

zrange = np.arange(0, 1.7, 0.1) #range to bin histogram over

##--Function to calculate dndz and associated statistics
def getdndz(idx):
    """Returns dndz, confusion matrix and statitics on bad label
    false positives given index number"""

    # split indices without replacement into train and test sub-samples.
    rng = np.random.default_rng(JOBID + idx)
    tmp_tst_rosetteID = rng.choice(rosetteIDs, tst_sz, replace=False)
    tmp_trn_rosetteID = np.setxor1d(rosetteIDs, tmp_tst_rosetteID)

    # within sub-samples, bootstrap indices of the rosettes
    # with replacement. This is the boostrap step.
    tmp_trn_rosetteID = rng.choice(tmp_trn_rosetteID, size=trn_sz, replace=True)
    tmp_tst_rosetteID = rng.choice(tmp_tst_rosetteID, size=tst_sz, replace=True)

    # identify indices of test and training sets
    indices_trn = []
    indices_tst = []

    for ii in range(len(tmp_trn_rosetteID)):
        indices_trn.append(
            np.where(X_transformed[:, 5] == tmp_trn_rosetteID[ii])[0])

    for ii in range(len(tmp_tst_rosetteID)):
        indices_tst.append(
            np.where(X_transformed[:, 5] == tmp_tst_rosetteID[ii])[0])

    #turn into numpy array
    indices_trn = np.concatenate(indices_trn)
    indices_tst = np.concatenate(indices_tst)

    #shuffle indices
    rng.shuffle(indices_trn)
    rng.shuffle(indices_tst)

    # splice training and test sets
    X_trn = X_transformed[indices_trn, :]
    X_tst = X_transformed[indices_tst, :]

    y_trn = y[indices_trn]
    y_tst = y[indices_tst]

    ##--Train Random Forest--##
    clf = RandomForestClassifier(class_weight={0: 0.325, 1: 0.35, 2: 0.325},
                       max_features='sqrt', min_samples_leaf=10,
                       min_samples_split=10, n_estimators=20)
    clf.fit(X_trn[:, 7:], y_trn.astype(int))

    #predict on the test set
    y_prd = clf.predict(X_tst[:, 7:])

    ##--score of training--##

    # print(f"idx: {idx}")
    # print("Confusion Matrix")
    # print("-----")
    # print(confusion_matrix(y_tst, y_prd, normalize='pred'))
    # print("-----")

    #--store histogram--##
    test_specz = X.iloc[indices_tst][y_prd == 1]['Z']

    # ignore all the bad labels that are false pos
    res = np.histogram(test_specz[test_specz != -99], bins=zrange, density = True)
    
    # count false positives with bad labels
    badFP = (test_specz == -99).sum()/len(test_specz)
    res += (badFP, )

    return res

##--run parallel jobs--##
tally = {}

start_time = time()

for i in range(ndist//nparallel):  # i refers to chunk of maps processed together
    idx = JOBID*int(100) + i * nparallel + np.arange(nparallel)
    with Pool(nparallel) as p:
        tally[i] = p.map(getdndz, idx)
    
end_time = time()
print(f"{ndist} jobs took {end_time - start_time} seconds.")

pickle.dump(tally, open(outp_dir + str(JOBID) + "_normed.npy", "wb"))
