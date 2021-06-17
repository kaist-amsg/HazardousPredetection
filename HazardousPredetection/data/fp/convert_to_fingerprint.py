##########import modules##########
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolfiles
from rdkit.Chem import Draw
from rdkit import RDLogger
from gu_decode_02 import GetMFPidx, GetMFP
import ast
import numpy as np
import random

from collections import defaultdict

RDLogger.DisableLog('rdApp.*')
####################

##########input_data##########
######target_toxin#####
reaction_data_positive = '../toxin/clean_positive_ammonia'
reaction_data_negative = '../toxin/clean_negative_ammonia'
fingerprint_directory = './toxin/ammonia2'
########

#####range of fingerprints#####
radius = 2
frequent_regulate = 5
##########
####################

##########read data##########
#####read positive data#####
#read data from USPTO

def write_list(lst, filename):
    fg = open(filename, 'w')
    for i in lst:
        fg.write(str(i))
        fg.write('\n')
    fg.close()

f = open(reaction_data_positive, 'r')
list_pos = []
while True:
    line = f.readline()
    if not line: break
    reaction = ast.literal_eval(line)
    if len(reaction[0]) <= 10:
        list_pos.append(reaction)
nump = len(list_pos)
f.close()

if nump > 10000:
    random.shuffle(list_pos)
    list_pos_new = list_pos[:10000]
    list_pos = list_pos_new
nump = len(list_pos)

div = int(nump/10)

random.shuffle(list_pos)
test_pos = list_pos[:div]
elp = list_pos[div:]
valid_pos = elp[:div]
train_pos = elp[div:]

write_list(test_pos, fingerprint_directory + '/test_positive')
write_list(valid_pos, fingerprint_directory + '/valid_positive')
write_list(train_pos, fingerprint_directory + '/train_positive_beforeoversample')

f = open(reaction_data_negative, 'r')
list_neg = []
while True:
    line = f.readline()
    if not line: break
    reaction = ast.literal_eval(line)
    if len(reaction[0]) <=10:
        list_neg.append(reaction)
numn = len(list_neg)
f.close()

list_neg_new = list_neg[:10000]
list_neg = list_neg_new
numn = len(list_neg)

div = int(numn/10)
random.shuffle(list_neg)
test_neg = list_neg[:div]
eln = list_neg[div:]
valid_neg = eln[:div]
train_neg = eln[div:]

write_list(test_neg, fingerprint_directory + '/test_negative')
write_list(valid_neg, fingerprint_directory + '/valid_negative')
write_list(train_neg, fingerprint_directory + '/train_negative_beforeoversample')

##########convert molecule into fingerprints#########
#####make library of unique substructures from training data#####
flib = open(fingerprint_directory +  '/library_' + str(frequent_regulate) + '_' + str(radius), 'w')

FPs = []
for reaction in train_pos:
    for reactant in reaction[0]:
        mol = Chem.MolFromSmiles(reactant)
        FPs.append(GetMFP(mol, radius))

for reaction in train_neg:
    for reactant in reaction[0]:
        mol = Chem.MolFromSmiles(reactant)
        FPs.append(GetMFP(mol, radius))

#make library of unique substructures
uniqueFPs_org = []
for fps in FPs:
    for fp in fps:
        uniqueFPs_org.append(fp)
uniqueFPs_org.sort()
uniqueFPs_or = [0]
subs_count = [0]
for FPP in uniqueFPs_org:
    if FPP == uniqueFPs_or[-1]:
        subs_count[-1] += 1
    else:
        uniqueFPs_or.append(FPP)
        subs_count.append(1)
uniqueFPs = []
for subs_index, frequent in enumerate(subs_count):
    if frequent > frequent_regulate:
        uniqueFPs.append(uniqueFPs_or[subs_index])

#write substructure library
print("training data, length of library")
print(len(uniqueFPs))
flib.write(str(uniqueFPs))
flib.write('\n')
flib.close()
##########

#####convert data into fingerprints#####
##########
tdl = ['train_positive', 'train_negative']
tddl = [train_pos, train_neg]
train_datt = [[],[]]

for idx, tgd in enumerate(tdl):
    print("converting " + tgd + " data")
    fp_list_positive = []
    outtr = open(fingerprint_directory + '/fp_' + tgd + '_' + str(radius) + '_' + str(frequent_regulate), 'w')
    for i in tddl[idx]:
        reaction = i
        fp_reactant = []
        for reactant in reaction[0]:
            X = np.zeros(len(uniqueFPs))
            mol = Chem.MolFromSmiles(reactant)
            if mol == None:
                print("error while converting SMILES to Mol")
                raise
            fps = GetMFP(mol, radius)
            for fp in fps:
                try:
                    j = uniqueFPs.index(fp)
                    X[j] = fps[fp]
                except:
                    continue
            X[X>0]=1
            fp_reactant.append(list(np.array(X)))
        tout = [fp_reactant] + reaction
        train_datt[idx].append(tout)
        outtr.write(str(tout))
        outtr.write('\n')
    outtr.close()
ovs = int(numn/nump)
print(ovs)
train_tot = []
train_tot = train_tot + train_datt[0]*ovs
train_tot = train_tot + train_datt[1]
random.shuffle(train_tot)
outtr = open(fingerprint_directory + '/fp_training_' + str(radius) + '_' + str(frequent_regulate), 'w')
for i in train_tot:
    outtr.write(str(i))
    outtr.write('\n')
outtr.close()

train_datt = []
train_tot = []
train_pos = []
train_neg = []

tdl = ['valid_positive', 'valid_negative', 'test_positive', 'test_negative']
tddl = [valid_pos, valid_neg, test_pos, test_neg]
for idx, tgd in enumerate(tdl):
    print("converting " + tgd + " data")
    fp_list_positive = []
    outtr = open(fingerprint_directory + '/fp_' + tgd + '_' + str(radius) + '_' + str(frequent_regulate), 'w')
    for i in tddl[idx]:
        reaction = i
        fp_reactant = []
        for reactant in reaction[0]:
            X = np.zeros(len(uniqueFPs))
            mol = Chem.MolFromSmiles(reactant)
            if mol == None:
                print("error while converting SMILES to Mol")
                raise
            fps = GetMFP(mol, radius)
            for fp in fps:
                try:
                    j = uniqueFPs.index(fp)
                    X[j] = fps[fp]
                except:
                    continue
            X[X>0]=1
            fp_reactant.append(list(np.array(X)))
        tout = [fp_reactant] + reaction
        outtr.write(str(tout))
        outtr.write('\n')
    outtr.close()
