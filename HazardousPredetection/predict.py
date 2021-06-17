import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy
import numpy as np
from numpy import array
import ast
import random

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolfiles
from rdkit.Chem import Draw
from rdkit import RDLogger
from collections import defaultdict

RDLogger.DisableLog('rdApp.*')

from gu_decode_02 import GetMFPidx, GetMFP

#Hyperparameter Setting
radius = 2
decay = 0.01
maxlen = 10
frequent_regulate = 5

#Path Information
library_path = './data/fp/toxin/ammonia2/library_' + str(frequent_regulate) + '_' + str(radius)
predict_reaction_path = './data/fp/toxin/ammonia2/test_positive'
model_save_path = './data/fp/toxin/ammonia2/trained'
prediction_save_path = './data/fp/toxin/ammonia2/'

#read fingerprint library data
f = open(library_path, 'r', encoding = 'UTF8')
library = []
while True:
    line = f.readline()
    if not line: break
    cv_inf = ast.literal_eval(line)
    library.append(cv_inf)
f.close()

final_result = []

fp_len = len(library[0])
print(fp_len)

#model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size = (fp_len,1), stride = fp_len)
        self.conv2 = nn.Conv2d(1, 128, kernel_size = (256,1), stride = 256)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,1)

    def forward(self, x):
        x = x.view(1,1,int(fp_len*maxlen),1)
        x = F.relu(self.conv1(x))
        x = x.view(1,1,256*maxlen,1)
        x = self.conv2(x)
        x = x.view(1,1,128,maxlen)
        x = F.max_pool2d(x, kernel_size = (1,maxlen))
        x = x.view(128)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
 
print(Net())

#binary classifier cross entropy error
criterion = nn.BCELoss()

print("reading reaction data")
reaction_list = []
f = open(predict_reaction_path, 'r', encoding = 'UTF8')

while True:
    line = f.readline()
    if not line: break
    reaction =ast.literal_eval(line)
    reaction_list.append(reaction[0])
f.close()

uniqueFPs = library[0]

reaction_fps = []
for i in reaction_list:
    reaction = i
    fp_reactant = []
    for reactant in reaction:
        X = np.zeros(fp_len)
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
    reaction_fps.append(fp_reactant)

fp_list = []
for reaction in reaction_fps:
    in_des = []
    for reactants in reaction:
        in_des = in_des + reactants
    while len(in_des) < fp_len*maxlen:
        in_des += [0]*fp_len
    fp_list.append(torch.FloatTensor(in_des))

PATH = model_save_path + '.pth'

net = Net()
net.load_state_dict(torch.load(PATH))

g = open(prediction_save_path + 'predictions', 'w')
    #print final result
for idx, i in enumerate(fp_list):
    g.write(str([reaction_list[idx], float(net(i))]))
    g.write('\n')
print("finished")
g.close()
