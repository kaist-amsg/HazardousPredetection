import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy
import numpy as np
from numpy import array
import ast
import random

import sklearn
from sklearn import metrics

#Hyperparameter Setting
radius = 2
decay = 0.01
maxlen = 10
frequent_regulate = 5

#Path Information
library_path = './data/fp/toxin/ammonia/library_' + str(frequent_regulate) + '_' + str(radius)
train_path = './data/fp/toxin/ammonia/fp_training_2_5'
validation_positive_path = './data/fp/toxin/ammonia/fp_valid_positive_2_5'
validation_negative_path = './data/fp/toxin/ammonia/fp_valid_negative_2_5'
model_save_path = './data/fp/toxin/ammonia/trained'

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

#initialize weight of model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)

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

#train model
print("start training")

#read positive data
print("reading training data")
reaction_list = []
f = open(train_path, 'r', encoding = 'UTF8')

while True:
    line = f.readline()
    if not line: break
    reaction =ast.literal_eval(line)
    reaction_list.append([reaction[0], reaction[-1]])
f.close()

training_data = []
for reaction in reaction_list:
    in_des = []
    for reactants in reaction[0]:
        in_des = in_des + reactants
    while len(in_des) < fp_len*maxlen:
        in_des += [0]*fp_len
    training_data.append([torch.FloatTensor(in_des), reaction[1]])

print("reading validation data")
reaction_list = []
f = open(validation_positive_path, 'r', encoding = 'UTF8')

while True:
    line = f.readline()
    if not line: break
    reaction =ast.literal_eval(line)
    reaction_list.append([reaction[0], reaction[-1]])
f.close()

val_pos = []
for reaction in reaction_list:
    in_des = []
    for reactants in reaction[0]:
        in_des = in_des + reactants
    while len(in_des) < fp_len*maxlen:
        in_des += [0]*fp_len
    val_pos.append([torch.FloatTensor(in_des), reaction[1]])

reaction_list = []
f = open(validation_negative_path, 'r', encoding = 'UTF8')

while True:
    line = f.readline()
    if not line: break
    reaction =ast.literal_eval(line)
    reaction_list.append([reaction[0], reaction[-1]])
f.close()

val_neg = []
for reaction in reaction_list:
    in_des = []
    for reactants in reaction[0]:
        in_des = in_des + reactants
    while len(in_des) < fp_len*maxlen:
        in_des += [0]*fp_len
    val_neg.append([torch.FloatTensor(in_des), reaction[1]])

val_list = val_pos + val_neg

dataset = training_data
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=True, num_workers=1)

    #retrain model when every output is 1 or 0
val_error = 0.0
while val_error == 0.0:
    print("set inital weights")
        #initialize model
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr = 0.001, betas = (0.9,0.999), weight_decay = decay)
    net.apply(weights_init)
    val_past = 0.00
    count = 0
        #train model
    for j,data in enumerate(dataloader, 0):
        i = training_data[j]

        input = i[0]
        output = net(input)
        target = torch.FloatTensor([float(i[1])])

        net.zero_grad()
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #get validation error
    num_success = 0
    num_tot = 0
    for val_dat in val_pos:
        num_tot += 1
        if float(net(val_dat[0])) >= 0.5:
            num_success = num_success + 1
    val_error = num_success/num_tot
    print("initial validation accuracy is "+ str(val_error))

    #train model with batch size 6
print("start train with 6 batches")
optimizer = optim.Adam(net.parameters(), lr = 0.0001, betas = (0.9,0.999), weight_decay = decay)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle=True, num_workers=4)
count = 0
auc_past = 0
for epoch in range(100):
    epoch_ = epoch
    for j,data in enumerate(dataloader, 0):
        i = training_data[j]

        input = i[0]
        output = net(input)
        target = torch.FloatTensor([float(i[1])])

        net.zero_grad()
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #get validation error
    y_true = []
    y_score = []
    for val_dat in val_list:
        y_true.append(float(val_dat[-1]))
        y_score.append(float(net(val_dat[0])))
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    fpr, tpr,thresholds = metrics.roc_curve(y_true, y_score, pos_label = 1)
    auc_val = metrics.auc(fpr, tpr)
    print(str(epoch) + "th epoch validation accuracy auc "+ str(auc_val))
        #with val, early stop
    if auc_val <= auc_past:
        auc_past = auc_val
        break
    else:
        auc_past = auc_val

    #train model with batch size 1
print("start single batch training")
optimizer = optim.Adam(net.parameters(), lr = 0.0001, betas = (0.9,0.999), weight_decay = decay)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=True, num_workers=4)
count = 0
for epoch in range(epoch_+1, 100):
    for j,data in enumerate(dataloader, 0):
        i = training_data[j]

        input = i[0]
        output = net(input)
        target = torch.FloatTensor([float(i[1])])

        net.zero_grad()
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #get validation error
    y_true = []
    y_score = []
    for val_dat in val_list:
        y_true.append(float(val_dat[-1]))
        y_score.append(float(net(val_dat[0])))
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    fpr, tpr,thresholds = metrics.roc_curve(y_true, y_score, pos_label = 1)
    auc_val = metrics.auc(fpr, tpr)
    print(str(epoch) + "th epoch validation accuracy auc "+ str(auc_val))

        #with val, learning rate decaying
    if auc_val <= auc_past:
        auc_past = auc_val
        if count == 0:
            optimizer = optim.Adam(net.parameters(), lr = 0.00005, betas = (0.9,0.999), weight_decay = decay)
            print("switch optimizer")
            count = 1
        #early stop
        else:
            break
    else:
        auc_past = auc_val

    #save trained model
PATH = model_save_path + '.pth'
torch.save(net.state_dict(), PATH)

print("finished")
