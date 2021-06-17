import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy
import numpy as np
from numpy import array
import ast
import random

#Hyperparameter Setting
radius = 2
decay = 0.01
maxlen = 10
frequent_regulate = 5

#Path Information
library_path = './data/fp/toxin/ammonia2/library_' + str(frequent_regulate) + '_' + str(radius)
fp_test_positive_path = './data/fp/toxin/ammonia2/fp_test_positive_2_5'
fp_test_negative_path = './data/fp/toxin/ammonia2/fp_test_negative_2_5'
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
    #optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr = 0.0001, betas = (0.9,0.999) )

print("reading test_positive data")
reaction_list = []
f = open(fp_test_positive_path, 'r', encoding = 'UTF8')

while True:
    line = f.readline()
    if not line: break
    reaction =ast.literal_eval(line)
    reaction_list.append([reaction[0], reaction[-1]])
f.close()

test_positive = []
for reaction in reaction_list:
    in_des = []
    for reactants in reaction[0]:
        in_des = in_des + reactants
    while len(in_des) < fp_len*maxlen:
        in_des += [0]*fp_len
    test_positive.append([torch.FloatTensor(in_des), reaction[1]])


print("reading test_negative data")
reaction_list = []
f = open(fp_test_negative_path, 'r', encoding = 'UTF8')

while True:
    line = f.readline()
    if not line: break
    reaction =ast.literal_eval(line)
    reaction_list.append([reaction[0], reaction[-1]])
f.close()

test_negative = []
for reaction in reaction_list:
    in_des = []
    for reactants in reaction[0]:
        in_des = in_des + reactants
    while len(in_des) < fp_len*maxlen:
        in_des += [0]*fp_len
    test_negative.append([torch.FloatTensor(in_des), reaction[1]])

PATH = model_save_path + '.pth'

net = Net()
net.load_state_dict(torch.load(PATH))


g = open(prediction_save_path + 'final_accuracy', 'w')
    #print final result
num_success = 0
for i in test_positive:
    if float(net(i[0])) >= 0.5:
        num_success = num_success + 1
print("toxic production test acc")
print(len(test_positive))
print(num_success)
pos_error = num_success/len(test_positive)

num_success = 0
for i in test_negative:
    if float(net(i[0])) < 0.5:
        num_success = num_success + 1
print("failed test acc")
print(len(test_negative))
print(num_success)
fail_error = num_success/len(test_negative)

final_result.append([pos_error, fail_error])
print("finished")

g.write(str(final_result))
g.close()
