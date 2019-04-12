# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 02:44:04 2019

@author: Frank
"""

# Importing libraries
import csv
import string
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import time
from torch.distributions.categorical import Categorical
import torch
import random
import math
import json

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pltsample

all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
n_letters = len(all_letters) + 1 # Plus EOS marker
n_categories = 1

learning_rate = 0.0001
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('cuda',use_cuda)  
#train_on_gpu = torch.cuda.is_available()
#if(train_on_gpu):
#    print('Training on GPU!')
#else: 
#    print('No GPU available, training on CPU; consider making n_epochs very small.')

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)



def get_data():
    category_lines = {}
    all_categories = ['st']
    category_lines['st']=[]
    filterwords=['NEXTEPISODE']
    
    with open('star_trek_transcripts_all_episodes.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            for el in row:
                if (el not in filterwords) and (len(el)>1):
#                     print(el)
                    v = el.strip().replace(';','').replace('\"','') #.replace('=','') #.replace('/','').replace('+','') 
                    category_lines['st'].append(v)
        
    print(len(all_categories), len(category_lines['st']))
    print('done')
    return category_lines, all_categories


def TrainingExample(line):
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return input_line_tensor, target_line_tensor

# Declaring the model
class CharRNN(nn.Module):
    
    def __init__(self,n_input, n_hidden=256, n_layers=2, lr=0.001):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.lr = lr
        #define the LSTM
        self.lstm = nn.LSTM(self.n_input, self.n_hidden, self.n_layers, 
                            batch_first=True)
        
        #define a dropout layer
        self.dropout = nn.Dropout(0.1)
        
        #define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, self.n_input)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
    def initHidden(self, device):
        return ((torch.zeros(self.n_layers, 1, self.n_hidden).to(device)),
                (torch.zeros(self.n_layers, 1, self.n_hidden).to(device)))
        
    def forward(self, x, hidden,c):
       
        #get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        
        #pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)
        
        #put x through the fully-connected layer
        out = self.fc(out)
        
        out = self.softmax(out/c)
        
            
            # return the final output and the hidden state
        return out, hidden
        
        
# Declaring the train method
def train(net, data, lr, print_every,start,device,final_output):
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    
    criterion = nn.NLLLoss()
    iter = 0
    
    total_train_loss = 0
#    n_chars = len(net.chars)
        # initialize hidden state
    h = net.initHidden(device)
    
    for line in data:
        opt.zero_grad()
        loss = 0
        iter += 1
        input_line_tensor, target_line_tensor = TrainingExample(line)
        target_line_tensor.unsqueeze_(-1)
        input_line_tensor, target_line_tensor = input_line_tensor.to(device), target_line_tensor.to(device)
        
        for i in range(input_line_tensor.size(0)):
            output, hidden = net(input_line_tensor[i], h,1)
#            print(output.shape)
#            print(target_line_tensor[i].shape)
            l = criterion(output, target_line_tensor[i])
            loss += l
            total_train_loss += l
            
        loss.backward()
        opt.step()
        if iter % print_every == 0:
            print('%s (%d %d%%)' % (timeSince(start), iter, iter / len(data) * 100))
            print('Samples:')
            start_letter = ''.join(random.choice(string.ascii_uppercase) for _ in range(15))
            generated = samples(net, start_letter ,device)
            net.train()
            final_output += '%s (%d %d%%)' % (timeSince(start), iter, iter / len(data) * 100) + '\n'
            final_output += 'Samples:' + '\n' 
            final_output += generated + '\n'

    return total_train_loss.item()/len(data) , final_output

def test(test_data, net,device):
    net.eval()
    criterion = nn.NLLLoss()
    with torch.no_grad():
        loss = 0
        correct = 0
        total = 0
        total_loss = 0
        for line in test_data:
            input_line_tensor, target_line_tensor = TrainingExample(line)
            target_line_tensor.unsqueeze_(-1)
            hidden = net.initHidden(device)
            loss = 0
            input_line_tensor, target_line_tensor = input_line_tensor.to(device), target_line_tensor.to(device)
            for i in range(input_line_tensor.size(0)):
                total += 1
                output, hidden = net(input_line_tensor[i], hidden,1)
                prediction = torch.exp(output)
                m = Categorical(prediction)
                topi = m.sample()[0]
                if int(topi) == int(target_line_tensor[i][0]):
                    correct += 1
                l = criterion(output, target_line_tensor[i])
                loss += l
            total_loss += loss
    return total_loss.item()/len(test_data), correct/total
       
        
# Sample from a category and starting letter
def sample(net,start_letter,device):
    net.eval()
    with torch.no_grad():  # no need to track history in sampling
#         category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = net.initHidden(device)

        output_name = start_letter
        input = input.to(device)
        for i in range(max_length):
            output, hidden = net(input[0], hidden,0.5)
#             topv, topi = output.topk(1)
#             topi = topi[0][0]
            output = torch.exp(output)
            m = Categorical(output)
            topi = m.sample()[0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter).to(device)
        
        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(net,start_letters,device):
    output = ''
    for start_letter in start_letters:
#         print(start_letter)
        letter_output = sample(net,start_letter,device)
        output += letter_output + '\n'
    print(output)
    return output
        

speech, category = get_data()

X_train, X_test = train_test_split(speech['st'],test_size = 0.2)          
                
n_hidden=200
n_layers=2

net = CharRNN(n_letters, n_hidden, n_layers)
print(net)
start_time = time.time()
n_epochs = 5 # start smaller if you are just testing initial behavior
net.to(device)
train_losses = []
total_loss = 0 # Reset every plot_every iters
test_losses = []
accuracy_ls = []
max_length = 150

start_letter = "BCDFGHJKLMNOPQRSTUVWX"
for e in range(1,1+n_epochs):
    final_output = ''
    print("EPOCH " + str(e) +" is starting")
    final_output += "EPOCH " + str(e) +" is starting" + '\n'
    train_loss,train_output = train(net, X_train,0.001, 3000,start_time,device,final_output)
    train_losses.append(train_loss)
    print('%s Epoch Train Loss: %.4f' % (timeSince(start_time), train_loss))
    train_output += '%s Epoch Train Loss: %.4f' % (timeSince(start_time), train_loss) + '\n'
    total_loss = 0
    
    test_loss,accuracy = test(X_test,net,device)
    test_losses.append(test_loss)
    accuracy_ls.append(accuracy)
    
    print('%s Epoch Test Loss: %.4f, Accuracy: %.4f ' % (timeSince(start_time), test_loss, accuracy*100))
    print('Epoch Samples:')
    train_output += '%s Epoch Test Loss: %.4f, Accuracy: %.4f ' % (timeSince(start_time), test_loss, accuracy*100) + '\n'
    train_output += 'Epoch Samples:' + '\n'
    epoch_generated = samples(net,start_letter ,device)
    train_output += epoch_generated + '\n'
    best_model_wts = net.state_dict()
    torch.save(best_model_wts,'./model{}.pt'.format(e))
#    print(train_output)
    f = open('output_epoch{}.txt'.format(e),'w')
    f.write(train_output)
    f.close()

plt.figure()
plt.title('Train Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_losses)
plt.show()


plt.figure()
plt.title('Test Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(test_losses)
plt.show()


plt.figure()
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(accuracy_ls)
plt.show()



    
